"""Per-layer LeapQuant state: the 8-bit checkpoint lives inside vLLM's fp32 state slot, the window buffer next to it.

Layout of one (slot, head) inside vLLM's fp32 [V, K] state slot:
    int8 residual [V][K] | smoothing scales Sk f32[K], Sv f32[V] | Compensator Tokens U^T | Q^T fp16 [R][K+V]
The window buffer (keys, values, weights of the last L tokens) and the per-slot counters are separate tensors.
"""
import functools

import torch

from leapquant import kernels as KN

_make_dequant = functools.lru_cache(maxsize=None)(KN.make_dequant)
_make_quantize = functools.lru_cache(maxsize=None)(KN.make_quantize)

WINDOW = 16          # tokens per window (L)
RANK = 4             # Compensator Tokens per head (R)


def _ident(t):
    """Same storage + same view -> same layer state."""
    return (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()))


def _view(ssm, dtype, shape, stride, offset):
    return torch.empty(0, dtype=dtype, device=ssm.device).set_(ssm.untyped_storage()).as_strided(shape, stride, offset)


def state_views(ssm, R):
    """All views of the 8-bit checkpoint aliasing vLLM's fp32 state tensor [NS, HV, V, K] (no allocation)."""
    NS, HV, V, K = ssm.shape
    SS = ssm.stride(0); off = ssm.storage_offset(); sc = off + V * K // 4
    return dict(
        Sq=_view(ssm, torch.int8, (NS, HV, V, K), (4 * SS, 4 * V * K, K, 1), 4 * off),
        q32=_view(ssm, torch.int32, (NS, HV, V, K // 4), (SS, V * K, K // 4, 1), off),
        sq16=_view(ssm, torch.int16, (NS, HV, V, K // 2), (2 * SS, 2 * V * K, K // 2, 1), 2 * off),
        Sk=_view(ssm, torch.float32, (NS, HV, K), (SS, V * K, 1), sc),
        Sv=_view(ssm, torch.float32, (NS, HV, V), (SS, V * K, 1), sc + K),
        skv=_view(ssm, torch.float32, (NS, HV, K + V), (SS, V * K, 1), sc),
        uq=_view(ssm, torch.float16, (NS, HV, R, K + V), (2 * SS, 2 * V * K, K + V, 1), 2 * (sc + K + V)))


class LeapQuantState:
    """Window buffers + compiled kernels of one linear-attention layer.

    kda=False: Gated DeltaNet (Qwen).  kda=True: Kimi Delta Attention (GLM), gate_lower_bound = GLM's bounded gate.
    nmax: the most rows one decode call can carry (sizes the flush kernel's shared-memory work list).
    """

    def __init__(self, ssm, H, kda=False, gate_lower_bound=0.0, nmax=0, L=WINDOW, R=RANK):
        assert ssm.dtype == torch.float32, f"LeapQuant needs the fp32 recurrent state cache, got {ssm.dtype}"
        NS, HV, V, K = ssm.shape
        SS = ssm.stride(0)
        assert ssm.stride()[1:] == (V * K, K, 1), f"state inner dims must be packed, got strides {ssm.stride()}"
        assert V * K // 4 + K + V + R * (K + V) // 2 <= V * K, "checkpoint must fit inside the fp32 slot"
        dev = ssm.device
        self.key = _ident(ssm)
        self.kda, self.NS, self.HV, self.H, self.K, self.V, self.L, self.R = kda, NS, HV, H, K, V, L, R
        KA = K if kda else 1

        self.Kbuf = torch.zeros(NS, HV, L, K, device=dev, dtype=torch.bfloat16)
        self.Ubuf = torch.zeros(NS, HV, L, V, device=dev, dtype=torch.bfloat16)
        self.Wbuf = torch.zeros(NS, HV, L, device=dev)
        self.Pbuf = torch.zeros(NS, HV, KA, device=dev) if kda else torch.ones(NS, HV, 1, device=dev)
        # KDA: each entry's per-step decay factor exp(g_j) (fp16); GDN: unused gate argument of the prefill kernel
        self.Gbuf = (torch.ones(NS, HV, L, KA, device=dev, dtype=torch.float16) if kda
                     else torch.zeros(NS, HV, L, 1, device=dev))
        self.Gd = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16, device=dev)
        self.hcnt = torch.zeros(NS, HV, dtype=torch.int32, device=dev)
        self.Q0 = torch.randn(V, R, generator=torch.Generator().manual_seed(0)).to(dev)

        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        if kda:
            self.kstep = KN.make_kda_step(NS, HV, H, K, V, L, SS, R=R, SMS=sms, NSTAGE=6, NSTAGEQ=4, NCG=2, NNW=3,
                                             GLB=float(gate_lower_bound))
        else:
            pipe = dict(NSTAGE=6, NCG=3, NNW=2) if (K, V) == (128, 128) else {}
            self.kstep = KN.make_gdn_step(NS, HV, H, K, V, L, SS, R=R, SMS=sms, **pipe)
        self.kflush = KN.make_flush(NS, HV, K, V, L, SS, R=R, ITERS=1, SMS=sms, NCG=3, KDA=kda, NMAX=nmax)
        if not kda:                                   # KDA prefill round trip runs in torch (_kda_rebuild / _kda_checkpoint)
            self.kdeq = _make_dequant(NS, HV, K, V, L, SS, buf_dtype=KN.BF16, R=R)
            self.kquant = _make_quantize(NS, HV, K, V, L, SS, R=R, ITERS=1)
        self._vkey = None
        self._f32 = {}

    def _p32(self, t):
        """fp32 copy of a layer parameter, cached per tensor (several layers can share one state tensor)."""
        c = self._f32.get(t.data_ptr())
        if c is None:
            c = self._f32[t.data_ptr()] = t.detach().to(torch.float32).contiguous()
        return c

    def views(self, ssm):
        k = _ident(ssm)
        if self._vkey != k:
            self._vkey, self._v = k, state_views(ssm, self.R)
        return self._v

    def decode(self, ssm, mixed, a, b, A_log, dt_bias, idx, out):
        """One decode token per row.  a: GDN (N, HV) | KDA (N, HV, K) raw gate;  dt_bias: GDN (HV,) | KDA (HV*K,)."""
        A_log = A_log if A_log.dtype == torch.float32 else self._p32(A_log)
        dt_bias = dt_bias if dt_bias.dtype == torch.float32 else self._p32(dt_bias)
        KA = self.K if self.kda else 1
        a = a.reshape(a.shape[0], self.HV, KA).contiguous(); dt_bias = dt_bias.reshape(self.HV, KA).contiguous()
        idx = idx if idx.dtype == torch.int32 else idx.to(torch.int32)
        v = self.views(ssm)
        if self.kda:
            self.kstep(v["q32"], v["Sk"], v["Sv"], v["skv"], self.Kbuf, self.Ubuf, self.Wbuf, self.Pbuf, self.Gbuf, self.hcnt,
                       mixed, a, b, A_log, dt_bias, v["uq"], idx, out)
            G = self.Gbuf.view(torch.bfloat16)
        else:
            self.kstep(v["q32"], v["Sk"], v["Sv"], v["skv"], self.Kbuf, self.Ubuf, self.Wbuf, self.Pbuf, self.hcnt,
                       mixed, a, b, A_log, dt_bias, v["uq"], idx, out)
            G = self.Gd
        self.kflush(v["q32"], v["Sk"], v["Sv"], v["skv"], self.Kbuf, self.Ubuf, self.Wbuf, self.Pbuf, G, self.hcnt,
                    v["sq16"], v["uq"], self.Q0, idx)

    def _kda_rebuild(self, ssm, sl):
        """KDA: S = exp(P) * (Q U^T + deq(res)) + sum_j w_j u_j (x) (k_j * D_j), D_j = product of the later per-step decays."""
        v = self.views(ssm)
        S0 = (torch.einsum("nhrv,nhrc->nhvc", v["uq"][sl][..., self.K:].float(), v["uq"][sl][..., :self.K].float())
              + v["Sq"][sl].float() * (v["Sk"][sl][..., None, :] / 127.0) * v["Sv"][sl][..., :, None])
        E = self.Gbuf[sl].float(); h = self.hcnt[sl]
        E = torch.where(torch.arange(self.L, device=E.device)[None, None, :, None] < h[..., None, None], E, torch.ones_like(E))
        D = torch.flip(torch.cumprod(torch.flip(E, [2]), 2), [2]) / E
        kdec = self.Kbuf[sl].float() * D
        return (torch.exp(self.Pbuf[sl])[..., None, :] * S0
                + torch.einsum("nhj,nhjv,nhjc->nhvc", self.Wbuf[sl], self.Ubuf[sl].float(), kdec))

    def _kda_checkpoint(self, ssm, sl):
        """KDA: 8-bit checkpoint of the fp32 states in the slots sl (Compensator Tokens + smoothed INT8 residual)."""
        v = self.views(ssm)
        S = ssm[sl].float()                                       # [n, HV, V, K]
        U, Q = KN.lowrank_factor(S.transpose(-1, -2), self.R)    # rank-R fit in the [K, V] orientation
        Uh, Qh = U.half(), Q.half()
        res = S - (Qh.float() @ Uh.float().transpose(-1, -2))
        sk = torch.sqrt(res.abs().mean(-2)).clamp_min(1e-12)     # smoothing scales
        sv = (res.abs() / sk[..., None, :]).amax(-1).clamp_min(1e-12)
        v["Sq"][sl] = torch.clamp(torch.round(res / (sk[..., None, :] * sv[..., :, None]) * 127.0), -127, 127).to(torch.int8)
        v["Sk"][sl] = sk; v["Sv"][sl] = sv
        v["uq"][sl] = torch.cat([Uh.transpose(-1, -2), Qh.transpose(-1, -2)], -1)
        self.Wbuf[sl] = 0.0; self.Pbuf[sl] = 0.0; self.Gbuf[sl] = 1.0; self.hcnt[sl] = 0

    def dequantize(self, ssm, idx):
        """Materialize the fp32 state of the slots idx in place (before vLLM's own prefill kernel runs on them)."""
        if idx.numel() == 0:
            return
        if self.kda:
            sl = idx.to(torch.int64)
            ssm[sl] = self._kda_rebuild(ssm, sl)
            return
        v = self.views(ssm)
        self.kdeq(ssm, v["Sq"], v["Sk"], v["Sv"], self.Kbuf, self.Ubuf, self.Wbuf, self.Pbuf, self.Gbuf, self.hcnt,
                  v["uq"], idx.to(torch.int32).contiguous())

    def quantize(self, ssm, idx):
        """Write the 8-bit checkpoint of the fp32 states in the slots idx and reset their windows (after a prefill)."""
        if idx.numel() == 0:
            return
        if self.kda:
            self._kda_checkpoint(ssm, idx.to(torch.int64))
            return
        v = self.views(ssm)
        self.kquant(ssm, v["Sq"], v["Sk"], v["Sv"], v["uq"], self.Wbuf, self.Pbuf, self.hcnt, self.Q0,
                    idx.to(torch.int32).contiguous())


_STATES = {}
_GEN = [0]


def get_state(layer, ssm, **kw):
    """The LeapQuantState of `layer` for its current state tensor (rebuilt when vLLM swaps the cache after profiling)."""
    st = layer.__dict__.get("_leapquant")
    k = _ident(ssm)
    if st is not None and st.key == k:
        return st
    if st is not None and st.gen == _GEN[0]:            # cache re-allocated: drop all buffers of the old one
        _GEN[0] += 1; _STATES.clear(); torch.cuda.empty_cache()
    st = _STATES.get(k)
    if st is None:
        cc = getattr(layer, "cache_config", None)
        if cc is not None and getattr(cc, "mamba_cache_mode", "none") != "none":
            raise RuntimeError("LeapQuant keeps each sequence's window buffer next to its state slot and does not support "
                               "prefix caching of the recurrent state: run vLLM with --no-enable-prefix-caching")
        st = _STATES[k] = LeapQuantState(ssm, **kw)
        st.gen = _GEN[0]
    layer._leapquant = st
    return st
