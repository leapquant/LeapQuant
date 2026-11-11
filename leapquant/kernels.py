"""TileLang kernels of LeapQuant's 8-bit recurrent state (rank-4 Compensator Tokens + smoothed INT8 residual, window L = 16).

  make_gdn_step      GDN decode step: reads the low-bit state + the window's buffered updates, appends the new token
  make_kda_step      KDA decode step (per-key-channel gate)
  make_flush         end-of-window flush: rebuilds the state, extracts the Compensator Tokens, smooths and re-quantizes the residual
  make_dequant / make_quantize    fp32 round trip of the slots touched by a prefill
"""
import functools
import torch, tilelang, tilelang.language as T
from tilelang.layout import make_swizzled_layout

F32, BF16, I32, I8 = "float32", "bfloat16", "int32", "int8"
QMAX = 127.0


def _ring_types(NS, HV, L, K, V, KA=1):
    """Parameter types of the window buffer (Kbuf, Ubuf, Wbuf, Pbuf, hcnt)."""
    return (T.Tensor((NS, HV, L, K), BF16), T.Tensor((NS, HV, L, V), BF16), T.Tensor((NS, HV, L), F32),
            T.Tensor((NS, HV, KA), F32), T.Tensor((NS, HV), I32))

SOFTPLUS_THRESHOLD = 20.0

EPS = 1e-12

STEP_CFG = {(128, 128): (6, 3, 2), (64, 128): (10, 5, 5), (64, 64): (8, 4, 4)}

@tilelang.jit(pass_configs={"tl.disable_thread_storage_sync": True})
def make_gdn_step(NS, HV, H, K, V, L, SS, R=4, SMS=148, NSTAGE=None, NCG=None, NNW=None, scale=None, MAXIT=256):
    """Gated DeltaNet decode step on the 8-bit state.

    The state at the start of the window is S0 = Q U^T + deq(residual): R fp16 Compensator Tokens (U^T | Q^T, [R][K+V]) plus an
    INT8 residual with smoothing scales Sk / Sv.  Each step reads S0 and the window buffer (the bf16 keys / values of the tokens
    seen since the last flush), computes the output, and appends the new token to the buffer; the Compensator Tokens are handled
    exactly like R extra buffer entries with weight 1.  The flush (make_flush) rebuilds and re-quantizes full windows.

    Persistent, warp-specialised kernel, one CTA per SM looping over (sequence, head) programs:
      producer warp 0 : TMA ring of the int8 tile, scales, factors and buffer into NSTAGE shared-memory stages
      norm warps      : l2-norm statistics, normalized k / q, the fp16 mma operand [kf, qf] / max, buffer weights, scalars
      consumers       : int8 tile -> fp16 (exact) -> mma.m16n8k16 against [kf qf]; buffer dots; output; buffer appends
    Numerics: kf / qf are fp16 after scaling by 1 / max|kf, qf|, the int8 -> fp16 conversion is exact, accumulation is fp32.
    Head dims (K, V) in {(128, 128), (64, 128), (64, 64)}; NSTAGE / NCG / NNW default to STEP_CFG[(K, V)]."""
    if scale is None:
        scale = K ** -0.5
    cfg = STEP_CFG.get((K, V), (6, 3, 2))
    NSTAGE = cfg[0] if NSTAGE is None else NSTAGE
    NCG = cfg[1] if NCG is None else NCG
    NNW = cfg[2] if NNW is None else NNW
    GQA = HV // H
    N = T.symbolic("N")
    KW = K // 4
    HP = 128 // V                                   # heads per program: the 128 consumer rows are HP heads x V rows
    HVP = HV // HP                                  # programs per token
    NI = K // 32                                    # lane-strided columns per lane (norm warp, factor dots)
    NKC = (K // 16) * 20                            # kn_s / qn_s length (16-column chunks padded to 20)
    NCONS = 128 * NCG
    assert NSTAGE % NCG == 0
    SPG = NSTAGE // NCG
    assert NSTAGE % NNW == 0, "every stage must be normed by ONE warp in fill order (program it -> warp it % NNW): otherwise a stage's fills alternate between warps, and a warp >= 2 fills ahead passes its mbarrier parity wait early (phase aliasing) -> stale data / deadlock"
    NPROD = 32 * (1 + NNW)                          # TMA warp + NNW norm warps
    MIX = 2 * H * K + HV * V
    F16 = "float16"
    assert (K, V) in ((128, 128), (64, 128), (64, 64)), f"make_gdn_step: (K, V) = ({K}, {V}), supported (128, 128) / (64, 128) / (64, 64)"
    assert (L <= 16 or L == 32) and (R <= 4 or (R % 4 == 0 and R <= 16))
    assert K == 128 or (L == 16 and R <= 4), "K = 64: L = 16 and R <= 4 only (the window / rank ablation paths are K = 128)"
    assert HP == 1 or (HV % HP == 0 and GQA % HP == 0), "V = 64 packs heads hv, hv + 1 of ONE key head: needs HV and HV // H even"
    MAGICX = T.int32(-2139062144)                  # 0x80808080: int8 -> offset byte
    H2MAG = T.int32(0x64646464)                    # fp16 exponent byte: 0x64xx = 1024 + xx
    UQS_SHAPE = (NSTAGE, R, K + V) if HP == 1 else (NSTAGE, HP, R, K + V)
    SKV_SHAPE = (NSTAGE, K + V) if HP == 1 else (NSTAGE, HP, K + V)
    UBS_SHAPE = (NSTAGE, L, V) if HP == 1 else (NSTAGE, HP, L, V)
    WBS_SHAPE = (NSTAGE, 32) if HP == 1 else (NSTAGE, HP, L)
    WS_SHAPE = (NSTAGE, L) if HP == 1 else (NSTAGE, HP, L)
    DS_SHAPE = (NCG, 2, L + R) if HP == 1 else (NCG, 2, HP, L + R)
    DKQ_SHAPE = (NCG, 2, 2 * (L + R)) if HP == 1 else (NCG, 2, HP, 2 * (L + R))
    NQL, QLW = V, KW       # the int8 residual tile as V lines of K / 4 words
    SKVT = F32
    KB_T, UB_T, WB_T, PB_T, HC_T = _ring_types(NS, HV, L, K, V)

    @T.prim_func
    def step(Sq: T.StridedTensor((NS, HV, NQL, QLW), (SS, V * K, QLW, 1), I32),
             Sk: T.StridedTensor((NS, HV, K), (SS, V * K, 1), F32),
             Sv: T.StridedTensor((NS, HV, V), (SS, V * K, 1), F32),
             Skv: T.StridedTensor((NS, HV, K + V), (SS, V * K, 1), SKVT),
             Kbuf: KB_T, Ubuf: UB_T,
             Wbuf: WB_T, Pbuf: PB_T, hcnt: HC_T,
             mixed: T.Tensor((N, MIX), BF16), a: T.Tensor((N, HV, 1), BF16), b: T.Tensor((N, HV), BF16),
             A_log: T.Tensor((HV,), F32), dt_bias: T.Tensor((HV, 1), F32),
             UQ: T.StridedTensor((NS, HV, R, K + V), (2 * SS, 2 * V * K, K + V, 1), F16),
             idx: T.Tensor((N,), I32), o: T.Tensor((N, HV, V), BF16)):
        with T.Kernel(SMS, threads=NPROD + NCONS) as bid:
            Qs = T.alloc_shared((NSTAGE * HP, NQL, QLW), I32)       # HP = 2: tile s * HP + hd holds head hv + hd of stage s
            UQ_s = T.alloc_shared((UQS_SHAPE), F16)
            skv_s = T.alloc_shared(SKV_SHAPE, SKVT)
            qx_s = T.alloc_shared((NSTAGE, K), BF16)
            kx_s = T.alloc_shared((NSTAGE, K), BF16)
            vx_s = T.alloc_shared((NSTAGE, HP * V), BF16)
            Kb_s = T.alloc_shared((NSTAGE * HP, L, K), BF16)
            Ub_s = T.alloc_shared(UBS_SHAPE, BF16)
            if L >= 8:
                T.annotate_layout({Kb_s: make_swizzled_layout(Kb_s)})     # the buffer dots read 32 B per lane over 4 rows per warp: 4-way bank conflicts unswizzled
                                                                     # (UQ_s cannot be swizzled: its 512 B rows are not the TMA swizzle granularity)
            Wb_s = T.alloc_shared(WBS_SHAPE, F32)
            # ---- per-stage data produced by the norm warp ----
            bf_s = T.alloc_shared((NSTAGE, 8, K + 16), F16)   # B operand: rows 0/1 = hi(kf, qf)/kfmax, 2/3 = lo parts, 4..7 = 0 (HP = 2: head 1's 4 rows)
                                                             # (+16: row stride 2K + 32 B -> the 8 r0 rows of a fragment load land on different banks)
            kn_s = T.alloc_shared((NSTAGE, NKC), F32)        # chunk-padded: column c at (c // 16) * 20 + c % 16 (conflict-free 64 B chunk reads)
            qn_s = T.alloc_shared((NSTAGE, NKC), F32)
            w_s = T.alloc_shared(WS_SHAPE, F32)
            sc_s = T.alloc_shared((NSTAGE, 8), F32)           # 4 hd + (0 kfmax, 1 kq, 2 pn, 3 beta)
            # ---- per-group scratch, double-buffered by program parity (no end-of-program barrier) ----
            mqy_s = T.alloc_shared((NCG, 2, HP * V, 4), F32)   # [mq_hi, yq_hi, mq_lo, yq_lo] per row
            if K == 128:
                dk_s = T.alloc_shared(DS_SHAPE, F32)
                dq_s = T.alloc_shared(DS_SHAPE, F32)
            else:                                            # K = 64: [w_j dk_j, w_j dq_j] at 2 j, [dU_r, dQ_r] at 2 (L + r) (two entries per 16 B row-loop load)
                dkq_s = T.alloc_shared(DKQ_SHAPE, F32)
            slot_s = T.alloc_shared((MAXIT,), I32)
            h_s = T.alloc_shared((MAXIT * HP,), I32)           # per head: [it * HP + hd]
            P_s = T.alloc_shared((MAXIT * HP,), F32)
            a_s = T.alloc_shared((MAXIT * HP,), F32)
            b_s = T.alloc_shared((MAXIT * HP,), F32)
            loaded = T.alloc_barrier([32] * NSTAGE)
            normed = T.alloc_barrier([32] * NSTAGE)
            consumed = T.alloc_barrier([128] * NSTAGE)
            T.annotate_layout({Qs: make_swizzled_layout(Qs)})
            tx = T.get_thread_binding()
            nprog = N * HVP
            nit = T.ceildiv(nprog - bid, SMS)
            if tx < NPROD:
                if HP == 1:
                    for j in T.serial(T.ceildiv(nit, NPROD)):
                        it = tx + NPROD * j
                        if it < nit:
                            p = bid + it * SMS
                            n = p // HV
                            hv = p % HV
                            slot = idx[n]
                            slot_s[it] = slot
                            h_s[it] = hcnt[T.max(slot, 0), hv]
                            P_s[it] = Pbuf[T.max(slot, 0), hv, 0]
                            xg = T.cast(a[n, hv, 0], F32) + dt_bias[hv, 0]
                            spg = T.if_then_else(xg <= SOFTPLUS_THRESHOLD, T.log(1.0 + T.exp(xg)), xg)
                            a_s[it] = T.exp(-T.exp(A_log[hv]) * spg)
                            b_s[it] = (T.cast(T.cast(T.sigmoid(T.cast(b[n, hv], F32)), BF16), F32))
                else:
                    for j in T.serial(T.ceildiv(nit * HP, NPROD)):     # one (program, head) entry e = it * HP + hd per thread
                        e = tx + NPROD * j
                        if e < nit * HP:
                            p = bid + (e // HP) * SMS
                            n = p // HVP
                            hv = p % HVP * HP + e % HP
                            slot = idx[n]
                            if e % HP == 0:
                                slot_s[e // HP] = slot
                            h_s[e] = hcnt[T.max(slot, 0), hv]
                            P_s[e] = Pbuf[T.max(slot, 0), hv, 0]
                            xg = T.cast(a[n, hv, 0], F32) + dt_bias[hv, 0]
                            spg = T.if_then_else(xg <= SOFTPLUS_THRESHOLD, T.log(1.0 + T.exp(xg)), xg)
                            a_s[e] = T.exp(-T.exp(A_log[hv]) * spg)
                            b_s[e] = (T.cast(T.cast(T.sigmoid(T.cast(b[n, hv], F32)), BF16), F32))
                T.sync_threads(15, NPROD)
                if tx >= 0 and tx < 32:        # ---- TMA warp ----
                    tp = T.alloc_local((2,), "int64")
                    tpacc = T.alloc_local((2,), F32)
                    tpacc[0] = 0.0
                    tpacc[1] = 0.0
                    for it in T.serial(nit):
                        p = bid + it * SMS
                        gc = it % NCG
                        ig = it // NCG
                        s = gc * SPG + ig % SPG
                        n = p // HVP
                        if HP == 1:
                            hv = p % HV
                        else:
                            hv = p % HVP * HP
                        hk = hv // GQA
                        slot = slot_s[it]
                        T.mbarrier_wait_parity(consumed[s], ((ig // SPG) & 1) ^ 1)
                        if slot > 0:
                            if HP == 1:
                                T.tma_copy(Sq[slot, hv, :, :], Qs[s, :, :], barrier=loaded[s])
                                T.tma_copy(Skv[slot, hv, :], skv_s[s, :], barrier=loaded[s])
                                T.tma_copy(mixed[n, hk * K:(hk + 1) * K], qx_s[s, :], barrier=loaded[s])
                                T.tma_copy(mixed[n, H * K + hk * K:H * K + (hk + 1) * K], kx_s[s, :], barrier=loaded[s])
                                T.tma_copy(mixed[n, 2 * H * K + hv * V:2 * H * K + (hv + 1) * V], vx_s[s, :], barrier=loaded[s])
                                T.tma_copy(Kbuf[slot, hv, :, :], Kb_s[s, :, :], barrier=loaded[s])
                                T.tma_copy(Ubuf[slot, hv, :, :], Ub_s[s, :, :], barrier=loaded[s])
                                T.tma_copy(Wbuf[slot, hv, :], Wb_s[s, 0:L], barrier=loaded[s])
                                T.tma_copy(UQ[slot, hv, :, :], UQ_s[s, :, :], barrier=loaded[s])
                            else:
                                for hd in T.unroll(HP):                      # the swizzled tiles: one 2-D box per head
                                    T.tma_copy(Sq[slot, hv + hd, :, :], Qs[s * HP + hd, :, :], barrier=loaded[s])
                                    T.tma_copy(Kbuf[slot, hv + hd, :, :], Kb_s[s * HP + hd, :, :], barrier=loaded[s])
                                T.tma_copy(Skv[slot, hv:hv + HP, :], skv_s[s, :, :], barrier=loaded[s])
                                T.tma_copy(mixed[n, hk * K:(hk + 1) * K], qx_s[s, :], barrier=loaded[s])
                                T.tma_copy(mixed[n, H * K + hk * K:H * K + (hk + 1) * K], kx_s[s, :], barrier=loaded[s])
                                T.tma_copy(mixed[n, 2 * H * K + hv * V:2 * H * K + (hv + HP) * V], vx_s[s, :], barrier=loaded[s])
                                T.tma_copy(Ubuf[slot, hv:hv + HP, :, :], Ub_s[s, :, :, :], barrier=loaded[s])
                                T.tma_copy(Wbuf[slot, hv:hv + HP, :], Wb_s[s, :, :], barrier=loaded[s])
                                T.tma_copy(UQ[slot, hv:hv + HP, :, :], UQ_s[s, :, :, :], barrier=loaded[s])
                        T.mbarrier_arrive(loaded[s])
                for nw in range(NNW):
                  if tx >= 32 * (1 + nw) and tx < 32 * (2 + nw):       # ---- norm warp nw: programs it = nw, nw + NNW, ... ----
                    lane = tx - 32 * (1 + nw)
                    ql = T.alloc_local((4,), F32)
                    kl = T.alloc_local((4,), F32)
                    kfl = T.alloc_local((4,), F32)
                    qfl = T.alloc_local((4,), F32)
                    red = T.alloc_local((4,), F32)
                    tn = T.alloc_local((2,), "int64")
                    tnacc = T.alloc_local((2,), F32)
                    tnacc[0] = 0.0
                    tnacc[1] = 0.0
                    if HP == 1:
                        if nw == 0:
                            for i in T.serial(NSTAGE * 4 * K // 32):          # zero rows 4..7 of every stage's B operand once
                                e = lane + 32 * i
                                bf_s[e // (4 * K), 4 + (e % (4 * K)) // K, e % K] = T.cast(0.0, F16)
                    for itn in T.serial(T.ceildiv(nit - nw, NNW)):
                        it = nw + itn * NNW
                        p = bid + it * SMS
                        gc = it % NCG
                        ig = it // NCG
                        s = gc * SPG + ig % SPG
                        if HP == 1:
                            hv = p % HV
                        else:
                            hv = p % HVP * HP
                        slot = slot_s[it]
                        T.mbarrier_wait_parity(loaded[s], (ig // SPG) & 1)
                        if slot > 0:
                            if HP == 1:
                                red[0] = 0.0
                                red[1] = 0.0
                                red[2] = 0.0
                                for i in T.unroll(NI):
                                    ql[i] = T.cast(qx_s[s, lane + 32 * i], F32)
                                    kl[i] = T.cast(kx_s[s, lane + 32 * i], F32)
                                    red[0] += ql[i] * ql[i]
                                    red[1] += kl[i] * kl[i]
                                    red[2] += kl[i] * ql[i]
                                nq = T.warp_reduce_sum(red[0])
                                nk = T.warp_reduce_sum(red[1])
                                nkq = T.warp_reduce_sum(red[2])
                                inq = T.rsqrt(nq + 1e-6) * scale
                                ink = T.rsqrt(nk + 1e-6)
                                red[3] = 0.0
                                for i in T.unroll(NI):
                                    c = lane + 32 * i
                                    kn = kl[i] * ink
                                    qn = ql[i] * inq
                                    kn_s[s, (c // 16) * 20 + c % 16] = kn
                                    qn_s[s, (c // 16) * 20 + c % 16] = qn
                                    kfl[i] = kn * skv_s[s, c] * (1.0 / QMAX)
                                    qfl[i] = qn * skv_s[s, c] * (1.0 / QMAX)
                                    red[3] = T.max(red[3], T.max(T.abs(kfl[i]), T.abs(qfl[i])))
                                kfmax = T.max(T.warp_reduce_max(red[3]), 1e-30)
                                rk = T.call_extern(F32, "__frcp_rn", kfmax)
                                for i in T.unroll(NI):
                                    c = lane + 32 * i
                                    khi = T.cast(kfl[i] * rk, F16)
                                    qhi = T.cast(qfl[i] * rk, F16)
                                    bf_s[s, 0, c] = khi
                                    bf_s[s, 1, c] = qhi
                                    bf_s[s, 2, c] = T.cast(kfl[i] * rk - T.cast(khi, F32), F16)
                                    bf_s[s, 3, c] = T.cast(qfl[i] * rk - T.cast(qhi, F32), F16)
                                h = h_s[it]
                                at = a_s[it]
                                if lane < L:
                                    wj = Wb_s[s, lane] * T.if_then_else(lane < h, at, 1.0)
                                    w_s[s, lane] = wj
                                    if lane < h:
                                        Wbuf[slot, hv, lane] = wj
                                if lane == 0:
                                    sc_s[s, 0] = kfmax
                                    sc_s[s, 1] = nkq * inq * ink
                                    sc_s[s, 2] = P_s[it] * at
                                    sc_s[s, 3] = b_s[it]
                            else:                                            # HP = 2: the two heads share q / k (one key head)
                                red[0] = 0.0
                                red[1] = 0.0
                                red[2] = 0.0
                                for i in T.unroll(NI):
                                    ql[i] = T.cast(qx_s[s, lane + 32 * i], F32)
                                    kl[i] = T.cast(kx_s[s, lane + 32 * i], F32)
                                    red[0] += ql[i] * ql[i]
                                    red[1] += kl[i] * kl[i]
                                    red[2] += kl[i] * ql[i]
                                nq = T.warp_reduce_sum(red[0])
                                nk = T.warp_reduce_sum(red[1])
                                nkq = T.warp_reduce_sum(red[2])
                                inq = T.rsqrt(nq + 1e-6) * scale
                                ink = T.rsqrt(nk + 1e-6)
                                for i in T.unroll(NI):
                                    c = lane + 32 * i
                                    kl[i] = kl[i] * ink                         # kl / ql := kn / qn
                                    ql[i] = ql[i] * inq
                                    kn_s[s, (c // 16) * 20 + c % 16] = kl[i]
                                    qn_s[s, (c // 16) * 20 + c % 16] = ql[i]
                                for hd in T.unroll(HP):                         # per head: column scales -> B operand rows 4 hd ..
                                    red[3] = 0.0
                                    for i in T.unroll(NI):
                                        c = lane + 32 * i
                                        kfl[i] = kl[i] * skv_s[s, hd, c] * (1.0 / QMAX)
                                        qfl[i] = ql[i] * skv_s[s, hd, c] * (1.0 / QMAX)
                                        red[3] = T.max(red[3], T.max(T.abs(kfl[i]), T.abs(qfl[i])))
                                    kfmax = T.max(T.warp_reduce_max(red[3]), 1e-30)
                                    rk = T.call_extern(F32, "__frcp_rn", kfmax)
                                    for i in T.unroll(NI):
                                        c = lane + 32 * i
                                        khi = T.cast(kfl[i] * rk, F16)
                                        qhi = T.cast(qfl[i] * rk, F16)
                                        bf_s[s, 4 * hd, c] = khi
                                        bf_s[s, 4 * hd + 1, c] = qhi
                                        bf_s[s, 4 * hd + 2, c] = T.cast(kfl[i] * rk - T.cast(khi, F32), F16)
                                        bf_s[s, 4 * hd + 3, c] = T.cast(qfl[i] * rk - T.cast(qhi, F32), F16)
                                    h = h_s[it * HP + hd]
                                    at = a_s[it * HP + hd]
                                    if lane < L:
                                        wj = Wb_s[s, hd, lane] * T.if_then_else(lane < h, at, 1.0)
                                        w_s[s, hd, lane] = wj
                                        if lane < h:
                                            Wbuf[slot, hv + hd, lane] = wj
                                    if lane == 0:
                                        sc_s[s, 4 * hd] = kfmax
                                        sc_s[s, 4 * hd + 1] = nkq * inq * ink
                                        sc_s[s, 4 * hd + 2] = P_s[it * HP + hd] * at
                                        sc_s[s, 4 * hd + 3] = b_s[it * HP + hd]
                        T.mbarrier_arrive(normed[s])
            else:
                g = (tx - NPROD) // 128
                t = (tx - NPROD) % 128
                warp = t // 32
                lane = t % 32
                q4 = lane % 4
                r0 = lane // 4
                bar = 8 + g
                if HP > 1:
                    hd = t // V                              # head of this thread's row (= warp // 2)
                    r64 = t % V                              # its row in that head
                afrag = T.alloc_local((4,), I32)             # A fragment (4 x fp16x2)
                bh = T.alloc_local((4,), F16)                # B fragment (2 x fp16x2)
                cfrag = T.alloc_local((8,), F32)             # 2 m-tiles x 4
                acc = T.alloc_local((8,), F32)
                tk = T.alloc_local((8,), "int64")
                tacc = T.alloc_local((8,), F32)
                if K < 128:                                  # K = 64: vector staging of the buffer dots / row steps (16 B ring, 32 B kn / qn loads)
                    kbv = T.alloc_local((8,), BF16)
                    knv = T.alloc_local((8,), F32)
                    qnv = T.alloc_local((8,), F32)
                    wv = T.alloc_local((4,), F32)
                    mv = T.alloc_local((4,), F32)
                for i in T.unroll(8):
                    tacc[i] = 0.0
                ncons = T.ceildiv(nit - g, NCG)
                for it0 in T.serial(ncons):
                    it = g + it0 * NCG
                    p = bid + it * SMS
                    s = g * SPG + it0 % SPG
                    pb = it0 % 2
                    n = p // HVP
                    if HP == 1:
                        hv = p % HV
                    else:
                        hv = p % HVP * HP
                    T.mbarrier_wait_parity(loaded[s], (it0 // SPG) & 1)
                    T.mbarrier_wait_parity(normed[s], (it0 // SPG) & 1)
                    slot = slot_s[it]
                    if HP == 1:
                        h = h_s[it]
                    else:
                        h = h_s[it * HP + hd]
                    if slot <= 0:
                        if HP == 1:
                            o[n, hv, t] = T.cast(0.0, BF16)
                        else:
                            o[n, hv + hd, r64] = T.cast(0.0, BF16)
                    else:
                      if HP == 1:
                        v0 = T.cast(vx_s[s, t], F32)
                        kfmax = sc_s[s, 0]
                        kq = sc_s[s, 1]
                        pn = sc_s[s, 2]
                        beta = sc_s[s, 3]
                        if K == V:
                            kn = kn_s[s, (t // 16) * 20 + t % 16]
                        else:
                            kn = kn_s[s, ((t % K) // 16) * 20 + t % 16]     # K < V: only threads t < K append (below)
                        # ---- checkpoint stream on tensor cores: C[v, 0:2] = sum_c q8[v, c] [kf qf][c] / kfmax ----
                        for i in T.unroll(8):
                            cfrag[i] = 0.0
                        for kk in T.unroll(KW // 4):                       # 16-column k-steps
                            for i in T.vectorized(4):
                                bh[i] = bf_s[s, r0, kk * 16 + q4 * 4 + i]
                            for mt in T.unroll(2):
                                rr = warp * 32 + mt * 16 + r0
                                wa = Qs[s, rr, kk * 4 + q4] ^ MAGICX
                                wb = Qs[s, rr + 8, kk * 4 + q4] ^ MAGICX
                                afrag[0] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wa, H2MAG, T.int32(0x4140))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[1] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wb, H2MAG, T.int32(0x4140))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[2] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wa, H2MAG, T.int32(0x4342))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[3] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wb, H2MAG, T.int32(0x4342))) - T.Broadcast(T.float16(1152.0), 2))
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", afrag.data, 0, ((bh)).data, 0, cfrag.data, mt * 4, T.bool(False))
                        if q4 < 2:                                         # q4 = 0: columns n = 0, 1 (hi parts); q4 = 1: n = 2, 3 (lo parts)
                            for mt in T.unroll(2):
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0, 2 * q4] = cfrag[mt * 4]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0, 2 * q4 + 1] = cfrag[mt * 4 + 1]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0 + 8, 2 * q4] = cfrag[mt * 4 + 2]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0 + 8, 2 * q4 + 1] = cfrag[mt * 4 + 3]
                        if K == 128:
                            # ---- buffer dots: dk[j] = K[j].kn, dq[j] = K[j].qn (8 lanes per row j) ----
                            jr = t // 8
                            part = t % 8
                            acc[2] = 0.0
                            acc[3] = 0.0
                            if L < 16:                                         # window ablation: rows jr >= L are idle (read row jr % L, no write)
                                for c1 in T.unroll(16):
                                    kb = T.cast(Kb_s[s, jr % L, part * 16 + c1], F32)
                                    acc[2] += kb * kn_s[s, part * 20 + c1]
                                    acc[3] += kb * qn_s[s, part * 20 + c1]
                            else:
                                for c1 in T.unroll(16):
                                    kb = T.cast(Kb_s[s, jr, part * 16 + c1], F32)
                                    acc[2] += kb * kn_s[s, part * 20 + c1]
                                    acc[3] += kb * qn_s[s, part * 20 + c1]
                            acc[2] += T.shfl_xor(acc[2], 1)
                            acc[3] += T.shfl_xor(acc[3], 1)
                            acc[2] += T.shfl_xor(acc[2], 2)
                            acc[3] += T.shfl_xor(acc[3], 2)
                            acc[2] += T.shfl_xor(acc[2], 4)
                            acc[3] += T.shfl_xor(acc[3], 4)
                            if L < 16:
                                if part == 0:
                                    if jr < L:
                                        dk_s[g, pb, jr] = acc[2]
                                        dq_s[g, pb, jr] = acc[3]
                            elif part == 0:
                                dk_s[g, pb, jr] = acc[2]
                                dq_s[g, pb, jr] = acc[3]
                            if L > 16:                                         # window ablation (L = 32): second pass, rows 16..31
                                acc[2] = 0.0
                                acc[3] = 0.0
                                for c1 in T.unroll(16):
                                    kb = T.cast(Kb_s[s, jr + 16, part * 16 + c1], F32)
                                    acc[2] += kb * kn_s[s, part * 20 + c1]
                                    acc[3] += kb * qn_s[s, part * 20 + c1]
                                acc[2] += T.shfl_xor(acc[2], 1)
                                acc[3] += T.shfl_xor(acc[3], 1)
                                acc[2] += T.shfl_xor(acc[2], 2)
                                acc[3] += T.shfl_xor(acc[3], 2)
                                acc[2] += T.shfl_xor(acc[2], 4)
                                acc[3] += T.shfl_xor(acc[3], 4)
                                if part == 0:
                                    dk_s[g, pb, jr + 16] = acc[2]
                                    dq_s[g, pb, jr + 16] = acc[3]
                            if R > 4:                                        # rank ablation (R = 8 / 16): warp w takes factors w, w + 4, ...
                                for rr in T.unroll(R // 4):
                                    acc[6] = 0.0
                                    acc[7] = 0.0
                                    for i in T.unroll(4):
                                        uf = T.cast(UQ_s[s, warp + 4 * rr, lane + 32 * i], F32)
                                        acc[6] += uf * kn_s[s, ((lane + 32 * i) // 16) * 20 + lane % 16]
                                        acc[7] += uf * qn_s[s, ((lane + 32 * i) // 16) * 20 + lane % 16]
                                    acc[6] = T.warp_reduce_sum(acc[6])
                                    acc[7] = T.warp_reduce_sum(acc[7])
                                    if lane == 0:
                                        dk_s[g, pb, L + warp + 4 * rr] = acc[6]
                                        dq_s[g, pb, L + warp + 4 * rr] = acc[7]
                            else:
                                if warp < R:                                   # factor dots: dU[r] = U^T[r].kn, U^T[r].qn (one warp per r)
                                    acc[6] = 0.0
                                    acc[7] = 0.0
                                    for i in T.unroll(4):
                                        uf = T.cast(UQ_s[s, warp, lane + 32 * i], F32)
                                        acc[6] += uf * kn_s[s, ((lane + 32 * i) // 16) * 20 + lane % 16]
                                        acc[7] += uf * qn_s[s, ((lane + 32 * i) // 16) * 20 + lane % 16]
                                    acc[6] = T.warp_reduce_sum(acc[6])
                                    acc[7] = T.warp_reduce_sum(acc[7])
                                    if lane == 0:
                                        dk_s[g, pb, L + warp] = acc[6]
                                        dq_s[g, pb, L + warp] = acc[7]
                        else:
                            if warp < 2:                                       # K = 64 buffer dots: row jr = t // 4, 4 lanes x 16 columns
                                jr = t // 4
                                part = t % 4
                                acc[2] = 0.0
                                acc[3] = 0.0
                                for c0 in T.unroll(2):                         # 8-column halves: one 16 B ring load, 2 x 32 B kn / qn loads (same FMA order)
                                    for c in T.vectorized(8):
                                        kbv[c] = Kb_s[s, jr, part * 16 + c0 * 8 + c]
                                    for c in T.vectorized(8):
                                        knv[c] = kn_s[s, part * 20 + c0 * 8 + c]
                                    for c in T.vectorized(8):
                                        qnv[c] = qn_s[s, part * 20 + c0 * 8 + c]
                                    for c in T.unroll(8):
                                        kb = T.cast(kbv[c], F32)
                                        acc[2] += kb * knv[c]
                                        acc[3] += kb * qnv[c]
                                # dk / dq butterflies (xor 1, 2) evaluated together: the same two trees, half the shuffles
                                acc[0] = T.if_then_else(lane % 2 == 1, acc[2], acc[3])     # sent (odd lanes keep dq)
                                acc[1] = T.if_then_else(lane % 2 == 1, acc[3], acc[2])     # kept
                                acc[1] = acc[1] + T.shfl_xor(acc[0], 1)
                                acc[1] = acc[1] + T.shfl_xor(acc[1], 2)
                                if part < 2:                                   # part 0: dk, 1: dq; the ring weight folded in (the K = 128 row loop's w_j dk_j product)
                                    dkq_s[g, pb, 2 * jr + part] = w_s[s, jr] * acc[1]
                            else:                                              # factor dots on warps 2, 3: factors warp - 2 (lanes 0-15) and warp (lanes 16-31)
                                for rr in T.unroll(2):
                                    acc[4 + 2 * rr] = 0.0
                                    acc[5 + 2 * rr] = 0.0
                                    if 2 * rr + warp - 2 < R:
                                        for i in T.unroll(NI):
                                            uf = T.cast(UQ_s[s, 2 * rr + warp - 2, lane + 32 * i], F32)
                                            acc[4 + 2 * rr] += uf * kn_s[s, ((lane + 32 * i) // 16) * 20 + lane % 16]
                                            acc[5 + 2 * rr] += uf * qn_s[s, ((lane + 32 * i) // 16) * 20 + lane % 16]
                                # the four 32-lane butterflies (xor 16, 8, 4, 2, 1) evaluated together: same trees, 6 shuffles instead of 20
                                acc[0] = T.if_then_else(lane >= 16, acc[4], acc[6])
                                acc[1] = T.if_then_else(lane >= 16, acc[5], acc[7])
                                acc[2] = T.if_then_else(lane >= 16, acc[6], acc[4])
                                acc[3] = T.if_then_else(lane >= 16, acc[7], acc[5])
                                acc[2] = acc[2] + T.shfl_xor(acc[0], 16)
                                acc[3] = acc[3] + T.shfl_xor(acc[1], 16)
                                acc[0] = T.if_then_else(lane % 16 >= 8, acc[2], acc[3])
                                acc[1] = T.if_then_else(lane % 16 >= 8, acc[3], acc[2])
                                acc[1] = acc[1] + T.shfl_xor(acc[0], 8)
                                acc[1] = acc[1] + T.shfl_xor(acc[1], 4)
                                acc[1] = acc[1] + T.shfl_xor(acc[1], 2)
                                acc[1] = acc[1] + T.shfl_xor(acc[1], 1)
                                if lane % 8 == 0:                              # lane 0 / 8: dU / dQ of factor warp - 2, lane 16 / 24: of factor warp
                                    if 2 * (lane // 16) + warp - 2 < R:
                                        dkq_s[g, pb, 2 * (L + 2 * (lane // 16) + warp - 2) + (lane // 8) % 2] = acc[1]
                        T.sync_threads(bar, 128)
                        # ---- contribution, correction, output, append (thread-per-row v = t) ----
                        # (MX: row v = t was computed by this thread's own warp -> undo its offset 2^(emax - 140))
                        sv = (skv_s[s, K + t])
                        if K == 128:
                            mq = (mqy_s[g, pb, t, 0] + mqy_s[g, pb, t, 2]) * kfmax * sv
                            yq = (mqy_s[g, pb, t, 1] + mqy_s[g, pb, t, 3]) * kfmax * sv
                            acc[4] = 0.0
                            acc[5] = 0.0
                            for j in T.unroll(L):
                                ub = T.cast(Ub_s[s, j, t], F32)
                                acc[4] += ub * (w_s[s, j] * dk_s[g, pb, j])
                                acc[5] += ub * (w_s[s, j] * dq_s[g, pb, j])
                            acc[6] = 0.0                                    # rank-R part of the checkpoint: Q^T[r][v] (U^T[r].kn)
                            acc[7] = 0.0
                            for r in T.unroll(R):
                                qf = T.cast((UQ_s[s, r, K + t]), F32)
                                acc[6] += qf * dk_s[g, pb, L + r]
                                acc[7] += qf * dq_s[g, pb, L + r]
                        else:                                                   # K = 64: 16 B mq / yq load, 16 B = [w dk, w dq] of two entries per load
                            for c in T.vectorized(4):
                                mv[c] = mqy_s[g, pb, t, c]
                            mq = (mv[0] + mv[2]) * kfmax * sv
                            yq = (mv[1] + mv[3]) * kfmax * sv
                            acc[4] = 0.0
                            acc[5] = 0.0
                            for jj in T.unroll(L // 2):
                                for c in T.vectorized(4):
                                    wv[c] = dkq_s[g, pb, 4 * jj + c]
                                ub0 = T.cast(Ub_s[s, 2 * jj, t], F32)
                                ub1 = T.cast(Ub_s[s, 2 * jj + 1, t], F32)
                                acc[4] += ub0 * wv[0]
                                acc[5] += ub0 * wv[1]
                                acc[4] += ub1 * wv[2]
                                acc[5] += ub1 * wv[3]
                            acc[6] = 0.0
                            acc[7] = 0.0
                            for r in T.unroll(R):
                                qf = T.cast(UQ_s[s, r, K + t], F32)
                                acc[6] += qf * dkq_s[g, pb, 2 * (L + r)]
                                acc[7] += qf * dkq_s[g, pb, 2 * (L + r) + 1]
                        u = beta * (v0 - (pn * (mq + acc[6]) + acc[4]))
                        y = pn * (yq + acc[7]) + acc[5] + kq * u
                        o[n, hv, t] = T.cast(y, BF16)
                        Ubuf[slot, hv, h, t] = T.cast(u, BF16)
                        if K == V:
                            Kbuf[slot, hv, h, t] = T.cast(kn, BF16)
                        else:
                            if t < K:
                                Kbuf[slot, hv, h, t] = T.cast(kn, BF16)
                        if t == 0:
                            Wbuf[slot, hv, h] = 1.0
                            Pbuf[slot, hv, 0] = pn
                            hcnt[slot, hv] = h + 1
                      else:                                                  # ---- HP = 2: rows t = 64 hd + r64 of head hv + hd ----
                        v0 = T.cast(vx_s[s, t], F32)
                        kfmax = sc_s[s, 4 * hd]
                        kq = sc_s[s, 4 * hd + 1]
                        pn = sc_s[s, 4 * hd + 2]
                        beta = sc_s[s, 4 * hd + 3]
                        kn = kn_s[s, (r64 // 16) * 20 + r64 % 16]
                        for i in T.unroll(8):
                            cfrag[i] = 0.0
                        for kk in T.unroll(KW // 4):
                            for i in T.vectorized(4):
                                bh[i] = bf_s[s, r0, kk * 16 + q4 * 4 + i]
                            for mt in T.unroll(2):
                                rr = (warp % 2) * 32 + mt * 16 + r0              # row of head warp // 2
                                wa = Qs[s * HP + warp // 2, rr, kk * 4 + q4] ^ MAGICX
                                wb = Qs[s * HP + warp // 2, rr + 8, kk * 4 + q4] ^ MAGICX
                                afrag[0] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wa, H2MAG, T.int32(0x4140))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[1] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wb, H2MAG, T.int32(0x4140))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[2] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wa, H2MAG, T.int32(0x4342))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[3] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wb, H2MAG, T.int32(0x4342))) - T.Broadcast(T.float16(1152.0), 2))
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", afrag.data, 0, bh.data, 0, cfrag.data, mt * 4, T.bool(False))
                        if q4 // 2 == warp // 2:                           # head hd keeps mma columns 4 hd .. 4 hd + 3 (q4 = 2 hd: hi, 2 hd + 1: lo)
                            for mt in T.unroll(2):
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0, 2 * (q4 % 2)] = cfrag[mt * 4]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0, 2 * (q4 % 2) + 1] = cfrag[mt * 4 + 1]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0 + 8, 2 * (q4 % 2)] = cfrag[mt * 4 + 2]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0 + 8, 2 * (q4 % 2) + 1] = cfrag[mt * 4 + 3]
                        jr = (t // 4) % L                                  # buffer dots: ring row jr of head hd, 4 lanes x 16 columns
                        part = t % 4
                        acc[2] = 0.0
                        acc[3] = 0.0
                        for c0 in T.unroll(2):                             # 8-column halves: one 16 B ring load, 2 x 32 B kn / qn loads (same FMA order)
                            for c in T.vectorized(8):
                                kbv[c] = Kb_s[s * HP + hd, jr, part * 16 + c0 * 8 + c]
                            for c in T.vectorized(8):
                                knv[c] = kn_s[s, part * 20 + c0 * 8 + c]
                            for c in T.vectorized(8):
                                qnv[c] = qn_s[s, part * 20 + c0 * 8 + c]
                            for c in T.unroll(8):
                                kb = T.cast(kbv[c], F32)
                                acc[2] += kb * knv[c]
                                acc[3] += kb * qnv[c]
                        acc[0] = T.if_then_else(lane % 2 == 1, acc[2], acc[3])     # dk / dq butterflies (xor 1, 2) together (same trees)
                        acc[1] = T.if_then_else(lane % 2 == 1, acc[3], acc[2])
                        acc[1] = acc[1] + T.shfl_xor(acc[0], 1)
                        acc[1] = acc[1] + T.shfl_xor(acc[1], 2)
                        if part < 2:                                       # part 0: dk, 1: dq; ring weight folded in (the K = 128 row loop's product)
                            dkq_s[g, pb, hd, 2 * jr + part] = w_s[s, hd, jr] * acc[1]
                        if warp < R:                                       # factor dots: warp r takes factor r of both heads
                            for hb in T.unroll(HP):
                                acc[4 + 2 * hb] = 0.0
                                acc[5 + 2 * hb] = 0.0
                                for i in T.unroll(NI):
                                    uf = T.cast(UQ_s[s, hb, warp, lane + 32 * i], F32)
                                    acc[4 + 2 * hb] += uf * kn_s[s, ((lane + 32 * i) // 16) * 20 + lane % 16]
                                    acc[5 + 2 * hb] += uf * qn_s[s, ((lane + 32 * i) // 16) * 20 + lane % 16]
                            # the four 32-lane butterflies (xor 16, 8, 4, 2, 1) together: head 0 on lanes 0-15, head 1 on 16-31, then
                            # dk on lanes with bit 3 clear, dq with it set -- the same trees, 6 shuffles instead of 20
                            acc[0] = T.if_then_else(lane >= 16, acc[4], acc[6])
                            acc[1] = T.if_then_else(lane >= 16, acc[5], acc[7])
                            acc[2] = T.if_then_else(lane >= 16, acc[6], acc[4])
                            acc[3] = T.if_then_else(lane >= 16, acc[7], acc[5])
                            acc[2] = acc[2] + T.shfl_xor(acc[0], 16)
                            acc[3] = acc[3] + T.shfl_xor(acc[1], 16)
                            acc[0] = T.if_then_else(lane % 16 >= 8, acc[2], acc[3])
                            acc[1] = T.if_then_else(lane % 16 >= 8, acc[3], acc[2])
                            acc[1] = acc[1] + T.shfl_xor(acc[0], 8)
                            acc[1] = acc[1] + T.shfl_xor(acc[1], 4)
                            acc[1] = acc[1] + T.shfl_xor(acc[1], 2)
                            acc[1] = acc[1] + T.shfl_xor(acc[1], 1)
                            if lane % 8 == 0:
                                dkq_s[g, pb, lane // 16, 2 * (L + warp) + (lane // 8) % 2] = acc[1]
                        T.sync_threads(bar, 128)
                        sv = skv_s[s, hd, K + r64]
                        for c in T.vectorized(4):
                            mv[c] = mqy_s[g, pb, t, c]
                        mq = (mv[0] + mv[2]) * kfmax * sv
                        yq = (mv[1] + mv[3]) * kfmax * sv
                        acc[4] = 0.0
                        acc[5] = 0.0
                        for jj in T.unroll(L // 2):                        # two ring entries per 16 B [w dk, w dq] load
                            for c in T.vectorized(4):
                                wv[c] = dkq_s[g, pb, hd, 4 * jj + c]
                            ub0 = T.cast(Ub_s[s, hd, 2 * jj, r64], F32)
                            ub1 = T.cast(Ub_s[s, hd, 2 * jj + 1, r64], F32)
                            acc[4] += ub0 * wv[0]
                            acc[5] += ub0 * wv[1]
                            acc[4] += ub1 * wv[2]
                            acc[5] += ub1 * wv[3]
                        acc[6] = 0.0
                        acc[7] = 0.0
                        for r in T.unroll(R):
                            qf = T.cast(UQ_s[s, hd, r, K + r64], F32)
                            acc[6] += qf * dkq_s[g, pb, hd, 2 * (L + r)]
                            acc[7] += qf * dkq_s[g, pb, hd, 2 * (L + r) + 1]
                        u = beta * (v0 - (pn * (mq + acc[6]) + acc[4]))
                        y = pn * (yq + acc[7]) + acc[5] + kq * u
                        o[n, hv + hd, r64] = T.cast(y, BF16)
                        Ubuf[slot, hv + hd, h, r64] = T.cast(u, BF16)
                        Kbuf[slot, hv + hd, h, r64] = T.cast(kn, BF16)
                        if r64 == 0:
                            Wbuf[slot, hv + hd, h] = 1.0
                            Pbuf[slot, hv + hd, 0] = pn
                            hcnt[slot, hv + hd] = h + 1
                    T.fence_proxy_async()
                    T.mbarrier_arrive(consumed[s])
    return step

@tilelang.jit(pass_configs={"tl.disable_thread_storage_sync": True})
def make_kda_step(NS, HV, H, K, V, L, SS, R=4, SMS=148, NSTAGE=6, NSTAGEQ=4, NCG=2, NNW=2, scale=128 ** -0.5, MAXIT=256, GLB=0.0):
    """Kimi Delta Attention version of make_gdn_step (per-key-channel gate).

    The norm warp computes the new cumulative log-gate gn[c] = Pbuf[c] + log-decay(a[c] + dt_bias[c]), pc = exp(gn) and the
    step factor en = exp(gn - Pbuf) per channel, and folds pc into the checkpoint operand (kf = kn sk / 127 pc) and the factor
    operand (kp = kn pc).  Gbuf holds the per-step decay factor e_j = exp(g_j) of each buffer entry (fp16); the decay of entry j
    is the backward running product D_j = en * prod_{i>j} e_i, folded into the staged buffer tile.
    GLB != 0: GLM's bounded gate, log-decay = GLB * sigmoid(exp(A_log) (a + dt_bias)); GLB == 0: Kimi's
    -exp(A_log) softplus(a + dt_bias)."""
    # The gate is 4 transcendentals per key channel inside the norm warp, on every consumer's critical path:
    # use the one-MUFU __expf / __logf (~2 ulp, far below the bf16 buffer's own error) instead of TileLang's accurate ones.
    FEXP = (lambda x: T.call_extern(F32, "__expf", x))
    FLOG = (lambda x: T.call_extern(F32, "__logf", x))
    # GLB != 0 (GLM): bounded gate, log-decay GLB * sigmoid(exp(A_log) (a + dt_bias)) in (GLB, 0); GLB == 0 (Kimi):
    # -exp(A_log) softplus(a + dt_bias).
    KGATE_SP = (lambda xg, ea: 1.0 / (1.0 + FEXP(-ea * xg))) if GLB != 0.0 else (lambda xg, ea: T.if_then_else(xg <= SOFTPLUS_THRESHOLD, FLOG(1.0 + FEXP(xg)), xg))
    KGATE_GN = (lambda pb, ea, spg: pb + GLB * spg) if GLB != 0.0 else (lambda pb, ea, spg: pb - ea * spg)
    KGATE_EN = (lambda ea, spg: FEXP(GLB * spg)) if GLB != 0.0 else (lambda ea, spg: FEXP(-ea * spg))
    GBT = "float16"                  # per-entry decay factors (fp16)
    GQA = HV // H
    N = T.symbolic("N")
    KW = K // 4
    NCONS = 128 * NCG
    assert NSTAGE % NCG == 0
    SPG = NSTAGE // NCG
    assert NSTAGEQ % NCG == 0
    SPGQ = NSTAGEQ // NCG                           # the int8 tile has its own (shallower) ring: it is dead after the checkpoint stream
    assert NSTAGE % NNW == 0, "every stage must be normed by ONE warp in fill order (program it -> warp it % NNW): otherwise a stage's fills alternate between warps, and a warp >= 2 fills ahead passes its mbarrier parity wait early (phase aliasing) -> stale data / deadlock"
    NTW = 1                         # TMA warps
    NPROD = 32 * (NTW + NNW)
    MIX = 2 * H * K + HV * V
    F16 = "float16"
    assert K == 128 and V == 128 and L <= 16
    MAGICX = T.int32(-2139062144)                  # 0x80808080: int8 -> offset byte
    H2MAG = T.int32(0x64646464)                    # fp16 exponent byte: 0x64xx = 1024 + xx

    @T.prim_func
    def step(Sq: T.StridedTensor((NS, HV, V, KW), (SS, V * K, KW, 1), I32),
             Sk: T.StridedTensor((NS, HV, K), (SS, V * K, 1), F32),
             Sv: T.StridedTensor((NS, HV, V), (SS, V * K, 1), F32),
             Skv: T.StridedTensor((NS, HV, K + V), (SS, V * K, 1), F32),
             Kbuf: T.Tensor((NS, HV, L, K), BF16), Ubuf: T.Tensor((NS, HV, L, V), BF16),
             Wbuf: T.Tensor((NS, HV, L), F32), Pbuf: T.Tensor((NS, HV, K), F32), Gbuf: T.Tensor((NS, HV, L, K), GBT), hcnt: T.Tensor((NS, HV), I32),
             mixed: T.Tensor((N, MIX), BF16), a: T.Tensor((N, HV, K), BF16), b: T.Tensor((N, HV), BF16),
             A_log: T.Tensor((HV,), F32), dt_bias: T.Tensor((HV, K), F32),
             UQ: T.StridedTensor((NS, HV, R, K + V), (2 * SS, 2 * V * K, K + V, 1), F16),
             idx: T.Tensor((N,), I32), o: T.Tensor((N, HV, V), BF16)):
        with T.Kernel(SMS, threads=NPROD + NCONS) as bid:
            Qs = T.alloc_shared((NSTAGEQ, V, KW), I32)
            UQ_s = T.alloc_shared((NSTAGE, R, K + V), F16)
            skv_s = T.alloc_shared((NSTAGE, K + V), F32)
            qx_s = T.alloc_shared((NSTAGE, K), BF16)
            kx_s = T.alloc_shared((NSTAGE, K), BF16)
            vx_s = T.alloc_shared((NSTAGE, V), BF16)
            Kb_s = T.alloc_shared((NSTAGE, L, K), BF16)
            Ub_s = T.alloc_shared((NSTAGE, L, V), BF16)
            T.annotate_layout({Kb_s: make_swizzled_layout(Kb_s)})     # the buffer dots read 32 B per lane over 4 rows per warp: 4-way bank conflicts unswizzled
                                                                     # (UQ_s cannot be swizzled: its 512 B rows are not the TMA swizzle granularity)
            Wb_s = T.alloc_shared((NSTAGE, 32), F32)
            Gb_s = T.alloc_shared((NSTAGE, L, K), GBT)         # per-entry cumulative log-gate at insertion (KDA)
            pb_s = T.alloc_shared((NSTAGE, K), F32)            # slot's cumulative log-gate since the checkpoint
            ax_s = T.alloc_shared((NSTAGE, K), BF16)           # this token's raw gate input a[n, hv, :]
            dtb_s = T.alloc_shared((NSTAGE, K), F32)           # dt_bias[hv, :] (TMA'd per program: no global latency in the norm warp)
            alog_s = T.alloc_shared((HV,), F32)
            # ---- per-stage data produced by the norm warp ----
            bf_s = T.alloc_shared((NSTAGE, 8, K + 16), F16)   # B operand: rows 0/1 = hi(kf, qf)/kfmax, 2/3 = lo parts, 4..7 = 0
                                                             # (+16: row stride 288 B -> the 8 r0 rows of a fragment load land on different banks; K = 8-way conflicts)
            kn_s = T.alloc_shared((NSTAGE, (K // 16) * 20), F32)   # chunk-padded: column c at (c // 16) * 20 + c % 16 (conflict-free 64 B chunk reads)
            qn_s = T.alloc_shared((NSTAGE, (K // 16) * 20), F32)
            en_s = T.alloc_shared((NSTAGE, K), F32)            # this step's decay factor exp(g_now)
            pc_s = T.alloc_shared((NSTAGE, K), F32)            # pc = exp(cumulative log-gate): applied to the factor dots in fp32 (U^T stays the stored fp16)
            bn_s = T.alloc_shared((NSTAGE, (1), (1)), BF16)   # ring-dot B operand
            # DFOLD: the per-entry decay is folded into the staged ring tile in place, so there is no D_s array:
            # the dot then reads only Kb_s (as the GDN kernel does), and the 20 KiB this used to cost is what
            # made NCG=2 the only affordable width.
            D_s = T.alloc_shared(((1), (1), (1), (1)), F16)
            w_s = T.alloc_shared((NSTAGE, L), F32)
            sc_s = T.alloc_shared((NSTAGE, 8), F32)           # 0 kfmax, 1 kq, 2 pn, 3 beta
            # ---- per-group scratch, double-buffered by program parity (no end-of-program barrier) ----
            mqy_s = T.alloc_shared((NCG, 2, V, 4), F32)        # [mq_hi, yq_hi, mq_lo, yq_lo] per row
            dk_s = T.alloc_shared((NCG, 2, L + R), F32)
            dq_s = T.alloc_shared((NCG, 2, L + R), F32)
            slot_s = T.alloc_shared((MAXIT,), I32)
            h_s = T.alloc_shared((MAXIT,), I32)
            b_s = T.alloc_shared((MAXIT,), F32)
            loaded = T.alloc_barrier([32] * NSTAGE)
            normed = T.alloc_barrier([32] * NSTAGE)
            consumed = T.alloc_barrier([128] * NSTAGE)
            loadedQ = T.alloc_barrier([32] * NSTAGEQ)
            consumedQ = T.alloc_barrier([128] * NSTAGEQ)
            T.annotate_layout({Qs: make_swizzled_layout(Qs)})
            tx = T.get_thread_binding()
            nprog = N * HV
            nit = T.ceildiv(nprog - bid, SMS)
            if tx < NPROD:
                for j in T.serial(T.ceildiv(nit, NPROD)):
                    it = tx + NPROD * j
                    if it < nit:
                        p = bid + it * SMS
                        n = p // HV
                        hv = p % HV
                        slot = idx[n]
                        slot_s[it] = slot
                        h_s[it] = hcnt[T.max(slot, 0), hv]
                        b_s[it] = (T.cast(T.cast(T.sigmoid(T.cast(b[n, hv], F32)), BF16), F32))
                for j in T.serial(T.ceildiv(HV, NPROD)):
                    if tx + NPROD * j < HV:
                        alog_s[tx + NPROD * j] = T.exp(A_log[tx + NPROD * j])
                T.sync_threads(15, NPROD)
                # TMA warps as literal 32-thread regions: a Python loop becomes a TIR loop here and the election extent would be 128
                if tx >= 0 and tx < 32:        # ---- TMA warp 0: all stages in fill order ----
                  tp = T.alloc_local((2,), "int64")
                  tpacc = T.alloc_local((2,), F32)
                  tpacc[0] = 0.0
                  tpacc[1] = 0.0
                  for ig in T.serial((nit)):
                      it = (ig)
                      p = bid + it * SMS
                      gc = (it % NCG)
                      ig2 = (it // NCG)
                      s = gc * SPG + ig2 % SPG
                      sq = gc * SPGQ + ig2 % SPGQ
                      n = p // HV
                      hv = p % HV
                      hk = hv // GQA
                      slot = slot_s[it]
                      T.mbarrier_wait_parity(consumedQ[sq], ((ig2 // SPGQ) & 1) ^ 1)
                      if slot > 0:
                          T.tma_copy(Sq[slot, hv, :, :], Qs[sq, :, :], barrier=loadedQ[sq])
                      T.mbarrier_arrive(loadedQ[sq])
                      T.mbarrier_wait_parity(consumed[s], ((ig2 // SPG) & 1) ^ 1)
                      if slot > 0:
                          T.tma_copy(Skv[slot, hv, :], skv_s[s, :], barrier=loaded[s])
                          T.tma_copy(mixed[n, hk * K:(hk + 1) * K], qx_s[s, :], barrier=loaded[s])
                          T.tma_copy(mixed[n, H * K + hk * K:H * K + (hk + 1) * K], kx_s[s, :], barrier=loaded[s])
                          T.tma_copy(mixed[n, 2 * H * K + hv * V:2 * H * K + (hv + 1) * V], vx_s[s, :], barrier=loaded[s])
                          T.tma_copy(Kbuf[slot, hv, :, :], Kb_s[s, :, :], barrier=loaded[s])
                          T.tma_copy(Ubuf[slot, hv, :, :], Ub_s[s, :, :], barrier=loaded[s])
                          T.tma_copy(Wbuf[slot, hv, :], Wb_s[s, 0:L], barrier=loaded[s])
                          T.tma_copy(UQ[slot, hv, :, :], UQ_s[s, :, :], barrier=loaded[s])
                          T.tma_copy(Gbuf[slot, hv, :, :], Gb_s[s, :, :], barrier=loaded[s])
                          T.tma_copy(Pbuf[slot, hv, :], pb_s[s, :], barrier=loaded[s])
                          T.tma_copy(a[n, hv, :], ax_s[s, :], barrier=loaded[s])
                          T.tma_copy(dt_bias[hv, :], dtb_s[s, :], barrier=loaded[s])
                      T.mbarrier_arrive(loaded[s])
                for nw in range(NNW):
                  if tx >= 32 * (NTW + nw) and tx < 32 * (NTW + 1 + nw):       # ---- norm warp nw: programs it = nw, nw + NNW, ... ----
                    lane = tx - 32 * (NTW + nw)
                    ql = T.alloc_local((4,), F32)
                    kl = T.alloc_local((4,), F32)
                    kfl = T.alloc_local((4,), F32)
                    qfl = T.alloc_local((4,), F32)
                    red = T.alloc_local((4,), F32)
                    tn = T.alloc_local((2,), "int64")
                    tnacc = T.alloc_local((2,), F32)
                    dn_l = T.alloc_local((1,), F32)
                    tnacc[0] = 0.0
                    tnacc[1] = 0.0
                    if nw == 0:
                        for i in T.serial(NSTAGE * 4 * K // 32):          # zero rows 4..7 of every stage's B operand once
                            e = lane + 32 * i
                            bf_s[e // (4 * K), 4 + (e % (4 * K)) // K, e % K] = T.cast(0.0, F16)
                    for itn in T.serial(T.ceildiv(nit - nw, NNW)):
                        it = nw + itn * NNW
                        p = bid + it * SMS
                        gc = it % NCG
                        ig = it // NCG
                        s = gc * SPG + ig % SPG
                        hv = p % HV
                        slot = slot_s[it]
                        T.mbarrier_wait_parity(loaded[s], (ig // SPG) & 1)
                        if slot > 0:
                            red[0] = 0.0
                            red[1] = 0.0
                            red[2] = 0.0
                            for i in T.unroll(4):
                                ql[i] = T.cast(qx_s[s, lane + 32 * i], F32)
                                kl[i] = T.cast(kx_s[s, lane + 32 * i], F32)
                                red[0] += ql[i] * ql[i]
                                red[1] += kl[i] * kl[i]
                                red[2] += kl[i] * ql[i]
                            nq = T.warp_reduce_sum(red[0])
                            nk = T.warp_reduce_sum(red[1])
                            nkq = T.warp_reduce_sum(red[2])
                            inq = T.rsqrt(nq + 1e-6) * scale
                            ink = T.rsqrt(nk + 1e-6)
                            red[3] = 0.0
                            ea = alog_s[hv]
                            for i in T.unroll(4):
                                c = lane + 32 * i
                                kn = kl[i] * ink
                                qn = ql[i] * inq
                                xg = T.cast(ax_s[s, c], F32) + dtb_s[s, c]
                                spg = KGATE_SP(xg, ea)
                                gn = KGATE_GN(pb_s[s, c], ea, spg)
                                pc = FEXP(gn)
                                # exp(gn - pb) algebraically, NOT as a difference: pb reaches -1365, and
                                # subtracting two numbers of that size to recover a ~0.3 exponent loses most of it
                                en = KGATE_EN(ea, spg)
                                en_s[s, c] = en
                                Pbuf[slot, hv, c] = gn                                    # the norm warp owns the channel vectors: appends go here
                                Gbuf[slot, hv, h_s[it], c] = T.cast(en, GBT)
                                kn_s[s, (c // 16) * 20 + c % 16] = kn
                                qn_s[s, (c // 16) * 20 + c % 16] = qn
                                pc_s[s, c] = pc                                           # (scaling U^T in place in fp16 cost 2^-11 on the rank-R term: 10.46)
                                kfl[i] = kn * skv_s[s, c] * (1.0 / QMAX) * pc
                                qfl[i] = qn * skv_s[s, c] * (1.0 / QMAX) * pc
                                red[3] = T.max(red[3], T.max(T.abs(kfl[i]), T.abs(qfl[i])))
                            kfmax = T.max(T.warp_reduce_max(red[3]), 1e-30)
                            rk = T.call_extern(F32, "__frcp_rn", kfmax)
                            for i in T.unroll(4):
                                c = lane + 32 * i
                                khi = T.cast(kfl[i] * rk, F16)
                                qhi = T.cast(qfl[i] * rk, F16)
                                bf_s[s, 0, c] = khi
                                bf_s[s, 1, c] = qhi
                                bf_s[s, 2, c] = T.cast(kfl[i] * rk - T.cast(khi, F32), F16)
                                bf_s[s, 3, c] = T.cast(qfl[i] * rk - T.cast(qhi, F32), F16)
                            if lane < L:
                                w_s[s, lane] = Wb_s[s, lane]
                            if lane == 0:
                                sc_s[s, 0] = kfmax
                                sc_s[s, 1] = nkq * inq * ink
                                sc_s[s, 2] = 0.0
                                sc_s[s, 3] = b_s[it]
                        T.mbarrier_arrive(normed[s])
            else:
                g = (tx - NPROD) // 128
                t = (tx - NPROD) % 128
                warp = t // 32
                lane = t % 32
                q4 = lane % 4
                r0 = lane // 4
                bar = 8 + g
                afrag = T.alloc_local((4,), I32)             # A fragment (4 x fp16x2)
                bh = T.alloc_local((4,), F16)                # B fragment (2 x fp16x2)
                cfrag = T.alloc_local((8,), F32)             # 2 m-tiles x 4
                acc = T.alloc_local((8,), F32)
                knl = T.alloc_local(((16),), F32)      # batched dot operands
                qnl = T.alloc_local(((16),), F32)
                kbl = T.alloc_local(((16),), BF16)
                afr2 = T.alloc_local(((1),), BF16)   # ring-dot A fragment (ldmatrix.x4)
                bfr2 = T.alloc_local(((1),), BF16)   # ring-dot B fragment
                dfrag = T.alloc_local(((1),), F32)
                tk = T.alloc_local((8,), "int64")
                tacc = T.alloc_local((8,), F32)
                for i in T.unroll(8):
                    tacc[i] = 0.0
                ncons = T.ceildiv(nit - g, NCG)
                for it0 in T.serial(ncons):
                    it = g + it0 * NCG
                    p = bid + it * SMS
                    s = g * SPG + it0 % SPG
                    sq = g * SPGQ + it0 % SPGQ
                    pb = it0 % 2
                    n = p // HV
                    hv = p % HV
                    T.mbarrier_wait_parity(loaded[s], (it0 // SPG) & 1)
                    T.mbarrier_wait_parity(loadedQ[sq], (it0 // SPGQ) & 1)
                    T.mbarrier_wait_parity(normed[s], (it0 // SPG) & 1)
                    slot = slot_s[it]
                    h = h_s[it]
                    if slot <= 0:
                        o[n, hv, t] = T.cast(0.0, BF16)
                        T.mbarrier_arrive(consumedQ[sq])
                    else:
                        v0 = T.cast(vx_s[s, t], F32)
                        kfmax = sc_s[s, 0]
                        kq = sc_s[s, 1]
                        beta = sc_s[s, 3]
                        kn = kn_s[s, (t // 16) * 20 + t % 16]
                        # ---- per-entry decay D_j[c] = en[c] * prod_{i>j, i<h} e_i[c] (thread per column c = t), backward product ----
                        # (entries j >= h hold e_j = 1: the flush resets Gbuf to 1, so the product needs no predicate)
                        dcur = T.alloc_local((1,), F32)
                        e_l = T.alloc_local(((L),), F32)
                        kb_l = T.alloc_local(((L),), F32)
                        for jj in T.unroll(L):                           # all factor loads first (smem loads cannot be hoisted above the stores)
                          e_l[jj] = T.cast(Gb_s[s, jj, t], F32)
                        for jj in T.unroll(L):
                            kb_l[jj] = T.cast(Kb_s[s, jj, t], F32)
                        dcur[0] = en_s[s, t]
                        for jj in T.unroll((L)):
                            Kb_s[s, L - 1 - jj, t] = T.cast(
                                kb_l[L - 1 - jj] * T.if_then_else(L - 1 - jj < h, dcur[0], 0.0), BF16)
                            dcur[0] = dcur[0] * e_l[L - 1 - jj]
                        # ---- checkpoint stream on tensor cores: C[v, 0:2] = sum_c q8[v, c] [kf qf][c] / kfmax ----
                        for i in T.unroll(8):
                            cfrag[i] = 0.0
                        for kk in T.unroll(KW // 4):                       # 16-column k-steps
                            for i in T.vectorized(4):
                                bh[i] = bf_s[s, r0, kk * 16 + q4 * 4 + i]
                            for mt in T.unroll(2):
                                rr = warp * 32 + mt * 16 + r0
                                wa = Qs[sq, rr, kk * 4 + q4] ^ MAGICX
                                wb = Qs[sq, rr + 8, kk * 4 + q4] ^ MAGICX
                                afrag[0] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wa, H2MAG, T.int32(0x4140))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[1] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wb, H2MAG, T.int32(0x4140))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[2] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wa, H2MAG, T.int32(0x4342))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[3] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wb, H2MAG, T.int32(0x4342))) - T.Broadcast(T.float16(1152.0), 2))
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", afrag.data, 0, bh.data, 0, cfrag.data, mt * 4, T.bool(False))
                        if q4 < 2:                                         # q4 = 0: columns n = 0, 1 (hi parts); q4 = 1: n = 2, 3 (lo parts)
                            for mt in T.unroll(2):
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0, 2 * q4] = cfrag[mt * 4]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0, 2 * q4 + 1] = cfrag[mt * 4 + 1]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0 + 8, 2 * q4] = cfrag[mt * 4 + 2]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0 + 8, 2 * q4 + 1] = cfrag[mt * 4 + 3]
                        T.fence_proxy_async()                              # the int8 tile is dead for this program: release its ring slot early
                        T.mbarrier_arrive(consumedQ[sq])
                        T.sync_threads(bar, 128)                           # D_s (and nothing else) must be complete before the dots
                        # ---- buffer dots: dk[j] = K[j].kn, dq[j] = K[j].qn (8 lanes per row j) ----
                        jr = t // 8
                        part = t % 8
                        acc[2] = 0.0
                        acc[3] = 0.0
                        # the dot is 48 scalar smem loads per thread on an issue-bound kernel; the chunk
                        # padding (part * 20) makes both 16-float runs 16 B aligned, so they are 4 vector
                        # loads each, and the 16 ring values are two 16 B loads
                        for v4 in T.unroll(4):
                            for c1 in T.vectorized(4):
                                knl[v4 * 4 + c1] = kn_s[s, part * 20 + v4 * 4 + c1]
                        for v4 in T.unroll(4):
                            for c1 in T.vectorized(4):
                                qnl[v4 * 4 + c1] = qn_s[s, part * 20 + v4 * 4 + c1]
                        for v8 in T.unroll(2):
                            for c1 in T.vectorized(8):
                                kbl[v8 * 8 + c1] = Kb_s[s, jr, part * 16 + v8 * 8 + c1]
                        for c1 in T.unroll(16):
                            kb = T.cast(kbl[c1], F32)
                            acc[2] += kb * knl[c1]
                            acc[3] += kb * qnl[c1]
                        acc[2] += T.shfl_xor(acc[2], 1)
                        acc[3] += T.shfl_xor(acc[3], 1)
                        acc[2] += T.shfl_xor(acc[2], 2)
                        acc[3] += T.shfl_xor(acc[3], 2)
                        acc[2] += T.shfl_xor(acc[2], 4)
                        acc[3] += T.shfl_xor(acc[3], 4)
                        if part == 0:
                            # scale by w_j HERE, once: the contribution loop below is thread-per-row, so
                            # leaving it there costs 128 x 16 x 2 redundant multiplies per program.
                            dk_s[g, pb, jr] = acc[2] * ((w_s[s, jr]))
                            dq_s[g, pb, jr] = acc[3] * ((w_s[s, jr]))
                        if warp < R:                                       # factor dots: dU[r] = U^T[r].kn, U^T[r].qn (one warp per r)
                            acc[6] = 0.0
                            acc[7] = 0.0
                            for i in T.unroll(4):
                                uf = T.cast(UQ_s[s, warp, lane + 32 * i], F32) * pc_s[s, lane + 32 * i]
                                acc[6] += uf * kn_s[s, ((lane + 32 * i) // 16) * 20 + lane % 16]
                                acc[7] += uf * qn_s[s, ((lane + 32 * i) // 16) * 20 + lane % 16]
                            acc[6] = T.warp_reduce_sum(acc[6])
                            acc[7] = T.warp_reduce_sum(acc[7])
                            if lane == 0:
                                dk_s[g, pb, L + warp] = acc[6]
                                dq_s[g, pb, L + warp] = acc[7]
                        T.sync_threads(bar, 128)
                        # ---- contribution, correction, output, append (thread-per-row v = t) ----
                        sv = skv_s[s, K + t]
                        mq = (mqy_s[g, pb, t, 0] + mqy_s[g, pb, t, 2]) * kfmax * sv
                        yq = (mqy_s[g, pb, t, 1] + mqy_s[g, pb, t, 3]) * kfmax * sv
                        acc[4] = 0.0
                        acc[5] = 0.0
                        for j in T.unroll(L):
                            ub = T.cast(Ub_s[s, j, t], F32)                 # dk_s / dq_s already carry w_j
                            acc[4] += ub * dk_s[g, pb, j] * ((1.0))
                            acc[5] += ub * dq_s[g, pb, j] * ((1.0))
                        acc[6] = 0.0                                        # rank-R part of the checkpoint: Q^T[r][v] (U^T[r].kn)
                        acc[7] = 0.0
                        for r in T.unroll(R):
                            qf = T.cast(UQ_s[s, r, K + t], F32)
                            acc[6] += qf * dk_s[g, pb, L + r]
                            acc[7] += qf * dq_s[g, pb, L + r]
                        u = beta * (v0 - (mq + acc[6] + acc[4]))
                        y = yq + acc[7] + acc[5] + kq * u
                        o[n, hv, t] = T.cast(y, BF16)
                        Ubuf[slot, hv, h, t] = T.cast(u, BF16)
                        Kbuf[slot, hv, h, t] = T.cast(kn, BF16)
                        if t == 0:
                            Wbuf[slot, hv, h] = 1.0
                            hcnt[slot, hv] = h + 1
                    T.fence_proxy_async()
                    T.mbarrier_arrive(consumed[s])
    return step

@tilelang.jit(pass_configs={"tl.disable_thread_storage_sync": True})
def make_flush(NS, HV, K, V, L, SS, R=4, ITERS=1, SMS=148, NCG=3, MAXIT=128, PAD=8, KDA=False, MMT="bf16", PNMIN=1e-20, NMAX=0, NTP=0, RSG=None):
    """End-of-window flush: rebuild the state of every head whose window is full, re-extract the Compensator Tokens and
    re-quantize the smoothed residual.  KDA=True selects the per-channel gate.

    For each due (sequence, head) program the kernel rebuilds B = decay * deq(S0) + sum_j w_j u_j k_j^T in two 64-column halves
    and keeps it in shared memory as fp16; the old rank-R part A = decay * Q_old U_old^T is carried algebraically.  ITERS rounds
    of subspace iteration from the fixed start Q0 (CholeskyQR2 orthogonalization) give the new rank-R factors Q, U; the residual
    E = B + A - Q U^T is smoothed by s_k[c] = sqrt(mean_v |E[v, c]|), quantized to INT8 with per-row scales, and the window is reset.
    Due programs are compacted on the device, so the launch is CUDA-graph safe; NMAX bounds the rows of one call (0 = NS).
    Head dims (K, V) in {(128, 128), (64, 128), (64, 64)}; NTP / RSG are bitwise-neutral schedule choices for the 64-dim shapes."""
    # Gbuf holds fp16 factors; it is declared bf16 so the TMA target (Kbl_s, later the lo key tile) matches -- the bits are reinterpreted on read
    assert (K, V) in ((128, 128), (64, 128), (64, 64)), f"make_flush: (K, V) = {(K, V)}"
    NEW = (K, V) != (128, 128)
    assert (1 <= R <= 4 or (R == 8 and not KDA and not NEW)) and PAD == 8
    assert not NEW or (not KDA and L <= 16), "K or V = 64: GDN, L <= 16, R <= 4"
    KB_T, UB_T, WB_T, PB_T, HC_T = _ring_types(NS, HV, L, K, V, KA=K if KDA else 1)
    assert KDA or L <= 16 or L == 32
    NB = K // 64                                    # 64-column blocks of the key dimension (the rebuild / residual halves)
    HP = 128 // V                                   # heads per program (V = 64: two heads stacked into the 128 rows)
    VR = V * HP                                     # rows per program (= 128)
    MTP = K // 64                                   # m-tiles per warp of P = B^T Q (K x R): columns 16 MTP w ..
    NUB = HP * NB                                   # U^T blocks of UQ_s (head-major), then 2 Q^T blocks (V = 128: two halves; V = 64: one per head)
    NUQB = NUB + 2
    LPR = K // 16                                   # quantisation: lanes per row (16 columns each)
    RPP = 32 // LPR                                 # rows per pass
    VMD = float(V)
    F16 = "float16"
    N = T.symbolic("N")
    KW = K // 4
    NPROD = 32
    NCONS = 128 * NCG
    NT = NPROD + NCONS
    NWARPS = NT // 32
    NHG = (HV + 31) // 32                              # head groups of 32 lanes: the compaction's virtual rows are (sequence, group)
    NDUE = min(NS - 8, NMAX) if NMAX else NS - 8      # rows that can be due at once: one call's rows (NMAX = the engine's batch cap) or the whole table
    if HP == 1:
        MAXIT = max(MAXIT, (NDUE * HV + SMS - 1) // SMS + 1)   # dlist capacity: a CTA's share of all programs when every such row is due
    else:
        MAXIT = max(MAXIT, HP * (((NDUE * HV + HP - 1) // HP + SMS - 1) // SMS) + 1)   # HP heads per program
    KMAX = min((NS * NHG + NWARPS - 1) // NWARPS, 40)  # n-steps per warp kept in registers by the compaction (3 x 40 registers; N <= 40 * NWARPS =
                                                 # 520 sequences at NCG = 3, vLLM's max_num_seqs 512); larger batches take a serial tail
                                                 # with the flags recomputed in the rank pass (64 steps spilled: +3 us none-due)
    NSTAGE = NCG
    LP = 16 if (L < 16 and not KDA) else L          # ring rows in smem: the rebuild mmas take 16 entries
    NKC = LP // 16 if not KDA else 1                # 16-entry chunks of the ring rebuild (L = 32: 2)
    RA = 4 if R <= 4 else 8                         # factor rows of the R x R corrections / CholeskyQR (R = 8: 8)
    NG = RA * (RA + 1) // 2                         # packed lower-triangular Gram / Cholesky entries
    NTP = NTP or (4 if K == 64 else 8)              # n-tiles (8 columns) per rebuild / residual pass: 8 = a 64-column half (64 fp32 accumulators per thread);
    assert NTP in (4, 8) and (NTP == 8 or (not KDA and NKC == 1))   # 4 = 32 columns (32 accumulators: K = 64 at 128 registers without spills, all due -10 %)
    NPS = K // (8 * NTP)                            # column passes (128: two halves)
    RSG = NEW if RSG is None else bool(RSG)          # Gram partials by reduce-scatter (bitwise the butterfly; K / V = 64 default, opt-in at 128 x 128)
    KP = K + PAD                                 # St_s row stride: conflict-free fragment stores, ldmatrix rows and 16 B row reads; the pad holds the fp16 Q tile
    KV8 = 2 * VR + 8                             # UQn_s row stride (halfwords): 528 B -> conflict-free ldmatrix rows (= K + V + 8 at 128 x 128;
                                                 # kept at K / V = 64: the fp32 Q plane needs VR rows, rsk VR entries)
    P16OFF = 4 * KV8                             # fp16 P tile (K x 8) at halfword offset 4 rows into UQn_s (after the fp32 Q plane)
    assert 4 * KV8 * 2 >= (VR + 4) * 4 * 4 and 4 * KV8 >= K * 8
    NQL, QLW = V, KW       # the int8 residual tile as V lines of K / 4 words
    SKVT = F32
    D16S, D16T, D16D = (((NS, HV, V, K // 2), (2 * SS, 2 * V * K, K // 2, 1), "int16"))

    @T.prim_func
    def flush(Sq: T.StridedTensor((NS, HV, NQL, QLW), (SS, V * K, QLW, 1), I32),
              Sk: T.StridedTensor((NS, HV, K), (SS, V * K, 1), F32),
              Sv: T.StridedTensor((NS, HV, V), (SS, V * K, 1), F32),
              Skv: T.StridedTensor((NS, HV, K + V), (SS, V * K, 1), SKVT),
              Kbuf: KB_T, Ubuf: UB_T,
              Wbuf: WB_T, Pbuf: PB_T, Gbuf: T.Tensor((NS, HV, L, K) if KDA else (1, 1, 1, 1), BF16), hcnt: HC_T,
              Sq16: T.StridedTensor(D16S, D16T, D16D),                                                # int16 view of the int8 tile (16 B row stores)
              UQ: T.StridedTensor((NS, HV, R, K + V), (2 * SS, 2 * V * K, K + V, 1), F16),           # fp16 factors U^T | Q^T after the scales
              Q0: T.Tensor((V, R), F32),                                                             # deterministic start of the subspace iteration
              idx: T.Tensor((N,), I32)):
        with T.Kernel(SMS, threads=NPROD + NCONS) as bid:
            Qs = T.alloc_shared((NSTAGE, HP * NQL, QLW), I32)         # HP = 2: head h in rows 64h..
            skv_s = T.alloc_shared((NSTAGE, HP * (K + V)), SKVT)     # HP = 2: head h at h (K + V)
            Kb_s = T.alloc_shared((NSTAGE, HP * NB, LP, 64), BF16)   # NB 128 B-wide column blocks per head, 128 B swizzle (conflict-free ldmatrix)
            Ub_s = T.alloc_shared((NSTAGE, 2, LP, 64), BF16)         # V = 128: two 64-row halves; V = 64: one per head
            Wb_s = T.alloc_shared((NSTAGE, 32 * HP), F32)            # 128 B per stage and head (TMA destination alignment)
            UQ_s = T.alloc_shared((NSTAGE, NUQB, 8, 64), F16)  # old factors U^T | Q^T in 128 B-wide column blocks (swizzled: conflict-free ldmatrix rows), k-rows padded to 8 (rows R.. zero)
            St_s = T.alloc_shared((NCG, VR, KP), F16)          # B (fp16) for the iteration, then the residual E in place; columns K.. = the fp16 Q tile (V x 8, columns R.. zero)
            UQn_s = T.alloc_shared((NCG, 8, KV8), F16)         # new -U^T (rows R.. zero) for the residual mma; aliases: PQ_s (fp32 Q rows), P16 (fp16 P tile), cpart (column partials)
            PQ_s = T.view(UQn_s, (NCG, 2, KV8 // 2, 4), F32)   # plane 0 = Q (V x R) fp32 for the CholeskyQR rows
            PQ8 = T.view(UQn_s, (NCG, 8 * KV8 // 16, 8), F32)  # R = 8: Q (V x 8) fp32 over the whole group buffer
            UQn1 = T.view(UQn_s, (NCG, 8 * KV8), F16)          # flat view: the fp16 P tile P16[c][r] at P16OFF + 8 c + r
            cpart = T.view(UQn_s, (NCG, 8, KV8 // 2), F32)     # per-warp column partials of |E| (rows 0..3)
            rsk = T.view(UQn_s, (NCG, 8, KV8 // 2), F32)       # reciprocal column scales: row 6 of the cpart view (free once the partials are read)
            dlist = T.alloc_shared((MAXIT,), I32)          # this CTA's due programs p (global rank == bid mod SMS), in rank order
            ndue_s = T.alloc_shared((1,), I32)
            wsum_s = T.alloc_shared((NWARPS,), I32)        # per-warp due counts of the block-wide compaction
            if R > 4:
                gp_s = T.alloc_shared((NCG, 4, 40), F32)   # R = 8: per-warp Gram partials (the 8-column fp32 Q plane fills UQn_s)
            if KDA:
                Kbl_s = T.alloc_shared((NSTAGE, 2, L, K // 2), BF16)     # per-entry decay factors (TMA target, fp16 bits) -> lo part of the decayed keys
                pb_s = T.alloc_shared((NSTAGE, K), F32)
                pc_s = T.alloc_shared((NCG, K), F32)
            loaded = T.alloc_barrier([32] * NSTAGE)
            consumed = T.alloc_barrier([128] * NSTAGE)
            if KDA:
                T.annotate_layout({Qs: make_swizzled_layout(Qs), Kb_s: make_swizzled_layout(Kb_s), Ub_s: make_swizzled_layout(Ub_s), Kbl_s: make_swizzled_layout(Kbl_s), UQ_s: make_swizzled_layout(UQ_s)})
            else:
                T.annotate_layout({Qs: make_swizzled_layout(Qs), Kb_s: make_swizzled_layout(Kb_s), Ub_s: make_swizzled_layout(Ub_s), UQ_s: make_swizzled_layout(UQ_s)})
            tx = T.get_thread_binding()
            # ---- block-wide balanced compaction (no atomics): every CTA scans all N * HV due flags -- warp w takes the virtual rows (sequence n, head group of 32)
            #      n = w, w + NWARPS, ..., lane hv the head (one broadcast idx load + one coalesced 128 B hcnt load per n; an uncoalesced
            #      gather costs 32 L1 lookups per warp-load and was 8x slower) -- ranks the due programs with ballot/popc in the fixed order
            #      (warp, n-step, hv) and keeps the ranks == bid (mod SMS).  The flush is latency-bound at 1/16 duty (1-3 programs per
            #      group): the static assignment's binomial imbalance (max 12 of mean 3.6 per CTA at bs 256) cost ~1.4x; this gives +-1.
            #      dlist holds q = slot * HV + hv. ----
            tc = T.alloc_local((4,), "int64")
            wlane = tx % 32
            wid = tx // 32
            kn = T.alloc_local((1,), I32)
            kn[0] = T.ceildiv(N * NHG, NWARPS)                # n-steps per warp (virtual rows r = n * NHG + head group)
            cnt = T.alloc_local((1,), I32)
            cnt[0] = 0
            slk = T.alloc_local((KMAX,), I32)
            hck = T.alloc_local((KMAX,), I32)
            mk = T.alloc_local((KMAX,), "uint32")
            for k in T.unroll(KMAX):                            # all loads in flight at once (explicit clamps: no guard branches)
                slk[k] = idx[T.max(T.min((wid + NWARPS * k) // NHG, N - 1), 0)]
            for k in T.unroll(KMAX):
                hck[k] = hcnt[T.min(T.max(slk[k], 0), NS - 1), T.min(((wid + NWARPS * k) % NHG) * 32 + wlane, HV - 1)]
            for k in T.unroll(KMAX):
                f = T.if_then_else((k < kn[0]) & (wid + NWARPS * k < N * NHG) & (slk[k] > 0) & (hck[k] == L)
                                   & (((wid + NWARPS * k) % NHG) * 32 + wlane < HV), 1, 0)
                mk[k] = T.call_extern("uint32", "__ballot_sync", T.uint32(0xFFFFFFFF), f)
                cnt[0] += T.call_extern(I32, "__popc", mk[k])
            sl8 = T.alloc_local((8,), I32)
            hc8 = T.alloc_local((8,), I32)
            for ck in T.serial(T.ceildiv(kn[0] - KMAX, 8)):    # tail (N > KMAX * NWARPS only): serial chunks, flags recomputed in the rank pass
                for j in T.unroll(8):
                    sl8[j] = idx[T.max(T.min((wid + NWARPS * (KMAX + ck * 8 + j)) // NHG, N - 1), 0)]
                for j in T.unroll(8):
                    hc8[j] = hcnt[T.min(T.max(sl8[j], 0), NS - 1), T.min(((wid + NWARPS * (KMAX + ck * 8 + j)) % NHG) * 32 + wlane, HV - 1)]
                for j in T.unroll(8):
                    k = KMAX + ck * 8 + j
                    f = T.if_then_else((k < kn[0]) & (wid + NWARPS * k < N * NHG) & (sl8[j] > 0) & (hc8[j] == L)
                                       & (((wid + NWARPS * k) % NHG) * 32 + wlane < HV), 1, 0)
                    cnt[0] += T.call_extern(I32, "__popc", T.call_extern("uint32", "__ballot_sync", T.uint32(0xFFFFFFFF), f))
            if wlane == 0:
                wsum_s[wid] = cnt[0]
            T.sync_threads(14, NPROD + NCONS)
            base = T.alloc_local((1,), I32)
            base[0] = 0
            for ww in T.serial(NWARPS):
                base[0] += T.if_then_else(ww < wid, wsum_s[ww], 0)
            tot = T.alloc_local((1,), I32)
            tot[0] = 0
            for ww in T.serial(NWARPS):
                tot[0] += wsum_s[ww]
            T.sync_threads(14, NPROD + NCONS)                 # wsum_s reads done before dlist (which may share its smem) is written
            lmask = T.cast((T.int64(1) << T.cast(wlane, "int64")) - T.int64(1), "uint32")
            if cnt[0] > 0:                                    # warps without due programs skip the pass (masks past kn are zero)
                rst = T.alloc_local((2,), I32)                    # next rank == bid (mod SMS) at or above base, and its dlist index (one division per thread)
                if HP == 1:
                    rst[0] = base[0] + (bid - base[0] % SMS + SMS) % SMS
                    rst[1] = rst[0] // SMS
                else:                                             # HP = 2: program p = rank // 2 (ranks 2p, 2p + 1); target the next p == bid (mod SMS) with 2p + 1 >= base
                    rst[0] = base[0] // 2 + (bid - (base[0] // 2) % SMS + SMS) % SMS
                    rst[1] = rst[0] // SMS
                for k in T.unroll(KMAX):                        # static indexing of the mask registers; one predicate per step, no division
                    m = mk[k]
                    rank = base[0] + T.call_extern(I32, "__popc", m & lmask)
                    if HP == 1:
                        if (((m >> T.cast(wlane, "uint32")) & T.uint32(1)) == T.uint32(1)) & (rank == rst[0]):
                            dlist[rst[1]] = idx[(wid + NWARPS * k) // NHG] * HV + ((wid + NWARPS * k) % NHG) * 32 + wlane
                    else:
                        if (((m >> T.cast(wlane, "uint32")) & T.uint32(1)) == T.uint32(1)) & ((rank >> 1) == rst[0]):
                            dlist[rst[1] * 2 + (rank & 1)] = idx[(wid + NWARPS * k) // NHG] * HV + ((wid + NWARPS * k) % NHG) * 32 + wlane
                    base[0] += T.call_extern(I32, "__popc", m)
                    if HP == 1:
                        if rst[0] < base[0]:                    # the warp's rank sequence passed the target (warp-uniform; at most one target per step)
                            rst[0] += SMS
                            rst[1] += 1
                    else:
                        if rst[0] * 2 + 1 < base[0]:            # both ranks of the target program passed (a step spans <= 32 ranks < 2 SMS)
                            rst[0] += SMS
                            rst[1] += 1
                for ck in T.serial(T.ceildiv(kn[0] - KMAX, 8)):
                    for j in T.unroll(8):
                        sl8[j] = idx[T.max(T.min((wid + NWARPS * (KMAX + ck * 8 + j)) // NHG, N - 1), 0)]
                    for j in T.unroll(8):
                        hc8[j] = hcnt[T.min(T.max(sl8[j], 0), NS - 1), T.min(((wid + NWARPS * (KMAX + ck * 8 + j)) % NHG) * 32 + wlane, HV - 1)]
                    for j in T.unroll(8):
                        k = KMAX + ck * 8 + j
                        f = T.if_then_else((k < kn[0]) & (wid + NWARPS * k < N * NHG) & (sl8[j] > 0) & (hc8[j] == L)
                                           & (((wid + NWARPS * k) % NHG) * 32 + wlane < HV), 1, 0)
                        m = T.call_extern("uint32", "__ballot_sync", T.uint32(0xFFFFFFFF), f)
                        if f == 1:
                            rank = base[0] + T.call_extern(I32, "__popc", m & lmask)
                            if HP == 1:
                                if rank % SMS == bid:
                                    dlist[rank // SMS] = sl8[j] * HV + ((wid + NWARPS * k) % NHG) * 32 + wlane
                            else:
                                if (rank >> 1) % SMS == bid:
                                    dlist[((rank >> 1) // SMS) * 2 + (rank & 1)] = sl8[j] * HV + ((wid + NWARPS * k) % NHG) * 32 + wlane
                        base[0] += T.call_extern(I32, "__popc", m)
            if tx == 0:
                if HP == 1:
                    ndue_s[0] = T.max(T.ceildiv(tot[0] - bid, SMS), 0)
                else:
                    ndue_s[0] = T.max(T.ceildiv((tot[0] + 1) // 2 - bid, SMS), 0)     # programs: ceil(due heads / 2)
            if R < 8:
                if tx < NPROD:
                    for i in T.serial(NSTAGE * NUQB * (8 - R) * 64 // NPROD):     # zero the padding k-rows of UQ_s once (only rows < R are ever written)
                        e = tx + NPROD * i
                        UQ_s[e // (NUQB * (8 - R) * 64), (e // ((8 - R) * 64)) % NUQB, R + (e % ((8 - R) * 64)) // 64, e % 64] = T.cast(0.0, F16)
            if LP != L:
                if tx < NPROD:
                    for i in T.serial(NSTAGE * 2 * (LP - L) * 64 // NPROD):   # zero the ring's padding rows L..15 once (the TMA writes rows < L only)
                        e = tx + NPROD * i
                        if HP * NB == 2:
                            Kb_s[e // (2 * (LP - L) * 64), (e // ((LP - L) * 64)) % 2, L + (e % ((LP - L) * 64)) // 64, e % 64] = T.cast(0.0, BF16)
                        Ub_s[e // (2 * (LP - L) * 64), (e // ((LP - L) * 64)) % 2, L + (e % ((LP - L) * 64)) // 64, e % 64] = T.cast(0.0, BF16)
                    if HP * NB == 1:
                        for i in T.serial(NSTAGE * (LP - L) * 64 // NPROD):
                            e = tx + NPROD * i
                            Kb_s[e // ((LP - L) * 64), 0, L + (e % ((LP - L) * 64)) // 64, e % 64] = T.cast(0.0, BF16)
                    for i in T.serial(NSTAGE):
                        if tx < LP - L:
                            Wb_s[i, L + tx] = 0.0
                            if HP == 2:
                                Wb_s[i, 32 + L + tx] = 0.0
            T.sync_threads(14, NPROD + NCONS)
            if tx < NPROD:
                T.fence_proxy_async()
                for k in T.serial(ndue_s[0]):
                    if HP == 1:
                        q = dlist[k]
                    else:
                        q = dlist[2 * k]                                         # head A of program k (ranks 2p, 2p + 1 with p = bid + SMS k)
                        qb = T.if_then_else(2 * (bid + SMS * k) + 1 < tot[0], dlist[2 * k + 1], q)   # head B; a missing one repeats A (writes masked)
                    s = k % NCG
                    ig = k // NCG
                    hv = q % HV
                    slot = q // HV
                    T.mbarrier_wait_parity(consumed[s], (ig & 1) ^ 1)
                    if not NEW:
                        T.tma_copy(Sq[slot, hv, :, :], Qs[s, :, :], barrier=loaded[s])
                        T.tma_copy(Skv[slot, hv, :], skv_s[s, :], barrier=loaded[s])
                        if LP != L:
                            T.tma_copy(Kbuf[slot, hv, :, 0:K // 2], Kb_s[s, 0, 0:L, :], barrier=loaded[s])
                            T.tma_copy(Kbuf[slot, hv, :, K // 2:K], Kb_s[s, 1, 0:L, :], barrier=loaded[s])
                            T.tma_copy(Ubuf[slot, hv, :, 0:V // 2], Ub_s[s, 0, 0:L, :], barrier=loaded[s])
                            T.tma_copy(Ubuf[slot, hv, :, V // 2:V], Ub_s[s, 1, 0:L, :], barrier=loaded[s])
                        else:
                            T.tma_copy(Kbuf[slot, hv, :, 0:K // 2], Kb_s[s, 0, :, :], barrier=loaded[s])
                            T.tma_copy(Kbuf[slot, hv, :, K // 2:K], Kb_s[s, 1, :, :], barrier=loaded[s])
                            T.tma_copy(Ubuf[slot, hv, :, 0:V // 2], Ub_s[s, 0, :, :], barrier=loaded[s])
                            T.tma_copy(Ubuf[slot, hv, :, V // 2:V], Ub_s[s, 1, :, :], barrier=loaded[s])
                        T.tma_copy(Wbuf[slot, hv, :], Wb_s[s, 0:L], barrier=loaded[s])
                        for cb in T.unroll(4):
                            T.tma_copy(UQ[slot, hv, :, cb * 64:(cb + 1) * 64], UQ_s[s, cb, 0:R, :], barrier=loaded[s])
                        if KDA:
                            T.tma_copy(Gbuf[slot, hv, :, 0:K // 2], Kbl_s[s, 0, :, :], barrier=loaded[s])
                            T.tma_copy(Gbuf[slot, hv, :, K // 2:K], Kbl_s[s, 1, :, :], barrier=loaded[s])
                            T.tma_copy(Pbuf[slot, hv, :], pb_s[s, :], barrier=loaded[s])
                    elif HP == 1:                                                # K = 64, V = 128: one key block, two value halves, 3 factor blocks
                        T.tma_copy(Sq[slot, hv, :, :], Qs[s, :, :], barrier=loaded[s])
                        T.tma_copy(Skv[slot, hv, :], skv_s[s, :], barrier=loaded[s])
                        T.tma_copy(Kbuf[slot, hv, :, :], Kb_s[s, 0, 0:L, :], barrier=loaded[s])
                        T.tma_copy(Ubuf[slot, hv, :, 0:64], Ub_s[s, 0, 0:L, :], barrier=loaded[s])
                        T.tma_copy(Ubuf[slot, hv, :, 64:128], Ub_s[s, 1, 0:L, :], barrier=loaded[s])
                        T.tma_copy(Wbuf[slot, hv, :], Wb_s[s, 0:L], barrier=loaded[s])
                        for cb in T.unroll(NUQB):
                            T.tma_copy(UQ[slot, hv, :, cb * 64:(cb + 1) * 64], UQ_s[s, cb, 0:R, :], barrier=loaded[s])
                    else:                                                        # V = 64: heads A (q) and B (qb) of the program
                        hvb = qb % HV
                        slotb = qb // HV
                        T.tma_copy(Sq[slot, hv, :, :], Qs[s, 0:64, :], barrier=loaded[s])
                        T.tma_copy(Sq[slotb, hvb, :, :], Qs[s, 64:128, :], barrier=loaded[s])
                        T.tma_copy(Skv[slot, hv, :], skv_s[s, 0:K + V], barrier=loaded[s])
                        T.tma_copy(Skv[slotb, hvb, :], skv_s[s, K + V:2 * (K + V)], barrier=loaded[s])
                        T.tma_copy(Kbuf[slot, hv, :, :], Kb_s[s, 0, 0:L, :], barrier=loaded[s])
                        T.tma_copy(Kbuf[slotb, hvb, :, :], Kb_s[s, 1, 0:L, :], barrier=loaded[s])
                        T.tma_copy(Ubuf[slot, hv, :, :], Ub_s[s, 0, 0:L, :], barrier=loaded[s])
                        T.tma_copy(Ubuf[slotb, hvb, :, :], Ub_s[s, 1, 0:L, :], barrier=loaded[s])
                        T.tma_copy(Wbuf[slot, hv, :], Wb_s[s, 0:L], barrier=loaded[s])
                        T.tma_copy(Wbuf[slotb, hvb, :], Wb_s[s, 32:32 + L], barrier=loaded[s])
                        T.tma_copy(UQ[slot, hv, :, 0:K], UQ_s[s, 0, 0:R, :], barrier=loaded[s])           # U^T blocks: head A, head B
                        T.tma_copy(UQ[slotb, hvb, :, 0:K], UQ_s[s, 1, 0:R, :], barrier=loaded[s])
                        T.tma_copy(UQ[slot, hv, :, K:K + V], UQ_s[s, 2, 0:R, :], barrier=loaded[s])       # Q^T blocks: head A, head B
                        T.tma_copy(UQ[slotb, hvb, :, K:K + V], UQ_s[s, 3, 0:R, :], barrier=loaded[s])
                    T.mbarrier_arrive(loaded[s])
            else:
                g = (tx - NPROD) // 128
                t = (tx - NPROD) % 128
                w = t // 32
                lane = t % 32
                q4 = lane % 4
                r0 = lane // 4
                bar = 8 + g
                s = g
                if HP == 2:
                    hw = w // 2                             # V = 64: this warp's (and thread's) head: rows 64 hw .. 64 hw + 63
                qw = T.alloc_local((4,), I32)
                svr_l = T.alloc_local((4,), F32)
                rmax = T.alloc_local((4,), F32)
                afrag = T.alloc_local((32 * NKC if not KDA else 16,), BF16)
                araw = T.alloc_local((8,), BF16)
                wv = T.alloc_local((4 * NKC,), F32)
                bfrag = T.alloc_local((32 if KDA else 16,), BF16)  # 2 x (ring-buffer B tile, KDA: hi + lo) -- alternating per unrolled step
                afrag2 = T.alloc_local((8,), F16)           # rank-R A fragments (m16n8k8) per m-tile: old Q^T (residual)
                afrag3 = T.alloc_local((8,), F16)           # rank-R A fragments of the new factors (residual)
                cfrag = T.alloc_local((8 * NTP,), F32)      # one column pass of B / E in fragment layout: index (ntl * 4 + (mt * 2 + r)) * 2 + c1, ntl = local n-tile
                efrag = T.alloc_local((8,), F16)            # ldmatrix.x4 result: B values of (ntl, top), (ntl, bottom), (ntl+1, top), (ntl+1, bottom)
                colp = T.alloc_local((K // 4,), F32)        # column partials of |E| (K / 8 n-tiles x 2 columns of this lane)
                skl = T.alloc_local((16,), F32)             # this lane's 16 column scales of the current half (dequant)
                ah = T.alloc_local((32,), F16)              # m16n8k16 A fragments (ldmatrix.x4 from the fp16 St_s): 2 k-parities x 2 m-tiles
                bh2 = T.alloc_local((8,), F16)              # B fragments (Q or P column pairs, fp16, ldmatrix.x2.trans): 2 k-parities
                ao = T.alloc_local((16,), F16)              # A fragments of the old factor (rows 0..7 of a 16-row m-tile; rows 8..15 zero): 2 k-parities
                bfr = T.alloc_local((8,), F16)              # 2 x rank-R B fragment (residual)
                bhi = T.alloc_local((4,), I32)              # |E| B fragments (sign bits cleared) of one n-tile, 2 k-steps
                aone = T.alloc_local((8,), F16)             # A fragment of ones (column sums on the tensor cores)
                c32 = T.alloc_local((16,), F32)             # 2 m-tiles x 4, two accumulator chains (even/odd k-steps)
                mc = T.alloc_local((4,), F32)               # R x R correction matrix accumulator (M = Q_old^T Q, M2 = U_old^T P), full in every warp
                mm = T.alloc_local((2 * RA,), F32)          # M[a][2 q4 + i] for a = 0..RA-1 (shuffled from lanes 4 a + q4)
                uo = T.alloc_local((4 * RA,), F32)          # old factor entries U_old[a][c] / Q_old[a][v] of this lane's 4 rows
                pr = T.alloc_local((RA,), F32)              # this thread's row of Q
                gg = T.alloc_local((NG,), F32)              # Gram entries (0,0) (1,0) (1,1) (2,0) (2,1) (2,2) (3,0) (3,1) (3,2) (3,3) ...
                ll = T.alloc_local((NG,), F32)              # Cholesky factor, same packing
                if RSG:
                    rsx = T.alloc_local((10,), F32)         # reduce-scatter of the per-warp Gram partials (K / V = 64 default)
                ac = T.alloc_local((1,), F32)
                uh = T.alloc_local((RA,), F16)
                qh = T.alloc_local((RA,), F16)
                el = T.alloc_local((L,), F32)               # KDA: this column's per-entry decay factors
                dcur = T.alloc_local((1,), F32)
                eh = T.alloc_local((32,), F16)              # quantisation: 2 x 2 x 8 halfwords of E (alternating per row pass)
                ph = T.alloc_local((8,), F16)               # fp16 pairs for stmatrix (vectorised cast -> one F2FP.PACK per pair)
                ex = T.alloc_local((16,), F32)              # quantisation: this lane's 16 scaled residual values of the row
                rk = T.alloc_local((16,), F32)              # quantisation: this lane's 16 reciprocal column scales
                tk = T.alloc_local((10,), "int64")
                tk2 = T.alloc_local((6,), "int64")
                tacc = T.alloc_local((10,), F32)
                for i in T.unroll(10):
                    tacc[i] = 0.0
                for i in T.unroll(8):
                    aone[i] = T.cast(1.0, F16)
                for i in T.unroll(2):                        # the old-factor A fragments: register pairs 1 and 3 (m-rows 8..15) stay zero
                    ao[2 + i] = T.cast(0.0, F16)
                    ao[6 + i] = T.cast(0.0, F16)
                    ao[10 + i] = T.cast(0.0, F16)
                    ao[14 + i] = T.cast(0.0, F16)
                if R <= 4:
                    if HP == 1:
                        for r in T.unroll(4):                    # the fp16 Q tile's padding columns R..7 stay zero (this thread's row)
                            St_s[g, t, K + 4 + r] = T.cast(0.0, F16)
                        for r in T.unroll(4 - R):
                            St_s[g, t, K + R + r] = T.cast(0.0, F16)
                    else:                                        # HP = 2: the other head's 4 columns (block-diagonal Q tile) and this head's R..3
                        for r in T.unroll(4):
                            St_s[g, t, K + 4 * (1 - hw) + r] = T.cast(0.0, F16)
                        for r in T.unroll(4 - R):
                            St_s[g, t, K + 4 * hw + R + r] = T.cast(0.0, F16)
                else:
                    for r in T.unroll(8 - R):
                        St_s[g, t, K + R + r] = T.cast(0.0, F16)
                for it0 in T.serial(T.ceildiv(ndue_s[0] - g, NCG)):
                    k = g + it0 * NCG
                    if HP == 1:
                        q = dlist[k]
                    else:
                        hbk = 2 * (bid + SMS * k) + 1 < tot[0]   # the program's head B exists (an odd due count leaves the last program one head)
                        qa = dlist[2 * k]
                        qb = T.if_then_else(hbk, dlist[2 * k + 1], qa)
                        q = T.if_then_else(w < 2, qa, qb)         # this warp's head
                        hok = (w < 2) | hbk                       # its writes are live
                    hv = q % HV
                    T.mbarrier_wait_parity(loaded[s], it0 & 1)
                    slot = q // HV
                    if KDA:
                        # pre-pass (thread per column c = t): pc = exp(G); decayed keys w_j k_j D_j as bf16 hi (in place) + lo (over the factors);
                        # U^T pre-multiplied by pc (the checkpoint's column decay)
                        pcc = T.exp(pb_s[s, t])
                        pc_s[g, t] = pcc
                        for jj in T.unroll(L):
                            el[jj] = T.cast(T.reinterpret(F16, Kbl_s[s, t // (K // 2), jj, t % (K // 2)]), F32)
                        T.sync_threads(bar, 128)                  # every lane holds its factors before the lo tile overwrites them
                        dcur[0] = 1.0
                        for jj in T.unroll(L):                    # backward: D_{L-1} = 1, D_j = D_{j+1} e_{j+1}
                            j = L - 1 - jj
                            kd = T.cast(Kb_s[s, t // (K // 2), j, t % (K // 2)], F32) * (Wb_s[s, j] * dcur[0])
                            khi = T.cast(kd, BF16)
                            Kb_s[s, t // (K // 2), j, t % (K // 2)] = khi
                            Kbl_s[s, t // (K // 2), j, t % (K // 2)] = T.cast(kd - T.cast(khi, F32), BF16)
                            dcur[0] = dcur[0] * el[j]
                        for r in T.unroll(R):                     # pc U^T in place (fp16: 2^-11 of the rank-R term, measured irrelevant next to the int8 residual, 10.46)
                            UQ_s[s, t // 64, r, t % 64] = T.cast(T.cast(UQ_s[s, t // 64, r, t % 64], F32) * pcc, F16)
                        T.sync_threads(bar, 128)
                        for i in T.unroll(4):
                            svr_l[i] = skv_s[s, K + w * 32 + (i // 2) * 16 + (i % 2) * 8 + lane // 4] * (1.0 / QMAX)
                    else:
                        if HP == 1:
                            pn = T.max(Pbuf[slot, hv, 0], PNMIN)     # floor: a fully decayed head has a denormal product (2e-41 seen on 9B layer 11) and 1/pn = inf
                        else:
                            pna = T.max(Pbuf[qa // HV, qa % HV, 0], PNMIN)
                            pnb = T.max(Pbuf[qb // HV, qb % HV, 0], PNMIN)
                            pn = T.if_then_else(w < 2, pna, pnb)    # this warp's head (rebuild, Q rows, residual)
                            pnl = T.if_then_else(q4 < 2, pna, pnb)  # this lane's P-tile columns 2 q4, 2 q4 + 1 belong to head q4 // 2
                        rpn = 1.0 / pn                           # poisoned the residual (Sk = inf -> NaN outputs, 10.49); the floor changes E by <= 1e-20 |Q_old U_old^T|
                    if KDA:
                        pn = 1.0
                    if not KDA:
                        for i in T.unroll(4):
                            if HP == 1:
                                svr_l[i] = skv_s[s, K + w * 32 + (i // 2) * 16 + (i % 2) * 8 + lane // 4] * (pn / QMAX)
                            else:
                                svr_l[i] = skv_s[s, hw * (K + V) + K + (w % 2) * 32 + (i // 2) * 16 + (i % 2) * 8 + lane // 4] * (pn / QMAX)
                    # (1) A fragments of the ring buffer u_j (GDN: w_j-scaled, bf16 hi + lo; KDA: exact bf16)
                    if KDA:
                        for mt in T.unroll(2):
                            T.ptx_ldmatrix(T.bool(True), 4, T.access_ptr(Ub_s[s, (w * 32 + mt * 16) // (V // 2), (lane % 8) + 8 * (lane // 16), (w * 32 + mt * 16) % (V // 2) + 8 * ((lane // 8) % 2)], "r", extent=8), T.access_ptr(afrag[mt * 8], "w", extent=8))
                    elif NKC > 1:
                        for kc in T.unroll(NKC):                 # L = 32: one set of hi / lo A fragments per 16-entry chunk
                            for i in T.unroll(4):
                                wv[kc * 4 + i] = Wb_s[s, 16 * kc + 2 * q4 + (i % 2) + 8 * (i // 2)]
                            for mt in T.unroll(2):
                                T.ptx_ldmatrix(T.bool(True), 4, T.access_ptr(Ub_s[s, (w * 32 + mt * 16) // (V // 2), 16 * kc + (lane % 8) + 8 * (lane // 16), (w * 32 + mt * 16) % (V // 2) + 8 * ((lane // 8) % 2)], "r", extent=8), T.access_ptr(araw[0], "w", extent=8))
                                for i in T.unroll(8):
                                    wu = wv[kc * 4 + (i % 2) + 2 * (i // 4)] * T.cast(araw[i], F32)
                                    afrag[kc * 32 + mt * 16 + i] = T.cast(wu, BF16)
                                    afrag[kc * 32 + mt * 16 + 8 + i] = T.cast(wu - T.cast(afrag[kc * 32 + mt * 16 + i], F32), BF16)
                    else:
                        for i in T.unroll(4):
                            if HP == 1:
                                wv[i] = Wb_s[s, 2 * q4 + (i % 2) + 8 * (i // 2)]
                            else:
                                wv[i] = Wb_s[s, hw * 32 + 2 * q4 + (i % 2) + 8 * (i // 2)]
                        for mt in T.unroll(2):                   # Ub_s block = 64-row half (V = 128) or head (V = 64)
                            T.ptx_ldmatrix(T.bool(True), 4, T.access_ptr(Ub_s[s, (w * 32 + mt * 16) // 64, (lane % 8) + 8 * (lane // 16), (w * 32 + mt * 16) % 64 + 8 * ((lane // 8) % 2)], "r", extent=8), T.access_ptr(araw[0], "w", extent=8))
                            for i in T.unroll(8):
                                wu = wv[(i % 2) + 2 * (i // 4)] * T.cast(araw[i], F32)
                                afrag[mt * 16 + i] = T.cast(wu, BF16)
                                afrag[mt * 16 + 8 + i] = T.cast(wu - T.cast(afrag[mt * 16 + i], F32), BF16)
                    # (2) NPS column passes of NTP n-tiles (128: two 64-column halves; K = 64: two 32-column quarters, 32 accumulators):
                    #     B = ring buffer (mma) + pn * deq(S0), out to St_s (fp16, stmatrix); the old rank-R part stays algebraic
                    for h in T.unroll(NPS):
                        # dequantise first (independent FFMA chains, no accumulator dependency), then the ring-buffer mma accumulates on top
                        for ntl in T.unroll(NTP):
                            for c1 in T.vectorized(2):
                                if KDA:
                                    skl[ntl * 2 + c1] = skv_s[s, (h * NTP + ntl) * 8 + q4 * 2 + c1] * pc_s[g, (h * NTP + ntl) * 8 + q4 * 2 + c1]
                                elif HP == 1:
                                    skl[ntl * 2 + c1] = skv_s[s, (h * NTP + ntl) * 8 + q4 * 2 + c1]
                                else:
                                    skl[ntl * 2 + c1] = skv_s[s, hw * (K + V) + (h * NTP + ntl) * 8 + q4 * 2 + c1]
                        for rr in ((T.unroll(4))):
                            v0 = w * 32 + (rr // 2) * 16 + (rr % 2) * 8 + r0
                            for jcl in T.unroll(NTP // 2):         # 16-column chunk = n-tiles 2jc, 2jc+1; lane q4 needs word q4 // 2 of each n-tile
                                jc = h * (NTP // 2) + jcl
                                for nn in T.unroll(2):
                                    qw[nn] = Qs[s, v0, jc * 4 + nn * 2 + q4 // 2]
                                for nn in T.unroll(2):
                                    ntl = jcl * 2 + nn
                                    pkw = qw[nn] ^ T.int32(-2139062144)
                                    for c1 in T.unroll(2):        # the magic-number subtraction must stay a separate exact fp32 op: folding -K*skl into an FFMA
                                        x = T.reinterpret(F32, T.call_extern(I32, "__byte_perm", pkw, T.int32(0x4B000000), T.int32(0x7650) + (q4 % 2) * 2 + c1)) - 8388736.0   # costs ulp(2^23 skl) = half a code
                                        cfrag[(ntl * 4 + rr) * 2 + c1] = x * svr_l[rr] * skl[ntl * 2 + c1]
                        # stmatrix.x4: matrices (ntl, rows 0-7), (ntl, rows 8-15), (ntl+1, rows 0-7), (ntl+1, rows 8-15) of m-tile mt
                        if NKC > 1:
                            for npl in T.unroll(4):
                                n0 = (h * 4 + npl) * 16
                                for kc in T.unroll(NKC):
                                    bo = ((npl * NKC + kc) % 2) * 8
                                    T.ptx_ldmatrix(T.bool(True), 4, T.access_ptr(Kb_s[s, n0 // (K // 2), 16 * kc + (lane % 8) + 8 * ((lane // 8) % 2), n0 % (K // 2) + 8 * (lane // 16)], "r", extent=8), T.access_ptr(bfrag[bo], "w", extent=8))
                                    for mt in T.unroll(2):
                                        for nn in T.unroll(2):
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", MMT, MMT, "fp32", afrag.data, kc * 32 + mt * 16, bfrag.data, bo + nn * 4, cfrag.data, ((npl * 2 + nn) * 4 + mt * 2) * 2, T.bool(False))
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", MMT, MMT, "fp32", afrag.data, kc * 32 + mt * 16 + 8, bfrag.data, bo + nn * 4, cfrag.data, ((npl * 2 + nn) * 4 + mt * 2) * 2, T.bool(False))
                        for npl in (T.unroll(0) if NKC > 1 else T.unroll(NTP // 2)):
                            n0 = (h * (NTP // 2) + npl) * 16
                            bo = (npl % 2) * (16 if KDA else 8)
                            if HP == 1:
                                T.ptx_ldmatrix(T.bool(True), 4, T.access_ptr(Kb_s[s, n0 // 64, (lane % 8) + 8 * ((lane // 8) % 2), n0 % 64 + 8 * (lane // 16)], "r", extent=8), T.access_ptr(bfrag[bo], "w", extent=8))
                            else:                                  # this warp's head's key block
                                T.ptx_ldmatrix(T.bool(True), 4, T.access_ptr(Kb_s[s, hw * NB + n0 // 64, (lane % 8) + 8 * ((lane // 8) % 2), n0 % 64 + 8 * (lane // 16)], "r", extent=8), T.access_ptr(bfrag[bo], "w", extent=8))
                            if KDA:
                                T.ptx_ldmatrix(T.bool(True), 4, T.access_ptr(Kbl_s[s, n0 // (K // 2), (lane % 8) + 8 * ((lane // 8) % 2), n0 % (K // 2) + 8 * (lane // 16)], "r", extent=8), T.access_ptr(bfrag[bo + 8], "w", extent=8))
                            for mt in T.unroll(2):
                                for nn in T.unroll(2):
                                    if KDA:
                                        T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", afrag.data, mt * 8, bfrag.data, bo + nn * 4, cfrag.data, ((npl * 2 + nn) * 4 + mt * 2) * 2, T.bool(False))
                                        T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", afrag.data, mt * 8, bfrag.data, bo + 8 + nn * 4, cfrag.data, ((npl * 2 + nn) * 4 + mt * 2) * 2, T.bool(False))
                                    else:
                                        T.ptx_mma("float32", "m16n8k16", "row", "col", MMT, MMT, "fp32", afrag.data, mt * 16, bfrag.data, bo + nn * 4, cfrag.data, ((npl * 2 + nn) * 4 + mt * 2) * 2, T.bool(False))
                                        T.ptx_mma("float32", "m16n8k16", "row", "col", MMT, MMT, "fp32", afrag.data, mt * 16 + 8, bfrag.data, bo + nn * 4, cfrag.data, ((npl * 2 + nn) * 4 + mt * 2) * 2, T.bool(False))
                        for mt in ((T.unroll(2))):
                            for npl in T.unroll(NTP // 2):
                                for jj in T.unroll(4):                     # 4 fp16 pairs: (ntl, rows 0-7), (ntl, rows 8-15), (ntl+1, ...) -- one F2FP.PACK each
                                    for c1 in T.vectorized(2):
                                        ph[jj * 2 + c1] = T.cast(cfrag[((npl * 2 + jj // 2) * 4 + mt * 2 + jj % 2) * 2 + c1], F16)
                                T.call_extern("handle", "tl::ptx_stmatrix_m8n8_x4",
                                              T.access_ptr(St_s[g, w * 32 + mt * 16 + 8 * ((lane // 8) % 2) + lane % 8, (h * NTP + npl * 2 + lane // 16) * 8], "w", extent=8),
                                              T.reinterpret(I32, ph[0:2]), T.reinterpret(I32, ph[2:4]), T.reinterpret(I32, ph[4:6]), T.reinterpret(I32, ph[6:8]))
                    # the iteration's start Q0 (fp16 tile in the St_s pad columns; this thread's row) -- before the barrier that publishes B
                    for r in T.unroll(R):
                        if HP == 1:
                            St_s[g, t, K + r] = T.cast(Q0[t, r], F16)
                        else:
                            St_s[g, t, K + 4 * hw + r] = T.cast(Q0[t % 64, r], F16)
                    T.sync_threads(bar, 128)
                    # ================= subspace iteration on B (fp16 mma) with the algebraic rank-R corrections =================
                    for itr in T.serial(ITERS + 1):
                        # ---- P = B^T Q  (K x R): A[m=c][k=v] = St_s[v][c] via ldmatrix.x4.trans; B[k=v][n=r] = Q16[v][r] via ldmatrix.x2.trans;
                        #      alongside M = Q_old^T Q (R x R): A[m=r][k=v] = Q_old^T (rows 8..15 zero), same B fragments ----
                        for i in T.unroll(16):
                            c32[i] = 0.0
                        for i in T.unroll(4):
                            mc[i] = 0.0
                        for k0 in T.unroll(VR // 16):
                            kp = k0 % 2                                    # alternating fragment registers: the next step's ldmatrix can overlap this step's mma
                            T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(St_s[g, k0 * 16 + 8 * ((lane // 8) % 2) + lane % 8, K], "r", extent=4), T.access_ptr(bh2[kp * 4], "w", extent=4))
                            for mt in T.unroll(MTP):                       # K = 64: one m-tile per warp (columns 16 w ..)
                                m0 = w * (16 * MTP) + mt * 16
                                T.ptx_ldmatrix(T.bool(True), 4, T.access_ptr(St_s[g, k0 * 16 + 8 * (lane // 16) + lane % 8, m0 + 8 * ((lane // 8) % 2)], "r", extent=8), T.access_ptr(ah[(kp * 2 + mt) * 8], "w", extent=8))
                                if HP == 1:
                                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", ah.data, (kp * 2 + mt) * 8, bh2.data, kp * 4, c32.data, (k0 % 2) * 8 + mt * 4, T.bool(False))
                                else:                                      # HP = 2: one even / odd chain pair per head of the rows (k0 // 4): a head's P never sums the other head's rows
                                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", ah.data, (kp * 2 + mt) * 8, bh2.data, kp * 4, c32.data, ((k0 // 4) * 2 + k0 % 2) * 4, T.bool(False))
                            if HP == 1:
                                T.ptx_ldmatrix(T.bool(False), 1, T.access_ptr(UQ_s[s, (K + k0 * 16) // 64, lane % 8, (K + k0 * 16) % 64], "r", extent=2), T.access_ptr(ao[kp * 8], "w", extent=2))
                                T.ptx_ldmatrix(T.bool(False), 1, T.access_ptr(UQ_s[s, (K + k0 * 16 + 8) // 64, lane % 8, (K + k0 * 16 + 8) % 64], "r", extent=2), T.access_ptr(ao[kp * 8 + 4], "w", extent=2))
                            else:                                          # rows v of head k0 // 4: its Q^T block; factor rows a of the other head -> zero rows 4..7
                                T.ptx_ldmatrix(T.bool(False), 1, T.access_ptr(UQ_s[s, NUB + k0 // 4, (lane % 8 + 4 * (k0 // 4)) % 8, (k0 * 16) % 64], "r", extent=2), T.access_ptr(ao[kp * 8], "w", extent=2))
                                T.ptx_ldmatrix(T.bool(False), 1, T.access_ptr(UQ_s[s, NUB + k0 // 4, (lane % 8 + 4 * (k0 // 4)) % 8, (k0 * 16 + 8) % 64], "r", extent=2), T.access_ptr(ao[kp * 8 + 4], "w", extent=2))
                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", ao.data, kp * 8, bh2.data, kp * 4, mc.data, 0, T.bool(False))
                        if HP == 1:
                            for i in T.unroll(8):
                                c32[i] += c32[8 + i]
                        else:                                              # this lane's columns 2 q4, 2 q4 + 1 belong to head q4 // 2: its chains only
                            for i in T.unroll(4):
                                c32[i] = T.if_then_else(q4 < 2, c32[i] + c32[4 + i], c32[8 + i] + c32[12 + i])
                        # ---- P += pn U_old M in the fragments: rows c = m0 + r0 (+8), columns n = 2 q4 + i; M[a][n] sits in lane 4 a + q4 ----
                        if R > 4:
                            for a in T.unroll(RA):
                                mm[a * 2] = T.shfl_sync(mc[0], a * 4 + q4)
                                mm[a * 2 + 1] = T.shfl_sync(mc[1], a * 4 + q4)
                            for mt in T.unroll(2):
                                for hh in T.unroll(2):
                                    cc = w * 32 + mt * 16 + hh * 8 + r0
                                    for a in T.unroll(RA):
                                        uo[(mt * 2 + hh) * RA + a] = T.if_then_else(a < R, T.cast(UQ_s[s, cc // 64, a, cc % 64], F32), 0.0)
                                    for i in T.unroll(2):
                                        ac[0] = 0.0
                                        for a in T.unroll(RA):
                                            ac[0] += uo[(mt * 2 + hh) * RA + a] * mm[a * 2 + i]
                                        c32[mt * 4 + hh * 2 + i] += pn * ac[0]
                        elif HP == 1:
                            for a in T.unroll(4):
                                mm[a * 2] = T.shfl_sync(mc[0], a * 4 + q4)
                                mm[a * 2 + 1] = T.shfl_sync(mc[1], a * 4 + q4)
                            for mt in T.unroll(MTP):
                                for hh in T.unroll(2):
                                    cc = w * (16 * MTP) + mt * 16 + hh * 8 + r0
                                    for a in T.unroll(4):
                                        uo[(mt * 2 + hh) * 4 + a] = T.if_then_else(a < R, T.cast(UQ_s[s, cc // 64, a, cc % 64], F32), 0.0)
                                    for i in T.unroll(2):
                                        c32[mt * 4 + hh * 2 + i] += pn * (uo[(mt * 2 + hh) * 4] * mm[i] + uo[(mt * 2 + hh) * 4 + 1] * mm[2 + i] + uo[(mt * 2 + hh) * 4 + 2] * mm[4 + i] + uo[(mt * 2 + hh) * 4 + 3] * mm[6 + i])
                        else:                                              # HP = 2: lane columns belong to head q4 // 2: M[4 h + a][n] from lane 16 h + 4 a + q4, U_old of head h
                            for a in T.unroll(4):
                                mm[a * 2] = T.shfl_sync(mc[0], 16 * (q4 // 2) + a * 4 + q4)
                                mm[a * 2 + 1] = T.shfl_sync(mc[1], 16 * (q4 // 2) + a * 4 + q4)
                            for mt in T.unroll(MTP):
                                for hh in T.unroll(2):
                                    cc = w * (16 * MTP) + mt * 16 + hh * 8 + r0
                                    for a in T.unroll(4):
                                        uo[(mt * 2 + hh) * 4 + a] = T.if_then_else(a < R, T.cast(UQ_s[s, (q4 // 2) * NB + cc // 64, a, cc % 64], F32), 0.0)
                                    for i in T.unroll(2):
                                        c32[mt * 4 + hh * 2 + i] += pnl * (uo[(mt * 2 + hh) * 4] * mm[i] + uo[(mt * 2 + hh) * 4 + 1] * mm[2 + i] + uo[(mt * 2 + hh) * 4 + 2] * mm[4 + i] + uo[(mt * 2 + hh) * 4 + 3] * mm[6 + i])
                        # ---- the fp16 P tile P16[c][r] (columns >= R zero: lanes q4 >= R/2 write zeros) via stmatrix.x2 (rows 0-7, rows 8-15 of each m-tile) ----
                        for mt in T.unroll(MTP):
                            for c1 in T.vectorized(2):
                                if HP == 1:
                                    ph[c1] = T.cast(T.if_then_else(2 * q4 + c1 < R, c32[mt * 4 + c1], 0.0), F16)
                                    ph[2 + c1] = T.cast(T.if_then_else(2 * q4 + c1 < R, c32[mt * 4 + 2 + c1], 0.0), F16)
                                else:                                      # HP = 2: columns 4 h + r, r < R
                                    ph[c1] = T.cast(T.if_then_else(2 * (q4 % 2) + c1 < R, c32[mt * 4 + c1], 0.0), F16)
                                    ph[2 + c1] = T.cast(T.if_then_else(2 * (q4 % 2) + c1 < R, c32[mt * 4 + 2 + c1], 0.0), F16)
                            T.call_extern("handle", "tl::ptx_stmatrix_m8n8_x2",
                                          T.access_ptr(UQn1[g, P16OFF + (w * (16 * MTP) + mt * 16 + 8 * ((lane // 8) % 2) + lane % 8) * 8], "w", extent=8),
                                          T.reinterpret(I32, ph[0:2]), T.reinterpret(I32, ph[2:4]))
                        T.sync_threads(bar, 128)
                        if itr < ITERS:
                            # ---- Q = B P  (V x R): A[m=v][k=c] = St_s[v][c] via ldmatrix.x4; B[k=c][n=r] = P16[c][r] via ldmatrix.x2.trans; alongside M2 = U_old^T P ----
                            for i in T.unroll(16):
                                c32[i] = 0.0
                            for i in T.unroll(4):
                                mc[i] = 0.0
                            for k0 in T.unroll(K // 16):
                                kp = k0 % 2
                                T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(UQn1[g, P16OFF + (k0 * 16 + 8 * ((lane // 8) % 2) + lane % 8) * 8], "r", extent=4), T.access_ptr(bh2[kp * 4], "w", extent=4))
                                for mt in T.unroll(2):
                                    m0 = w * 32 + mt * 16
                                    T.ptx_ldmatrix(T.bool(False), 4, T.access_ptr(St_s[g, m0 + 8 * ((lane // 8) % 2) + lane % 8, k0 * 16 + 8 * (lane // 16)], "r", extent=8), T.access_ptr(ah[(kp * 2 + mt) * 8], "w", extent=8))
                                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", ah.data, (kp * 2 + mt) * 8, bh2.data, kp * 4, c32.data, (k0 % 2) * 8 + mt * 4, T.bool(False))
                                if HP == 1:
                                    T.ptx_ldmatrix(T.bool(False), 1, T.access_ptr(UQ_s[s, (k0 * 16) // 64, lane % 8, (k0 * 16) % 64], "r", extent=2), T.access_ptr(ao[kp * 8], "w", extent=2))
                                    T.ptx_ldmatrix(T.bool(False), 1, T.access_ptr(UQ_s[s, (k0 * 16 + 8) // 64, lane % 8, (k0 * 16 + 8) % 64], "r", extent=2), T.access_ptr(ao[kp * 8 + 4], "w", extent=2))
                                else:                                      # factor rows a: U_old^T row a % 4 of head a // 4 (rows R..3 zero)
                                    T.ptx_ldmatrix(T.bool(False), 1, T.access_ptr(UQ_s[s, ((lane % 8) // 4) * NB + (k0 * 16) // 64, lane % 4, (k0 * 16) % 64], "r", extent=2), T.access_ptr(ao[kp * 8], "w", extent=2))
                                    T.ptx_ldmatrix(T.bool(False), 1, T.access_ptr(UQ_s[s, ((lane % 8) // 4) * NB + (k0 * 16 + 8) // 64, lane % 4, (k0 * 16 + 8) % 64], "r", extent=2), T.access_ptr(ao[kp * 8 + 4], "w", extent=2))
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", ao.data, kp * 8, bh2.data, kp * 4, mc.data, 0, T.bool(False))
                            for i in T.unroll(8):
                                c32[i] += c32[8 + i]
                            # ---- Q += pn Q_old M2 in the fragments, then the fp32 rows to PQ_s plane 0 for the CholeskyQR ----
                            if R > 4:
                                for a in T.unroll(RA):
                                    mm[a * 2] = T.shfl_sync(mc[0], a * 4 + q4)
                                    mm[a * 2 + 1] = T.shfl_sync(mc[1], a * 4 + q4)
                                for mt in T.unroll(2):
                                    for hh in T.unroll(2):
                                        vv = w * 32 + mt * 16 + hh * 8 + r0
                                        for a in T.unroll(RA):
                                            uo[(mt * 2 + hh) * RA + a] = T.if_then_else(a < R, T.cast(UQ_s[s, (K + vv) // 64, a, (K + vv) % 64], F32), 0.0)
                                        for i in T.unroll(2):
                                            ac[0] = 0.0
                                            for a in T.unroll(RA):
                                                ac[0] += uo[(mt * 2 + hh) * RA + a] * mm[a * 2 + i]
                                            c32[mt * 4 + hh * 2 + i] += pn * ac[0]
                            elif HP == 1:
                                for a in T.unroll(4):
                                    mm[a * 2] = T.shfl_sync(mc[0], a * 4 + q4)
                                    mm[a * 2 + 1] = T.shfl_sync(mc[1], a * 4 + q4)
                                for mt in T.unroll(2):
                                    for hh in T.unroll(2):
                                        vv = w * 32 + mt * 16 + hh * 8 + r0
                                        for a in T.unroll(4):
                                            uo[(mt * 2 + hh) * 4 + a] = T.if_then_else(a < R, T.cast(UQ_s[s, (K + vv) // 64, a, (K + vv) % 64], F32), 0.0)
                                        for i in T.unroll(2):
                                            c32[mt * 4 + hh * 2 + i] += pn * (uo[(mt * 2 + hh) * 4] * mm[i] + uo[(mt * 2 + hh) * 4 + 1] * mm[2 + i] + uo[(mt * 2 + hh) * 4 + 2] * mm[4 + i] + uo[(mt * 2 + hh) * 4 + 3] * mm[6 + i])
                            else:                                       # HP = 2: rows of head hw; M2[4 hw + a][n] from lane 16 hw + 4 a + q4 (lanes q4 // 2 == hw keep the result)
                                for a in T.unroll(4):
                                    mm[a * 2] = T.shfl_sync(mc[0], 16 * hw + a * 4 + q4)
                                    mm[a * 2 + 1] = T.shfl_sync(mc[1], 16 * hw + a * 4 + q4)
                                for mt in T.unroll(2):
                                    for hh in T.unroll(2):
                                        vv = w * 32 + mt * 16 + hh * 8 + r0
                                        for a in T.unroll(4):
                                            uo[(mt * 2 + hh) * 4 + a] = T.if_then_else(a < R, T.cast(UQ_s[s, NUB + hw, a, vv % 64], F32), 0.0)
                                        for i in T.unroll(2):
                                            c32[mt * 4 + hh * 2 + i] += pn * (uo[(mt * 2 + hh) * 4] * mm[i] + uo[(mt * 2 + hh) * 4 + 1] * mm[2 + i] + uo[(mt * 2 + hh) * 4 + 2] * mm[4 + i] + uo[(mt * 2 + hh) * 4 + 3] * mm[6 + i])
                            if R > 4:                                   # the 8-column plane covers P16: every warp is done reading it first
                                T.sync_threads(bar, 128)
                                for mt in T.unroll(2):
                                    PQ8[g, w * 32 + mt * 16 + r0, 2 * q4] = c32[mt * 4]
                                    PQ8[g, w * 32 + mt * 16 + r0, 2 * q4 + 1] = c32[mt * 4 + 1]
                                    PQ8[g, w * 32 + mt * 16 + r0 + 8, 2 * q4] = c32[mt * 4 + 2]
                                    PQ8[g, w * 32 + mt * 16 + r0 + 8, 2 * q4 + 1] = c32[mt * 4 + 3]
                            elif HP == 1:
                                if q4 * 2 < R:
                                    for mt in T.unroll(2):
                                        PQ_s[g, 0, w * 32 + mt * 16 + r0, 2 * q4] = c32[mt * 4]
                                        PQ_s[g, 0, w * 32 + mt * 16 + r0, 2 * q4 + 1] = c32[mt * 4 + 1]
                                        PQ_s[g, 0, w * 32 + mt * 16 + r0 + 8, 2 * q4] = c32[mt * 4 + 2]
                                        PQ_s[g, 0, w * 32 + mt * 16 + r0 + 8, 2 * q4 + 1] = c32[mt * 4 + 3]
                            else:                                       # HP = 2: this head's 4 columns (the other head's are B_hw P_other: dropped)
                                if (q4 // 2 == hw) & ((q4 % 2) * 2 < R):
                                    for mt in T.unroll(2):
                                        PQ_s[g, 0, w * 32 + mt * 16 + r0, 2 * (q4 % 2)] = c32[mt * 4]
                                        PQ_s[g, 0, w * 32 + mt * 16 + r0, 2 * (q4 % 2) + 1] = c32[mt * 4 + 1]
                                        PQ_s[g, 0, w * 32 + mt * 16 + r0 + 8, 2 * (q4 % 2)] = c32[mt * 4 + 2]
                                        PQ_s[g, 0, w * 32 + mt * 16 + r0 + 8, 2 * (q4 % 2) + 1] = c32[mt * 4 + 3]
                            T.sync_threads(bar, 128)
                            # ---- CholeskyQR2 of Q (V rows = 128 threads), relative pivot floor for rank-deficient heads.  Pass 1 in fp32 (shuffle tree);
                            #      pass 2's Gram on the tensor cores in fp16 (8 mma; the factors are stored in fp16 anyway and the residual uses the same
                            #      fp16 factors, so a 2^-11 non-orthonormality costs nothing).  A 3xTF32 Gram for pass 1 was slower (48 dependent mma). ----
                            if R > 4:
                                for r in T.unroll(RA):
                                    pr[r] = T.if_then_else(r < R, PQ8[g, t, r], 0.0)
                                for cq in range(2):
                                    for r in T.unroll(RA):
                                        for c in T.unroll(r + 1):
                                            gg[r * (r + 1) // 2 + c] = pr[r] * pr[c]
                                    for off in T.unroll(5):
                                        for i in T.unroll(NG):
                                            gg[i] += T.shfl_xor(gg[i], 16 >> off)
                                    if lane == 0:
                                        for i in T.unroll(NG):
                                            gp_s[g, w, i] = gg[i]
                                    T.sync_threads(bar, 128)
                                    for i in T.unroll(NG):
                                        gg[i] = gp_s[g, 0, i] + gp_s[g, 1, i] + gp_s[g, 2, i] + gp_s[g, 3, i]
                                    for j in T.unroll(RA):              # rows of L: off-diagonal L[j][k], then 1 / L[j][j] (relative pivot floor)
                                        for k in T.unroll(j):
                                            ac[0] = gg[j * (j + 1) // 2 + k]
                                            for m in T.unroll(k):
                                                ac[0] -= ll[j * (j + 1) // 2 + m] * ll[k * (k + 1) // 2 + m]
                                            ll[j * (j + 1) // 2 + k] = ac[0] * ll[k * (k + 1) // 2 + k]
                                        ac[0] = gg[j * (j + 1) // 2 + j]
                                        for m in T.unroll(j):
                                            ac[0] -= ll[j * (j + 1) // 2 + m] * ll[j * (j + 1) // 2 + m]
                                        ll[j * (j + 1) // 2 + j] = T.if_then_else(ac[0] > 1e-6 * gg[j * (j + 1) // 2 + j] + 1e-30, T.rsqrt(T.max(ac[0], 1e-30)), 0.0)   # dependent column: dropped
                                    for j in T.unroll(RA):              # forward substitution on this thread's row
                                        ac[0] = pr[j]
                                        for m in T.unroll(j):
                                            ac[0] -= pr[m] * ll[j * (j + 1) // 2 + m]
                                        pr[j] = ac[0] * ll[j * (j + 1) // 2 + j]
                                    for r in T.unroll(R):
                                        St_s[g, t, K + r] = T.cast(pr[r], F16)
                                    T.sync_threads(bar, 128)
                            for r in T.unroll(4 if R <= 4 else 0):
                                pr[r] = T.if_then_else(r < R, PQ_s[g, 0, t, r], 0.0)
                            for cq in range(2 if R <= 4 else 0):
                                                              # after a floored pass-1 pivot (rank-deficient head, e.g. svals 4.8 / 0.2 / 8e-3 / 1e-3) the 4th column has norm 1e-3..1e-5,
                                                              # its fp16 Gram entry is subnormal/zero, rsqrt blows the column up by 1e3..1e15 and U_4 = B^T q_4 with it
                                for r in T.unroll(4):
                                    for c in T.unroll(r + 1):
                                        gg[r * (r + 1) // 2 + c] = pr[r] * pr[c]
                                if not RSG:
                                    for off in ((T.unroll(5))):
                                        for i in T.unroll(10):
                                            gg[i] += T.shfl_xor(gg[i], 16 >> off)
                                    if lane == 0:
                                        for i in T.unroll(10):
                                            cpart[g, 4, w * 16 + i] = gg[i]           # per-warp Gram partials (cpart row 4: free during the iteration)
                                else:
                                    # K / V = 64: reduce-scatter over the butterfly's xor pairs 16, 8, 4, 2, 1 -- every sum is formed by the same
                                    # additions in the same order (bitwise the butterfly's), the lane pairs exchange only the half they do not keep:
                                    # 12 shuffles + 12 adds instead of 50 + 50, one store instruction instead of ten.  Value i ends in the lanes
                                    # with 5 b4 + 3 b3 + 2 b2 + b1 = i (b = lane bits; sub-groups padded with zeros to 6, then 4).
                                    for j in T.unroll(5):
                                        rsx[j] = T.if_then_else(lane >= 16, gg[5 + j], gg[j]) + T.shfl_xor(T.if_then_else(lane >= 16, gg[j], gg[5 + j]), 16)
                                    rsx[5] = 0.0
                                    for j in T.unroll(3):
                                        rsx[6 + j] = T.if_then_else((lane & 8) != 0, rsx[3 + j], rsx[j]) + T.shfl_xor(T.if_then_else((lane & 8) != 0, rsx[j], rsx[3 + j]), 8)
                                    rsx[9] = 0.0
                                    for j in T.unroll(2):
                                        rsx[j] = T.if_then_else((lane & 4) != 0, rsx[8 + j], rsx[6 + j]) + T.shfl_xor(T.if_then_else((lane & 4) != 0, rsx[6 + j], rsx[8 + j]), 4)
                                    rsx[2] = T.if_then_else((lane & 2) != 0, rsx[1], rsx[0]) + T.shfl_xor(T.if_then_else((lane & 2) != 0, rsx[0], rsx[1]), 2)
                                    rsx[3] = rsx[2] + T.shfl_xor(rsx[2], 1)
                                    if ((lane & 1) == 0) & ((lane & 6) != 6) & (3 * ((lane >> 3) & 1) + ((lane >> 1) & 3) <= 4):
                                        cpart[g, 4, w * 16 + 5 * (lane >> 4) + 3 * ((lane >> 3) & 1) + ((lane >> 1) & 3)] = rsx[3]
                                T.sync_threads(bar, 128)
                                for i in T.unroll(10):
                                    if HP == 1:
                                        gg[i] = cpart[g, 4, i] + cpart[g, 4, 16 + i] + cpart[g, 4, 32 + i] + cpart[g, 4, 48 + i]
                                    else:                                       # HP = 2: this head's two warps
                                        gg[i] = cpart[g, 4, hw * 32 + i] + cpart[g, 4, hw * 32 + 16 + i]
                                ll[0] = T.rsqrt(T.max(gg[0], 1e-30))                       # reciprocals of the diagonal (1/L_jj)
                                ll[1] = gg[1] * ll[0]
                                ll[2] = T.rsqrt(T.max(T.max(gg[2] - ll[1] * ll[1], 1e-6 * gg[2]), 1e-30))
                                ll[3] = gg[3] * ll[0]
                                ll[4] = (gg[4] - ll[3] * ll[1]) * ll[2]
                                ll[5] = T.rsqrt(T.max(T.max(gg[5] - ll[3] * ll[3] - ll[4] * ll[4], 1e-6 * gg[5]), 1e-30))
                                ll[6] = gg[6] * ll[0]
                                ll[7] = (gg[7] - ll[6] * ll[1]) * ll[2]
                                ll[8] = (gg[8] - ll[6] * ll[3] - ll[7] * ll[4]) * ll[5]
                                ll[9] = T.rsqrt(T.max(T.max(gg[9] - ll[6] * ll[6] - ll[7] * ll[7] - ll[8] * ll[8], 1e-6 * gg[9]), 1e-30))
                                pr[0] = pr[0] * ll[0]
                                pr[1] = (pr[1] - pr[0] * ll[1]) * ll[2]
                                pr[2] = (pr[2] - pr[0] * ll[3] - pr[1] * ll[4]) * ll[5]
                                pr[3] = (pr[3] - pr[0] * ll[6] - pr[1] * ll[7] - pr[2] * ll[8]) * ll[9]
                                for r in T.unroll(R):
                                    if HP == 1:
                                        St_s[g, t, K + r] = T.cast(pr[r], F16)                       # the fp16 Q tile (pass 2's Gram operand, then the next product)
                                    else:
                                        St_s[g, t, K + 4 * hw + r] = T.cast(pr[r], F16)
                                T.sync_threads(bar, 128)
                    # ---- the fp16 factors (U^T[r][c] = P16[c][r], Q^T[r][v] = Q16[v][r]) to global and, after a barrier, -U^T over the P/Q planes for the residual mma ----
                    if not NEW:
                        for r in T.unroll(R):
                            uh[r] = UQn1[g, P16OFF + t * 8 + r]
                            qh[r] = St_s[g, t, K + r]
                            UQ[slot, hv, r, t] = uh[r]
                            UQ[slot, hv, r, K + t] = qh[r]
                        T.sync_threads(bar, 128)
                        for r in T.unroll(R):
                            UQn_s[g, r, t] = -uh[r]                   # -U^T: the residual mma accumulates the subtraction
                        for r in T.unroll(8 - R):
                            UQn_s[g, R + r, t] = T.cast(0.0, F16)
                        T.sync_threads(bar, 128)
                    elif HP == 1:                                 # K = 64 < V: threads t < K hold the columns of U^T
                        for r in T.unroll(R):
                            qh[r] = St_s[g, t, K + r]
                            UQ[slot, hv, r, K + t] = qh[r]
                        if t < K:
                            for r in T.unroll(R):
                                uh[r] = UQn1[g, P16OFF + t * 8 + r]
                                UQ[slot, hv, r, t] = uh[r]
                        T.sync_threads(bar, 128)
                        if t < K:
                            for r in T.unroll(R):
                                UQn_s[g, r, t] = -uh[r]
                            for r in T.unroll(8 - R):
                                UQn_s[g, R + r, t] = T.cast(0.0, F16)
                        T.sync_threads(bar, 128)
                    else:                                         # HP = 2: thread t = column t % 64 of head hw's U^T and row t % 64 of its Q^T
                        for r in T.unroll(R):
                            uh[r] = UQn1[g, P16OFF + (t % 64) * 8 + 4 * hw + r]
                            qh[r] = St_s[g, t, K + 4 * hw + r]
                        if hok:
                            for r in T.unroll(R):
                                UQ[slot, hv, r, t % 64] = uh[r]
                                UQ[slot, hv, r, K + t % 64] = qh[r]
                        T.sync_threads(bar, 128)
                        for r in T.unroll(R):                     # -U^T of head hw: rows 4 hw + r (the Q tile's block structure) of its own column block
                            UQn_s[g, 4 * hw + r, hw * 128 + t % 64] = -uh[r]   # at 128 hw, the other head's rows zero there: no cross-head product
                        for r in T.unroll(4 - R):                 # touches a non-zero (or non-finite) value
                            UQn_s[g, 4 * hw + R + r, hw * 128 + t % 64] = T.cast(0.0, F16)
                        for r in T.unroll(4):
                            UQn_s[g, 4 * (1 - hw) + r, hw * 128 + t % 64] = T.cast(0.0, F16)
                        T.sync_threads(bar, 128)
                    # ---- residual E = fp16(B) + pn Q_old U_old^T - Q U^T in NB halves (fp16 mma, fp32 accumulate), |E| column partials, E over B in St_s ----
                    for mt in T.unroll(2):        # A[m=v][k=r] = Q_old^T[r][v]: x2.trans -> (m 0-7), (m 8-15) of k-rows 0-7
                        if HP == 1:
                            T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(UQ_s[s, (K + w * 32 + mt * 16 + 8 * ((lane // 8) % 2)) // 64, lane % 8, (K + w * 32 + mt * 16 + 8 * ((lane // 8) % 2)) % 64], "r", extent=4), T.access_ptr(afrag2[mt * 4], "w", extent=4))
                        else:                     # HP = 2: head hw's Q^T block; factor rows of the other head -> zero rows 4..7
                            T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(UQ_s[s, NUB + hw, (lane % 8 + 4 * hw) % 8, (w % 2) * 32 + mt * 16 + 8 * ((lane // 8) % 2)], "r", extent=4), T.access_ptr(afrag2[mt * 4], "w", extent=4))
                        # A[m=v][k=r] = Q16[v][r] (row-major [m][k], k 0-7): x2 (non-trans) -> rows 0-7, rows 8-15
                        T.ptx_ldmatrix(T.bool(False), 2, T.access_ptr(St_s[g, w * 32 + mt * 16 + 8 * ((lane // 8) % 2) + lane % 8, K], "r", extent=4), T.access_ptr(afrag3[mt * 4], "w", extent=4))
                    for h in T.unroll(NPS):
                        # E = pn (B / pn + Q_old U_old^T) - Q U^T: the accumulator starts from B (scaled by 1 / pn on GDN, loaded with ldmatrix.x4 in the
                        # fragment layout), so both rank-R mma chains accumulate straight into it and there is no elementwise add afterwards
                        for mt in T.unroll(2):
                            for npl in T.unroll(NTP // 2):
                                T.ptx_ldmatrix(T.bool(False), 4, T.access_ptr(St_s[g, w * 32 + mt * 16 + 8 * ((lane // 8) % 2) + lane % 8, (h * NTP + npl * 2 + lane // 16) * 8], "r", extent=8), T.access_ptr(efrag[0], "w", extent=8))
                                for jj in T.unroll(4):
                                    for c1 in T.unroll(2):
                                        if KDA:
                                            cfrag[((npl * 2 + jj // 2) * 4 + mt * 2 + jj % 2) * 2 + c1] = T.cast(efrag[jj * 2 + c1], F32)
                                        else:
                                            cfrag[((npl * 2 + jj // 2) * 4 + mt * 2 + jj % 2) * 2 + c1] = T.cast(efrag[jj * 2 + c1], F32) * rpn
                        for npl in T.unroll(NTP // 2):
                            n0 = (h * (NTP // 2) + npl) * 16
                            if HP == 1:
                                T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(UQ_s[s, n0 // 64, lane % 8, n0 % 64 + 8 * ((lane // 8) % 2)], "r", extent=4), T.access_ptr(bfr[(npl % 2) * 4], "w", extent=4))
                            else:                 # HP = 2: head hw's U^T block, the other head's factor rows -> zero rows 4..7
                                T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(UQ_s[s, hw * NB + n0 // 64, (lane % 8 + 4 * hw) % 8, n0 % 64 + 8 * ((lane // 8) % 2)], "r", extent=4), T.access_ptr(bfr[(npl % 2) * 4], "w", extent=4))
                            for mt in T.unroll(2):
                                for nn in T.unroll(2):
                                    T.ptx_mma("float32", "m16n8k8", "row", "col", "fp16", "fp16", "fp32", afrag2.data, mt * 4, bfr.data, (npl % 2) * 4 + nn * 2, cfrag.data, ((npl * 2 + nn) * 4 + mt * 2) * 2, T.bool(False))
                        if not KDA:
                            for i in T.unroll(8 * NTP):
                                cfrag[i] = cfrag[i] * pn
                        for npl in T.unroll(NTP // 2):
                            n0 = (h * (NTP // 2) + npl) * 16
                            if HP == 1:
                                T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(UQn_s[g, lane % 8, n0 + 8 * ((lane // 8) % 2)], "r", extent=4), T.access_ptr(bfr[(npl % 2) * 4], "w", extent=4))
                            else:                 # HP = 2: head hw's -U^T block (the other head's rows zero)
                                T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(UQn_s[g, lane % 8, hw * 128 + n0 + 8 * ((lane // 8) % 2)], "r", extent=4), T.access_ptr(bfr[(npl % 2) * 4], "w", extent=4))
                            for mt in T.unroll(2):
                                for nn in T.unroll(2):
                                    T.ptx_mma("float32", "m16n8k8", "row", "col", "fp16", "fp16", "fp32", afrag3.data, mt * 4, bfr.data, (npl % 2) * 4 + nn * 2, cfrag.data, ((npl * 2 + nn) * 4 + mt * 2) * 2, T.bool(False))
                        for mt in T.unroll(2):
                            for npl in T.unroll(NTP // 2):
                                for jj in T.unroll(4):
                                    for c1 in T.vectorized(2):
                                        ph[jj * 2 + c1] = T.cast(cfrag[((npl * 2 + jj // 2) * 4 + mt * 2 + jj % 2) * 2 + c1], F16)
                                T.call_extern("handle", "tl::ptx_stmatrix_m8n8_x4",
                                              T.access_ptr(St_s[g, w * 32 + mt * 16 + 8 * ((lane // 8) % 2) + lane % 8, (h * NTP + npl * 2 + lane // 16) * 8], "w", extent=8),
                                              T.reinterpret(I32, ph[0:2]), T.reinterpret(I32, ph[2:4]), T.reinterpret(I32, ph[4:6]), T.reinterpret(I32, ph[6:8]))
                        # ---- |E| column partials of this warp's 32 rows on the tensor cores (this pass's NTP n-tiles): C[m][c] = sum_v 1 * |E[v][c]| with
                        #      A = ones and B = |E| read back from the fp16 E tile (ldmatrix.x2.trans + sign-bit clear), accumulated into the dead cfrag;
                        #      every m-row of C holds the sums -- lane rows r0 == 0 are used.  No shuffle reduction. ----
                        T.sync_warp()
                        for i in T.unroll(4 * NTP):
                            cfrag[i] = 0.0
                        for ntl in T.unroll(NTP):
                            for ks in T.unroll(2):
                                T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(St_s[g, w * 32 + ks * 16 + 8 * ((lane // 8) % 2) + lane % 8, (h * NTP + ntl) * 8], "r", extent=4), T.access_ptr(bh2[ks * 4], "w", extent=4))
                            for i in T.unroll(4):
                                bhi[i] = T.reinterpret(I32, bh2[i * 2:i * 2 + 2]) & T.int32(0x7FFF7FFF)
                            for ks in T.unroll(2):
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", aone.data, 0, bhi.data, ks * 2, cfrag.data, ntl * 4, T.bool(False))
                        for ntl in T.unroll(NTP):
                            for c1 in T.unroll(2):
                                colp[(h * NTP + ntl) * 2 + c1] = cfrag[ntl * 4 + c1]
                    T.sync_threads(bar, 128)                  # the residual mma operands (UQn_s, UQ_s) are consumed: the partials may take the buffer, the stage is free
                    T.fence_proxy_async()
                    T.mbarrier_arrive(consumed[s])
                    if r0 == 0:
                        for nt in T.unroll(K // 8):
                            for c1 in T.vectorized(2):
                                cpart[g, w, nt * 8 + q4 * 2 + c1] = colp[nt * 2 + c1]
                    T.sync_threads(bar, 128)
                    # (4) column scale s_k (thread per column) from the 4 warp partials, reciprocal to smem
                    if not NEW:
                        msum = (cpart[g, 0, t] + cpart[g, 1, t] + cpart[g, 2, t] + cpart[g, 3, t]) * (1.0 / VMD)
                        rskn = T.min(T.rsqrt(msum), 1.0 / EPS)                      # 1/max(sqrt(m), EPS) without the IEEE division slow path
                        Sk[slot, hv, t] = T.max(T.sqrt(msum), EPS)
                        rsk[g, 6, t] = rskn
                    elif HP == 1:                                                    # K = 64: threads t < K, the 4 warps' partials (V = 128 rows)
                        if t < K:
                            msum = (cpart[g, 0, t] + cpart[g, 1, t] + cpart[g, 2, t] + cpart[g, 3, t]) * (1.0 / VMD)
                            rskn = T.min(T.rsqrt(msum), 1.0 / EPS)
                            Sk[slot, hv, t] = T.max(T.sqrt(msum), EPS)
                            rsk[g, 6, t] = rskn
                    else:                                                            # HP = 2: column t % 64 of head hw from its two warps
                        msum = (cpart[g, 2 * hw, t % 64] + cpart[g, 2 * hw + 1, t % 64]) * (1.0 / VMD)
                        rskn = T.min(T.rsqrt(msum), 1.0 / EPS)
                        if hok:
                            Sk[slot, hv, t % 64] = T.max(T.sqrt(msum), EPS)
                        rsk[g, 6, t] = rskn                                          # head hw's scales at 64 hw ..
                    T.sync_threads(bar, 128)
                    # (5) row-wise quantisation: warp w rows 32w.., LPR = K / 16 lanes per row (16 columns each), RPP = 32 / LPR rows per pass;
                    #     row max over the LPR lanes, int8 codes packed in registers, one coalesced 16 B store per lane and row
                    for i in T.unroll(4):
                        for c1 in T.vectorized(4):
                            if HP == 1:
                                rk[i * 4 + c1] = rsk[g, 6, (lane % LPR) * 16 + i * 4 + c1]
                            else:
                                rk[i * 4 + c1] = rsk[g, 6, hw * 64 + (lane % LPR) * 16 + i * 4 + c1]
                    for i in T.unroll(32 // RPP):
                        v0 = w * 32 + i * RPP + lane // LPR
                        rmax[0] = 0.0
                        for hh in T.unroll(2):
                            for c1 in T.vectorized(8):
                                eh[((i % 2) * 2 + hh) * 8 + c1] = St_s[g, v0, (lane % LPR) * 16 + hh * 8 + c1]
                        for hh in T.unroll(2):
                            for c1 in T.unroll(8):
                                ex[hh * 8 + c1] = T.cast(eh[((i % 2) * 2 + hh) * 8 + c1], F32) * rk[hh * 8 + c1]
                                rmax[0] = T.max(rmax[0], T.abs(ex[hh * 8 + c1]))
                        rmax[0] = T.max(rmax[0], T.shfl_xor(rmax[0], 1))
                        rmax[0] = T.max(rmax[0], T.shfl_xor(rmax[0], 2))
                        if LPR == 8:
                            rmax[0] = T.max(rmax[0], T.shfl_xor(rmax[0], 4))
                        rmax[0] = T.max(rmax[0], EPS)
                        rmax[1] = QMAX * T.call_extern(F32, "__frcp_rn", rmax[0])
                        if HP == 1:
                            if lane % LPR == 0:
                                Sv[slot, hv, v0] = rmax[0]
                        else:
                            if hok & (lane % LPR == 0):
                                Sv[slot, hv, v0 - 64 * hw] = rmax[0]
                        for c1 in T.unroll(4):
                            qw[c1] = T.call_extern(I32, "__byte_perm",
                                                   T.call_extern(I32, "__byte_perm", T.reinterpret(I32, ex[c1 * 4] * rmax[1] + 12582912.0), T.reinterpret(I32, ex[c1 * 4 + 1] * rmax[1] + 12582912.0), T.int32(0x0040)),
                                                   T.call_extern(I32, "__byte_perm", T.reinterpret(I32, ex[c1 * 4 + 2] * rmax[1] + 12582912.0), T.reinterpret(I32, ex[c1 * 4 + 3] * rmax[1] + 12582912.0), T.int32(0x0040)),
                                                   T.int32(0x5410))
                        if HP == 1:
                            for c1 in T.vectorized(4):
                                Sq[slot, hv, v0, (lane % LPR) * 4 + c1] = qw[c1]
                        else:
                            if hok:
                                for c1 in T.vectorized(4):
                                    Sq[slot, hv, v0 - 64 * hw, (lane % LPR) * 4 + c1] = qw[c1]
                    if HP == 1:
                        if t < L:
                            Wbuf[slot, hv, t] = 0.0
                        if KDA:
                            Pbuf[slot, hv, t] = 0.0
                            for jj in T.unroll(L):
                                Gbuf[slot, hv, jj, t] = T.reinterpret(BF16, T.cast(1.0, F16))          # entry factors -> fp16 1.0
                        else:
                            if t == 0:
                                Pbuf[slot, hv, 0] = 1.0
                        if t == 0:
                            hcnt[slot, hv] = 0
                    else:                                                            # HP = 2: thread 64 hw + j resets head hw
                        if hok & (t % 64 < L):
                            Wbuf[slot, hv, t % 64] = 0.0
                        if hok & (t % 64 == 0):
                            Pbuf[slot, hv, 0] = 1.0
                            hcnt[slot, hv] = 0
                    T.sync_threads(bar, 128)          # St_s / UQn_s reuse by the next program of this group
    return flush


def _q_strides(SS, V, K):
    # byte strides of the int8 view of the fp32 [NS, HV, V, K] cache
    return (4 * SS, 4 * V * K, K, 1)


@tilelang.jit
def make_dequant(NS, HV, K, V, L, SS, BV=32, threads=128, kda=False, buf_dtype=F32, R=0):
    """Prefill round trip, part 1: fp32 S[slot] <- decay * deq(S0) + sum_j w_j u_j k_j^T for the listed slots, then reset the
    window.  The int8 checkpoint aliases the first rows of the fp32 slot, so it is staged through shared memory first."""
    N = T.symbolic("N")
    NT = V // BV
    IS_KDA = bool(kda); KA = K if IS_KDA else 1; PRESET = 0.0 if IS_KDA else 1.0
    KB_T, UB_T = T.Tensor((NS, HV, L, K), buf_dtype), T.Tensor((NS, HV, L, V), buf_dtype)
    WB_T, PB_T, GB_T, HC_T = T.Tensor((NS, HV, L), F32), T.Tensor((NS, HV, KA), F32), T.Tensor((NS, HV, L, KA), F32), T.Tensor((NS, HV), I32)
    S0_T = (T.StridedTensor((NS, HV, V, K), (SS, V * K, K, 1), F32))

    @T.prim_func
    def dequant_flush(S0: S0_T,
                              Sq: T.StridedTensor((NS, HV, V, K), _q_strides(SS, V, K), I8),
                              Sk: T.StridedTensor((NS, HV, K), (SS, V * K, 1), F32),
                              Sv: T.StridedTensor((NS, HV, V), (SS, V * K, 1), F32),
                              Kbuf: KB_T, Ubuf: UB_T,
                              Wbuf: WB_T, Pbuf: PB_T,
                              Gbuf: GB_T,
                              hcnt: HC_T,
                              UQ: T.StridedTensor((NS, HV, T.max(R, 1), K + V), (2 * SS, 2 * V * K, K + V, 1), "float16"),
                              idx: T.Tensor((N,), I32)):
        with T.Kernel(N * HV, threads=threads) as pid:
            n = pid // HV
            hv = pid % HV
            slot = idx[n]
            if slot > 0:
                h = hcnt[slot, hv]
                p = 1.0 if IS_KDA else Pbuf[slot, hv, 0]
                pc_sh = T.alloc_shared((K,), F32)
                q_all = T.alloc_shared((V, K), I8)
                sk_sh = T.alloc_shared((K,), F32)
                sv_sh = T.alloc_shared((V,), F32)
                w_sh = T.alloc_shared((L,), F32)
                kb_sh = T.alloc_shared((L, K), F32)
                ub_sh = T.alloc_shared((L, V), F32)
                tile = T.alloc_fragment((BV, K), F32)
                uq_sh = T.alloc_shared((T.max(R, 1), K + V), F32)          # rank-R factors (U^T | Q^T), fp16 -> fp32 once
                T.copy(Sq[slot, hv, :, :], q_all)
                if R > 0:
                    for r, c in T.Parallel(R, K + V):
                        uq_sh[r, c] = T.cast(UQ[slot, hv, r, c], F32)
                for c in T.Parallel(K):
                    sk_sh[c] = Sk[slot, hv, c]
                for c in T.Parallel(V):
                    sv_sh[c] = Sv[slot, hv, c]
                for j in T.Parallel(L):
                    w_sh[j] = T.if_then_else(j < h, Wbuf[slot, hv, j], 0.0)
                if buf_dtype == F32:
                    T.copy(Kbuf[slot, hv, :, :], kb_sh)
                    T.copy(Ubuf[slot, hv, :, :], ub_sh)
                else:
                    for j, c in T.Parallel(L, K):
                        kb_sh[j, c] = T.cast(Kbuf[slot, hv, j, c], F32)
                    for j, c in T.Parallel(L, V):
                        ub_sh[j, c] = T.cast(Ubuf[slot, hv, j, c], F32)
                if IS_KDA:
                    for c in T.Parallel(K):
                        pc_sh[c] = T.exp(Pbuf[slot, hv, c])
                    for j, c in T.Parallel(L, K):
                        kb_sh[j, c] = kb_sh[j, c] * T.if_then_else(j < h, T.exp(Pbuf[slot, hv, c] - Gbuf[slot, hv, j, c]), 1.0)
                else:
                    for c in T.Parallel(K):
                        pc_sh[c] = 1.0
                for v0 in T.serial(NT):
                    for r, c in T.Parallel(BV, K):
                        tile[r, c] = (T.cast(q_all[v0 * BV + r, c], F32) / QMAX) * sk_sh[c] * sv_sh[v0 * BV + r]
                    if R > 0:
                        for rr in T.serial(R):
                            for r, c in T.Parallel(BV, K):
                                tile[r, c] += uq_sh[rr, K + v0 * BV + r] * uq_sh[rr, c]
                    for r, c in T.Parallel(BV, K):
                        tile[r, c] = tile[r, c] * p * pc_sh[c]
                    for j in T.serial(L):
                        for r, c in T.Parallel(BV, K):
                            tile[r, c] += w_sh[j] * ub_sh[j, v0 * BV + r] * kb_sh[j, c]
                    T.copy(tile, S0[slot, hv, v0 * BV:(v0 + 1) * BV, :])
                for j in T.Parallel(L):
                    Wbuf[slot, hv, j] = 0.0
                for c in T.Parallel(KA):
                    Pbuf[slot, hv, c] = PRESET
                hcnt[slot, hv] = 0
    return dequant_flush


@tilelang.jit
def make_quantize(NS, HV, K, V, L, SS, R=4, ITERS=1, threads=128):
    """Prefill round trip, part 2: 8-bit checkpoint of the fp32 states in the listed slots (subspace iteration from Q0 with the
    flush kernel's numerics, fp16 Compensator Tokens, smoothed INT8 residual), then reset the window."""
    N = T.symbolic("N")
    F16 = "float16"
    WB_T, PB_T, HC_T = T.Tensor((NS, HV, L), F32), T.Tensor((NS, HV, 1), F32), T.Tensor((NS, HV), I32)
    S0_T = (T.StridedTensor((NS, HV, V, K), (SS, V * K, K, 1), F32))

    @T.prim_func
    def quantize(S0: S0_T,
                    Sq: T.StridedTensor((NS, HV, V, K), _q_strides(SS, V, K), I8),
                    Sk: T.StridedTensor((NS, HV, K), (SS, V * K, 1), F32),
                    Sv: T.StridedTensor((NS, HV, V), (SS, V * K, 1), F32),
                    UQ: T.StridedTensor((NS, HV, R, K + V), (2 * SS, 2 * V * K, K + V, 1), F16),
                    Wbuf: WB_T, Pbuf: PB_T,
                    hcnt: HC_T, Q0: T.Tensor((V, R), F32), idx: T.Tensor((N,), I32)):
        with T.Kernel(N * HV, threads=threads) as pid:
            n = pid // HV
            hv = pid % HV
            slot = idx[n]
            if slot > 0:
                S_sh = T.alloc_shared((V, K), F32)
                Uk_sh = T.alloc_shared((K, R), F32)
                Qv_sh = T.alloc_shared((V, R), F32)
                P_sh = T.alloc_shared((K, R), F32)
                sk_sh = T.alloc_shared((K,), F32)
                sv_sh = T.alloc_shared((V,), F32)
                col_fr = T.alloc_fragment((V, K), F32)
                colv_fr = T.alloc_fragment((V,), F32)
                colk_fr = T.alloc_fragment((K,), F32)
                red = T.alloc_fragment((1, K), F32)
                redv = T.alloc_fragment((1, V), F32)
                rs = T.alloc_fragment((1,), F32)
                sc = T.alloc_shared((4,), F32)
                T.copy(S0[slot, hv, :, :], S_sh)
                T.copy(Q0, Qv_sh)
                for it in T.serial(ITERS):
                    for c, j in T.Parallel(K, R):
                        acc = T.alloc_local((1,), F32)
                        acc[0] = 0.0
                        for v in T.serial(V):
                            acc[0] += S_sh[v, c] * Qv_sh[v, j]
                        P_sh[c, j] = acc[0]
                    for j in T.serial(R):
                        for cc in T.serial(j):
                            for z, c in T.Parallel(1, K):
                                red[z, c] = P_sh[c, cc] * P_sh[c, j]
                            T.reduce_sum(red, rs, dim=1)
                            sc[0] = rs[0]
                            for c in T.Parallel(K):
                                P_sh[c, j] = P_sh[c, j] - sc[0] * P_sh[c, cc]
                        for z, c in T.Parallel(1, K):
                            red[z, c] = P_sh[c, j] * P_sh[c, j]
                        T.reduce_sum(red, rs, dim=1)
                        sc[1] = T.rsqrt(T.max(rs[0], 1e-30))
                        for c in T.Parallel(K):
                            P_sh[c, j] = P_sh[c, j] * sc[1]
                    for v, j in T.Parallel(V, R):
                        acc = T.alloc_local((1,), F32)
                        acc[0] = 0.0
                        for c in T.serial(K):
                            acc[0] += S_sh[v, c] * P_sh[c, j]
                        Qv_sh[v, j] = acc[0]
                    for j in T.serial(R):
                        for cc in T.serial(j):
                            for z, v in T.Parallel(1, V):
                                redv[z, v] = Qv_sh[v, cc] * Qv_sh[v, j]
                            T.reduce_sum(redv, rs, dim=1)
                            sc[2] = rs[0]
                            for v in T.Parallel(V):
                                Qv_sh[v, j] = Qv_sh[v, j] - sc[2] * Qv_sh[v, cc]
                        for z, v in T.Parallel(1, V):
                            redv[z, v] = Qv_sh[v, j] * Qv_sh[v, j]
                        T.reduce_sum(redv, rs, dim=1)
                        sc[3] = T.rsqrt(T.max(rs[0], 1e-30))
                        for v in T.Parallel(V):
                            Qv_sh[v, j] = Qv_sh[v, j] * sc[3]
                for c, j in T.Parallel(K, R):
                    acc = T.alloc_local((1,), F32)
                    acc[0] = 0.0
                    for v in T.serial(V):
                        acc[0] += S_sh[v, c] * Qv_sh[v, j]
                    Uk_sh[c, j] = T.cast(T.cast(acc[0], F16), F32)            # fp16 factors
                for v, j in T.Parallel(V, R):
                    Qv_sh[v, j] = T.cast(T.cast(Qv_sh[v, j], F16), F32)
                for v, c in T.Parallel(V, K):
                    acc = T.alloc_local((1,), F32)
                    acc[0] = S_sh[v, c]
                    for r in T.serial(R):
                        acc[0] -= Qv_sh[v, r] * Uk_sh[c, r]
                    S_sh[v, c] = acc[0]
                for v, c in T.Parallel(V, K):
                    col_fr[v, c] = T.abs(S_sh[v, c])
                T.reduce_sum(col_fr, colk_fr, dim=0)
                for c in T.Parallel(K):
                    sk_sh[c] = T.max(T.sqrt(colk_fr[c] / V), EPS)
                for v, c in T.Parallel(V, K):
                    col_fr[v, c] = T.abs(S_sh[v, c] / sk_sh[c])
                T.reduce_max(col_fr, colv_fr, dim=1)
                for v in T.Parallel(V):
                    sv_sh[v] = T.max(colv_fr[v], EPS)
                for v, c in T.Parallel(V, K):
                    Sq[slot, hv, v, c] = T.cast(T.clamp(T.nearbyint(S_sh[v, c] / (sk_sh[c] * sv_sh[v]) * QMAX), -QMAX, QMAX), I8)
                for c in T.Parallel(K):
                    Sk[slot, hv, c] = sk_sh[c]
                for v in T.Parallel(V):
                    Sv[slot, hv, v] = sv_sh[v]
                for r, c in T.Parallel(R, K):
                    UQ[slot, hv, r, c] = T.cast(Uk_sh[c, r], F16)
                for r, v in T.Parallel(R, V):
                    UQ[slot, hv, r, K + v] = T.cast(Qv_sh[v, r], F16)
                for j in T.Parallel(L):
                    Wbuf[slot, hv, j] = 0.0
                Pbuf[slot, hv, 0] = 1.0
                hcnt[slot, hv] = 0
    return quantize


# ---- rank-R factorization (subspace iteration), used by the KDA prefill checkpoint ----
@tilelang.jit
def make_lowrank(K, V, R, ITERS, threads=128):
    N = T.symbolic("N")

    @T.prim_func
    def lowrank(S: T.Tensor((N, K, V), F32), Q0: T.Tensor((V, R), F32), U: T.Tensor((N, K, R), F32), Q: T.Tensor((N, V, R), F32)):
        with T.Kernel(N, threads=threads) as bx:
            S_sh = T.alloc_shared((K, V), F32)
            Q_sh = T.alloc_shared((V, R), F32)
            P_sh = T.alloc_shared((K, R), F32)
            red = T.alloc_fragment((1, K), F32)
            redv = T.alloc_fragment((1, V), F32)
            rs = T.alloc_fragment((1,), F32)
            sc = T.alloc_shared((4,), F32)
            T.copy(S[bx, :, :], S_sh)
            T.copy(Q0, Q_sh)
            for it in T.serial(ITERS):
                # P = S Q   (K x R)
                for i, j in T.Parallel(K, R):
                    acc = T.alloc_local((1,), F32)
                    acc[0] = 0.0
                    for v in T.serial(V):
                        acc[0] += S_sh[i, v] * Q_sh[v, j]
                    P_sh[i, j] = acc[0]
                # modified Gram-Schmidt on the columns of P
                for j in T.serial(R):
                    for c in T.serial(j):
                        for z, i in T.Parallel(1, K):
                            red[z, i] = P_sh[i, c] * P_sh[i, j]
                        T.reduce_sum(red, rs, dim=1)
                        sc[0] = rs[0]
                        for i in T.Parallel(K):
                            P_sh[i, j] = P_sh[i, j] - sc[0] * P_sh[i, c]
                    for z, i in T.Parallel(1, K):
                        red[z, i] = P_sh[i, j] * P_sh[i, j]
                    T.reduce_sum(red, rs, dim=1)
                    sc[1] = T.rsqrt(T.max(rs[0], 1e-30))
                    for i in T.Parallel(K):
                        P_sh[i, j] = P_sh[i, j] * sc[1]
                # Q = S^T P   (V x R)
                for v, j in T.Parallel(V, R):
                    acc = T.alloc_local((1,), F32)
                    acc[0] = 0.0
                    for i in T.serial(K):
                        acc[0] += S_sh[i, v] * P_sh[i, j]
                    Q_sh[v, j] = acc[0]
                for j in T.serial(R):
                    for c in T.serial(j):
                        for z, v in T.Parallel(1, V):
                            redv[z, v] = Q_sh[v, c] * Q_sh[v, j]
                        T.reduce_sum(redv, rs, dim=1)
                        sc[2] = rs[0]
                        for v in T.Parallel(V):
                            Q_sh[v, j] = Q_sh[v, j] - sc[2] * Q_sh[v, c]
                    for z, v in T.Parallel(1, V):
                        redv[z, v] = Q_sh[v, j] * Q_sh[v, j]
                    T.reduce_sum(redv, rs, dim=1)
                    sc[3] = T.rsqrt(T.max(rs[0], 1e-30))
                    for v in T.Parallel(V):
                        Q_sh[v, j] = Q_sh[v, j] * sc[3]
            # U = S Q
            for i, j in T.Parallel(K, R):
                acc = T.alloc_local((1,), F32)
                acc[0] = 0.0
                for v in T.serial(V):
                    acc[0] += S_sh[i, v] * Q_sh[v, j]
                U[bx, i, j] = acc[0]
            T.copy(Q_sh, Q[bx, :, :])
    return lowrank


_mk_lowrank = functools.lru_cache(maxsize=None)(make_lowrank)
_Q0_LR = {}


def lowrank_factor(x, r, iters=3):
    """x: [..., K, V] fp32 CUDA -> U [..., K, r], Q [..., V, r] (one fused kernel)."""
    *lead, K, V = x.shape
    key = (str(x.device), V, r)
    if key not in _Q0_LR:
        _Q0_LR[key] = torch.randn(V, r, generator=torch.Generator().manual_seed(0)).to(x.device)
    xf = x.reshape(-1, K, V).contiguous().float()
    U = torch.empty(xf.shape[0], K, r, device=x.device, dtype=torch.float32)
    Q = torch.empty(xf.shape[0], V, r, device=x.device, dtype=torch.float32)
    _mk_lowrank(K, V, r, iters)(xf, _Q0_LR[key], U, Q)
    return U.reshape(*lead, K, r), Q.reshape(*lead, V, r)

