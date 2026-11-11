"""LeapQuant on vLLM's Gated DeltaNet layers (Qwen3.5 / Qwen3.8).

Pure decode steps run LeapQuant's step + flush kernels on the 8-bit state.  In steps with prefill tokens, the decode tokens
still go through LeapQuant, and only the prefill slots take an fp32 round trip around vLLM's own prefill kernel.
"""
import torch

from leapquant.state import get_state


def install(nmax):
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as M
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    cls = M.QwenGatedDeltaNetAttention
    orig_core = cls._forward_core
    orig_fsg = M.fused_sigmoid_gating_delta_rule_update
    cur = [None]                       # the layer whose prefill step is running (routes its decode tokens to LeapQuant)

    def state(self):
        return get_state(self, self.kv_cache[1], H=self.num_k_heads // self.tp_size, nmax=nmax[0])

    def decode(self, mixed_qkv, b, a, core_attn_out, md):
        idx = md.non_spec_state_indices_tensor
        kv = self.kv_cache
        conv_state = kv[0] if is_conv_state_dim_first() else kv[0].transpose(-1, -2)
        n = md.num_actual_tokens
        mixed_qkv, b, a = mixed_qkv[:n], b[:n], a[:n]
        conv_w = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        mixed = causal_conv1d_update(mixed_qkv, conv_state, conv_w, self.conv1d.bias, self.activation,
                                     conv_state_indices=idx[:n], validate_data=False)
        st = state(self)
        st.decode(kv[1], mixed.contiguous(), a.contiguous(), b.contiguous(), self.A_log, self.dt_bias, idx[:n],
                  core_attn_out[:n].view(n, st.HV, st.V))

    def fsg(A_log, a, b, dt_bias, q, k, v, initial_state, inplace_final_state, cu_seqlens, ssm_state_indices,
            use_qk_l2norm_in_kernel=True, **kw):
        """vLLM's decode kernel inside a mixed prefill + decode step -> LeapQuant's step kernel."""
        self = cur[0]
        if self is None or kw.get("num_accepted_tokens") is not None or q.shape[0] != 1:
            return orig_fsg(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v, initial_state=initial_state,
                            inplace_final_state=inplace_final_state, cu_seqlens=cu_seqlens,
                            ssm_state_indices=ssm_state_indices, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kw)
        st = state(self)
        n = q.shape[1]
        mixed = torch.cat([q.reshape(n, -1), k.reshape(n, -1), v.reshape(n, -1)], dim=1).to(torch.bfloat16).contiguous()
        out = torch.empty(n, st.HV, st.V, dtype=torch.bfloat16, device=q.device)
        st.decode(self.kv_cache[1], mixed, a[:n].contiguous(), b[:n].contiguous(), A_log, dt_bias, ssm_state_indices[:n], out)
        return out.unsqueeze(0), None

    def forward_core(self, mixed_qkv, b, a, core_attn_out):
        md_raw = get_forward_context().attn_metadata
        md = md_raw.get(self.prefix) if isinstance(md_raw, dict) else None
        if md is None:                                   # profiling / warm-up runs
            return orig_core(self, mixed_qkv, b, a, core_attn_out)
        if (self.enable_packed_recurrent_decode and md.spec_sequence_masks is None
                and md.num_prefills == 0 and md.num_decodes > 0):
            return decode(self, mixed_qkv, b, a, core_attn_out, md)
        st = state(self)
        if md.spec_sequence_masks is None and md.prefill_state_indices is not None:
            tidx = md.prefill_state_indices.reshape(-1).clamp(min=0)       # prefill slots only
            route = True
        else:                                                               # speculative decoding: round trip every slot
            touched = [t.reshape(-1) for t in (md.non_spec_state_indices_tensor, md.spec_state_indices_tensor) if t is not None]
            if not touched:
                return orig_core(self, mixed_qkv, b, a, core_attn_out)
            tidx = torch.cat(touched).clamp_(min=0)
            route = False
        st.dequantize(self.kv_cache[1], tidx)
        cur[0] = self if route else None
        try:
            ret = orig_core(self, mixed_qkv, b, a, core_attn_out)
        finally:
            cur[0] = None
        st.quantize(self.kv_cache[1], tidx)
        return ret

    M.fused_sigmoid_gating_delta_rule_update = fsg
    cls._forward_core = forward_core
