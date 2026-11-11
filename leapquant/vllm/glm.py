"""LeapQuant on vLLM's GLM-5.3 Kimi Delta Attention layers (Glm5NextLinearAttention).

Pure decode steps run LeapQuant's KDA step + flush kernels on the 8-bit state.  In steps with prefill tokens, the decode
tokens go through LeapQuant and only the prefill slots take an fp32 round trip around vLLM's own prefill kernel.  The
replaced forward stays an eager break of vLLM's breakable CUDA graph.
"""
import torch

from leapquant.state import get_state


def install(nmax):
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.models.glm5next.nvidia import kda as K

    cls = K.Glm5NextLinearAttention
    orig = getattr(cls._forward, "__wrapped__", cls._forward)

    def state(self):
        return get_state(self, self.kv_cache[1], H=self.local_num_heads, kda=True,
                         gate_lower_bound=float(self.kda_lower_bound), nmax=nmax[0])

    def fwd(self, qkv_proj_states, g1, beta, core_attn_out):
        md_raw = get_forward_context().attn_metadata
        m = md_raw.get(self.prefix) if isinstance(md_raw, dict) else None
        if m is None or m.num_actual_tokens == 0:
            return
        n = m.num_actual_tokens
        assert self.kda_safe_gate and self.kda_lower_bound is not None, "LeapQuant supports the bounded KDA gate only"
        st = state(self)
        kv = self.kv_cache
        if m.spec_sequence_masks is not None and m.num_spec_decodes > 0:       # speculative decoding: round trip every slot
            touched = [t.reshape(-1) for t in (m.non_spec_state_indices_tensor, m.spec_state_indices_tensor) if t is not None]
            tidx = torch.cat(touched).clamp(min=0)
            st.dequantize(kv[1], tidx)
            orig(self, qkv_proj_states=qkv_proj_states, g1=g1, beta=beta, core_attn_out=core_attn_out)
            st.quantize(kv[1], tidx)
            return
        conv_state = kv[0] if self._conv_state_dim_first else kv[0].transpose(-1, -2)
        if self._merged_conv_weight is None:
            w = lambda mm: mm.weight.view(mm.weight.size(0), mm.weight.size(2))
            self._merged_conv_weight = torch.cat([w(self.q_conv1d), w(self.k_conv1d), w(self.v_conv1d)], dim=0).contiguous()
        if m.num_prefills > 0:
            nd = m.num_decode_tokens
            idx_all = m.non_spec_state_indices_tensor
            x = K.causal_conv1d_fn(qkv_proj_states[:n].transpose(0, 1), self._merged_conv_weight, self.q_conv1d.bias,
                                   activation="silu", conv_states=conv_state, has_initial_state=m.has_initial_state,
                                   cache_indices=idx_all, query_start_loc=m.non_spec_query_start_loc, metadata=m).transpose(0, 1)
            if nd > 0:                                   # decode tokens first, through LeapQuant
                st.decode(kv[1], x[:nd].contiguous(), g1[0, :nd], beta[0, :nd].contiguous(), self.A_log.view(-1), self.dt_bias,
                          idx_all[:nd], core_attn_out[0, :nd])
            pidx, phas, ploc = m.prefill_state_indices, m.prefill_has_initial_state, m.prefill_query_start_loc
            st.dequantize(kv[1], pidx.clamp(min=0))  # fp32 state for the prefill slots only
            init = K.gather_initial_states(kv[1], pidx, phas)
            rr = lambda t: t.reshape(1, -1, self.local_num_heads, self.head_dim)
            q, k, v = (rr(t) for t in x[nd:n].split(self.local_projection_size, dim=-1))
            if self.kda_prefill_backend == "flashkda":
                outp, last = self._flashkda_prefill(q=q, k=k, v=v, g=g1[:, nd:n], beta=beta[:, nd:n], initial_state=init,
                                                    cu_seqlens=ploc, out=None)
            else:
                outp, last = K.chunk_kda_with_fused_gate(
                    q=q, k=k, v=v, raw_g=g1[:, nd:n], beta=K._cast_sigmoid(beta[:, nd:n].squeeze(0)).unsqueeze(0),
                    A_log=self.A_log, g_bias=self.dt_bias, initial_state=init, output_final_state=True,
                    use_qk_l2norm_in_kernel=True, cu_seqlens=ploc, safe_gate=self.kda_safe_gate, lower_bound=self.kda_lower_bound)
            K.scatter_states(kv[1], last, pidx)
            core_attn_out[0, nd:n] = outp[0, :n - nd]
            st.quantize(kv[1], pidx.clamp(min=0))       # checkpoint written when the prefill ends
            return
        idx = m.non_spec_state_indices_tensor[:n]
        mixed = causal_conv1d_update(qkv_proj_states[:n], conv_state, self._merged_conv_weight, self.q_conv1d.bias,
                                     activation="silu", conv_state_indices=idx)
        st.decode(kv[1], mixed.contiguous(), g1[0, :n], beta[0, :n].contiguous(), self.A_log.view(-1), self.dt_bias,
                  idx, core_attn_out[0, :n])

    cls._forward = eager_break_during_capture(fwd)
