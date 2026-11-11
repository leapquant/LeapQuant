"""vLLM integration: once installed, the recurrent states of Qwen (GDN) and GLM (KDA) models are kept in 8 bits."""
import os

_INSTALLED = []
_NMAX = [0]          # largest decode batch one call can carry (sizes the flush kernel's work list)


def _record_nmax(cls):
    """Wrap the layer constructor to read max_num_seqs / CUDA-graph capture sizes from the vLLM config being built."""
    init = cls.__init__

    def __init__(self, *a, **k):
        init(self, *a, **k)
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        cc = cfg.compilation_config
        _NMAX[0] = max(_NMAX[0], cfg.scheduler_config.max_num_seqs, getattr(cc, "max_cudagraph_capture_size", None) or 0)

    cls.__init__ = __init__


def register():
    """vllm.general_plugins entry point, run in every vLLM process."""
    if _INSTALLED:
        return
    from leapquant.vllm import gdn, glm
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
    from vllm.models.glm5next.nvidia import kda
    _record_nmax(qwen_gdn_linear_attn.QwenGatedDeltaNetAttention)
    _record_nmax(kda.Glm5NextLinearAttention)
    gdn.install(_NMAX)
    glm.install(_NMAX)
    _INSTALLED.append(True)
    print(f"[LeapQuant] 8-bit recurrent states enabled (pid {os.getpid()})", flush=True)
