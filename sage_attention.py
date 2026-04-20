from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_sage_callable() -> tuple[Callable[..., torch.Tensor] | None, str]:
    try:
        from sageattention import sageattn  # type: ignore

        return sageattn, "sageattention.sageattn"
    except Exception as exc:  # pragma: no cover - depends on local env
        return None, f"unavailable ({exc})"


def detect_sageattention() -> str:
    _, detail = _load_sage_callable()
    return detail


def patch_anisora_attention() -> str:
    sageattn, detail = _load_sage_callable()
    if sageattn is None:
        return f"disabled: {detail}"

    import wan.modules.attention as attention_mod  # type: ignore
    import wan.modules.model as model_mod  # type: ignore

    if getattr(attention_mod, "_sageattention_patched", False):
        return f"enabled via {detail}"

    original_flash_attention = attention_mod.flash_attention

    def flash_attention_with_sage(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: torch.Tensor | None = None,
        k_lens: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        q_scale: float | None = None,
        causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        deterministic: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        version: int | None = None,
    ) -> torch.Tensor:
        can_use_sage = (
            q.device.type == "cuda"
            and q.shape[0] == 1
            and dropout_p == 0.0
            and not causal
            and window_size == (-1, -1)
            and softmax_scale is None
        )
        if not can_use_sage:
            return original_flash_attention(
                q=q,
                k=k,
                v=v,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=dtype,
                version=version,
            )

        try:
            half_dtypes = (torch.float16, torch.bfloat16)
            q_dtype = q.dtype
            target_dtype = v.dtype if v.dtype in half_dtypes else dtype

            valid_q = int(q_lens[0].item()) if q_lens is not None else q.shape[1]
            valid_k = int(k_lens[0].item()) if k_lens is not None else k.shape[1]
            if valid_q != valid_k:
                raise ValueError("sageattention fallback only supports equal q/k valid lengths")

            q_view = q[0, :valid_q].to(target_dtype)
            k_view = k[0, :valid_k].to(target_dtype)
            v_view = v[0, :valid_k].to(target_dtype)
            if q_scale is not None:
                q_view = q_view * q_scale

            out = sageattn(
                q_view.unsqueeze(0),
                k_view.unsqueeze(0),
                v_view.unsqueeze(0),
                tensor_layout="NHD",
            ).squeeze(0)

            if valid_q < q.shape[1]:
                pad_shape = (q.shape[1] - valid_q, *out.shape[1:])
                out = torch.cat([out, torch.zeros(pad_shape, device=out.device, dtype=out.dtype)], dim=0)

            return out.unsqueeze(0).to(q_dtype)
        except Exception as exc:  # pragma: no cover - hardware specific fallback path
            logger.warning("Anisora SageAttention fallback failed, using native attention: %s", exc)
            return original_flash_attention(
                q=q,
                k=k,
                v=v,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=dtype,
                version=version,
            )

    attention_mod.flash_attention = flash_attention_with_sage
    model_mod.flash_attention = flash_attention_with_sage
    attention_mod._sageattention_patched = True
    return f"enabled via {detail}"


def patch_ltx_attention() -> str:
    sageattn, detail = _load_sage_callable()
    if sageattn is None:
        return f"disabled: {detail}"

    import ltx_core.model.transformer.attention as attention_mod  # type: ignore

    if getattr(attention_mod, "_sageattention_patched", False):
        return f"enabled via {detail}"

    def _make_wrapper(original_call: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def _wrapped(self: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None) -> torch.Tensor:
            if mask is not None or q.device.type != "cuda":
                return original_call(self, q, k, v, heads, mask)
            try:
                batch, _, inner_dim = q.shape
                dim_head = inner_dim // heads
                q_view = q.view(batch, -1, heads, dim_head).to(v.dtype)
                k_view = k.view(batch, -1, heads, dim_head).to(v.dtype)
                v_view = v.view(batch, -1, heads, dim_head)
                out = sageattn(q_view, k_view, v_view, tensor_layout="NHD")
                return out.reshape(batch, -1, heads * dim_head)
            except Exception as exc:  # pragma: no cover - hardware specific fallback path
                logger.warning("LTX SageAttention fallback failed, using native attention: %s", exc)
                return original_call(self, q, k, v, heads, mask)

        return _wrapped

    attention_mod.PytorchAttention.__call__ = _make_wrapper(attention_mod.PytorchAttention.__call__)
    attention_mod.XFormersAttention.__call__ = _make_wrapper(attention_mod.XFormersAttention.__call__)
    attention_mod._sageattention_patched = True
    return f"enabled via {detail}"
