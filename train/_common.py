"""Shared training settings."""
from __future__ import annotations

import torch


def precision() -> tuple[dict, torch.dtype]:
    """Mixed-precision flags and weight dtype for the current hardware: bf16 on A100, fp16 on the free T4
    (no bf16), plain fp32 on CPU / Apple MPS, used for local smoke tests."""
    if torch.cuda.is_available():
        bf16 = torch.cuda.is_bf16_supported()
        return {"bf16": bf16, "fp16": not bf16}, torch.bfloat16 if bf16 else torch.float16
    return {"bf16": False, "fp16": False}, torch.float32
