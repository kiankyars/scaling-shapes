"""Model factories. We use HF GPT-NeoX so the architecture matches Pythia's
exactly, which makes our results comparable to the published Pythia
memorization literature.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rotary_pct: float = 0.25
    max_position_embeddings: int = 2048
    vocab_size: int = 50304


PRESETS: dict[str, ModelConfig] = {
    # Pythia-160M (close to "125M" target, same family).
    "125m": ModelConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
    ),
    "350m": ModelConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
    ),
    "1b": ModelConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=8,
    ),
}


def build_model(preset: str, *, dtype: str = "bfloat16"):
    import torch
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    cfg = PRESETS[preset]
    hf_cfg = GPTNeoXConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        rotary_pct=cfg.rotary_pct,
        max_position_embeddings=cfg.max_position_embeddings,
        use_cache=False,
        torch_dtype=dtype,
    )
    model = GPTNeoXForCausalLM(hf_cfg)
    if dtype == "bfloat16":
        model = model.to(dtype=torch.bfloat16)
    return model
