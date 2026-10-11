"""Loading a trained checkpoint into :class:`~milo.model.slm.SLM`."""

import torch

from milo.config import model_config
from milo.model.slm import SLM


def load_model(checkpoint_path: str, device: str = "cuda") -> SLM:
    """Load Milo from a checkpoint, with the architecture from ``configs/model.yaml``."""
    cfg = model_config()
    model = SLM(
        cfg["vocab_size"], cfg["d_model"],
        cfg["n_q_heads"], cfg["n_kv_heads"],
        cfg["n_layers"], cfg["seq_len"],
        cfg["dropout"], cfg["share_blocks"],
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    _check_shapes(state, cfg, checkpoint_path)
    model.load_state_dict(state)
    del ckpt, state

    model.to(device)
    model.eval()
    return model


def _check_shapes(state: dict, cfg: dict, path: str) -> None:
    """Fail with the mismatching dimension rather than a wall of state-dict keys."""
    emb = state.get("token_embedding.weight")
    if emb is None:
        return
    vocab, d_model = emb.shape
    if (vocab, d_model) != (cfg["vocab_size"], cfg["d_model"]):
        raise ValueError(
            f"{path} has vocab_size={vocab}, d_model={d_model} but configs/model.yaml says "
            f"vocab_size={cfg['vocab_size']}, d_model={cfg['d_model']}"
        )
