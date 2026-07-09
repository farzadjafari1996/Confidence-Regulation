"""HuggingFace Llama wrapper exposing GPT-2/paper-compatible weight and hook interface."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformer_lens import utils as tl_utils


class LlamaCfg:
    """Minimal cfg surface used by shared phase runners."""

    def __init__(self, hf_config) -> None:
        self.n_layers = hf_config.num_hidden_layers
        self.d_mlp = hf_config.intermediate_size
        self.d_model = hf_config.hidden_size
        self.d_vocab = hf_config.vocab_size
        self.normalization_type = "RMSPre"
        self.final_rms = True


class LlamaWrapper:
    """
    Last-layer SwiGLU neurons: post = silu(gate) * up, w_out[i] = down_proj[:, i].

    Provides W_out/W_U, ln_final, loss_fn, and run_with_cache compatible with gpt2/core ablation.
    """

    def __init__(self, model, tokenizer, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = LlamaCfg(model.config)
        self._last_layer_idx = self.cfg.n_layers - 1
        self._w_out: torch.Tensor | None = None
        self._w_u: torch.Tensor | None = None
        self._b_u: torch.Tensor | None = None

    @property
    def W_out(self) -> torch.Tensor:
        """Shape (n_layers, d_mlp, d_model); only the last layer is populated."""
        if self._w_out is None:
            w = torch.zeros(
                self.cfg.n_layers,
                self.cfg.d_mlp,
                self.cfg.d_model,
                dtype=torch.float32,
                device=self.device,
            )
            down = self.model.model.layers[-1].mlp.down_proj.weight
            w[-1] = down.T.float()
            self._w_out = w
        return self._w_out

    @property
    def W_U(self) -> torch.Tensor:
        if self._w_u is None:
            self._w_u = self.model.lm_head.weight.T.float()
        return self._w_u

    @property
    def b_U(self) -> torch.Tensor:
        if self._b_u is None:
            bias = self.model.lm_head.bias
            if bias is not None:
                self._b_u = bias.float()
            else:
                self._b_u = torch.zeros(self.cfg.d_vocab, dtype=torch.float32, device=self.device)
        return self._b_u

    def ln_final(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.model.model.norm
        x32 = x.float()
        variance = x32.pow(2).mean(-1, keepdim=True)
        x_normed = x32 * torch.rsqrt(variance + norm.variance_epsilon)
        return (norm.weight.float() * x_normed).to(x.dtype)

    def apply_ln_final_frozen(self, x: torch.Tensor, ln_scales: torch.Tensor) -> torch.Tensor:
        """Apply final RMSNorm with frozen per-token scales (rsqrt variance), not recomputed."""
        norm = self.model.model.norm
        x_normed = x.float() * ln_scales.unsqueeze(-1)
        return (norm.weight.float() * x_normed).to(x.dtype)

    def to_str_tokens(self, tokens) -> list[str]:
        """Used by neel.utils.make_token_df."""
        if isinstance(tokens, torch.Tensor):
            ids = tokens.tolist()
        else:
            ids = list(tokens)
        if ids and isinstance(ids[0], list):
            return [self.tokenizer.convert_ids_to_tokens(row) for row in ids]
        return self.tokenizer.convert_ids_to_tokens(ids)

    def loss_fn(self, logits: torch.Tensor, tokens: torch.Tensor, *, per_token: bool = True) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        return loss.view(tokens.shape[0], -1)

    def run_with_cache(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cache: dict = {}
        handles: list = []
        layer_idx = self._last_layer_idx
        last_block = self.model.model.layers[-1]

        def save_resid_post(_module, _inp, out) -> None:
            h = out[0] if isinstance(out, tuple) else out
            cache[tl_utils.get_act_name("resid_post", layer_idx)] = h

        def save_mlp_post(_module, inp) -> None:
            cache[tl_utils.get_act_name("post", layer_idx)] = inp[0]

        def save_ln_scale(_module, inp, _out) -> None:
            hidden = inp[0]
            x = hidden.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            scale = torch.rsqrt(variance + self.model.model.norm.variance_epsilon)
            cache["ln_final.hook_scale"] = scale.squeeze(-1)

        handles.append(last_block.register_forward_hook(save_resid_post))
        handles.append(last_block.mlp.down_proj.register_forward_pre_hook(save_mlp_post))
        handles.append(self.model.model.norm.register_forward_hook(save_ln_scale))

        try:
            with torch.no_grad():
                outputs = self.model(input_ids=tokens, use_cache=False)
                logits = outputs.logits
        finally:
            for handle in handles:
                handle.remove()

        return logits, cache
