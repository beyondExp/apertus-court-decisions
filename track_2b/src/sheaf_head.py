"""Sheaf glue head for a frozen Apertus reader.

The reader stays frozen. This module is the only trained piece. A document and
each allowed answer are mapped by their own linear restriction, and the score
is the strongest cosine between a document token and an answer token, plus a
small bias on the pooled answer.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


class SheafGlueHead(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.state_map = torch.nn.Linear(d_model, d_model, bias=False)
        self.option_map = torch.nn.Linear(d_model, d_model, bias=False)
        torch.nn.init.eye_(self.state_map.weight)
        torch.nn.init.eye_(self.option_map.weight)
        self.log_scale = torch.nn.Parameter(torch.tensor(2.302585))
        self.option_bias = torch.nn.Linear(d_model, 1, bias=False)
        torch.nn.init.zeros_(self.option_bias.weight)

    def glue(
        self,
        hidden: torch.Tensor,
        state_mask: torch.Tensor,
        option: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score each answer against one document.

        hidden: (B, T, D), state_mask: (B, T) bool
        option: (B, N, L, D), option_mask: (B, N, L) bool
        returns logits (B, N) and the best cosine before the scale (B, N)
        """
        hidden = hidden.float()
        option = option.float()
        state = F.normalize(self.state_map(hidden), dim=-1)
        answers = F.normalize(self.option_map(option), dim=-1)
        dots = torch.einsum("btd,bnld->bntl", state, answers)
        dots = dots.masked_fill(~state_mask[:, None, :, None], -1.0e4)
        dots = dots.masked_fill(~option_mask[:, :, None, :], -1.0e4)
        best = dots.amax(dim=(-1, -2))
        counted = option_mask.to(dtype=answers.dtype).unsqueeze(-1)
        pooled = (answers * counted).sum(dim=2) / counted.sum(dim=2).clamp(min=1.0)
        bias = self.option_bias(pooled).squeeze(-1)
        logits = self.log_scale.exp() * best + bias
        return logits, best


def load_head(path: Path, d_model: int = 4096) -> SheafGlueHead:
    state = torch.load(path, map_location="cpu", weights_only=True)
    head = SheafGlueHead(d_model)
    float_state = {
        key: value.float() if torch.is_floating_point(value) else value
        for key, value in state.items()
    }
    head.load_state_dict(float_state)
    head.eval()
    return head
