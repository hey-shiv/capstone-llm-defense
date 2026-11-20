"""Centralised critic (the 'CT' in CTDE).

MAPPO uses one value function that sees the *joint* state - the attack the
attacker produced and the response the defender produced - rather than each
agent's local observation. A frozen MiniLM encoder keeps this cheap enough to
run alongside two LLM actors on a laptop.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from config import CRITIC_ENCODER
from policy import DEVICE

N_SCALARS = 5


class CentralCritic(nn.Module):
    def __init__(self, d_scalar: int = N_SCALARS):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(CRITIC_ENCODER)
        self.enc = AutoModel.from_pretrained(CRITIC_ENCODER)
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.to(DEVICE).eval()
        d = self.enc.config.hidden_size
        self.trunk = nn.Sequential(
            nn.Linear(2 * d + d_scalar, 256), nn.Tanh()
        ).to(DEVICE)
        self.out = nn.Linear(256, 2).to(DEVICE)   # per-agent heads, shared trunk

    @torch.no_grad()
    def embed(self, texts: list[str]) -> torch.Tensor:
        enc = self.tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(DEVICE)
        out = self.enc(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return pooled

    def forward(self, attacks: list[str], responses: list[str], feats: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.embed(attacks), self.embed(responses), feats.to(DEVICE)], dim=-1)
        v = self.trunk(z)
        return self.out(v)          # (B, 2): [defender value, attacker value]

    def parameters_trainable(self):
        return [p for p in self.parameters() if p.requires_grad]
