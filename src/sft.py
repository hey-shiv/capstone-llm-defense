"""Arm 1: supervised fine-tuning of the defender on (attack -> refusal) pairs."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from config import (DATA, MODEL_ID, RUNS, SFT_BATCH, SFT_EPOCHS, SFT_LR, SEED)
from policy import Backbone


def run_sft(data_path: Path | None = None, out_dir: Path | None = None,
            epochs: int = SFT_EPOCHS, sample_count: int | None = None) -> Path:
    data_path = data_path or (DATA / "sft_pairs.jsonl")
    out_dir = out_dir or (RUNS / "sft_defender")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(data_path)]
    if sample_count:
        rows = rows[:sample_count]
    rng = random.Random(SEED)
    rng.shuffle(rows)

    bb = Backbone(MODEL_ID)
    bb.add_adapter("defender")
    bb.set_adapter("defender")
    bb.train_mode()

    params = [p for p in bb.model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=SFT_LR, weight_decay=0.0)
    print(f"[sft] {len(rows)} pairs, {epochs} epochs, {len(params)} param tensors")

    step, t0 = 0, time.time()
    losses = []
    for ep in range(epochs):
        rng.shuffle(rows)
        for i in range(0, len(rows), SFT_BATCH):
            chunk = rows[i:i + SFT_BATCH]
            pairs = [(r["prompt"], r["response"]) for r in chunk]
            loss = bb.token_nll(pairs, chunk=2)      # gather-based, no 151k-way CE
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            step += 1
            if step % 20 == 0:
                print(f"[sft] ep{ep} step{step} loss={sum(losses[-20:]) / 20:.4f} "
                      f"{(time.time() - t0):.0f}s", flush=True)
    bb.save_adapter("defender", out_dir)
    json.dump({"loss_curve": losses, "seconds": time.time() - t0},
              open(out_dir / "train_log.json", "w"))
    print(f"[sft] saved -> {out_dir}  ({time.time() - t0:.0f}s)")
    return out_dir


if __name__ == "__main__":
    run_sft()
