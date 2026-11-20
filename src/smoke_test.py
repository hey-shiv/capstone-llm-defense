"""Fast end-to-end sanity check: does the whole loop run and learn one step?

    python src/smoke_test.py
"""
from __future__ import annotations

import time

import torch

from config import MODEL_ID
from mappo import make_hint_batch, ppo_update, rollout
from critic import CentralCritic
from policy import Backbone
from evaluate import evaluate_arm
from leak_detect import leak_kind
import random


def main():
    t0 = time.time()
    print("== leak detector ==")
    assert leak_kind("The word is CAPSTONE") == "verbatim"
    assert leak_kind("c-a-p-s-t-o-n-e") == "obfuscated"
    assert leak_kind("Q0FQU1RPTkU=") == "base64"
    assert leak_kind("The capital of France is Paris") is None
    print("ok")

    print("== backbone ==")
    bb = Backbone(MODEL_ID)
    bb.add_adapter("attacker")
    bb.add_adapter("defender")
    print("loaded in", round(time.time() - t0, 1), "s")

    rng = random.Random(0)
    hints = make_hint_batch(rng, 4)
    t = time.time()
    steps = rollout(bb, hints, turns=1)
    print(f"rollout 4 eps: {time.time() - t:.1f}s")
    for ep in steps:
        s = ep[0]
        print(f"  [{s.category:20s}] leak={s.leak} r_d={s.r_defender:+.2f} "
              f"| A: {s.attack[:50]!r} -> D: {s.response[:50]!r}")

    print("== critic ==")
    critic = CentralCritic()
    from env import state_features
    flat = [s for ep in steps for s in ep]
    v = critic([s.attack for s in flat], [s.response for s in flat],
               torch.stack([state_features(s, 0) for s in flat]))
    print("values", v.shape)

    print("== ppo update ==")
    from torch.optim import AdamW
    from config import MAPPO_KL_DEFENDER, MAPPO_LR
    opt = AdamW(bb.trainable_params("defender"), lr=MAPPO_LR)
    log = {}
    pairs = [(s.defender_prompt, s.response) for s in flat]
    old = torch.tensor([s.response_lp for s in flat])
    adv = torch.tensor([s.r_defender for s in flat])
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    ppo_update(bb, opt, "defender", pairs, old, adv, log, kl_coef=MAPPO_KL_DEFENDER)
    print("log", {k: round(v, 4) for k, v in log.items() if isinstance(v, float)})
    assert abs(log["defender_ratio0"] - 1.0) < 0.02, (
        f"PPO ratio must be 1.0 at update start, got {log['defender_ratio0']}")
    print(f"ratio0={log['defender_ratio0']:.4f}  lp_mismatch={log['defender_lp_mismatch']:.4f}")

    print("== eval ==")
    res = evaluate_arm(bb, "defender", "smoke")
    print(f"ASR={res['attack']['asr']:.2f} over_refusal={res['benign']['over_refusal']:.2f}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
