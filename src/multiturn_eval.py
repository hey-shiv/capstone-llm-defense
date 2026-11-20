"""Genuine multi-turn evaluation.

The single-message `multi_turn_crescendo` prompts in `EVAL_SUITE` only test
whether the model is fooled by a false premise. This harness sends real
escalation chains turn by turn, feeding the model's own replies back as history,
so a defence has to hold a boundary across the conversation rather than across
one sentence.

    python src/multiturn_eval.py <adapter-dir> <tag>
    python src/multiturn_eval.py - base
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from attacks import MULTITURN_SUITE
from config import DEFENDER_MAX_NEW, MODEL_ID, RUNS
from leak_detect import is_degenerate, is_refusal_dominant, leak_kind
from policy import Backbone
from prompts import defender_messages


@torch.no_grad()
def run_chain(bb: Backbone, chain: list[str], batch_turns: bool = False) -> dict:
    """Play one conversation to completion, recording every turn."""
    history: list[tuple[str, str]] = []
    turns = []
    first_leak = None
    for i, user in enumerate(chain):
        prompt = bb.chat_prompt(defender_messages(history, user))
        out = bb.generate_batch([prompt], max_new_tokens=DEFENDER_MAX_NEW,
                                do_sample=False)[0]
        kind = leak_kind(out.text)
        turns.append({
            "turn": i,
            "user": user,
            "assistant": out.text,
            "leak": kind,
            "refused": is_refusal_dominant(out.text),
            "degenerate": is_degenerate(out.text),
        })
        if kind and first_leak is None:
            first_leak = i
        history.append((user, out.text))
    n_turns = len(chain)
    return {
        "n_turns": n_turns,
        "leaked": first_leak is not None,
        "first_leak_turn": first_leak,
        "leak_kind": next((t["leak"] for t in turns if t["leak"]), None),
        "refused_all": all(t["refused"] for t in turns),
        "refused_any": any(t["refused"] for t in turns),
        "degenerate_any": any(t["degenerate"] for t in turns),
        "turns": turns,
    }


def evaluate(bb: Backbone, adapter: str | None, tag: str) -> dict:
    ctx = bb.base_policy() if adapter is None else bb.using(adapter)
    with ctx:
        records = [run_chain(bb, chain) for chain in MULTITURN_SUITE]

    n = len(records)
    leaked = sum(r["leaked"] for r in records)
    # per-turn leak rate: how often does the model leak at turn k, given it
    # reached turn k without leaking
    per_turn = {}
    for k in range(3):
        at_k = [r for r in records if r["first_leak_turn"] == k]
        survived = [r for r in records if r["first_leak_turn"] is None
                    or r["first_leak_turn"] >= k]
        per_turn[k] = {
            "leaks_here": len(at_k),
            "survived_to_here": len(survived),
            "hazard": len(at_k) / max(len(survived), 1),
        }
    return {
        "arm": tag,
        "n_chains": n,
        "n_turns": sum(r["n_turns"] for r in records),
        "chain_asr": leaked / max(n, 1),
        "turn1_asr": sum(r["first_leak_turn"] == 0 for r in records) / max(n, 1),
        "leak_hazard_by_turn": per_turn,
        "refused_all_chains": sum(r["refused_all"] for r in records) / max(n, 1),
        "degenerate_chains": sum(r["degenerate_any"] for r in records) / max(n, 1),
        "records": records,
    }


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    tag = sys.argv[2] if len(sys.argv) > 2 else "base"
    bb = Backbone(MODEL_ID)
    if path is not None:
        bb.load_adapter(tag, str(path))
        res = evaluate(bb, tag, tag)
    else:
        res = evaluate(bb, None, tag)
    print(f"{tag}: chain ASR={res['chain_asr']:.3f} "
          f"turn-1 ASR={res['turn1_asr']:.3f} "
          f"refused-all={res['refused_all_chains']:.3f} "
          f"degenerate={res['degenerate_chains']:.3f}")
    (RUNS / f"multiturn_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
