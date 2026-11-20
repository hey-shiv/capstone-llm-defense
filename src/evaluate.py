"""Evaluation harness: held-out attack suite + benign over-refusal suite."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from attacks import BENIGN_SUITE, all_eval_prompts
from config import DATA, DEFENDER_MAX_NEW, MODEL_ID, RUNS
from leak_detect import is_degenerate, is_refusal, leak_kind
from policy import Backbone
from prompts import defender_messages


@torch.no_grad()
def generate_all(bb: Backbone, prompts: list[str], batch: int = 8,
                 max_new: int = DEFENDER_MAX_NEW, sample: bool = False) -> list[str]:
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = [bb.chat_prompt(defender_messages([], p)) for p in prompts[i:i + batch]]
        res = bb.generate_batch(chunk, max_new_tokens=max_new,
                                temperature=0.9 if sample else None,
                                do_sample=sample)
        outs.extend(r.text for r in res)
    return outs


def summarise(prompts: list[str], outputs: list[str], cats: list[str]) -> dict:
    from leak_detect import is_refusal_dominant
    per_cat: dict[str, dict] = {}
    for cat, out in zip(cats, outputs):
        d = per_cat.setdefault(cat, {"n": 0, "leaks": 0, "refusals": 0, "degenerate": 0})
        d["n"] += 1
        d["leaks"] += int(leak_kind(out) is not None)
        d["refusals"] += int(is_refusal_dominant(out))
        d["degenerate"] += int(is_degenerate(out))

    for d in per_cat.values():
        d["asr"] = d["leaks"] / d["n"]
        d["refusal_rate"] = d["refusals"] / d["n"]

    n = len(outputs)
    leaks = sum(int(leak_kind(o) is not None) for o in outputs)
    return {
        "n": n,
        "asr": leaks / max(n, 1),
        "refusal_rate": sum(int(is_refusal_dominant(o)) for o in outputs) / max(n, 1),
        "per_category": per_cat,
    }


def evaluate_arm(bb: Backbone, adapter: str | None, tag: str,
                 batch: int = 8, sample: bool = False) -> dict:
    """adapter=None evaluates the frozen base model."""
    if adapter is None:
        ctx = bb.base_policy()
    else:
        ctx = bb.using(adapter)

    with ctx:
        att_rows = all_eval_prompts()
        cats = [c for c, _ in att_rows]
        prompts = [p for _, p in att_rows]
        attack_outs = generate_all(bb, prompts, batch=batch, sample=sample)
        benign_outs = generate_all(bb, BENIGN_SUITE, batch=batch, sample=sample)

    attack_sum = summarise(prompts, attack_outs, cats)
    benign_sum = summarise(BENIGN_SUITE, benign_outs, ["benign"] * len(BENIGN_SUITE))
    from leak_detect import is_refusal_dominant
    benign_refusal = sum(int(is_refusal_dominant(o)) for o in benign_outs) / len(benign_outs)
    benign_marker = sum(int(is_refusal(o)) for o in benign_outs) / len(benign_outs)

    kinds: dict[str, int] = {}
    for o in attack_outs:
        k = leak_kind(o)
        if k:
            kinds[k] = kinds.get(k, 0) + 1

    from leak_detect import is_refusal_dominant
    kinds_list = [leak_kind(o) for o in attack_outs]
    refused = [is_refusal_dominant(o) for o in attack_outs]
    engaged = [not r for r in refused]
    # Conditional leak rate: how often does the defender leak *when it chooses to
    # answer* rather than deflect. This is the number that decides whether
    # answering an attack is worth the risk (see REPORT.md).
    cond_leak = (sum(1 for k, e in zip(kinds_list, engaged) if k and e)
                 / max(sum(engaged), 1))

    result = {
        "arm": tag,
        "sample": sample,
        "attack": {**attack_sum, "engaged_n": sum(engaged),
                   "leak_given_engaged": cond_leak},
        "benign": {**benign_sum, "over_refusal": benign_refusal,
                   "refusal_marker_rate": benign_marker},
        "leak_kinds": kinds,
        "examples": [
            {"prompt": p, "output": o, "category": c, "leak": k,
             "refused": r, "degenerate": is_degenerate(o),
             "refusal_marker": is_refusal(o)}
            for p, o, c, k, r in zip(prompts, attack_outs, cats, kinds_list, refused)
        ],
        "benign_examples": [
            {"prompt": p, "output": o, "refused": is_refusal(o),
             "refusal_dominant": is_refusal_dominant(o)}
            for p, o in zip(BENIGN_SUITE, benign_outs)
        ],
    }
    return result


def load_and_eval(adapter_path: Path | None, tag: str, sample: bool = False) -> dict:
    bb = Backbone(MODEL_ID)
    if adapter_path is not None:
        bb.load_adapter(tag, str(adapter_path))
        return evaluate_arm(bb, tag, tag, sample=sample)
    return evaluate_arm(bb, None, tag, sample=sample)


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    tag = sys.argv[2] if len(sys.argv) > 2 else "base"
    res = load_and_eval(path, tag)
    print(f"{tag}: ASR={res['attack']['asr']:.3f} "
          f"refusal={res['attack']['refusal_rate']:.3f} "
          f"over-refusal={res['benign']['over_refusal']:.3f}")
    (RUNS).mkdir(exist_ok=True)
    json.dump(res, open(RUNS / f"eval_{tag}.json", "w"), indent=1)
