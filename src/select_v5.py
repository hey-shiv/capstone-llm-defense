"""v5: per-run checkpoint selection on VAL, then TEST and TEST2 (fresh, held-out benign families)."""
import json, sys
from pathlib import Path
import quadrant_eval as qe
from attacks import *
from attacks2 import TEST2_OOD_BENIGN, test2_ood_attacks
from config import RUNS, MODEL_ID
from evaluate import generate_all
from policy import Backbone

VAL = {"Q1_id_benign": (VAL_ID_BENIGN, True), "Q2_id_attack": (VAL_ID_ATTACK, False),
       "Q3_ood_benign": (VAL_OOD_BENIGN, True), "Q4_ood_attack": (VAL_OOD_ATTACK, False)}
TEST = {"Q1_id_benign": (ID_BENIGN_TEST, True), "Q2_id_attack": (ID_ATTACK_TEST, False),
        "Q3_ood_benign": (OOD_BENIGN_TEST, True), "Q4_ood_attack": ([p for _, p in ood_attack_prompts()], False)}
TEST2 = {"Q3_ood_benign": (TEST2_OOD_BENIGN, True), "Q4_ood_attack": ([p for _, p in test2_ood_attacks()], False)}

def evaluate(path, tag, sets):
    bb = Backbone(MODEL_ID)
    if path: bb.load_adapter(tag, str(path)); ctx = bb.using(tag)
    else: ctx = bb.base_policy()
    res = {}
    with ctx:
        for q, (ps, benign) in sets.items():
            rows = qe.score(ps, generate_all(bb, ps, batch=8), benign)
            res[q] = {"n": len(rows), "correct": sum(r["correct"] for r in rows),
                      "accuracy": sum(r["correct"] for r in rows) / len(rows),
                      "leaks": sum(r["leak"] is not None for r in rows), "rows": rows}
    return res

def line(r): return " ".join(f"{q[:2]}={v['correct']}/{v['n']}(L{v['leaks']})" for q, v in r.items())

if __name__ == "__main__":
    prefix = sys.argv[1]
    runs = sorted(p.name for p in RUNS.iterdir() if p.is_dir() and p.name.startswith(prefix) and (p / "defender").exists())
    sel = {}
    for run in runs:
        cands = [(it, RUNS / run / f"ckpt{it}" / "defender") for it in (15, 30, 45, 60, 75) if (RUNS / run / f"ckpt{it}" / "defender").exists()]
        cands.append((999, RUNS / run / "defender"))
        best = None
        for it, p in cands:
            r = evaluate(p, f"{run}_{it}", VAL)
            leaks = sum(v["leaks"] for v in r.values()); acc = sum(v["accuracy"] for v in r.values()) / 4
            print(f"VAL {run} it{it}: acc={acc:.3f} leaks={leaks} {line(r)}", flush=True)
            sc = (leaks == 0, acc, it)
            if best is None or sc > best[0]: best = (sc, it, p)
        sel[run] = {"iter": best[1], "path": str(best[2])}
    json.dump(sel, open(RUNS / f"selection_{prefix}.json", "w"), indent=1)
    arms = {"base": None, "sft_small": RUNS / "sft_small/defender", "sft_aug": RUNS / "sft_aug/defender"}
    if (RUNS / "sft_big/defender").exists(): arms["sft_big"] = RUNS / "sft_big/defender"
    arms.update({k: Path(v["path"]) for k, v in sel.items()})
    for tag, p in arms.items():
        r1 = evaluate(p, tag, TEST); r2 = evaluate(p, tag + "_t2", TEST2)
        print(f"FINAL {tag}: TEST {line(r1)} | TEST2 {line(r2)}", flush=True)
        json.dump({"TEST": r1, "TEST2": r2}, open(RUNS / f"final5_{tag}.json", "w"))
