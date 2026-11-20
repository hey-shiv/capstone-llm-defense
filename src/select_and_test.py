"""Validation-based checkpoint selection, then a single test evaluation.
Rule (fixed before looking at test): per MAPPO run, pick the checkpoint with the best mean
validation accuracy over Q1-Q4 subject to zero leaks on val; ties -> later checkpoint."""
import json, os, sys
from pathlib import Path
import quadrant_eval as qe
from attacks import *
from config import RUNS, MODEL_ID
from evaluate import generate_all
from policy import Backbone

import importlib
def eval_adapter(path, tag, val):
    os.environ["VAL"] = "1" if val else ""
    importlib.reload(qe)
    return qe.run(tag, path) if False else None

# simple local evaluator reusing quadrant_eval.score
def evaluate(path, tag, val):
    from attacks import VAL_ID_BENIGN, VAL_ID_ATTACK, VAL_OOD_BENIGN, VAL_OOD_ATTACK
    Q = ({"Q1_id_benign": (VAL_ID_BENIGN, True), "Q2_id_attack": (VAL_ID_ATTACK, False),
          "Q3_ood_benign": (VAL_OOD_BENIGN, True), "Q4_ood_attack": (VAL_OOD_ATTACK, False)} if val else
         {"Q1_id_benign": (ID_BENIGN_TEST, True), "Q2_id_attack": (ID_ATTACK_TEST, False),
          "Q3_ood_benign": (OOD_BENIGN_TEST, True), "Q4_ood_attack": ([p for _, p in ood_attack_prompts()], False)})
    bb = Backbone(MODEL_ID)
    if path: bb.load_adapter(tag, str(path)); ctx = bb.using(tag)
    else: ctx = bb.base_policy()
    res = {}
    with ctx:
        for q, (ps, benign) in Q.items():
            rows = qe.score(ps, generate_all(bb, ps, batch=8), benign)
            res[q] = {"accuracy": sum(r["correct"] for r in rows) / len(rows),
                      "leak_rate": sum(r["leak"] is not None for r in rows) / len(rows), "rows": rows}
    del bb
    return res

def mean_acc(r): return sum(v["accuracy"] for v in r.values()) / 4
def leaks(r): return sum(v["leak_rate"] for v in r.values())

if __name__ == "__main__":
    runs = sorted(p.name for p in RUNS.iterdir() if p.is_dir() and p.name.startswith("mappo") and (p / "defender").exists())
    selection = {}
    for run in runs:
        cands = [(it, RUNS / run / f"ckpt{it}" / "defender") for it in (15, 30, 45) if (RUNS / run / f"ckpt{it}" / "defender").exists()]
        cands.append((60, RUNS / run / "defender"))
        best = None
        for it, p in cands:
            r = evaluate(p, f"{run}_{it}", True)
            sc = (leaks(r) == 0, mean_acc(r), it)
            print(f"VAL {run} it{it}: acc={mean_acc(r):.3f} leak={leaks(r):.2f} " + str({q: round(v['accuracy'], 2) for q, v in r.items()}), flush=True)
            if best is None or sc > best[0]: best = (sc, it, p)
        selection[run] = {"iter": best[1], "path": str(best[2]), "val_mean": best[0][1]}
    json.dump(selection, open(RUNS / "selection.json", "w"), indent=1)
    arms = {"base": None, "sft_small": RUNS / "sft_small/defender", "sft_aug": RUNS / "sft_aug/defender"}
    arms.update({run: Path(s["path"]) for run, s in selection.items()})
    out = {}
    for tag, p in arms.items():
        r = evaluate(p, tag, False); out[tag] = r
        print(f"TEST {tag}: " + str({q: round(v['accuracy'], 2) for q, v in r.items()}) + " leak " + str({q: round(v['leak_rate'], 2) for q, v in r.items()}), flush=True)
        json.dump(r, open(RUNS / f"final_{tag}.json", "w"), indent=1)
