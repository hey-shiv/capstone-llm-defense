"""End-to-end driver: data -> SFT -> MAPPO -> evaluation -> figures.

Run from the project root:
    python src/run_all.py                # full pipeline
    python src/run_all.py --stage eval   # re-run evaluation only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RUNS = ROOT / "runs"


def sh(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"  ({time.time() - t:.0f}s)", flush=True)


def stage_data():
    sh([sys.executable, str(SRC / "datagen.py")])


def stage_sft():
    sh([sys.executable, str(SRC / "sft.py")])


def stage_mappo():
    for seed, out in ((0, "mappo"), (1, "mappo_s1")):
        sh([sys.executable, str(SRC / "mappo.py"), "--iters", "20",
            "--seed", str(seed), "--out", f"runs/{out}"])


def stage_eval():
    arms = [
        ("base", None),
        ("sft", RUNS / "sft_defender" / "defender"),
        ("mappo", RUNS / "mappo" / "defender"),
        ("mappo_s1", RUNS / "mappo_s1" / "defender"),
        ("mappo_v1", RUNS / "mappo_v1" / "defender"),
    ]
    for tag, path in arms:
        if path is not None and not Path(path).exists():
            print(f"[skip] {tag}: {path} missing")
            continue
        args = [sys.executable, str(SRC / "evaluate.py")]
        args += [str(path) if path else "-", tag]
        sh(args)


def stage_plots():
    sh([sys.executable, str(SRC / "plots.py")])


def stage_report():
    sh([sys.executable, str(SRC / "usefulness.py")])
    for tag, path in (("base", None),
                      ("sft", RUNS / "sft_defender" / "defender"),
                      ("mappo", RUNS / "mappo" / "defender"),
                      ("mappo_s1", RUNS / "mappo_s1" / "defender"),
                      ("mappo_v1", RUNS / "mappo_v1" / "defender")):
        if path is not None and not Path(path).exists():
            continue
        sh([sys.executable, str(SRC / "multiturn_eval.py"),
            str(path) if path else "-", tag])
    sh([sys.executable, str(SRC / "plots.py")])
    for run in ("mappo", "mappo_v1"):
        if (RUNS / run / "attacker").exists():
            sh([sys.executable, str(SRC / "sample_attack.py"), "--run", run])
    sh([sys.executable, str(SRC / "make_report.py")])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "sft", "mappo", "eval", "plots", "report"])
    a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    if a.stage in ("all", "data"):
        stage_data()
    if a.stage in ("all", "sft"):
        stage_sft()
    if a.stage in ("all", "mappo"):
        stage_mappo()
    if a.stage in ("all", "eval"):
        stage_eval()
    if a.stage in ("all", "plots"):
        stage_plots()
    if a.stage in ("all", "report"):
        stage_report()
