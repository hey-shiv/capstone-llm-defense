"""Figures + tables from the run artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGS, RUNS

ARMS = ["base", "sft", "mappo_v1", "mappo"]
LABELS = {
    "base": "No defence (prompt only)",
    "sft": "SFT defender",
    "mappo_v1": "MAPPO v1 (buggy ratio)",
    "mappo": "MAPPO v2 (fixed)",
    "mappo_s1": "MAPPO v2, seed 1",
}
COLORS = {"base": "#9aa4b2", "sft": "#e8833a",
          "mappo_v1": "#ef4444", "mappo": "#3b82f6", "mappo_s1": "#8b5cf6"}


def load_evals() -> dict:
    out = {}
    for a in ARMS:
        p = RUNS / f"eval_{a}.json"
        if p.exists():
            out[a] = json.loads(p.read_text())
    return out


def _present(evals: dict) -> list[str]:
    return [a for a in ARMS if a in evals]


def fig_asr_by_category(evals: dict):
    arms = _present(evals)
    cats = list(next(iter(evals.values()))["attack"]["per_category"].keys())
    x = np.arange(len(cats))
    w = 0.8 / max(len(arms), 1)
    plt.figure(figsize=(12, 5.2))
    for i, a in enumerate(arms):
        vals = [evals[a]["attack"]["per_category"][c]["asr"] for c in cats]
        xs = x + (i - (len(arms) - 1) / 2) * w
        plt.bar(xs, vals, w, label=LABELS[a], color=COLORS[a])
        for xv, v in zip(xs, vals):
            if v == 0:
                plt.text(xv, 0.012, "0", ha="center", va="bottom", fontsize=7,
                         color=COLORS[a], fontweight="bold")
    plt.xticks(x, [c.replace("_", "\n") for c in cats], fontsize=9)
    plt.ylabel("Attack success rate")
    plt.ylim(0, 1)
    plt.legend(frameon=False, fontsize=8)
    plt.title("Secret extraction rate by attack family (held-out suite)")
    plt.tight_layout()
    plt.savefig(FIGS / "asr_by_category.png", dpi=160)
    plt.close()


def _series(log: dict, key: str):
    """A per-iteration series, if the artifact has one of the right shape."""
    v = log.get(key)
    return v if isinstance(v, list) and len(v) == len(log["iters"]) else None


def _kl_from_console(run: str):
    """Recover per-iteration KL from the training console log.

    The v1/v2-in-flight artifacts only recorded final KL values. The trainer
    prints `kl_d=`/`kl_a=` every other iteration, so the trajectory is still
    recoverable from the log file for runs that have one.
    """
    p = RUNS / f"{run}.log"
    if not p.exists():
        return None
    import re
    d_it, d_kl, a_kl = [], [], []
    for line in p.read_text(errors="ignore").splitlines():
        m = re.search(r"it\s*(\d+).*?kl_d=([+-][\d.]+).*?kl_a=([+-][\d.]+)", line)
        if m:
            d_it.append(int(m.group(1)))
            d_kl.append(float(m.group(2)))
            a_kl.append(float(m.group(3)))
    return (d_it, d_kl, a_kl) if d_it else None


def fig_curve():
    """MAPPO training dynamics, v1 vs v2 when both are available."""
    runs = [(name, RUNS / name / "train_log.json")
            for name in ("mappo_v1", "mappo", "mappo_s1")]
    runs = [(n, p) for n, p in runs if p.exists()]
    if not runs:
        return
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for name, path in runs:
        log = json.loads(path.read_text())
        it = log["iters"]
        lab = LABELS.get(name, name)
        axes[0].plot(it, log["leak_rate"], lw=1.8, label=lab)
        deg = _series(log, "attacker_degenerate")
        if deg is not None:
            axes[1].plot(it, deg, lw=1.8, label=lab)
        rd = _series(log, "r_defender")
        if rd is not None:
            axes[2].plot(it, rd, lw=1.8, label=lab)
        kl = _series(log, "attacker_kl_hist")
        if kl is not None:
            axes[3].plot(it, kl, lw=1.8, label=lab)
            continue
        parsed = _kl_from_console(name)
        if parsed is not None:
            x, _, a_kl = parsed
            axes[3].plot(x, a_kl, lw=1.8, label=lab, marker="o", ms=3)
    axes[0].set_title("train leak rate")
    axes[1].set_title("attacker degenerate-output rate")
    axes[2].set_title("defender reward")
    axes[3].set_title("attacker KL to reference (per token)")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=8)
    plt.suptitle("MAPPO self-play dynamics: collapsed v1 vs corrected v2")
    plt.tight_layout()
    plt.savefig(FIGS / "mappo_training.png", dpi=160)
    plt.close()


def fig_tradeoff(evals: dict):
    usef = {}
    p = RUNS / "usefulness.json"
    if p.exists():
        usef = json.loads(p.read_text())
    plt.figure(figsize=(6.8, 5))
    for a in _present(evals):
        e = evals[a]
        y = e["attack"]["asr"]
        x = usef.get(a, {}).get("uninformative_rate", e["benign"]["over_refusal"])
        plt.scatter([x], [y], s=200, color=COLORS[a], label=LABELS[a], zorder=3)
        short = {"base": "BASE", "sft": "SFT", "mappo_v1": "MAPPO v1", "mappo": "MAPPO v2"}
        plt.annotate(short.get(a, a), (x, y), textcoords="offset points",
                     xytext=(10, 8), fontsize=9)
    plt.xlabel("Benign answers that are unusable (F1 < 0.15) — lower is better")
    plt.ylabel("Attack success rate — lower is better")
    plt.xlim(-0.04, 1.04)
    plt.ylim(-0.04, 1.04)
    plt.grid(alpha=0.25)
    plt.title("Security vs capability")
    plt.legend(frameon=False, fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGS / "tradeoff.png", dpi=160)
    plt.close()


def seed_variance() -> list[dict]:
    """MAPPO v2 per-seed summary, if more than one seed was evaluated."""
    rows = []
    for tag in ("mappo", "mappo_s1", "mappo_s2"):
        p = RUNS / f"eval_{tag}.json"
        if not p.exists():
            continue
        e = json.loads(p.read_text())
        rows.append({"run": tag,
                     "asr": e["attack"]["asr"],
                     "over_refusal": e["benign"]["over_refusal"]})
    return rows


def write_table(evals: dict):
    usef = {}
    p = RUNS / "usefulness.json"
    if p.exists():
        usef = json.loads(p.read_text())
    rows = []
    for a in _present(evals):
        e = evals[a]
        u = usef.get(a, {})
        rows.append({
            "arm": a,
            "asr": round(e["attack"]["asr"], 3),
            "leak_given_engaged": round(e["attack"].get("leak_given_engaged", float("nan")), 3),
            "refusal_rate": round(e["attack"]["refusal_rate"], 3),
            "over_refusal": round(e["benign"]["over_refusal"], 3),
            "refusal_marker_rate": round(e["benign"].get("refusal_marker_rate", 0.0), 3),
            "benign_f1": round(u.get("mean_f1", float("nan")), 3),
            "benign_uninformative": round(u.get("uninformative_rate", float("nan")), 3),
            "generalisation_multi_turn": round(
                e["attack"]["per_category"].get("multi_turn_crescendo", {}).get("asr", float("nan")), 3),
            "generalisation_encoding": round(
                e["attack"]["per_category"].get("encoding_obfuscation", {}).get("asr", float("nan")), 3),
        })
    payload = {"arms": rows, "seed_variance": seed_variance()}
    (RUNS / "summary.json").write_text(json.dumps(payload, indent=1))

    with open(RUNS / "summary.md", "w") as f:
        f.write("| Arm | ASR | Refusal rate | Over-refusal | Benign F1 | Uninformative | ASR: multi-turn | ASR: encoding |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {LABELS.get(r['arm'], r['arm'])} | {r['asr']:.3f} | "
                    f"{r['refusal_rate']:.3f} | {r['over_refusal']:.3f} | "
                    f"{r['benign_f1']:.3f} | {r['benign_uninformative']:.3f} | "
                    f"{r['generalisation_multi_turn']:.3f} | "
                    f"{r['generalisation_encoding']:.3f} |\n")
        sv = payload["seed_variance"]
        if len(sv) > 1:
            f.write("\n### MAPPO v2 seed variance\n\n| Run | ASR | Over-refusal |\n|---|---|---|\n")
            for r in sv:
                f.write(f"| {r['run']} | {r['asr']:.3f} | {r['over_refusal']:.3f} |\n")
    print((RUNS / "summary.md").read_text())


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    evals = load_evals()
    if not evals:
        raise SystemExit("no eval_*.json found")
    fig_asr_by_category(evals)
    fig_curve()
    fig_tradeoff(evals)
    write_table(evals)
    print("figures ->", FIGS)
