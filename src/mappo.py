"""Arm 2: Multi-Agent PPO (MAPPO) with a centralised critic.

CTDE layout
-----------
  actors    : two LoRA policies (attacker, defender). Decentralised - each
              sees only its own prompt.
  critic    : one frozen-MiniLM trunk with two value heads that sees the
              JOINT state (attack, response, turn features).
  advantage : A_i = r_i + gamma * V_i(s') - V_i(s), computed per agent.
  objective : clipped surrogate on the length-normalised log-ratio, plus a KL
              penalty to the frozen base policy, per agent.

Episodes are mixed 80/20 adversarial/benign so that the defender is rewarded
for remaining useful, which is what separates "safe" from "refuses everything".

v1 -> v2 fix
-----------
v1 stored the *mean* per-token log-prob at rollout but compared it against a
*summed* log-prob at update time:

    ratio = exp((new_sum - old_mean) / N)

For an unchanged policy that evaluates to exp(mean*(1 - 1/N)) ~= 0.56 instead
of 1.0. Every ratio then sits far below the 1-eps clip boundary, so for every
*negative* advantage the clipped surrogate is a constant and the gradient is
exactly zero. The policy could only be pushed toward behaviour it already
preferred - never away from anything - which is precisely the runaway-collapse
signature observed in v1.

v2 recomputes the behaviour log-probabilities at the start of each update, so
ratio == 1.0 exactly before the first epoch. The agreement between those
recomputed values and the rollout-time values is logged as `lp_mismatch`; it is
an end-to-end check that generation and scoring see the same tokens.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from attacks import CATEGORIES, HARD_BENIGN_TRAIN, ID_ATTACK_FAMILIES, TRAIN_HINTS
from datagen import BENIGN_QA
BENIGN_TRAIN = [q for q, _ in BENIGN_QA]
FAMILIES = list(ID_ATTACK_FAMILIES)
BENIGN_HARD = []
from config import (CRITIC_LR, DEFENDER_MAX_NEW, GEN_MAX_NEW, MAPPO_BATCH,
                    MAPPO_CLIP, MAPPO_DISCOUNT, MAPPO_ITERS, MAPPO_KL_ATTACKER,
                    MAPPO_KL_DEFENDER, MAPPO_LR, MAPPO_TURNS, MODEL_ID, RUNS, SEED)
from critic import CentralCritic
from env import Step, judge, reward_audit, state_features
from leak_detect import is_degenerate
from policy import DEVICE, Backbone
from prompts import attacker_messages, defender_messages

BENIGN_FRAC = 0.30


def make_hint_batch(rng: random.Random, n: int) -> list[tuple[str, str]]:
    out = []
    for _ in range(n):
        if rng.random() < BENIGN_FRAC:
            pool = BENIGN_HARD if (BENIGN_HARD and rng.random() < 0.6) else BENIGN_TRAIN
            out.append((rng.choice(pool), "benign"))
        else:
            cat = rng.choice(FAMILIES)
            out.append((rng.choice(TRAIN_HINTS[cat]), cat))
    return out


@torch.no_grad()
def rollout(bb: Backbone, hints: list[tuple[str, str]], turns: int) -> list[list[Step]]:
    """One episode per hint, fully batched."""
    B = len(hints)
    benign = [c == "benign" for _, c in hints]
    histories: list[list[tuple[str, str]]] = [[] for _ in range(B)]
    priors: list[list[str]] = [[] for _ in range(B)]
    steps: list[list[Step]] = [[] for _ in range(B)]

    for _ in range(turns):
        adv_idx = [i for i in range(B) if not benign[i]]
        attack_text = {i: hints[i][0] for i in range(B)}
        a_prompt_map: dict[int, str] = {}
        a_out_map: dict[int, object] = {}
        if adv_idx:
            a_prompts = [bb.chat_prompt(attacker_messages(hints[i][0], hints[i][1], priors[i]))
                         for i in adv_idx]
            bb.set_adapter("attacker")
            a_outs = bb.generate_batch(a_prompts, max_new_tokens=GEN_MAX_NEW,
                                       temperature=1.0, top_p=0.95)
            for j, i in enumerate(adv_idx):
                attack_text[i] = a_outs[j].text
                a_prompt_map[i] = a_prompts[j]
                a_out_map[i] = a_outs[j]

        d_prompts = [bb.chat_prompt(defender_messages(histories[i], attack_text[i]))
                     for i in range(B)]
        bb.set_adapter("defender")
        d_outs = bb.generate_batch(d_prompts, max_new_tokens=DEFENDER_MAX_NEW,
                                   temperature=1.0, top_p=0.95)

        for i in range(B):
            st = judge(attack_text[i], d_outs[i].text, hints[i][1], priors[i],
                       benign=benign[i])
            # stored values are SUMS; ppo_update recomputes them independently
            st.response_lp = float(d_outs[i].seq_logprob)
            if not benign[i]:
                st.attack_lp = float(a_out_map[i].seq_logprob)
                st.attack_prompt = a_prompt_map[i]
            st.defender_prompt = d_prompts[i]
            steps[i].append(st)
            histories[i].append((st.attack, st.response))
            priors[i].append(st.attack)

    return steps


def advantages_from(steps: list[list[Step]], values: torch.Tensor, turns: int,
                    gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(steps)
    adv_d = torch.zeros(B, turns)
    adv_a = torch.zeros(B, turns)
    for t in range(turns - 1, -1, -1):
        next_v = values[:, t + 1, :] if t + 1 < turns else torch.zeros(B, 2)
        for i, ep in enumerate(steps):
            st = ep[t]
            adv_d[i, t] = st.r_defender + gamma * float(next_v[i, 0]) - float(values[i, t, 0])
            adv_a[i, t] = st.r_attacker + gamma * float(next_v[i, 1]) - float(values[i, t, 1])
    return adv_d, adv_a


def _lengths(bb: Backbone, pairs) -> torch.Tensor:
    out = []
    for p, r in pairs:
        np_ = len(bb.tok(p, add_special_tokens=False).input_ids)
        nf = len(bb.tok(p + r, add_special_tokens=False).input_ids)
        out.append(max(nf - np_, 1))
    return torch.tensor(out, device=DEVICE, dtype=torch.float32)


@torch.no_grad()
def _behaviour_logprobs(bb: Backbone, pairs, mb: int) -> torch.Tensor:
    """Summed log-probabilities of `pairs` under the current (pre-update) policy."""
    return torch.cat([bb._score(pairs[i:i + mb], chunk=mb)
                      for i in range(0, len(pairs), mb)]).detach().cpu()


def ppo_update(bb: Backbone, opt: AdamW, name: str, pairs: list[tuple[str, str]],
               rollout_lp: torch.Tensor, adv: torch.Tensor, log: dict,
               kl_coef: float, epochs: int = 2, mb: int = 2, ref_adapter: str | None = None) -> None:
    """Clipped-surrogate PPO with gradient accumulation over micro-batches."""
    if not pairs:
        return
    bb.set_adapter(name)
    bb.eval_mode()
    lens = _lengths(bb, pairs)
    N = len(pairs)
    a = adv.to(DEVICE)

    # Behaviour policy. Recomputed rather than reused from the rollout so that
    # the ratio is exactly 1.0 before the first epoch; `rollout_lp` is then a
    # cross-check on the generation/scoring plumbing.
    old = _behaviour_logprobs(bb, pairs, mb)
    lp_mismatch = float((old - rollout_lp.float()).abs().mean())

    with torch.no_grad():
        if ref_adapter:                       # trust region around the SFT policy, not the base model
            with bb.using(ref_adapter):
                ref_sum = _behaviour_logprobs(bb, pairs, mb)
            bb.set_adapter(name)
        else:
            with bb.base_policy():
                ref_sum = _behaviour_logprobs(bb, pairs, mb)

    bb.train_mode()
    params = [p for p in bb.model.parameters() if p.requires_grad]

    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        pg_acc, kl_acc, ratio_acc, clip_acc = 0.0, 0.0, 0.0, 0.0
        for i in range(0, N, mb):
            sl = slice(i, i + mb)
            k = min(mb, N - i)
            new_sum = bb._score(pairs[sl], chunk=mb)
            ratio = ((new_sum - old[sl].to(DEVICE)) / lens[sl]).clamp(-10, 10).exp()
            surrogate = torch.min(ratio * a[sl],
                                  ratio.clamp(1 - MAPPO_CLIP, 1 + MAPPO_CLIP) * a[sl])
            kl = ((new_sum - ref_sum[sl].to(DEVICE)) / lens[sl]).mean()
            loss = (-surrogate.mean() + kl_coef * kl) * (k / N)
            loss.backward()
            pg_acc += float(surrogate.mean().detach()) * -k / N
            kl_acc += float(kl.detach()) * k / N
            ratio_acc += float(ratio.mean().detach()) * k / N
            clip_acc += float(((ratio - 1).abs() > MAPPO_CLIP).float().mean()) * k / N
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        opt.step()
        if ep == 0:
            log[f"{name}_ratio0"] = ratio_acc
        log[f"{name}_pg_loss"] = pg_acc
        log[f"{name}_kl"] = kl_acc
        log[f"{name}_ratio"] = ratio_acc
        log[f"{name}_clip_frac"] = clip_acc
        log[f"{name}_lp_mismatch"] = lp_mismatch
    bb.eval_mode()


def train_mappo(episodes_total: int | None = None,
                batch: int = MAPPO_BATCH, turns: int = MAPPO_TURNS,
                out_dir: Path | None = None, seed: int = SEED,
                kl_attacker: float = MAPPO_KL_ATTACKER,
                kl_defender: float = MAPPO_KL_DEFENDER,
                init_defender: Path | None = None, ref_sft: bool = False) -> Path:
    out_dir = out_dir or (RUNS / "mappo")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    episodes_total = episodes_total or (MAPPO_ITERS * batch)

    bb = Backbone(MODEL_ID)
    bb.add_adapter("attacker")
    # Warm start: attach the trained defender under the same name the trainer
    # expects, instead of adding a fresh one and then no-op'ing a load on top of
    # it (which is what the previous version silently did).
    if init_defender is not None and Path(init_defender).exists():
        bb.load_adapter("defender", str(init_defender), trainable=True)
        print(f"[mappo] warm-started defender from {init_defender}", flush=True)
    else:
        if init_defender is not None:
            print(f"[mappo] warm start requested but {init_defender} is missing; "
                  "training from the base policy", flush=True)
        bb.add_adapter("defender")
    ref_name = None
    if ref_sft and init_defender is not None:
        bb.load_adapter("sftref", str(init_defender), trainable=False)
        ref_name = "sftref"
        print("[mappo] KL reference = SFT policy", flush=True)
    critic = CentralCritic()
    bb.set_adapter("defender")
    opt_a = AdamW(bb.trainable_params("attacker"), lr=MAPPO_LR)
    opt_d = AdamW(bb.trainable_params("defender"), lr=MAPPO_LR)
    opt_c = AdamW(critic.parameters_trainable(), lr=CRITIC_LR)
    bb.set_adapter("defender")

    log: dict = {"iters": [], "leak_rate": [], "r_defender": [], "r_attacker": [],
                 "benign_refusal": [], "attacker_degenerate": [], "samples": [],
                 "episodes": 0, "seconds": 0.0, "seed": seed,
                 "reward_audit": reward_audit(BENIGN_FRAC),
                 "config": {"batch": batch, "turns": turns, "lr": MAPPO_LR,
                            "kl_attacker": kl_attacker, "kl_defender": kl_defender,
                            "clip": MAPPO_CLIP}}
    t0 = time.time()
    iters = max(1, episodes_total // batch)

    for it in range(iters):
        hints = make_hint_batch(rng, batch)
        steps = rollout(bb, hints, turns)

        flat = [s for ep in steps for s in ep]
        attacks = [s.attack for s in flat]
        responses = [s.response for s in flat]
        feats = torch.stack([state_features(s, t) for ep in steps for t, s in enumerate(ep)])

        values = critic(attacks, responses, feats).view(len(steps), turns, 2).detach()
        adv_d, adv_a = advantages_from(steps, values, turns, MAPPO_DISCOUNT)

        flat_d = adv_d.flatten()
        flat_d = ((flat_d - flat_d.mean()) / (flat_d.std() + 1e-6)
                  if flat_d.numel() > 1 else torch.zeros_like(flat_d))
        flat_d = torch.nan_to_num(flat_d)

        keep = [i for i, s in enumerate(flat) if not s.benign]
        adv_flat = adv_a.flatten()[keep]          # normalise over the attacker's own steps
        adv_flat = ((adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-6)
                    if adv_flat.numel() > 1 else torch.zeros_like(adv_flat))

        # critic regression against realised rewards
        ret = torch.tensor([[s.r_defender, s.r_attacker] for s in flat], device=DEVICE)
        pred = critic(attacks, responses, feats)
        c_loss = ((pred - ret) ** 2).mean()
        opt_c.zero_grad(set_to_none=True)
        c_loss.backward()
        opt_c.step()

        d_pairs = [(s.defender_prompt, s.response) for s in flat]
        old_d = torch.tensor([s.response_lp for s in flat])
        ppo_update(bb, opt_d, "defender", d_pairs, old_d, flat_d, log,
                   kl_coef=kl_defender, ref_adapter=ref_name)

        a_pairs = [(flat[i].attack_prompt, flat[i].attack) for i in keep]
        old_a = torch.tensor([flat[i].attack_lp for i in keep])
        ppo_update(bb, opt_a, "attacker", a_pairs, old_a, adv_flat, log,
                   kl_coef=kl_attacker)

        adv = [s for s in flat if not s.benign]
        leaks = [bool(s.leak) for s in adv]
        benign_steps = [s for s in flat if s.benign]
        log["iters"].append(it)
        log["leak_rate"].append(sum(leaks) / max(len(leaks), 1))
        log["r_defender"].append(ret[:, 0].mean().item())
        log["r_attacker"].append(ret[:, 1].mean().item())
        log["benign_refusal"].append(
            sum(bool(s.refused) for s in benign_steps) / max(len(benign_steps), 1))
        log["attacker_degenerate"].append(
            sum(is_degenerate(s.attack) for s in adv) / max(len(adv), 1))
        log["critic_loss"] = float(c_loss.detach())
        # per-iteration KL so the trajectory is recoverable from the artifact
        # rather than only from the console log (v1 stored finals only)
        log.setdefault("attacker_kl_hist", []).append(float(log.get("attacker_kl", 0.0)))
        log.setdefault("defender_kl_hist", []).append(float(log.get("defender_kl", 0.0)))
        log["episodes"] += len(steps)
        log["seconds"] = time.time() - t0
        log["samples"].append({
            "it": it,
            "attack": adv[0].attack[:200] if adv else "",
            "response": adv[0].response[:200] if adv else "",
            "leak": adv[0].leak if adv else None,
            "attacker_reward": adv[0].r_attacker if adv else 0.0,
            "defender_reward": adv[0].r_defender if adv else 0.0,
        })

        if it % 2 == 0 or it == iters - 1:
            print(f"[mappo] it{it:3d} leak={log['leak_rate'][-1]:.2f} "
                  f"R_d={log['r_defender'][-1]:+.2f} R_a={log['r_attacker'][-1]:+.2f} "
                  f"benign_ref={log['benign_refusal'][-1]:.2f} "
                  f"deg_a={log['attacker_degenerate'][-1]:.2f} "
                  f"r0_d={log.get('defender_ratio0', 0):.3f} "
                  f"kl_d={log.get('defender_kl', 0):+.3f} "
                  f"kl_a={log.get('attacker_kl', 0):+.3f} {time.time() - t0:.0f}s",
                  flush=True)
        if (it + 1) % 15 == 0 and it + 1 < iters:
            bb.save_adapter("defender", out_dir / f"ckpt{it + 1}")
        if DEVICE == "mps":
            torch.mps.empty_cache()

    bb.save_adapter("defender", out_dir)
    bb.save_adapter("attacker", out_dir)
    json.dump(log, open(out_dir / "train_log.json", "w"), indent=1)
    print(f"[mappo] saved -> {out_dir} ({time.time() - t0:.0f}s)")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--iters", type=int, default=MAPPO_ITERS)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--all-families", action="store_true", help="attacker explores all 8 tactic families")
    ap.add_argument("--hard-benign", action="store_true", help="add attack-looking benign prompts to the benign pool")
    ap.add_argument("--hard-pool", action="store_true", help="use ~300 generated attack-looking benign prompts")
    ap.add_argument("--benign-frac", type=float, default=None)
    ap.add_argument("--ref-sft", action="store_true", help="KL reference = the warm-start SFT adapter")
    ap.add_argument("--kl-attacker", type=float, default=MAPPO_KL_ATTACKER)
    ap.add_argument("--kl-defender", type=float, default=MAPPO_KL_DEFENDER)
    ap.add_argument("--init-defender", type=str, default="",
                    help="optional warm-start adapter dir (e.g. runs/sft_defender/defender)")
    a = ap.parse_args()
    if a.all_families:
        FAMILIES = list(CATEGORIES)
    if a.hard_pool:
        from attacks2 import train_hard_benign_clean
        BENIGN_HARD = train_hard_benign_clean()
    if a.benign_frac is not None:
        BENIGN_FRAC = a.benign_frac
    if a.hard_benign:
        BENIGN_TRAIN = BENIGN_TRAIN + HARD_BENIGN_TRAIN
    train_mappo(episodes_total=a.iters * MAPPO_BATCH,
                out_dir=Path(a.out) if a.out else (RUNS / "mappo"),
                seed=a.seed, kl_attacker=a.kl_attacker, kl_defender=a.kl_defender,
                init_defender=Path(a.init_defender) if a.init_defender else None,
                ref_sft=a.ref_sft)
