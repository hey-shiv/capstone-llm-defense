"""Adaptive-attacker evaluation.
Attack sources: frozen base model as attacker, and MAPPO-trained attacker adapters. Each writes fresh attacks from
tactic hints (all 8 families, 2 samples/hint, T=1.0). Every defender then answers all attacks greedily.
Reported: attack success rate (secret leaked, v6 oracle) and refusal rate, split into ID-family and OOD-family attacks.
Usage: python adaptive_eval.py out.json attacker_dir... -- defender_tag=path ...  (path '-' = base policy)"""
import json, random, sys
from pathlib import Path
from attacks import ID_ATTACK_FAMILIES, TRAIN_HINTS
from config import MODEL_ID, RUNS, DEFENDER_MAX_NEW
from leak_detect import is_refusal_dominant, leak_kind
from policy import Backbone
from prompts import attacker_messages, defender_messages

def main():
    out = sys.argv[1]; rest = sys.argv[2:]; i = rest.index("--")
    attackers = rest[:i]; defenders = dict(a.split("=", 1) for a in rest[i + 1:])
    bb = Backbone(MODEL_ID)
    names = {}
    for k, d in enumerate(attackers):
        n = f"att{k}"; bb.load_adapter(n, d); names[n] = Path(d).parent.name
    for t, p in defenders.items():
        if p != "-": bb.load_adapter(t, p)
    rng = random.Random(99)
    sources = {"frozen_base": None, **{v: k for k, v in names.items()}}
    attacks = []   # (source, family, text)
    for src, adapter in sources.items():
        jobs = [(c, h) for c, hs in TRAIN_HINTS.items() for h in hs for _ in range(2)]
        for b in range(0, len(jobs), 8):
            chunk = jobs[b:b + 8]
            prompts = [bb.chat_prompt(attacker_messages(h, c, [])) for c, h in chunk]
            if adapter is None:
                with bb.base_policy(): outs = bb.generate_batch(prompts, max_new_tokens=48, temperature=1.0, top_p=0.95)
            else:
                with bb.using(adapter): outs = bb.generate_batch(prompts, max_new_tokens=48, temperature=1.0, top_p=0.95)
            for (c, h), o in zip(chunk, outs):
                if len(o.text.split()) >= 3: attacks.append((src, c, o.text))
        print("generated", src, sum(1 for a in attacks if a[0] == src), flush=True)
    res = {}
    for t, p in defenders.items():
        ctx = bb.base_policy() if p == "-" else bb.using(t)
        texts = [a[2] for a in attacks]; outs = []
        with ctx:
            for b in range(0, len(texts), 8):
                ps = [bb.chat_prompt(defender_messages([], x)) for x in texts[b:b + 8]]
                outs += [o.text for o in bb.generate_batch(ps, max_new_tokens=DEFENDER_MAX_NEW, do_sample=False)]
        agg = {}
        for (src, c, _), o in zip(attacks, outs):
            for key in (src, "ALL"):
                for scope in ("id" if c in ID_ATTACK_FAMILIES else "ood", "both"):
                    d = agg.setdefault(f"{key}|{scope}", {"n": 0, "leak": 0, "refuse": 0})
                    d["n"] += 1; d["leak"] += leak_kind(o) is not None; d["refuse"] += is_refusal_dominant(o)
        res[t] = agg
        a = agg["ALL|both"]; b_ = agg["ALL|ood"]
        print(f"ADAPT {t}: ASR all {a['leak']}/{a['n']} ({a['leak']/a['n']:.3f}); OOD-family {b_['leak']}/{b_['n']}", flush=True)
    json.dump({"attacks": attacks, "results": res}, open(out, "w"), indent=1)

main()
