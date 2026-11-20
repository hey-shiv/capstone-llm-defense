# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Plain static HTML/CSS/JS in `web/static/`, served by the FastAPI app in `web/web_server.py`. Local only (127.0.0.1). Chosen by the user.

## Users

Capstone reviewers (professors, peers) watching a live demo, and the author running it. They want to see, in a few minutes, whether a trained defender keeps a secret that a prompted-only model leaks.

## Product Purpose

A local demo for a capstone on defending an LLM that guards a secret word. It sends one attack prompt to several defenders side by side (no-defence base model, SFT defender, MAPPO defender) running locally on Qwen2.5-0.5B-Instruct with LoRA adapters, and shows each reply, whether the secret leaked, and the evaluation results from `runs/`.

## Positioning

Real local generation from the project's own trained adapters, with a leak detector that also catches encoded leaks (leetspeak, base64, ROT13, reversal, acrostics). Not a mock-up.

## Operating Context

Runs on the author's Mac (Apple Silicon, MPS). Model weights load from the local Hugging Face cache; if missing they download in the background. Secret word is CAPSTONE. Results data: `runs/summary.json`, `runs/eval_*.json`, `runs/multiturn_*.json`, `figs/`.

## Capabilities and Constraints

- Defenders: base (adapters disabled), SFT (`runs/sft_defender/defender`), MAPPO (`runs/mappo/defender`; `mappo_v1`, `mappo_s1` also exist).
- Slow first load; generation takes seconds. UI must show loading and download state honestly.
- Numbers shown must come from `runs/summary.md`/json, never invented.
- Assumption (inferred from the brief): audience is reviewers, per the user's answer.

## Evidence on Hand

`runs/summary.md` results table (ASR, refusal, over-refusal, benign F1), `figs/*.png`, attacker samples in `runs/attacker_samples*.json`. No testimonials, users or deployment claims exist; do not invent any.

## Product Principles

1. Show the real thing: live output and real measured numbers, flagged honestly when a run is noisy (MAPPO seed variance is large).
2. A leak is the event that matters; make it unmistakable and make a refusal read as the quiet default.
3. State is always visible: model loading, downloading, generating, failed.
4. Local and self-contained: no external fonts, CDNs or network calls at runtime.
