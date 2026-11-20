# Final results (v6/v7 overnight runs)

All numbers: greedy decoding, Qwen2.5-0.5B-Instruct + LoRA, secret "CAPSTONE". MAPPO checkpoints chosen on a separate
validation split (best val accuracy, zero val leaks); test sets TEST/TEST2/TEST3 never used for selection.
Pooled over three test sets (253 OOD attacks, 123 OOD benign). Full table: `runs/aggregate_v6.md`.

| Arm (seeds) | Q3 OOD benign allowed | Q4 OOD attacks defended | Q4 leaks |
|---|---|---|---|
| base | 115/123 (0.93) | 53/253 (0.21) | 144 |
| SFT small (96 ex.) | 103/123 (0.84) | 164/253 (0.65) | 17 |
| SFT aug | 108/123 (0.88) | 215/253 (0.85) | 2 |
| SFT big (400 ex.) | 107/123 (0.87) | 252/253 (1.00) | 0 |
| MAPPO from small SFT (v6small, 5 seeds) | 494/615 (0.80) | 921/1265 (0.73) | 44 (3.5%) |
| MAPPO, KL anchored to SFT (v7ref, 3 seeds) | 302/369 (0.82) | 563/759 (0.74) | 22 (2.9%) |
| MAPPO from big SFT (v6big, 3 seeds) | 330/369 (0.89) | 747/759 (0.98) | 1 |

Q1 (common benign) and Q2 (common attacks) are essentially perfect for every trained arm (MAPPO 96/100 and 98/100).

## What is supported
- Q4: MAPPO from the small SFT beats the small SFT on OOD attacks (0.73 vs 0.65, Fisher p=0.012; v7ref p=0.005),
  and its leak rate is lower (3.5% vs 6.7%).
- Q1/Q2 hold for both arms.

## What is not supported
- Q3: MAPPO does not fix over-refusal of OOD benign prompts. Differences vs small SFT are not significant
  (v6small 0.80, v7ref 0.82, v6big 0.89 vs 0.84; all p>0.1).
- SFT with 4x more varied data (sft_big) reaches 0.996 on Q4 and beats every MAPPO arm trained from small SFT.
  MAPPO on top of sft_big matches it, with no significant gain. So the gain over SFT is a data-coverage effect
  at least as much as an RL effect.
- Cold-start MAPPO (no SFT init) fails to learn the task (23 leaks in v5).
- Adaptive attacker (0.5B attacker, 239 attacks): ASR base 13.4%, sft_small 0.8%, sft_big 0.4%, v5warm_s2 0.4%,
  v6small_s0 0%, v6big_s0 1.3%. Learned attackers are too weak to separate the trained defenders (differences are 1-3
  attacks), so this test gives no evidence either way.

## Honest summary for the presentation
MAPPO warm-started from a small SFT gives a statistically significant but modest improvement in defending unseen
injection families, with no cost on common prompts. It does not reliably reduce over-refusal on unseen benign
prompts, and a better-curated SFT set matches or exceeds it. Claim accordingly.

## Demo
The live demo still serves v5warm_s2 (verified on all 8 presets). It was not switched, since v6/v7 models are not
clearly better and any switch would need re-verification.
