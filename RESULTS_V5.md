# v5 results (selection on VAL; TEST and fresh TEST2 evaluated once). Counts = correct/n (L = secret leaks)

| Arm | TEST Q1 | Q2 | Q3 | Q4 | TEST2 Q3 (n=30) | TEST2 Q4 (n=60) |
|---|---|---|---|---|---|---|
| base | 20/20 | 9/20 | 21/24 | 15/60 (L32) | 28/30 | 12/60 (L33) |
| SFT small (96 pairs) | 20/20 | 19/20 | 20/24 | 39/60 (L4) | 25/30 | 43/60 (L1) |
| SFT aug (96 pairs, varied attacks + hard benign) | 20/20 | 18/20 | 21/24 | 47/60 | 26/30 | 51/60 |
| **SFT big (400 pairs, same data sources)** | 20/20 | 20/20 | 18/24 | **59/60** | 29/30 | **60/60** |
| MAPPO cold (no SFT init) | 20/20 | 9/20 | 21/24 | 14/60 (L23) | 28/30 | 13/60 (L30) |
| MAPPO warm s0 | 20/20 | 20/20 | 18/24 | 46/60 (L4) | 25/30 | 51/60 (L1) |
| MAPPO warm s1 | 18/20 | 20/20 | 18/24 | 40/60 | 28/30 | 41/60 (L2) |
| MAPPO warm s2 | 20/20 | 20/20 | 23/24 | 43/60 (L2) | 28/30 | 46/60 (L2) |

MAPPO warm pooled over 3 seeds vs SFT small: Q4 TEST 129/180 vs 39/60, TEST2 138/180 vs 43/60 (differences of ~5-6 points, not significant);
TEST2 Q3 81/90 vs 25/30. Cold-start MAPPO does not learn the task in this budget. Larger SFT beats every MAPPO arm on Q4.
Caveats: 0.5B model; leak oracle misses cross-script transliterations; Morse-code style items fail for all arms (capability, not safety).
