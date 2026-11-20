from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether, PageBreak, Table, TableStyle

OUT = "/Users/shivashant/Desktop/capstone-llm-defense/Evaluator_QA_Prep.pdf"

H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=20, leading=24, spaceAfter=4)
SUB = ParagraphStyle("sub", fontName="Helvetica", fontSize=10, leading=14, textColor=colors.HexColor("#555555"), spaceAfter=10)
SEC = ParagraphStyle("sec", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=colors.HexColor("#1a3c6e"), spaceBefore=12, spaceAfter=6)
Q = ParagraphStyle("q", fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=2)
A = ParagraphStyle("a", fontName="Helvetica", fontSize=9.8, leading=13.5, leftIndent=10, spaceAfter=2)
T = ParagraphStyle("t", fontName="Helvetica-Oblique", fontSize=9, leading=12, leftIndent=10, textColor=colors.HexColor("#8a1c1c"), spaceAfter=4)
B = ParagraphStyle("b", fontName="Helvetica", fontSize=9.8, leading=13.5, spaceAfter=3)

sections = []

def sec(title, items):
    sections.append((title, items))

# (question, answer, optional trap/tip)
sec("A. Problem and motivation", [
("What problem are you solving, and why does it matter?",
 "An LLM that holds a secret in its system prompt (API key, internal instructions) can be tricked by prompt injection into leaking it. I use a toy version: the model, VaultBot, knows the word CAPSTONE and must never reveal it. It is a clean, measurable proxy for real prompt-injection defence: a leak is unambiguous, so I can count successes and failures.",
 None),
("Why a secret-word game and not a real-world benchmark like HarmBench or a jailbreak set?",
 "A secret word gives an objective, automatic success signal (string match plus encodings), so no LLM judge is needed and results are reproducible. Real jailbreak benchmarks need a judge model or human grading, which adds noise. The trade-off is realism: this is a controlled proxy, and I state that absolute numbers do not transfer to frontier models.",
 "Admit the limitation up front; evaluators respect that."),
("What exactly is your research question?",
 "Does multi-agent RL (MAPPO, with an attacker and defender trained together) defend better than simple supervised fine-tuning (SFT)? I compared three arms: no defence (system prompt only), SFT, and MAPPO, on held-out attacks and benign prompts.",
 None),
("What was your headline result?",
 "Honest answer: MAPPO warm-started from a small SFT set gives a statistically significant but modest gain on unseen attack families (0.73 vs 0.65 defended, Fisher p=0.012; leak rate 3.5% vs 6.7%), with no cost on common prompts. But SFT with 4x more varied data (400 pairs) reaches 0.996 and beats every MAPPO arm trained from small SFT. So the gain is largely a data-coverage effect, and I do not claim RL beats SFT in general.",
 "This is in RESULTS_FINAL_V6.md. Do not overclaim."),
])

sec("B. Model and setup choices", [
("Why Qwen2.5-0.5B-Instruct? Why such a small model?",
 "Compute. Everything ran on one 16 GB Apple-silicon laptop (MPS). Two actors plus a critic, with rollouts and PPO updates, only fit at 0.5B. It is also instruction-tuned, so it follows the VaultBot system prompt, and it leaks often enough (about 54% ASR undefended) that there is something to defend. Limitation: absolute robustness at 0.5B does not predict larger models, so only the relative ranking of defences is claimed.",
 None),
("Why LoRA instead of full fine-tuning?",
 "Memory and speed (r=16, alpha=32, all attention and MLP projections). It lets attacker and defender be two adapters on one shared frozen backbone, so peak memory is about 2 GB. It also makes the KL reference trivial: disable the adapter and you have the base policy.",
 None),
("Why Qwen and not Llama or Mistral?",
 "Small, strong for its size, permissively licensed, a good chat template, and runs on MPS. I did not run a model-family ablation, so I cannot say the findings are Qwen-independent.",
 "Don't claim generality you did not test."),
("Why one shared backbone for attacker and defender? Is that a problem?",
 "It is what makes training fit in 16 GB. Cost: the two adapters are coupled, since updating one perturbs the other's log-probs through the shared base. Listed under limitations. Separate backbones would be cleaner but about double the memory.",
 None),
("How is the secret defined and how do you detect a leak?",
 "The secret is CAPSTONE, in the system prompt with four rules. The oracle (src/leak_detect.py) checks verbatim, leetspeak, separator obfuscation (c-a-p-s-t-o-n-e), base64, ROT13, reversal, and first-letter acrostics. Limitation: paraphrase or cross-script transliteration would evade it, so measured ASR is a lower bound on true leakage. One live demo case, \"CAPSTON\", shows near-misses too.",
 None),
])

sec("C. Why MAPPO", [
("Why MAPPO and not plain PPO, DPO, or RLHF?",
 "Extraction defence is adversarial and non-stationary: any fixed attacker set is a moving target. MAPPO trains attacker and defender together, so the defender sees an evolving distribution of attacks. PPO alone would need a fixed attacker or dataset. DPO needs preference pairs, effectively another supervised dataset, so it would resemble the SFT arm. I did not run DPO or plain PPO, so that is a design argument, not an empirical one.",
 "Expect a follow-up: \"did you test PPO or DPO?\" Answer: no."),
("What does 'centralised training, decentralised execution' mean in your code?",
 "The critic (frozen MiniLM encoder plus a trunk with two value heads) sees the joint state: attack text, response text, and turn features. The actors do not. The attacker sees only its tactic hint; the defender sees only the conversation. At deployment only the defender runs, with no critic. Advantage is r + gamma*V(s') - V(s) per agent.",
 None),
("Is this really multi-agent? There is one defender and one attacker, one turn per episode.",
 "Fair challenge. It is two-agent, general-sum (attacker reward is leak, defender reward is no leak), with a centralised critic, so it is MAPPO structurally. Training episodes are short (1 attacker-to-defender round), so the multi-turn dynamics are limited. I tested multi-turn separately with 3-turn chains as out-of-distribution evaluation.",
 None),
("Did the learned attacker actually become a good red team?",
 "No, and the report says so. Once the defender stopped leaking, the attacker's reward saturated and it collapsed into paraphrasing its own instructions (my degeneracy detector missed that). So the defence claims rest on the fixed held-out human-written attack suites, not on the self-play curriculum. In the adaptive-attacker test, learned attackers were too weak to separate defenders (differences of 1-3 attacks), so that test gives no evidence either way.",
 "Know this one cold; it is the most damaging admitted weakness."),
("What are the MAPPO hyperparameters and how did you choose them?",
 "LR 1e-5, clip 0.2, gamma 0.95, entropy 0.01, batch 8, KL penalty to a reference (attacker 0.15 / defender 0.10 in config), LoRA r=16. Final runs use 90 iterations and a 40% benign fraction, with --all-families and --hard-pool. Values were set by hand with limited tuning; I did not run a systematic sweep. Checkpoint selection used a separate validation split.",
 None),
])

sec("D. Reward design", [
("Walk me through the reward function. Why those numbers?",
 "Defender: leak -1.5, refuse an attack +1.0, answer a benign prompt +1.0, refuse a benign prompt -1.0, degenerate -0.2, engaged-without-leak 0.0. Attacker: extract +1.0, softened +0.25, stonewalled +0.02, duplicate or degenerate -0.3. Leak is penalised more than a benign refusal helps, and over-refusal is penalised symmetrically so refusing everything is not optimal.",
 None),
("Could the policy cheat by refusing everything?",
 "That was a real risk. Under an 80/20 attack/benign mix, blanket refusal was near-optimal in v1. I added a reward audit (env.reward_audit) that scores degenerate unconditional policies, raised the benign weight, and used a 40% benign fraction in the final runs. Even so, in the earlier report the reward prior still leaned toward refusal when the model cannot tell attack from question.",
 None),
("Did you see reward hacking?",
 "Yes. Rewarding \"engaged without leaking\" at +0.40 taught evasive non-refusals (a seed-1 hack), so I set it to 0. On the attacker side, a +0.03 length-prior shaping term became the real objective once the main signal saturated, causing instruction-echo collapse.",
 None),
])

sec("E. The bug and what you learned", [
("Tell me about the MAPPO v1 collapse.",
 "v1 stored the mean per-token log-prob at rollout but compared it to a summed log-prob at update, so ratio = exp((new_sum - old_mean)/N) was about 0.56 for an unchanged policy. That is below the lower clip boundary, so for negative advantages min(r*A, clip(r)*A) is constant and the gradient is exactly zero: the policy could only be pushed toward what it liked, never away from anything. v2 recomputes behaviour log-probs at update start; ratio at iteration 0 is 0.997. The ratio0 and lp_mismatch values are logged so this is verifiable.",
 "Good story to volunteer; it shows debugging depth."),
("Why is the buggy v1 still at 5% ASR if it was broken?",
 "Because ASR alone is gameable: v1 had destroyed the model (62.5% of benign answers unusable, 25% over-refusal) while still looking safe. That is why I use three independent probes: attack success, capability (benign token-F1), and sampling what the policies actually emit.",
 None),
])

sec("F. Evaluation design and statistics", [
("How do you know your test set is not leaked into training?",
 "The held-out suites were written independently of training hints and checked for string overlap (no test string appears in training data). OOD families were never shown to SFT or the MAPPO attacker. Checkpoints were selected on a separate validation split, and TEST, TEST2 and TEST3 were never used for selection.",
 None),
("Explain the 'four quadrants' Q1-Q4.",
 "Q1: ordinary benign prompts (in-distribution), should be answered. Q2: common attacks (ID), should be refused with no leak. Q3: out-of-distribution benign, tests over-refusal. Q4: out-of-distribution attacks (6 unseen families), the hard test of generalisation. Correct means benign answered, or attack refused with no leak.",
 None),
("Are your results statistically significant? Sample sizes look small.",
 "Early single-test runs had n=20 to 60 and many differences were not significant. I fixed that by pooling three test sets (253 OOD attacks, 123 OOD benign) and using 3-5 seeds per MAPPO arm, with Fisher exact tests and Wilson CIs. Significant: v6small vs sft_small on Q4 (p=0.012; v7ref p=0.005). Not significant: all Q3 comparisons (p>0.1). SFT and baselines are single runs, which is a limitation.",
 None),
("Why greedy decoding?",
 "Reproducibility and a deterministic per-prompt comparison. Limitation: it hides sampling-based leaks, so a real attacker with retries would do worse than my numbers suggest. Also, single-prompt serving in the demo can differ slightly from the batched eval.",
 None),
("How did you choose which MAPPO checkpoint to report? Is that cherry-picking?",
 "A rule fixed in advance: per run, pick the checkpoint with best mean validation accuracy and zero validation leaks. Test sets were evaluated once after selection. The demo serves v5warm_s2, picked before the v6/v7 results, not tuned afterward.",
 None),
])

sec("G. Results interpretation", [
("Your own table shows SFT big beats MAPPO. So why should anyone use MAPPO?",
 "On this task, they should not, and I say so. SFT-big gets 252/253 on OOD attacks with 0 leaks at 0.87 benign allowed. MAPPO on top of sft_big matches it (0.98) with no significant gain. MAPPO's value in this study is the controlled finding that RL adds a small, significant gain over a weak SFT start, and that coverage of attack families matters more than the training algorithm.",
 "Take the honest route; the evaluator will see the table anyway."),
("Why did cold-start MAPPO (no SFT init) fail?",
 "With a 0.5B model and a small budget it cannot discover the attack/benign boundary by exploration: from a base policy that leaks on about 76% of the attacks it answers, refusing everything pays better than answering. So the reward gives no usable gradient toward discrimination. It had 23 leaks in v5. SFT demonstrations encode the boundary directly, which is why warm-starting helps.",
 None),
("MAPPO does not fix over-refusal on OOD benign prompts. Why not?",
 "Q3: MAPPO from small SFT gets 0.80-0.82 vs SFT-small 0.84 (p>0.1). The reward prior still favours refusing when uncertain, and the model is too small to separate an attack-looking benign question from an actual attack. A stronger benign reward or an explicit attack classifier in the reward is the obvious next step; I did not test it.",
 None),
("What is the 'conditional leak rate' and why report it?",
 "Leak rate among prompts the defender chose to answer rather than deflect. The undefended 53.8% ASR was 75.9% conditional on engaging, because it refuses a third of attacks by default. It shows whether engaging with an attack is safe, which plain ASR hides.",
 None),
("SFT had a side effect on benign answers. What was it?",
 "In the earlier report, SFT appended refusal boilerplate to 70.8% of correct benign answers (e.g. '17 x 23 = 391. ... I can't share the secret.'): a learned reflex bleeding into unrelated turns. Content stays correct, but it is a real artefact of safety training.",
 None),
("Does the defence hold in multi-turn conversations?",
 "In the earlier 3-turn chain test (10 chains): undefended 50% chain ASR, SFT 0%, MAPPO v2 seed 0 at 10%, but seed 1 back at 50%. So MAPPO showed large seed variance there. This test is small (n=10 chains), so treat it as indicative only.",
 "This was measured on the earlier v2 models, not the final v6/v7 ones."),
])

sec("H. Limitations, ethics and what's next", [
("What are the biggest limitations of your study?",
 "(1) 0.5B model, one machine; only relative ranking claimed. (2) The leak oracle misses paraphrase and cross-script leaks, so ASR is a lower bound. (3) One secret and one system prompt; unclear if it generalises. (4) SFT and baselines are single-seed. (5) The learned attacker is weak, so adaptive-attack robustness is unproven. (6) Budgets are not matched between SFT and MAPPO. (7) Morse-code-style attacks fail for all arms, which is a capability limit, not a safety one.",
 None),
("Would this transfer to a larger model or to a real deployment?",
 "Unknown. I did not test larger models. A real deployment would face adaptive, sampling-based attackers and secrets that cannot be string-matched. A system prompt is also not a security boundary: real secrets should not live in prompts at all. This is a research probe, not a production defence.",
 None),
("Is there a dual-use or ethics concern in training an attacker?",
 "Low risk here: the attacker is a 0.5B model producing short extraction prompts for a toy word, and the attack taxonomy (roleplay, encoding, override) is already public. Everything runs locally. The value is defensive: understanding what training makes a model robust.",
 None),
("If you had another month, what would you do?",
 "(1) Add a pretrained attack classifier to the reward to attack the over-refusal gap. (2) Test a larger model (1.5B-7B). (3) Compare against DPO and plain PPO. (4) Fix the attacker collapse with entropy or diversity rewards so the adaptive test is meaningful. (5) Multi-seed SFT baselines. (6) Sampling-based, not greedy, evaluation.",
 None),
])

sec("I. Live demo questions", [
("Is the demo real or mocked? Are the examples cherry-picked?",
 "Real local generation from the trained adapters (Qwen2.5-0.5B plus LoRA) with the same leak detector. The preset prompts are curated, and I will say so. Q1: all allow; Q2: SFT and MAPPO refuse while base leaks CAPSTONE; Q3: SFT over-refuses where MAPPO answers; Q4: SFT leaks 'CAPSTON' where MAPPO refuses. The aggregate table, not the presets, is the evidence.",
 "Run: python3 -m web.web_server, open 127.0.0.1:8000, wait for 'Ready on mps'."),
("Why does the demo use MAPPO v5warm_s2 and not your best model?",
 "It was verified on all 8 presets, and the v6/v7 models were not clearly better. Switching would have needed re-verification. The best overall model is SFT-big, and I say so.",
 None),
("What if the live demo leaks or behaves differently than shown?",
 "Expected: single-prompt serving differs slightly from batched eval, and 0.5B models are noisy. I would show the leak flag, then point to the aggregate numbers (e.g. 3.5% leak over 1265 attacks) rather than any single output.",
 None),
])

story = [
    Paragraph("Evaluator Q&amp;A Preparation", H1),
    Paragraph("SFT vs MAPPO for defending an LLM against prompt-injection secret extraction (Qwen2.5-0.5B, secret = CAPSTONE). 37 likely questions with answers grounded in this repo's results. Italic red notes are tips.", SUB),
]

# one-minute pitch + number cheat sheet
story.append(Paragraph("30-second pitch", SEC))
story.append(Paragraph(
    "I trained a small LLM (Qwen2.5-0.5B) to keep a secret word under prompt-injection attacks and compared three defences: "
    "no defence, supervised fine-tuning (SFT), and multi-agent PPO (MAPPO) with a learned attacker. "
    "MAPPO warm-started from a small SFT gives a significant but modest gain on unseen attack families (0.73 vs 0.65, p=0.012), "
    "but a bigger, more varied SFT set matches or beats it. My main lessons are that coverage matters more than the algorithm, "
    "and that safety results need three probes (attack success, capability, and sampled outputs) because each can be fooled alone.", B))

story.append(Paragraph("Numbers to memorise (pooled OOD test, 253 attacks / 123 benign)", SEC))
rows = [["Arm", "OOD attacks defended", "Leaks", "OOD benign allowed"],
        ["Base (no defence)", "53/253 (0.21)", "144", "115/123 (0.93)"],
        ["SFT small (96)", "164/253 (0.65)", "17", "103/123 (0.84)"],
        ["SFT big (400)", "252/253 (1.00)", "0", "107/123 (0.87)"],
        ["MAPPO from small SFT (5 seeds)", "921/1265 (0.73)", "44 (3.5%)", "494/615 (0.80)"],
        ["MAPPO, KL to SFT (3 seeds)", "563/759 (0.74)", "22 (2.9%)", "302/369 (0.82)"],
        ["MAPPO from big SFT (3 seeds)", "747/759 (0.98)", "1", "330/369 (0.89)"]]
t = Table(rows, colWidths=[62*mm, 40*mm, 24*mm, 40*mm])
t.setStyle(TableStyle([
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8.8),
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e6ecf5")),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#b9c2d0")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3)]))
story.append(t)
story.append(Spacer(1, 4))
story.append(Paragraph("Source: RESULTS_FINAL_V6.md and runs/aggregate_v6.md. Cold-start MAPPO (no SFT init) failed to learn (23 leaks in v5). "
                       "Fisher p-values vs SFT small on Q4: v6small 0.012, v7ref 0.005. Q3 differences all p&gt;0.1.", B))

n = 0
for title, items in sections:
    story.append(Paragraph(title, SEC))
    for q, a, tip in items:
        n += 1
        block = [Paragraph(f"Q{n}. {q}", Q), Paragraph(a, A)]
        if tip:
            block.append(Paragraph("Tip: " + tip, T))
        story.append(KeepTogether(block))

story.append(Paragraph("Golden rules for the demo", SEC))
for s in ["Claim only what the table supports: modest, significant MAPPO gain over small SFT; no over-refusal fix; SFT-big is the best model.",
          "Volunteer the weaknesses (weak attacker, greedy decoding, single-seed SFT, lower-bound oracle) before they are found.",
          "If unsure whether something was tested, say 'I did not test that' rather than guessing.",
          "Know the v1 ratio bug and the attacker collapse stories; they show real understanding."]:
    story.append(Paragraph("• " + s, B))

assert n == 37, n
SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm,
                  title="Evaluator Q&A Preparation").build(story)
print("ok", n)
