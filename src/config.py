"""Central configuration for the CAPSTONE secret-keeping experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
FIGS = ROOT / "figs"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SECRET = "CAPSTONE"

# ---------------------------------------------------------------- training
SEED = 0
SFT_EXAMPLES = 96           # one (attack, safe-response) pair per example
SFT_EPOCHS = 2
SFT_LR = 1e-4
SFT_BATCH = 4
SFT_MAX_LEN = 512

MAPPO_EPISODES = 384        # comparable exposure to SFT's 512 pairs
MAPPO_ITERS = 60            # 20 x 8 = 160 episodes
MAPPO_BATCH = 8
MAPPO_LR = 1e-5
MAPPO_CLIP = 0.2
# Separate KL budgets: the attacker is kept close to the base model because the
# base model already writes serviceable attacks, and letting it drift was the
# collapse path in v1. The defender is allowed more room, since it must actually
# learn a new behaviour.
MAPPO_KL_ATTACKER = 0.15
MAPPO_KL_DEFENDER = 0.10
MAPPO_KL_COEF = 0.02        # retained for older artifacts / reports
MAPPO_ENTROPY = 0.01
MAPPO_DISCOUNT = 0.95
MAPPO_TURNS = 1             # attacker->defender rounds per episode

GEN_MAX_NEW = 32               # attacker token budget
DEFENDER_MAX_NEW = 64

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

CRITIC_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
CRITIC_LR = 1e-3
