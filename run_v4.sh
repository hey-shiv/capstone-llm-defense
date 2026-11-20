cd /Users/shivashant/Desktop/capstone-llm-defense
python3 src/datagen.py --aug > runs/datagen_aug.log 2>&1
python3 -c "
import sys; sys.path.insert(0,'src')
from pathlib import Path
from sft import run_sft
run_sft(Path('data/sft_aug.jsonl'), Path('runs/sft_aug'))" > runs/sft_aug.log 2>&1
M="python3 src/mappo.py --iters 60"
W="--init-defender runs/sft_small/defender"
for s in 0 1; do
 $M --seed $s --all-families --hard-benign $W --out runs/mappoall_warm_s$s > runs/mappoall_warm_s$s.log 2>&1
 $M --seed $s --all-families --hard-benign --out runs/mappoall_cold_s$s > runs/mappoall_cold_s$s.log 2>&1
 $M --seed $s $W --out runs/mappo_warm_s$s > runs/mappo_warm_s$s.log 2>&1
 $M --seed $s --out runs/mappo_cold_s$s > runs/mappo_cold_s$s.log 2>&1
done
echo done > runs/ALLDONE4
