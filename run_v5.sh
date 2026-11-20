cd /Users/shivashant/Desktop/capstone-llm-defense
python3 src/datagen.py --big > runs/datagen_big.log 2>&1
python3 -c "
import sys; sys.path.insert(0,'src')
from pathlib import Path
from sft import run_sft
run_sft(Path('data/sft_big.jsonl'), Path('runs/sft_big'))" > runs/sft_big.log 2>&1
for s in 0 1 2; do
 python3 src/mappo.py --iters 75 --seed $s --all-families --hard-pool --benign-frac 0.4 --init-defender runs/sft_small/defender --out runs/v5warm_s$s > runs/v5warm_s$s.log 2>&1
done
python3 src/mappo.py --iters 75 --seed 0 --all-families --hard-pool --benign-frac 0.4 --out runs/v5cold_s0 > runs/v5cold_s0.log 2>&1
cd src && python3 select_v5.py v5 > ../runs/select_v5.log 2>&1
echo fin > ../runs/V5DONE
