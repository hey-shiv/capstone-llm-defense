cd /Users/shivashant/Desktop/capstone-llm-defense
python3 src/datagen.py > runs/datagen.log 2>&1
python3 src/sft.py > runs/sft.log 2>&1
mv runs/sft_defender runs/sft_small
for s in 0 1; do
 python3 src/mappo.py --iters 100 --seed $s --out runs/mappo_cold_s$s > runs/mappo_cold_s$s.log 2>&1
 python3 src/mappo.py --iters 100 --seed $s --out runs/mappo_warm_s$s --init-defender runs/sft_small/defender > runs/mappo_warm_s$s.log 2>&1
done
echo ALLDONE > runs/ALLDONE
