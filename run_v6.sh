cd /Users/shivashant/Desktop/capstone-llm-defense
C="--all-families --hard-pool --benign-frac 0.4 --iters 90"
for s in 0 1 2 3 4; do
 python3 src/mappo.py $C --seed $s --init-defender runs/sft_small/defender --out runs/v6small_s$s > runs/v6small_s$s.log 2>&1
done
for s in 0 1 2; do
 python3 src/mappo.py $C --seed $s --init-defender runs/sft_big/defender --out runs/v6big_s$s > runs/v6big_s$s.log 2>&1
done
cd src && python3 select_v6.py v6 > ../runs/select_v6.log 2>&1
echo fin > ../runs/V6DONE
