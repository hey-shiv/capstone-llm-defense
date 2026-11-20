cd /Users/shivashant/Desktop/capstone-llm-defense
while [ ! -f runs/V6DONE ]; do sleep 30; done
C="--all-families --hard-pool --benign-frac 0.4 --iters 90 --ref-sft"
for s in 0 1 2; do
 python3 src/mappo.py $C --seed $s --init-defender runs/sft_small/defender --out runs/v7ref_s$s > runs/v7ref_s$s.log 2>&1
done
cd src && python3 select_v6.py v6 v7 > ../runs/select_v7.log 2>&1
python3 aggregate_v6.py > ../runs/aggregate_v7.log 2>&1
echo fin > ../runs/V7DONE
