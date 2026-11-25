#!/usr/bin/env bash
# Round 3 (panel review): GNN baselines (GCN, GIN, GAT) and tuned XGBoost / GIN on every experiment.
# Two waves so that at most ~5 Python processes run at once (RAM is tight).
# Watch progress with:  Get-Content logs\<name>.log -Wait
cd "$(dirname "$0")"
mkdir -p logs
echo "round 3 started $(date)"

# wave 1: quality / speed / size on Level 1, 2, 3 and the graph-construction study
python run_benchmark.py level1 --models round3 > logs/r3_bench_level1.log 2>&1 &
python run_benchmark.py level2 --models round3 > logs/r3_bench_level2.log 2>&1 &
python run_level3.py --models round3        > logs/r3_level3.log 2>&1 &
python run_graph_study.py 2                  > logs/r3_graph_study.log 2>&1 &
wait
echo "wave 1 done $(date)"

# wave 2: robustness and learning curves
python run_robustness.py evaluate round3                > logs/r3_robustness.log 2>&1 &
python run_learning_curves.py level1 --models round3    > logs/r3_curves_level1.log 2>&1 &
python run_learning_curves.py level2 --models round3    > logs/r3_curves_level2.log 2>&1 &
wait
echo "wave 2 done $(date)"

for f in logs/r3_*.log; do
  if grep -q "Traceback" "$f"; then echo "FAILED: $f"; tail -3 "$f"; else echo "ok: $f"; fi
done
