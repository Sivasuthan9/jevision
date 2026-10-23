#!/usr/bin/env bash
# Experiment 2: counterfactual cache patching (four models) and the evidence-size control (two models).
# Runtime on one GPU: about 2.5 h per model per pair set (800 pairs, 32-36 patched rows each).
# Set OUT=results/exp2_cache_patching to overwrite the shipped results; by default new runs go to outputs/.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-outputs/exp2}

python scripts/generate_patching_pairs.py --set main --out data/pairs_main
python scripts/generate_patching_pairs.py --set size-control --out data/pairs_size_control

for m in h2o-lightning-4b qwen3.5-4b visual-jev-4b qwen3-vl-4b; do
  python scripts/run_patching.py --model $m --pairs data/pairs_main --out $OUT/${m}_main.json
done
for m in h2o-lightning-4b qwen3-vl-4b; do
  python scripts/run_patching.py --model $m --pairs data/pairs_size_control --out $OUT/${m}_size-control.json
done

python scripts/analyze_patching.py --results-dir $OUT
python scripts/analyze_size_control.py --results-dir $OUT
python scripts/make_figures.py --results-dir $OUT --out-dir $OUT/figures
