#!/usr/bin/env bash
# Experiment 1: readout vs probes on synthetic counting and small-text tasks (H2O-Lightning-4B and its base).
# Runtime on one GPU: about 11 min per model for the main set, about 5 min per model for each smaller set.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/generate_readout_data.py --set main --out data/readout_main
python scripts/generate_readout_data.py --set text-hard --out data/readout_text_hard
python scripts/generate_readout_data.py --set random-codes --out data/readout_random_codes

for m in h2o-lightning-4b qwen3.5-4b; do
  python scripts/extract_features.py --model $m --data data/readout_main --out outputs/exp1/${m}_main.pt --n-generate 200
  python scripts/extract_features.py --model $m --data data/readout_text_hard --out outputs/exp1/${m}_text-hard.pt
  python scripts/extract_features.py --model $m --data data/readout_random_codes --out outputs/exp1/${m}_random-codes.pt
done
python scripts/extract_features.py --model h2o-lightning-4b --data data/readout_main --task count --options words --out outputs/exp1/h2o-lightning-4b_count-words.pt
python scripts/extract_features.py --model h2o-lightning-4b --data data/readout_main --task count --options words-reversed --out outputs/exp1/h2o-lightning-4b_count-words-reversed.pt

python scripts/analyze_probes.py --data data/readout_main outputs/exp1/*_main.pt
python scripts/analyze_readout_bias.py --meta data/readout_main/items.json outputs/exp1/*_main.pt outputs/exp1/*_count-words*.pt
python scripts/analyze_readout_bias.py --meta data/readout_text_hard/items.json outputs/exp1/*_text-hard.pt
python scripts/analyze_random_codes.py --data data/readout_random_codes outputs/exp1/*_random-codes.pt
