# Results files

## exp1_readout_probes/

| File | Content |
|---|---|
| `h2o-lightning-4b_main_summary.json`, `qwen3.5-4b_main_summary.json` | `analyze_probes.py` summaries for the main stimulus set, per task (`count`, `text`): readout accuracy, probe accuracy by layer at the answer position (`probe_ans`) and on pooled image tokens (`probe_img`), pixel-histogram and shuffled-label controls, logit-lens accuracy by layer, median decision depth |
| `analysis_output.txt` | printed output of the three experiment 1 analysis scripts on the extracted features |

The extracted features (`.pt`, about 4 GB) are not stored; `scripts/reproduce_experiment1.sh` recreates them.

## exp2_cache_patching/

One JSON file per model and pair set, written by `run_patching.py`:

| File | Model | Pair set |
|---|---|---|
| `h2o-lightning-4b_main.json` | H2O-Lightning-4B | main (800 pairs) |
| `qwen3.5-4b_main.json` | Qwen3.5-4B base | main |
| `visual-jev-4b_main.json` | Visual-Jev-4B | main |
| `qwen3-vl-4b_main.json` | Qwen3-VL-4B base | main |
| `h2o-lightning-4b_size-control.json` | H2O-Lightning-4B | size control (800 pairs) |
| `qwen3-vl-4b_size-control.json` | Qwen3-VL-4B base | size control |
| `analysis_output.txt` | printed output of `analyze_patching.py` and `analyze_size_control.py` | |

Format:

```jsonc
{
  "layers": 32,                       // decoder layers of the model
  "grid": [0, 2, 4, ...],             // layers at which patches were applied
  "items": [
    {"id": "text_000", "skill": "text",
     "okA": true, "okB": true,        // both clean images answered correctly (only then are patches run)
     "late":   {"0": 1.0, "2": 0.99, ...},   // effect of the late-cache patch from layer l onward
     "branch": {"0": 0.0, "2": 0.01, ...}}   // effect of the branch patch at layer l
  ]
}
```

Effects are normalised logit differences: 0 means the model gives image A's answer, 1 means image B's answer
(see `docs/methods.md`). These runs used the pilot's stimuli. The Qwen3.5-4B base run and both Qwen3-VL-4B runs were executed on a second
machine with the same code; the model revisions on that machine were not recorded.
