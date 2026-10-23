# jevision

**Where and how Jev-style vision decision models read the image.**

This repository contains the code, synthetic stimuli generators and results of a study of open-source "Jev-style"
vision-language models: models that answer multiple-choice questions about an image in a single forward pass and
return a probability for every option. It covers two completed experiments:

1. **Readout versus probes:** when such a model gets a visual question wrong, is the answer present in its hidden
   states but lost at the option-letter readout?
2. **Counterfactual cache patching:** at which layers does the model read the image, does that depend on the visual
   skill, and does Jev-style training change it?

The work is at the pilot stage. All results so far use synthetic images; see [Limitations](#limitations).

---

## Background

**Jev** is a commercial "decision model" from TypeSafe AI. Given a state (input) and typed questions with a fixed
set of answers, it returns a calibrated probability for each answer, quickly and without generating text. TypeSafe
calls its training method "reinforcement learning for calibrated decisions" (RLCD). Jev itself accepts text only, and
its architecture and training algorithm are not public.

**Jev-style vision models** are open models that copy this interface for images. In practice they are a
vision-language model (VLM) prompted with lettered options, whose answer is read from the next-token logits of the
option letters. This repository studies two of them, each against its base model:

| Model | Built from | How |
|---|---|---|
| H2O-Lightning-4B | Qwen3.5-4B (32 layers, hybrid attention + Gated DeltaNet) | LoRA trained on text-only decision data; vision encoder untouched |
| Visual-Jev-4B (arXiv 2609.25845) | Qwen3-VL-4B-Instruct (36 layers, pure attention) | LoRA trained on GQA and SNLI-VE decisions; vision encoder frozen |

Fast Jev-style vision models are reported to be weak on fine detail such as small text and counting. This project
asks where in the network the image is used, and whether failures come from the evidence or from the readout.

---

## Key results so far

**Experiment 2: each visual skill reads the image in its own band of layers, and reading stops early.**

![Late-cache curves](figures/late_cache_curves.png)

*Effect of replacing image A's internal states with image B's from layer ℓ onward (0 = the model gives A's answer,
1 = B's). Dots mark the read layer, where the effect falls through 0.5.*

| Read layer [95% CI] | H2O-Lightning-4B | Qwen3.5-4B base | Visual-Jev-4B | Qwen3-VL-4B base |
|---|---|---|---|---|
| text | 7.0 [7.0, 7.1] | 6.9 [6.9, 7.0] | 9.7 [9.5, 10.0] | 9.5 [9.3, 9.7] |
| spatial | 11.4 [11.4, 11.5] | 11.5 [11.5, 11.6] | 17.4 [17.1, 17.7] | 17.8 [17.5, 18.0] |
| presence | 14.5 [14.4, 14.5] | 11.6 [11.6, 11.6] | 18.7 [18.7, 18.7] | 18.7 [18.7, 18.7] |
| count | 14.9 [14.8, 15.0] | 14.0 [12.0, 14.3] | 18.6 [18.4, 18.8] | 18.2 [17.9, 18.3] |
| image no longer read from | layer 20 of 32 | layer 16 of 32 | layer 22 of 36 | layer 22 of 36 |

- **Skill bands.** Text is read first, at about a quarter of the depth, in both model families. Spatial relations,
  presence and counting follow, between about a third and just over half of the depth.
- **Reading stops early.** After 50–62% of the depth, swapping the image states changes the answer by at most 2%.
- **Not object size.** A size control shows the bands follow the skill: changing the evidence size moves the read
  layer by at most 1.1 layers, against 7.8–9.1 layers between skills.
- **Training effect does not replicate.** Jev-style training shifts reading later only in H2O (presence +2.9 layers,
  counting +1.7). Visual-Jev shows no shift (at most 0.4 layers).

**Experiment 1: the models' counting and format failures are readout biases, not lost evidence.**

- **Counting.** A linear recalibration of the 8 option-letter logits (0.652 for H2O) matches or beats a probe on the
  hidden state (0.608). The readout systematically undercounts.
- **Memorisation-proof control.** In a control with a different random code in every image, the answer-position state
  knows no more than the readout says (H2O: probe 0.742, readout 0.750).
- **Answer format.** The same 5–6 px text is at chance in 8-way choice (0.13; the model picks option A 71% of the time)
  but well above chance as a yes/no question (0.76 at 6 px).
- **Jev-style training reduces biases.** H2O's yes-rate falls from 68% (base) to 44%. On counting, the base model's
  letter readout loses to its own free-text answers (0.277 vs 0.494), and H2O closes that gap.

Full tables, caveats and interpretation: [docs/results.md](docs/results.md). Methods: [docs/methods.md](docs/methods.md).
Next steps: [docs/roadmap.md](docs/roadmap.md).

---

## Repository structure

```
jevision/
├── README.md
├── pyproject.toml              package metadata and dependencies
├── requirements.txt            exact versions used for the reported results
├── configs/
│   └── models.yaml             model short names -> Hugging Face repos and pinned revisions
├── src/jevision/
│   ├── config.py               model registry
│   ├── models.py               model loading, hand-written LoRA merge
│   ├── prompts.py              the Jev-style decision prompt and option-letter readout
│   ├── extraction.py           experiment 1: readout, logit lens and hidden states per item
│   ├── patching.py             experiment 2: counterfactual cache patching
│   ├── stimuli/
│   │   ├── common.py           drawing helpers
│   │   ├── readout_tasks.py    experiment 1 stimuli (counting, small text, random codes)
│   │   └── counterfactual.py   experiment 2 image pairs (main set and size control)
│   └── analysis/
│       ├── crossings.py        read/absorb layers, bootstrap intervals, paired comparisons
│       └── probes.py           linear probes, readout recalibration
├── scripts/                    command-line entry points (one per step) and full reproduction scripts
├── results/
│   ├── exp1_readout_probes/    probe summaries and the analysis output
│   └── exp2_cache_patching/    per-pair effect curves for every run and the analysis output
├── figures/                    figures made by scripts/make_figures.py
├── docs/                       methods, results, roadmap
└── tests/                      unit tests (no GPU or model needed)
```

Generated stimuli (`data/`) and extracted activations (`outputs/`, about 4 GB for experiment 1) are not tracked;
the scripts recreate them.

---

## Setup

**Requirements**
- Linux and Python 3.10 or newer (results were produced with Python 3.12.3).
- One NVIDIA GPU with CUDA. Measured peak GPU memory is 14.3 GiB for cache patching (default batch of 32 patched
  rows; use `--chunk` to lower it) and 9.5 GiB for feature extraction.
- About 26 GB of disk for the three model checkpoints (8–9 GB each; the Visual-Jev adapter is 127 MB).
- The DejaVu Sans font (`sudo apt install fonts-dejavu-core`). If it is missing, matplotlib's copy is used, which can
  change glyph rendering slightly.

**Install**

```bash
git clone https://github.com/Sivasuthan9/jevision.git
cd jevision
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130   # pick the build for your CUDA version
pip install -r requirements.txt
pip install -e .            # installs the jevision package from src/
pip install pytest && pytest   # optional: unit tests
```

**Models.** Scripts accept the short names in `configs/models.yaml` (`h2o-lightning-4b`, `qwen3.5-4b`,
`qwen3-vl-4b`, `visual-jev-4b`), any Hugging Face id, or a local directory. Pinned revisions download on first use.
To download in advance, for example:

```bash
hf download h2oai/h2o-lightning-4b --revision b43f759b6385d21182f87c3829931ec3d4dfa98a
```

Set `HF_HUB_OFFLINE=1` to run from the local cache only.

---

## Usage

### Quick smoke test (about 5 minutes)

```bash
python scripts/generate_patching_pairs.py --set main --out data/pairs_smoke --pairs-per-skill 2
python scripts/run_patching.py --model h2o-lightning-4b --pairs data/pairs_smoke --out outputs/smoke.json
```

### Re-analyse the shipped results (no GPU)

```bash
python scripts/analyze_patching.py --results-dir results/exp2_cache_patching
python scripts/analyze_size_control.py --results-dir results/exp2_cache_patching
python scripts/make_figures.py --results-dir results/exp2_cache_patching --out-dir figures
```

### Experiment 1: readout versus probes

```bash
python scripts/generate_readout_data.py --set main --out data/readout_main
python scripts/extract_features.py --model h2o-lightning-4b --data data/readout_main \
    --out outputs/exp1/h2o-lightning-4b_main.pt --n-generate 200
python scripts/analyze_probes.py --data data/readout_main outputs/exp1/h2o-lightning-4b_main.pt
```

Other stimulus sets are `text-hard` (5–8 px fonts) and `random-codes` (the memorisation-proof control, analysed with
`analyze_random_codes.py`). `analyze_readout_bias.py` tests whether a failure is a bias of the letter logits.
Everything is chained in [`scripts/reproduce_experiment1.sh`](scripts/reproduce_experiment1.sh). Runtime: about
11 minutes per model for the main set and 5 minutes per smaller set.

### Experiment 2: counterfactual cache patching

```bash
python scripts/generate_patching_pairs.py --set main --out data/pairs_main
python scripts/run_patching.py --model h2o-lightning-4b --pairs data/pairs_main \
    --out outputs/exp2/h2o-lightning-4b_main.json
python scripts/analyze_patching.py --results-dir outputs/exp2
```

The size control uses `--set size-control` and `analyze_size_control.py`. Everything (four models, both pair sets,
figures) is chained in [`scripts/reproduce_experiment2.sh`](scripts/reproduce_experiment2.sh). Runtime: about
2.5 hours per model per pair set.

All scripts print their options with `--help`.

---

## Verification

The code was refactored from the original pilot scripts. Before release, we checked that it reproduces them:

| Check | Result |
|---|---|
| Experiment 1 stimuli regenerated (1,600 main, 800 text-hard, 1,000 random-code items) | pixel-identical images, identical metadata |
| Experiment 2 pairs regenerated (main set and size control) | presence, count and spatial pairs pixel-identical; for text pairs see below |
| `analyze_patching.py` and `analyze_size_control.py` on the shipped results | reproduce every read layer, absorb layer, pair count and interval of the pilot analysis |
| Experiment 1 analysis scripts on the pilot's extracted features | probe summaries identical (112 of 112 values per model); readout-bias, random-code and generation numbers identical |
| `run_patching.py` on the pilot's pairs (H2O-Lightning-4B, 8 pairs; Visual-Jev-4B, 4 pairs) | effects identical to the shipped results (largest difference 0.0000) |
| `extract_features.py` on the pilot's items (H2O-Lightning-4B, 8 items) | readout, logit lens and hidden states identical to the pilot's saved features; same free-text answers |

**Text pairs.** The pilot drew the distractor codes of text pairs into an unordered Python set, whose order depends
on hash randomisation. The refactored generator sorts that set. As a result, in 129 of 200 main-set text pairs (133
of 200 in the size control), image B shows a different one-character variant than in the pilot. The random stream
is otherwise unchanged, so all other pairs are identical. The shipped results come from the pilot's stimuli; a re-run
on regenerated text pairs gives statistically equivalent but not bit-identical numbers.

---

## Limitations

- **Synthetic images only.** The confidence intervals cover the sampling of pairs, not other image types. Natural
  benchmarks (TextVQA, TallyQA, DocVQA) are the next step.
- **Two Jev-style models and their bases, 4B parameters each.** ImaJev-4B (the top model on the ImageJevBench
  leaderboard) has not been run yet.
- **Layer-level localisation, not a circuit.** Individual attention heads are not identified.
- **One prompt format and one readout** (letter logits). The grid covers every second layer.
- **Hybrid architecture.** In Qwen3.5 the recurrent layers carry state across positions, so a read layer there mixes
  attention reads and recurrent transfer.
- **Counting pairs are filtered by correctness.** Only 30–63% survive, so the counting curves describe easier counts.

---

## References

- TypeSafe AI, *Jev documentation* (typesafe.ai): the decision-model interface and the term RLCD.
- H2O.ai, *H2O-Lightning-4B* model card (huggingface.co/h2oai/h2o-lightning-4b).
- Yu and Yao, *Visual Jev*, arXiv 2609.25845: the Visual-Jev-4B model.
- Zhang, Yadav, Han and Shutova, *Cross-modal Information Flow in Multimodal Large Language Models*, arXiv
  2411.18620: a related two-stage integration pattern in LLaVA models.
- Gao and Wang, *OpenJev-RLCD: A Working RLCD Implementation*, arXiv 2609.38850: an open RLCD implementation for
  text reasoning models; its §3.1 notes that without a sampled rationale, the objective reduces to proper-score
  training of a direct scorer.

## License

No license has been chosen yet. Until one is added, all rights are reserved.
