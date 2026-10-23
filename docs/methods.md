# Methods

This page describes exactly what the code in this repository measures. Results are in [results.md](results.md).

## The decision format

All experiments use the prompt that H2O-Lightning-4B serves (`src/jevision/prompts.py`):

```
system:    You are a decision engine. You read a record and answer one question about it by choosing exactly
           one option. Reply with a single letter.
user:      <image> record: [image 1]
           question: How many red circles are in the image? Red squares and blue circles do not count.
           options:
           A) 2: 2
           B) 3: 3
           ...
assistant: Answer:
```

Yes/no ("Noul") questions use two options, `A) true: the statement holds` and `B) false: it does not`. The model
runs one forward pass. Its decision is the set of next-token logits of the option letters " A", " B", ... at the last
position (the **readout**). No text is generated. The image comes before the question, so under the causal attention
mask the image tokens' hidden states do not depend on the question. This is the "shared prefix" design that
Jev-style models use to answer many questions about one image cheaply.

## Models

| Name | What it is | Layers |
|---|---|---|
| `h2o-lightning-4b` | Jev-style model: Qwen3.5-4B with a LoRA trained on text decisions; vision encoder untouched | 32 (hybrid attention + Gated DeltaNet) |
| `qwen3.5-4b` | its base model | 32 |
| `visual-jev-4b` | Jev-style model: Qwen3-VL-4B-Instruct with a LoRA (rank 16, alpha 32) trained on GQA and SNLI-VE decisions (arXiv 2609.25845); the adapter is merged into the weights at load time | 36 (pure attention) |
| `qwen3-vl-4b` | its base model | 36 |

Revisions are pinned in `configs/models.yaml`. Qwen3-VL adds vision-encoder features to its first decoder layers
("DeepStack"), so patches at layers 0–2 of the Qwen3-VL models are only partial; no reported crossing lies there.

## Experiment 1: readout versus probes

**Question.** When a Jev-style model gets a visual question wrong, is the answer present in its hidden states but
lost at the option-letter readout?

**Stimuli** (`src/jevision/stimuli/readout_tasks.py`, all synthetic, answers known):
- *counting*: 2–9 red circles among red squares and blue circles, 640 × 640 px, 8-way choice (800 items);
- *text*: one of 8 confusable 4-character codes (for example RK7T, RK1T, RX7T) printed at 8–14 px among clutter
  labels, 768 × 768 px, 8-way choice (800 items); a hard version uses 5–8 px fonts (800 items);
- *random codes*: each image has its own random 4-character code at 5–8 px; yes/no question "does the code read X?",
  with X the true code (50%) or a one-character variant (1,000 items). Because no code repeats, a probe cannot
  succeed by memorising the appearance of a few codes.

Items are split 50/50 into train and test halves.

**Measurements** (`scripts/extract_features.py`, `scripts/analyze_*.py`):
- *readout accuracy*: argmax of the option-letter logits;
- *recalibrated readout*: a logistic regression on the 8 letter logits (on the random-code task, a threshold on the
  true-minus-false margin). If this matches a hidden-state probe, the readout already carries the information and
  its failure is a bias, not a loss;
- *probes*: logistic regression on the answer-position hidden state and on pooled image-token states, every 4th
  layer (PCA to 256 dimensions fitted on the train half; regularisation chosen by 3-fold cross-validation);
- *controls*: a probe on pixel colour histograms and a probe trained on shuffled labels;
- *logit lens*: the letter logits read through the final norm and output embedding at every layer;
- *readout vs free generation*: the same question without options, answered in free text, on the first 200 items.

## Experiment 2: counterfactual cache patching

**Question.** At which layers does the question read the image, does this depend on the visual skill, and does
Jev-style training change it?

**Stimuli** (`src/jevision/stimuli/counterfactual.py`): pairs of 768 × 768 images (A, B) that differ in exactly one
answer-relevant detail, with identical clutter:

| Skill | A vs B | Question format |
|---|---|---|
| presence | a small yellow triangle (26 px) present vs absent | yes/no |
| text | a 4-character code at 14 px; B changes one character | 4-way choice |
| count | n vs n + 1 red circles (n = 2–5), distractors identical | 5-way choice (2–6) |
| spatial | red circle left vs right of a blue square | yes/no |

The main set has 200 pairs per skill. The **size control** repeats each skill at a small and a large evidence size
(text 12 vs 28 px; triangle 20 vs 56 px; circles 16–20 vs 40–48 px in diameter; shapes 20 vs 56 px), with 100 pairs
per skill and size.

**Interventions** (`src/jevision/patching.py`), at every second layer ℓ:
- *late-cache patch at ℓ*: run image A, but from layer ℓ onward the image positions carry image B's hidden states.
  Because the image precedes the question, this is exactly "B's image cache from layer ℓ on". The effect stays high
  while the question still reads the image at layers ≥ ℓ.
- *branch patch at ℓ*: run image A, but at the input of layer ℓ give all post-image positions (question and answer)
  B's hidden states. The effect rises once the question positions already hold the evidence.

**Effect.** With z the letter logits at the answer slot and LD = z[y_A] − z[y_B],

    effect = (LD_patched − LD_A) / (LD_B − LD_A),

so 0 means the model gives A's answer and 1 means B's answer. Only pairs whose two clean images are both answered
correctly are patched.

**Summary statistics** (`src/jevision/analysis/crossings.py`):
- *read layer*: the layer where the mean late-cache curve falls through 0.5, interpolated linearly between grid
  points. Patching from this layer onward moves the answer less than half way to B's;
- *absorb layer*: where the mean branch curve rises through 0.5;
- 95% confidence intervals from 2,000 bootstrap resamples of pairs (1,000 for the size control);
- *paired comparison*: Jev-style model minus base model on the pairs both kept, resampling pairs jointly;
- *no-read depth*: the first grid layer from which the mean late-cache effect of every skill stays at or below 0.02.

**Implementation details.** All patched variants of one pair run as batches of 32 rows, and logits are computed
only at the last position. Batched and unbatched runs agreed within 0.06 in effect units (bf16).
