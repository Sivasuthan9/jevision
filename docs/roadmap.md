# Next steps

## Status in one paragraph

Experiment 2 located where Jev-style vision models read the image. Each visual skill has its own band of layers,
text is read first, and the image is no longer read after about 50–62% of the depth. This replicates in two model
families. On its own it is not enough for a paper: it localises layers rather than a full circuit, a related two-stage
pattern is already reported for LLaVA models, and Jev-style training does not change it consistently. Experiment 1
showed that, at the answer position, the models know no more than their readout says once memorisation is
controlled for. Their counting and format failures are readout biases. The open question is what limits them on
fine detail such as small text.

## Main direction: what can decision training fix, and what can it not?

Jev-style models are trained to give calibrated probabilities ("reinforcement learning for calibrated decisions",
RLCD, in TypeSafe's terms). For a model that reads its answer distribution in one pass, as all current Jev-style
vision models do, such training amounts to supervised training with a proper scoring rule. A proper scoring rule is
one whose expected value is best when the model reports its true belief. Reinforcement learning only adds something
when the model samples a step first, such as a rationale or a choice of where to look.

A model's decision error can be split exactly into three parts:

- **perception gap:** detail the vision encoder discards;
- **reading gap:** detail present in the encoder's tokens that never reaches the answer;
- **readout gap:** miscalibration of the final distribution.

With the vision encoder frozen, as in these models, training the language model with any objective can shrink only
the reading and readout gaps. Only more pixels (a closer look, higher resolution) or a trained encoder can shrink the
perception gap. Experiment 1 suggests the readout gap is small once the format is fixed. The question is whether
small-text failures are a perception gap or a reading gap.

## 1. The decisive test (about one GPU-day, no training)

- **Stimuli:** a font-size ladder (for example 3–10 px) of random codes, so probes cannot memorise. Unlimited
  synthetic data, for example 50,000 training images, so probes are not data-limited (P1's probes used 1,000).
- **Measurements:**
  - a probe on the frozen encoder's tokens at the code's location: what the encoder keeps;
  - a probe on the answer-position state: what reaches the answer;
  - the model's recalibrated readout: what it says.
- **Models:** Qwen3-VL-4B and Qwen3.5-4B, plus their Jev-style versions.
- **Outcomes:**
  - *A band of sizes where the encoder probe succeeds but the model fails ("encoded but unread").* Decision
    training could fix those failures without new pixels. This would be the main result.
  - *The encoder probe fails wherever the model fails.* Fine-detail failures are perception-bound, and decision
    training can only recalibrate them. This is a useful, if expected, result.

## 2. If the test is positive

- Measure the three gaps for every skill, on tasks whose true answer probabilities are known.
- Compare training regimes on the same data: cross-entropy, Brier score, GRPO with a correctness reward, and RLCD
  with rationales. Locate where each one changes the network, using layer-restricted weight patching and the
  experiment 2 machinery.

## 3. Needed regardless

- **Natural images:** TextVQA, TallyQA and DocVQA in multiple-choice form. The datasets are not yet on the
  experiment machine.
- **ImaJev-4B,** the top model on ImageJevBench, once it is available locally.
- **Head-level localisation** inside the read bands, if a circuit-level claim is needed.
