"""The Jev-style decision prompt (H2O-Lightning-4B's served format) and its option-letter readout.

The image comes first, then the question with lettered options, then an assistant prefill "Answer:". The
decision is read from the next-token logits of the option letters " A", " B", ... at the last position.
"""

SYSTEM_PROMPT = (
    "You are a decision engine. You read a record and answer one question about it by choosing exactly one "
    "option. Reply with a single letter."
)
LABELS = list("ABCDEFGH")
YES_NO_OPTIONS = "A) true: the statement holds\nB) false: it does not"
IMAGE_PLACEHOLDER = "<|image_pad|>"


def user_text(question: str, options: list[str] | None = None) -> str:
    """Options None means a yes/no ("Noul") question with A = true, B = false."""
    if options is None:
        body = YES_NO_OPTIONS
    else:
        body = "\n".join(f"{label}) {opt}: {opt}" for label, opt in zip(LABELS, options))
    return f"record: [image 1]\nquestion: {question}\noptions:\n{body}"


def build_prompt(processor, question: str, options: list[str] | None = None, prefill: str = "Answer:") -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text(question, options)}]},
        {"role": "assistant", "content": prefill},
    ]
    return processor.apply_chat_template(messages, tokenize=False, continue_final_message=True, enable_thinking=False)


def label_token_ids(processor) -> list[int]:
    """Token ids of " A".." H" as the next token after the prefill (each label must be exactly one token)."""
    tok = processor.tokenizer
    probe = build_prompt(processor, "x").replace(IMAGE_PLACEHOLDER, "")
    base = tok(probe, add_special_tokens=False)["input_ids"]
    ids = []
    for label in LABELS:
        full = tok(probe + " " + label, add_special_tokens=False)["input_ids"]
        assert full[: len(base)] == base and len(full) == len(base) + 1, f"label {label!r} is not a single token"
        ids.append(full[-1])
    return ids
