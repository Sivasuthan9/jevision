"""Experiment 1: one Jev-style forward pass per item, saving the readout and the hidden states needed for probes.

Saved per item:
  readout    : logits of the 8 option labels " A".." H" at the answer slot (the decision as served)
  lens       : the same 8 label logits read through the final norm and LM head at every layer (logit lens)
  h_ans      : hidden state at the answer position, every layer
  h_img_mean : image-token hidden states averaged over tokens, every layer
  h_img_max  : image-token hidden states max-pooled over tokens, every layer
  gen        : optional greedy free-text answer to the plain question (readout vs generation comparison)
"""
import time
from pathlib import Path

import torch
from PIL import Image

from .models import language_model
from .prompts import build_prompt, label_token_ids

COUNT_WORDS = ["two", "three", "four", "five", "six", "seven", "eight", "nine"]
Q_COUNT = "How many red circles are in the image? Red squares and blue circles do not count."
Q_TEXT = "What code is printed after the word CODE in the image?"
OPTION_FORMATS = ("digits", "words", "words-reversed")


def question_for(item, meta, option_format="digits"):
    """(question, options) for an item; options None means yes/no."""
    if item["task"] == "noul":
        return f"Does the code printed after the word CODE in the image read {item['asked']}?", None
    if item["task"] == "text":
        return Q_TEXT, meta["codes"]
    if option_format == "words":
        return Q_COUNT, COUNT_WORDS
    if option_format == "words-reversed":
        return Q_COUNT, COUNT_WORDS[::-1]
    return Q_COUNT, [str(c) for c in meta["counts"]]


@torch.inference_mode()
def extract_features(model, processor, meta, image_dir, option_format="digits", n_generate=0, limit=None,
                     log_every=100):
    assert option_format in OPTION_FORMATS, option_format
    device = next(model.parameters()).device
    tok = processor.tokenizer
    lm_head = model.get_output_embeddings()
    norm = language_model(model).norm
    label_ids = torch.tensor(label_token_ids(processor), device=device)
    image_token = model.config.image_token_id
    items = meta["items"][:limit] if limit else meta["items"]

    rec = {"ids": [], "readout": [], "lens": [], "h_ans": [], "h_img_mean": [], "h_img_max": [], "gen": {}}
    t0 = time.time()
    for i, it in enumerate(items):
        img = Image.open(Path(image_dir) / f"{it['id']}.png").convert("RGB")
        question, options = question_for(it, meta, option_format)
        inputs = processor(text=[build_prompt(processor, question, options)], images=[img], return_tensors="pt").to(device)
        out = model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple of (n_layers + 1) tensors [1, T, d]
        is_img = inputs["input_ids"][0] == image_token
        ans = torch.stack([h[0, -1] for h in hs])  # [L+1, d]
        img_states = torch.stack([h[0, is_img] for h in hs])  # [L+1, n_img, d]
        lens = lm_head(norm(ans.to(lm_head.weight.dtype)))[:, label_ids].float()  # [L+1, 8]
        readout = out.logits[0, -1, label_ids].float()
        if option_format == "words-reversed" and it["task"] == "count":  # map letters back to class order
            readout, lens = readout.flip(-1), lens.flip(-1)
        rec["ids"].append(it["id"])
        rec["readout"].append(readout.cpu())
        rec["lens"].append(lens.cpu())
        rec["h_ans"].append(ans.half().cpu())
        rec["h_img_mean"].append(img_states.mean(1).half().cpu())
        rec["h_img_max"].append(img_states.amax(1).half().cpu())
        if i < n_generate and it["task"] != "noul":  # free-form answer to the same question, without options
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question + " Answer briefly."}]}]
            gtext = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ginp = processor(text=[gtext], images=[img], return_tensors="pt").to(device)
            g = model.generate(**ginp, max_new_tokens=16, do_sample=False)
            rec["gen"][it["id"]] = tok.decode(g[0, ginp["input_ids"].shape[1]:], skip_special_tokens=True)
        if i % log_every == 0:
            print(i, f"{time.time() - t0:.0f}s", "n_img_tokens", int(is_img.sum()), flush=True)
    for k in ("readout", "lens", "h_ans", "h_img_mean", "h_img_max"):
        rec[k] = torch.stack(rec[k])
    return rec
