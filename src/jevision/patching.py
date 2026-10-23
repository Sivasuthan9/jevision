"""Experiment 2: counterfactual cache patching in a shared-cache decision VLM.

For a pair (A, B) that differs in one answer-relevant detail, the prompt is built image first, so the image tokens
precede the question and their hidden states do not depend on it. Two interventions, at every second layer l:

  late-cache patch at l : run image A, but from layer l onward the image positions carry B's hidden states. The
                          effect stays high while the question still reads the image at layers >= l.
  branch patch at l     : run image A, but at the input of layer l give all post-image positions (question and
                          answer) B's hidden states. The effect rises once the question already holds the evidence.

Effect = (LD_patched - LD_A) / (LD_B - LD_A), with LD = z[y_A] - z[y_B] the logit difference between A's and B's
answers at the answer slot: 0 means A's answer, 1 means B's. Only pairs answered correctly on both clean images
are patched. All patched variants of a pair run in batches of `chunk` rows, with logits only at the last position.
"""
import time
from pathlib import Path

import torch
from PIL import Image

from .models import language_model
from .prompts import build_prompt, label_token_ids


class CachePatcher:
    def __init__(self, model, processor):
        self.model, self.processor = model, processor
        self.device = next(model.parameters()).device
        self.lm_head = model.get_output_embeddings()
        self.layers = language_model(model).layers
        self.n_layers = len(self.layers)
        self.grid = list(range(0, self.n_layers + 1, 2))
        self.rows = [("late", l) for l in self.grid if l < self.n_layers] + \
                    [("branch", l) for l in self.grid if l < self.n_layers]
        self.label_ids = torch.tensor(label_token_ids(processor), device=self.device)
        self._patch = None  # {layer: [(row, positions, values), ...]}
        self._handles = [layer.register_forward_pre_hook(self._hook(i), with_kwargs=True)
                         for i, layer in enumerate(self.layers)]

    def _hook(self, i):
        def hook(module, args, kwargs):
            if not self._patch or i not in self._patch:
                return None
            in_kwargs = "hidden_states" in kwargs
            hs = (kwargs["hidden_states"] if in_kwargs else args[0]).clone()
            for row, pos, val in self._patch[i]:
                hs[row, pos] = val
            if in_kwargs:
                kwargs["hidden_states"] = hs
                return args, kwargs
            return (hs,) + tuple(args[1:]), kwargs
        return hook

    def remove_hooks(self):
        for h in self._handles:
            h.remove()

    @torch.inference_mode()
    def _last_logits(self, inputs, k, hidden=False):
        out = self.model.model(**inputs, use_cache=False, output_hidden_states=hidden)
        z = self.lm_head(out.last_hidden_state[:, -1]).float()[:, self.label_ids[:k]]
        return z, (out.hidden_states if hidden else None)

    def run_pair(self, item, image_dir, chunk=32):
        """Effects for one pair, or a record with okA/okB False if a clean image is answered wrongly."""
        proc = self.processor
        k = 2 if item["options"] is None else len(item["options"])
        text = build_prompt(proc, item["question"], item["options"])
        img_a = Image.open(Path(image_dir) / f"{item['id']}_A.png").convert("RGB")
        img_b = Image.open(Path(image_dir) / f"{item['id']}_B.png").convert("RGB")
        in_a = proc(text=[text], images=[img_a], return_tensors="pt").to(self.device)
        in_b = proc(text=[text], images=[img_b], return_tensors="pt").to(self.device)
        if in_a["input_ids"].shape != in_b["input_ids"].shape:
            return None
        self._patch = None
        z_a, h_a = self._last_logits(in_a, k, hidden=True)
        z_b, h_b = self._last_logits(in_b, k, hidden=True)
        z_a, z_b = z_a[0], z_b[0]
        ya, yb = item["yA"], item["yB"]
        rec = {"id": item["id"], "skill": item["skill"],
               "okA": int(z_a.argmax()) == ya, "okB": int(z_b.argmax()) == yb}
        if not (rec["okA"] and rec["okB"]):
            return rec
        ids = in_a["input_ids"][0]
        img_pos = (ids == self.model.config.image_token_id).nonzero().flatten()
        post_pos = torch.arange(int(img_pos.max()) + 1, ids.shape[0], device=self.device)
        ld_a, ld_b = float(z_a[ya] - z_a[yb]), float(z_b[ya] - z_b[yb])
        rec["late"], rec["branch"] = {}, {}
        for c0 in range(0, len(self.rows), chunk):
            rows = self.rows[c0:c0 + chunk]
            batch = proc(text=[text] * len(rows), images=[img_a] * len(rows), return_tensors="pt").to(self.device)
            patch = {}
            for row, (kind, l) in enumerate(rows):
                pos = img_pos if kind == "late" else post_pos
                patch.setdefault(l, []).append((row, pos, h_b[l][0, pos]))
            self._patch = patch
            z, _ = self._last_logits(batch, k)
            self._patch = None
            for row, (kind, l) in enumerate(rows):
                rec[kind][l] = (float(z[row, ya] - z[row, yb]) - ld_a) / (ld_b - ld_a)
        return rec

    def run(self, items, image_dir, chunk=32, log_every=25):
        results, kept, t0 = [], 0, time.time()
        for n, item in enumerate(items):
            rec = self.run_pair(item, image_dir, chunk)
            if rec is None:
                continue
            kept += int("late" in rec)
            results.append(rec)
            if n % log_every == 0:
                print(n, f"{time.time() - t0:.0f}s kept {kept}", flush=True)
        return {"layers": self.n_layers, "grid": self.grid, "items": results}
