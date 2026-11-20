"""Shared backbone + LoRA actors.

One Qwen2.5-0.5B-Instruct instance hosts both policy adapters. Two adapters
(attacker, defender) are enough for MAPPO's decentralised actors, and sharing
the base weights keeps memory inside a 16 GB laptop.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGETS

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEFAULT_DTYPE = torch.bfloat16 if DEVICE == "mps" else torch.float32


@dataclass
class GenOut:
    text: str
    token_logprobs: torch.Tensor   # (T,) logprob of each generated token
    prompt_text: str

    @property
    def seq_logprob(self) -> float:
        return float(self.token_logprobs.sum()) if len(self.token_logprobs) else 0.0


class Backbone:
    """Single base model + named LoRA adapters."""

    def __init__(self, model_id: str, dtype=None):
        dtype = dtype or DEFAULT_DTYPE
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.model.to(DEVICE)
        self.model.eval()
        self._wrapped = False
        self.adapters: list[str] = []

    # ------------------------------------------------------------- adapters
    def add_adapter(self, name: str) -> None:
        cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
        )
        if not self._wrapped:
            self.model = get_peft_model(self.model, cfg, adapter_name=name)
            self._wrapped = True
        else:
            self.model.add_adapter(name, cfg)
        self.adapters.append(name)

    def load_adapter(self, name: str, path: str, trainable: bool = False) -> None:
        """Attach a *saved* adapter directory under `name`.

        `trainable=True` is needed when the loaded adapter is going to keep
        training (e.g. SFT-then-PPO warm start); PEFT otherwise marks it frozen.
        """
        if name in self.adapters:
            # Already attached: this is a no-op, which is what callers that load
            # for inference want. Warn instead of silently ignoring a *different*
            # path, because that is how a warm start would quietly not happen.
            print(f"[policy] adapter {name!r} already attached; ignoring {path}")
            return
        if not self._wrapped:
            self.model = PeftModel.from_pretrained(self.model, str(path),
                                                   adapter_name=name,
                                                   is_trainable=trainable)
            self._wrapped = True
        else:
            self.model.load_adapter(str(path), adapter_name=name,
                                    is_trainable=trainable)
        self.adapters.append(name)

    def set_adapter(self, name: str) -> None:
        self.model.set_adapter(name)
        for n, p in self.model.named_parameters():
            if "lora_" in n:
                p.requires_grad = name in n

    @contextlib.contextmanager
    def using(self, name: str):
        prev = getattr(self.model, "active_adapter", None)
        self.set_adapter(name)
        try:
            yield self
        finally:
            if prev is not None and prev in self.adapters:
                self.set_adapter(prev if isinstance(prev, str) else prev[0])

    @contextlib.contextmanager
    def base_policy(self):
        """Reference policy = adapters disabled (the frozen instruct model)."""
        if not self._wrapped:
            yield self
            return
        with self.model.disable_adapter():
            yield self

    # ------------------------------------------------------------ formatting
    def chat_prompt(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def generate_batch(
        self,
        prompt_texts: list[str],
        max_new_tokens: int = 96,
        temperature: float = 0.9,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> list[GenOut]:
        """Left-padded batch generation, returns one GenOut per prompt."""
        self.model.eval()
        prev_side = self.tok.padding_side
        self.tok.padding_side = "left"
        enc = self.tok(
            prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(DEVICE)
        self.tok.padding_side = prev_side

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **({"temperature": temperature, "top_p": top_p} if do_sample else {}),
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tok.pad_token_id,
        )
        gen = out.sequences[:, enc["input_ids"].shape[1]:]        # (B, T)
        # stream per-step logprobs instead of materialising (T, B, V)
        step_lp = []
        for t, sc in enumerate(out.scores):
            lsm = torch.log_softmax(sc, dim=-1)
            step_lp.append(lsm.gather(-1, gen[:, t:t + 1]).squeeze(-1))
        tok_all = torch.stack(step_lp, dim=1) if step_lp else torch.zeros(gen.shape[0], 0)

        results: list[GenOut] = []
        eos = self.tok.eos_token_id
        for b in range(gen.shape[0]):
            ids = gen[b]
            stop = ids.shape[0]
            hit = (ids == eos).nonzero(as_tuple=False)
            if len(hit):
                # exclude the EOS token itself: `_chunk_lp` re-scores the decoded
                # text, which never contains EOS, so including it here would put a
                # constant offset into every stored log-probability.
                stop = int(hit[0, 0])
            ids = ids[:stop]
            n = min(stop, tok_all.shape[1])
            tok_lp = tok_all[b, :n].float().cpu() if n else torch.zeros(0)
            text = self.tok.decode(ids[:n], skip_special_tokens=True)
            results.append(GenOut(text=text.strip(), token_logprobs=tok_lp,
                                  prompt_text=prompt_texts[b]))
        return results

    def generate(self, prompt_text: str, **kw) -> GenOut:
        return self.generate_batch([prompt_text], **kw)[0]

    # --------------------------------------------------- scoring (for PPO)
    def _chunk_lp(self, pairs: list[tuple[str, str]]):
        """Per-token logprob and mask for a small chunk of (prompt, response)."""
        seqs = []
        for prompt_text, response_text in pairs:
            p_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
            f_ids = self.tok(prompt_text + response_text, add_special_tokens=False).input_ids
            start = min(len(p_ids), len(f_ids))
            if start >= len(f_ids):
                f_ids = f_ids + [self.tok.eos_token_id]
            seqs.append((f_ids, start))

        maxlen = max(len(s) for s, _ in seqs)
        pad_id = self.tok.pad_token_id
        input_ids, attn, lab = [], [], []
        for ids, start in seqs:
            pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            lab.append([-100] * start + ids[start:] + [-100] * pad)

        batch = {
            "input_ids": torch.tensor(input_ids, device=DEVICE),
            "attention_mask": torch.tensor(attn, device=DEVICE),
        }
        labels = torch.tensor(lab, device=DEVICE)
        logits = self.model(**batch, use_cache=False).logits      # (B, T, V)
        tgt = labels[:, 1:]
        lg = logits[:, :-1, :]
        mask = tgt.ne(-100)
        safe = tgt.clamp(min=0)
        chosen = lg.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        lp = chosen.float() - torch.logsumexp(lg, dim=-1).float()
        return lp, mask

    def _score(self, pairs: list[tuple[str, str]], chunk: int = 4) -> torch.Tensor:
        """Sum response logprob per pair. Chunked to bound peak memory."""
        totals = []
        for i in range(0, len(pairs), chunk):
            lp, mask = self._chunk_lp(pairs[i:i + chunk])
            totals.append((lp * mask).sum(dim=1))
        return torch.cat(totals)

    def token_nll(self, pairs: list[tuple[str, str]], chunk: int = 2) -> torch.Tensor:
        """Mean per-token NLL over the response spans (memory-light SFT loss)."""
        self.model.train()
        num, den = 0.0, 0
        for i in range(0, len(pairs), chunk):
            lp, mask = self._chunk_lp(pairs[i:i + chunk])
            num = num + (-(lp * mask).sum())
            den += int(mask.sum())
        return num / max(den, 1)

    @torch.no_grad()
    def score_texts(self, pairs: list[tuple[str, str]]) -> torch.Tensor:
        """Sum logprob of each response given its prompt. Returns (B,)."""
        self.model.eval()
        return self._score(pairs).detach().cpu()

    # ------------------------------------------------------------- training
    def trainable_params(self, name: str):
        params = [p for n, p in self.model.named_parameters()
                  if "lora_" in n and name in n]
        for p in params:
            p.requires_grad_(True)
        return params

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def save_adapter(self, name: str, parent_dir) -> None:
        """Save one adapter into `<parent_dir>/<name>/`."""
        self.model.save_pretrained(str(parent_dir), selected_adapters=[name])


def build_backbone(model_id: str, adapters: list[str]) -> Backbone:
    bb = Backbone(model_id)
    for a in adapters:
        bb.add_adapter(a)
    return bb
