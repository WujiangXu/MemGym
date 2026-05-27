"""Pluggable summarizer backends for `NaiveSummarizationMemory`.

The default path (`LitellmSummarizerBackend`) wraps `litellm.completion`
so existing callers keep working unchanged. `LocalHFSummarizerBackend`
is the H.3 hot-swap target: a loaded `transformers` model used
directly (no HTTP endpoint) so the RL loop can reach in for log-probs.

Why a Protocol and not just a callable: the reasoning-content fallback
(Qwen3.6-Thinking can route all output into `reasoning_content` when
the completion budget is hit before `</think>`) is a litellm-shape
concern, not something the local HF path has. Keeping a small typed
interface lets each backend own its idiosyncrasies.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from litellm import completion

logger = logging.getLogger(__name__)


@runtime_checkable
class SummarizerBackend(Protocol):
    """Minimal interface every summarizer backend must implement.

    `system` and `user` are already-formatted strings. Returning an
    empty string means "no summary this compaction"; the caller keeps
    the previous summary in that case.
    """

    def summarize(self, system: str, user: str, max_tokens: int = 1024) -> str:
        ...


class LitellmSummarizerBackend:
    """Default backend — routes the summarization call through `litellm.completion`.

    Matches pre-refactor behavior byte-for-byte so retiring this class
    is a no-op when the plumbing path is in use.
    """

    def __init__(self, model: str):
        self.model = model

    def summarize(self, system: str, user: str, max_tokens: int = 1024) -> str:
        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — summarization is best-effort
            logger.warning("litellm summarization failed: %s", exc)
            return ""

        msg = response.choices[0].message
        text = (getattr(msg, "content", None) or "").strip()
        if text:
            return text
        reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
        if reasoning:
            return reasoning[-1600:]
        return ""


class LocalHFSummarizerBackend:
    """Hot-swappable local HF summarizer — direct `transformers` inference.

    Two construction paths:

    1. `LocalHFSummarizerBackend(model_name_or_path="...")` — loads
       model + tokenizer via `transformers.AutoModelForCausalLM`.
       Used by the H.2 smoke-eval script (load once, run 20 instances).
    2. `LocalHFSummarizerBackend.from_model(model, tokenizer)` — plugs
       in an already-loaded HF model. Used by the H.3 RL loop where
       TRL holds the policy model already; we avoid a second load.

    Generation kwargs are mutable so the RL loop can set per-rollout
    sampling temperature / top_p without breaking the Protocol:
        backend.generation_kwargs["temperature"] = 0.9
        backend.summarize(system, user)

    Returning `""` means "no summary this compaction" — same contract
    as `LitellmSummarizerBackend`.
    """

    # Keep heavy deps deferred — tests mock-instantiate via `from_model`
    # without ever pulling transformers in.
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        max_prompt_tokens: int = 6144,
        use_chat_template: bool = True,
        use_lora: bool = True,
    ):
        import json
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(f"unsupported torch_dtype {torch_dtype!r}")

        # PEFT adapter auto-detect is gated on `use_lora=True`. When the
        # caller wants a full-finetune load, we skip the adapter branch
        # entirely — `model_name_or_path` is treated as a plain HF
        # checkpoint, even if an adapter_config.json happens to be on
        # disk. This keeps the full-FT RL path from accidentally reading
        # a LoRA-shaped checkpoint (different parameter shape would only
        # fail loudly later, during the optimizer init).
        adapter_cfg_path = None
        if use_lora and os.path.isdir(model_name_or_path):
            cand = os.path.join(model_name_or_path, "adapter_config.json")
            if os.path.isfile(cand):
                adapter_cfg_path = cand

        if adapter_cfg_path is not None:
            with open(adapter_cfg_path) as fh:
                adapter_cfg = json.load(fh)
            base_id = adapter_cfg.get("base_model_name_or_path")
            if not base_id:
                raise ValueError(
                    f"adapter_config.json at {adapter_cfg_path} has no "
                    "base_model_name_or_path; cannot infer base model."
                )
            tok_path = tokenizer_name_or_path or model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                base_id,
                torch_dtype=dtype_map[torch_dtype],
                device_map=device_map,
            )
            # is_trainable=True keeps the LoRA matrices requires_grad=True,
            # which a policy-gradient training loop needs to backprop through. Inference-only
            # callers wrap in torch.no_grad(), so this default is safe.
            self.model = PeftModel.from_pretrained(
                base, model_name_or_path, is_trainable=True,
            )
        else:
            tok_path = tokenizer_name_or_path or model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype_map[torch_dtype],
                device_map=device_map,
            )
        self.model.eval()

        self.max_prompt_tokens = max_prompt_tokens
        self.use_chat_template = use_chat_template
        self.generation_kwargs: Dict[str, Any] = {
            "do_sample": False,  # deterministic by default; RL loop toggles this
            "temperature": 1.0,
            "top_p": 1.0,
        }
        # Per-rank non-sharded inference copy. Set by the RL trainer after
        # FSDP-wrapping `self.model`; left None on single-GPU paths where
        # `self.model` is already non-sharded.
        self._inference_model: Optional[Any] = None

    @classmethod
    def from_model(
        cls,
        model: Any,
        tokenizer: Any,
        max_prompt_tokens: int = 6144,
        use_chat_template: bool = True,
    ) -> "LocalHFSummarizerBackend":
        """Plug an already-loaded model/tokenizer (RL loop / testing)."""
        inst = cls.__new__(cls)
        inst.model = model
        inst.tokenizer = tokenizer
        inst.max_prompt_tokens = max_prompt_tokens
        inst.use_chat_template = use_chat_template
        inst.generation_kwargs = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
        }
        inst._inference_model = None
        return inst

    def _rollout_model(self) -> Any:
        """Pick the model for rollout-time forward passes (no collectives).

        FSDP path: trainer attaches a non-sharded deepcopy at
        `_inference_model`. Single-GPU path: fall back to `self.model`,
        which is already non-sharded, so per-rank divergence is moot.
        """
        return self._inference_model if self._inference_model is not None else self.model

    def _build_prompt_ids(self, system: str, user: str):
        """Format + tokenize the prompt, truncating from the left so the
        most recent (load-bearing) content is retained under the budget.
        """
        tok = self.tokenizer
        if self.use_chat_template and hasattr(tok, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"{system}\n\n{user}\n\nSummary:"

        ids = tok(text, return_tensors="pt", add_special_tokens=not self.use_chat_template)
        input_ids = ids["input_ids"]
        if input_ids.shape[1] > self.max_prompt_tokens:
            input_ids = input_ids[:, -self.max_prompt_tokens :]
            if "attention_mask" in ids:
                ids["attention_mask"] = ids["attention_mask"][:, -self.max_prompt_tokens :]
        ids["input_ids"] = input_ids
        return ids

    def summarize(self, system: str, user: str, max_tokens: int = 1024) -> str:
        import torch

        # Route through the per-rank inference copy. The previous
        # implementation called `FSDP.summon_full_params` here, which is a
        # collective: every rank must reach it the same number of times in
        # lockstep. Cloud-agent rollouts produce variable-length episodes,
        # so per-rank `summarize()` call counts diverge and NCCL times out
        # on `_ALLGATHER_BASE`. The non-sharded inference copy avoids the
        # collective entirely — sync happens once per optimizer step at the
        # post-step barrier (see `sync_inference_from_fsdp`).
        rollout_model = self._rollout_model()

        inputs = self._build_prompt_ids(system, user)
        device = getattr(rollout_model, "device", None)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = rollout_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **self.generation_kwargs,
            )

        # Decode only the newly-generated tokens — `output_ids` includes
        # the prompt, which we slice off so chat-template scaffolding
        # doesn't leak into the summary string.
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text

    def compute_logprobs_for_completion(
        self,
        system: str,
        user: str,
        completion: str,
        *,
        mode: str = "train",
    ):
        """Recompute per-token log-probs of `completion` under the current policy.

        Returns a 1-D `torch.Tensor` of shape `[completion_len]`.

        Two modes:
            * ``mode="train"`` — forward through ``self.model`` (FSDP-wrapped
              under multi-GPU training). Result has grads attached. Called
              by the training update where every rank reaches this point in
              lockstep, so the FSDP collective is safe.
            * ``mode="rollout"`` — forward through ``self._inference_model``
              (per-rank non-sharded copy). No grad, no collective. Used by
              the snapshot pass during rollouts where per-rank step counts
              diverge.

        The prompt is formatted byte-identically to ``summarize()`` (same
        chat-template path, same left-truncation budget) so the log-probs
        line up with what the recording backend captured at rollout time.
        Differences in tokenization between rollout and recompute would
        silently corrupt the gradient — this is the contract that keeps
        them aligned.

        Empty completions return an empty tensor (no gradient signal).
        """
        import torch

        if mode not in ("train", "rollout"):
            raise ValueError(f"mode must be 'train' or 'rollout', got {mode!r}")

        target_model = self._rollout_model() if mode == "rollout" else self.model

        prompt_inputs = self._build_prompt_ids(system, user)
        prompt_ids = prompt_inputs["input_ids"]  # shape [1, P]

        completion_ids = self.tokenizer(
            completion,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]  # shape [1, C]

        if completion_ids.shape[1] == 0:
            device = getattr(target_model, "device", None) or torch.device("cpu")
            return torch.zeros(0, device=device)

        device = getattr(target_model, "device", None)
        if device is not None:
            prompt_ids = prompt_ids.to(device)
            completion_ids = completion_ids.to(device)

        # Concatenate prompt + completion, run a single forward pass.
        # Logits at position i predict token i+1, so the slice
        # `logits[:, P-1 : P+C-1]` lines up with the C completion tokens.
        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        if mode == "rollout":
            with torch.no_grad():
                outputs = target_model(input_ids=full_ids)
        else:
            outputs = target_model(input_ids=full_ids)
        logits = outputs.logits  # [1, P+C, V]

        prompt_len = prompt_ids.shape[1]
        completion_len = completion_ids.shape[1]
        target_logits = logits[0, prompt_len - 1 : prompt_len - 1 + completion_len, :]
        log_probs = torch.log_softmax(target_logits, dim=-1)
        target_ids = completion_ids[0]
        token_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return token_logprobs

    def sync_inference_from_fsdp(self) -> None:
        """Push current FSDP-sharded weights into the non-sharded inference copy.

        Called once per optimizer step at the post-``optimizer.step()`` barrier,
        when every rank is guaranteed to be at the same place in the loop —
        the single safe spot for an all-gather collective. After this call,
        ``self._inference_model`` reflects the just-optimized policy and
        the next rollout sees the new weights.

        The DTensor branch handles FSDP2 (each parameter is a
        ``DTensor`` whose ``.full_tensor()`` triggers the all-gather).
        The plain-tensor branch covers FSDP1 (its state_dict already
        materializes full tensors via ``FullStateDictConfig``).
        Wrapper-prefix stripping (``_fsdp_wrapped_module.``) covers the
        FSDP1 case where the wrapper prepends to every key.
        """
        if self._inference_model is None or self._inference_model is self.model:
            return  # non-FSDP path or trainer never attached an inference copy

        import torch

        try:
            from torch.distributed._tensor import DTensor  # FSDP2
        except ImportError:  # pragma: no cover — older torch
            DTensor = None  # type: ignore[assignment]

        with torch.no_grad():
            src_sd = self.model.state_dict()
            target_device = next(self._inference_model.parameters()).device
            full_sd: Dict[str, Any] = {}
            for k, v in src_sd.items():
                if DTensor is not None and isinstance(v, DTensor):
                    v = v.full_tensor()
                clean_k = k.replace("_fsdp_wrapped_module.", "")
                full_sd[clean_k] = v.to(target_device, non_blocking=True)
            # strict=False lets us tolerate mismatched LoRA-adapter keys
            # vs base; trainer is responsible for ensuring shapes match.
            self._inference_model.load_state_dict(full_sd, strict=False)


__all__ = [
    "SummarizerBackend",
    "LitellmSummarizerBackend",
    "LocalHFSummarizerBackend",
]
