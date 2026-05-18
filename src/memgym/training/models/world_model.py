"""Memory world model architecture.

Binary classifier: predicts whether a compression event will hurt agent performance.

Architecture: Qwen2.5-Coder-7B-Instruct + QLoRA (NF4) adapters
Input: Chat-formatted (task, context, compression event)
Output: SAFE/HARMFUL with reasoning

Multi-GPU: vanilla HF transformers + PEFT + bitsandbytes.
Launch with: accelerate launch --multi_gpu --num_processes 8
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the world model."""

    base_model: str = "Qwen/Qwen3.5-9B-Base"
    max_seq_length: int = 32768
    load_in_4bit: bool = True

    # Completion label pair the classifier was trained against. Must match
    # what the aug-pairs adapter wrote to the `completion` field.
    # Phase A sweeps these over {" Y"/" N", " yes"/" no", " good"/" bad"}.
    # Required single-token under the base tokenizer so CE-on-one-token is
    # symmetric across labels.
    safe_completion: str = " Y"
    harmful_completion: str = " N"

    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.0


class MemoryWorldModel:
    """Binary classifier for compression event safety.

    Uses HF transformers + bitsandbytes NF4 quantization + PEFT LoRA.
    Compatible with multi-GPU DDP via accelerate.
    """

    # Token strings used as single-token classification targets.
    SAFE_TOKEN = "SAFE"
    HARMFUL_TOKEN = "HARMFUL"

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        # Completion-suffix strings that `aug_pairs.py` writes as the SFT
        # completion field. Driven by ModelConfig so Phase A's label
        # sweep flips between " Y"/" N", " yes"/" no", " good"/" bad"
        # without touching call sites. Must be single-token under the
        # base tokenizer — `predict_logits_text` errors loudly if not.
        self.SAFE_COMPLETION = self.config.safe_completion
        self.HARMFUL_COMPLETION = self.config.harmful_completion
        self.model = None
        self.tokenizer = None
        self._is_setup = False
        # Cached after setup — single-token IDs for SAFE/HARMFUL used by
        # `predict_logits` for one-forward-pass classification.
        self._safe_token_id: Optional[int] = None
        self._harmful_token_id: Optional[int] = None

    def setup(self) -> None:
        """Load the base model with QLoRA adapters."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        logger.info("Loading %s with NF4 quantization...", self.config.base_model)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ) if self.config.load_in_4bit else None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
            attn_implementation="sdpa",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            model_max_length=self.config.max_seq_length,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = prepare_model_for_kbit_training(
            self.model, use_gradient_checkpointing=True,
        )

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Model setup complete: %s trainable / %s total params (%.2f%%)",
            f"{trainable:,}", f"{total:,}", trainable / total * 100,
        )

        self._cache_label_token_ids()
        self._is_setup = True

    def _cache_label_token_ids(self) -> None:
        """Resolve single-token IDs for SAFE/HARMFUL.

        Qwen2.5 BPE encodes both as standalone tokens at the start of an
        assistant turn; we look up the first token from each label string
        encoded WITHOUT special tokens. If a tokenizer ever splits these
        across multiple BPE pieces, `predict_logits` would silently
        compare partial-token logits, so we assert single-tokenness.
        """
        safe_ids = self.tokenizer.encode(self.SAFE_TOKEN, add_special_tokens=False)
        harm_ids = self.tokenizer.encode(self.HARMFUL_TOKEN, add_special_tokens=False)
        if len(safe_ids) != 1 or len(harm_ids) != 1:
            logger.warning(
                "Label tokens are not single-token: SAFE=%s HARMFUL=%s — "
                "predict_logits will use the first sub-token, which may be ambiguous.",
                safe_ids, harm_ids,
            )
        self._safe_token_id = safe_ids[0]
        self._harmful_token_id = harm_ids[0]
        logger.info(
            "Label token IDs: SAFE=%d HARMFUL=%d",
            self._safe_token_id, self._harmful_token_id,
        )

    def predict_logits(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, float]:
        """One-forward-pass binary classification.

        Runs a single forward at `add_generation_prompt=True`, takes the
        next-token logits at the SAFE/HARMFUL token IDs, softmaxes over
        those two, and returns (predicted_label, calibrated_probability).

        Replaces the old generate-then-parse flow which:
          * cost 256 autoregressive token steps per call,
          * returned `confidence=0.5` in ~90% of cases (untrustworthy AUROC).

        Returns (label_str, prob_safe) where `prob_safe` is in [0,1] and
        equals the softmax weight of the SAFE token over {SAFE, HARMFUL}.
        """
        if not self._is_setup:
            raise RuntimeError("Call setup() before predict_logits()")
        if self._safe_token_id is None or self._harmful_token_id is None:
            self._cache_label_token_ids()

        self.model.eval()
        enc = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            # logits_to_keep=1 projects only the last position through
            # lm_head. At 32K seq × 152K vocab, full-sequence projection
            # costs ~21 GiB (fp32 upcast inside accelerate's
            # convert_to_fp32) and OOMs A100-40GB; last-token-only is
            # ~600 KiB. Falls back gracefully on older transformers.
            try:
                outputs = self.model(input_ids=input_ids, logits_to_keep=1)
            except TypeError:
                outputs = self.model(input_ids=input_ids)
            logits = outputs.logits  # (1, k, V) — k=1 if kwarg honored, else T

        next_logits = logits[0, -1, :]  # (V,)
        pair = torch.stack([
            next_logits[self._safe_token_id],
            next_logits[self._harmful_token_id],
        ])
        probs = torch.softmax(pair, dim=-1)
        prob_safe = float(probs[0].item())
        prediction = self.SAFE_TOKEN if prob_safe >= 0.5 else self.HARMFUL_TOKEN
        return prediction, prob_safe

    def predict_logits_text(
        self, text: str, safe_threshold: float = 0.5,
    ) -> Tuple[str, float]:
        """Raw-text one-forward-pass classification.

        Matches the prompt+completion SFT training format produced by
        `aug_pairs.py`. The tricky bit: Qwen uses byte-level BPE, which
        greedily merges whitespace into the following token, so
        `tokenize("Answer: SAFE")` and `tokenize("Answer: ")`+`tokenize("SAFE")`
        produce DIFFERENT token sequences. The trainer saw the joint
        tokenization, and the "label" token it trained against is
        whatever the first-differing-position produces.

        We recover the correct label slots by probing: encode
        `text+"SAFE"` and `text+"HARMFUL"`, find the first position
        where they diverge, and that position IS the label. We feed
        the shared prefix to the model and compare logits at the two
        divergent tokens. Robust to any whitespace-merging quirks.
        """
        if not self._is_setup:
            raise RuntimeError("Call setup() before predict_logits_text()")

        enc_safe = self.tokenizer.encode(
            text + self.SAFE_COMPLETION, add_special_tokens=False,
        )
        enc_harm = self.tokenizer.encode(
            text + self.HARMFUL_COMPLETION, add_special_tokens=False,
        )
        diff = 0
        min_len = min(len(enc_safe), len(enc_harm))
        while diff < min_len and enc_safe[diff] == enc_harm[diff]:
            diff += 1
        if diff == min_len:
            # SAFE/HARMFUL encodings never diverge; fatal misconfig.
            raise ValueError(
                f"label encodings share a full prefix of length {min_len}; "
                f"enc_safe={enc_safe} enc_harm={enc_harm}"
            )
        safe_label_id = enc_safe[diff]
        harm_label_id = enc_harm[diff]

        self.model.eval()
        input_ids = torch.tensor(
            [enc_safe[:diff]], dtype=torch.long, device=self.model.device,
        )
        with torch.no_grad():
            # See predict_logits() — same OOM avoidance.
            try:
                outputs = self.model(input_ids=input_ids, logits_to_keep=1)
            except TypeError:
                outputs = self.model(input_ids=input_ids)
            next_logits = outputs.logits[0, -1, :]

        pair = torch.stack([next_logits[safe_label_id], next_logits[harm_label_id]])
        probs = torch.softmax(pair, dim=-1)
        prob_safe = float(probs[0].item())
        prediction = self.SAFE_TOKEN if prob_safe >= safe_threshold else self.HARMFUL_TOKEN
        return prediction, prob_safe

    def predict(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, float]:
        """Predict whether a compression event is SAFE or HARMFUL."""
        if not self._is_setup:
            raise RuntimeError("Call setup() before predict()")

        self.model.eval()

        enc = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
            )

        generated = outputs[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        response_upper = response.strip().upper()
        if "HARMFUL" in response_upper.split("\n")[0]:
            prediction = "HARMFUL"
        else:
            prediction = "SAFE"

        confidence = 1.0 if response_upper.startswith(f"PREDICTION: {prediction}") else 0.5
        return prediction, confidence

    def save_checkpoint(self, path: str | Path) -> None:
        """Save LoRA adapter weights."""
        if not self._is_setup:
            raise RuntimeError("Call setup() before save_checkpoint()")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        logger.info("Checkpoint saved to %s", path)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: Optional[ModelConfig] = None,
    ) -> "MemoryWorldModel":
        """Load a trained model from checkpoint."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        config = config or ModelConfig()
        instance = cls(config)

        checkpoint_path = Path(checkpoint_path)
        logger.info("Loading checkpoint from %s", checkpoint_path)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ) if config.load_in_4bit else None

        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
            attn_implementation="sdpa",
        )
        instance.model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
        # LoRA-only checkpoints (e.g. WeightedTrainer path) may omit
        # tokenizer files; fall back to the base model's tokenizer in that
        # case. Fall-back is safe because LoRA never modifies vocabulary.
        try:
            instance.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        except (OSError, ValueError) as exc:
            logger.warning(
                "tokenizer not found at checkpoint %s (%s); falling back to base %s",
                checkpoint_path, type(exc).__name__, config.base_model,
            )
            instance.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
        instance._cache_label_token_ids()
        instance._is_setup = True

        return instance
