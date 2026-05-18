"""Tokenization diagnostic for the aug-SFT prompt/completion format.

Byte-level BPE (Qwen, GPT-2/3 family) greedily merges whitespace into
following tokens, so `tokenizer("SAFE")` and `tokenizer("\n\nAnswer: SAFE")`
do NOT necessarily share a suffix token. If the trainer tokenized
prompt+completion as one string, the trained target was the suffix of
the full encoding — NOT the standalone-label encoding our
`_cache_label_token_ids` uses. This script prints both so we can see
the mismatch explicitly.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def show(s: str) -> list[int]:
        ids = tok.encode(s, add_special_tokens=False)
        toks = [tok.decode([i]) for i in ids]
        print(f"{s!r:42}  ids={ids}  toks={toks}")
        return ids

    print("=== standalone label encodings ===")
    show("SAFE")
    show(" SAFE")
    show("HARMFUL")
    show(" HARMFUL")

    print("\n=== full prompt+completion (what TRL concat-tokenized) ===")
    prefix = show("\n\nAnswer: ")
    full_safe = show("\n\nAnswer: SAFE")
    full_harm = show("\n\nAnswer: HARMFUL")

    print("\n=== trained-target token (prompt+completion suffix) ===")
    tail_safe = full_safe[len(prefix):]
    tail_harm = full_harm[len(prefix):]
    print(f"tail_safe    ids={tail_safe}  toks={[tok.decode([i]) for i in tail_safe]}")
    print(f"tail_harm    ids={tail_harm}  toks={[tok.decode([i]) for i in tail_harm]}")

    print("\n=== would they match what predict_logits_text looks up? ===")
    lut_safe = tok.encode("SAFE", add_special_tokens=False)
    lut_harm = tok.encode("HARMFUL", add_special_tokens=False)
    print(f"standalone 'SAFE'   = {lut_safe}   trained = {tail_safe}   "
          f"match={lut_safe == tail_safe}")
    print(f"standalone 'HARMFUL'= {lut_harm}   trained = {tail_harm}   "
          f"match={lut_harm == tail_harm}")

    print("\n=== single-token label candidates (after 'Answer:') ===")
    # For each candidate, check tokenization of (prompt_prefix + " " + candidate)
    # and whether the label part is a single token after the shared prefix.
    prefix = "\n\nAnswer:"  # NO trailing space; label carries its own leading space
    candidates = [
        ("SAFE", "HARMFUL"),
        ("Y", "N"),
        ("1", "0"),
        ("T", "F"),
        ("yes", "no"),
        ("Yes", "No"),
        ("OK", "NG"),
        ("good", "bad"),
    ]
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    for safe_s, harm_s in candidates:
        enc_s = tok.encode(prefix + " " + safe_s, add_special_tokens=False)
        enc_h = tok.encode(prefix + " " + harm_s, add_special_tokens=False)
        tail_s = enc_s[len(prefix_ids):]
        tail_h = enc_h[len(prefix_ids):]
        both_single = len(tail_s) == 1 and len(tail_h) == 1
        marker = "✓" if both_single else "✗"
        print(f"  {marker} ({safe_s!r:>6}, {harm_s!r:>6})  "
              f"tail_safe={tail_s}({len(tail_s)})  "
              f"tail_harm={tail_h}({len(tail_h)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
