"""Render train/eval curves from HF Trainer's `trainer_state.json`.

HF Trainer writes `<save_dir>/checkpoint-<step>/trainer_state.json` at
every save and `<save_dir>/trainer_state.json` at end of training. The
`log_history` array in that file contains every scalar logged via
`Trainer.log()` — train/loss, train/grad_norm, train/learning_rate, and
all `eval_*` metrics from the in-loop eval. We plot them here as a
multi-panel PNG so we don't need WandB to inspect the run shape.

Layout (2x3 grid):
    train/loss               train/grad_norm           train/learning_rate
    eval/loss                eval/auroc                eval/ece
    + a 2nd row of bonus eval panels if eval_safe_f1 / eval_harmful_f1
      are present (collapses to a tighter grid otherwise).

Output: `<save_dir>/training_curves.png` (300 DPI) and a JSON dump
of the parsed metrics so downstream tools can reuse the parsed shape.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _load_log_history(state_path: Path) -> list[dict]:
    state = json.loads(state_path.read_text())
    history = state.get("log_history")
    if not isinstance(history, list):
        raise ValueError(f"trainer_state.json at {state_path} has no log_history list")
    return history


def _split_streams(history: list[dict]) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Bucket each log row's keys into train / eval streams keyed by step."""
    train: dict[str, dict[int, float]] = defaultdict(dict)
    eval_: dict[str, dict[int, float]] = defaultdict(dict)
    for row in history:
        step = row.get("step")
        if step is None:
            continue
        for k, v in row.items():
            if k in ("step", "epoch"):
                continue
            if not isinstance(v, (int, float)):
                continue
            target = eval_ if k.startswith("eval_") or k.startswith("eval/") else train
            target[k][int(step)] = float(v)
    # Convert to parallel (steps, values) lists, sorted by step.
    def _flatten(b: dict[str, dict[int, float]]) -> dict[str, tuple[list[int], list[float]]]:
        out = {}
        for k, m in b.items():
            steps = sorted(m.keys())
            out[k] = (steps, [m[s] for s in steps])
        return out
    return _flatten(train), _flatten(eval_)


def _plot(train: dict, eval_: dict, out_png: Path, run_label: str) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    panels = [
        ("loss", train.get("loss"), "train/loss"),
        ("grad_norm", train.get("grad_norm"), "train/grad_norm"),
        ("lr", train.get("learning_rate"), "train/learning_rate"),
        ("eval_loss", eval_.get("eval_loss"), "eval/loss"),
        ("eval_auroc", eval_.get("eval_auroc"), "eval/auroc"),
        ("eval_ece", eval_.get("eval_ece"), "eval/ece"),
        ("eval_safe_f1", eval_.get("eval_safe_f1"), "eval/safe_f1"),
        ("eval_harmful_f1", eval_.get("eval_harmful_f1"), "eval/harmful_f1"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    fig.suptitle(f"Training metrics — {run_label}", fontsize=14)
    for ax, (_, data, title) in zip(axes.flat, panels):
        if data is None:
            ax.set_axis_off()
            ax.set_title(f"{title} (no data)", color="gray", fontsize=10)
            continue
        steps, vals = data
        marker = "o" if title.startswith("eval/") else None
        ax.plot(steps, vals, marker=marker, linewidth=1.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, required=True,
                        help="HF Trainer save_dir containing trainer_state.json "
                             "(or one or more checkpoint-<step> subdirs).")
    parser.add_argument("--out-png", type=Path, default=None,
                        help="Output PNG (default: <save-dir>/training_curves.png)")
    parser.add_argument("--out-json", type=Path, default=None,
                        help="Optional JSON dump of parsed train/eval streams "
                             "(default: <save-dir>/training_metrics.json)")
    parser.add_argument("--run-label", default=None,
                        help="Title suffix; defaults to the save-dir name")
    args = parser.parse_args()

    state_path = args.save_dir / "trainer_state.json"
    if not state_path.exists():
        # Fall back to the latest checkpoint subdir.
        ckpt_dirs = sorted(args.save_dir.glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[-1]))
        if not ckpt_dirs:
            print(f"ERROR: no trainer_state.json under {args.save_dir} or any "
                  f"checkpoint-* subdir", file=sys.stderr)
            return 2
        state_path = ckpt_dirs[-1] / "trainer_state.json"

    history = _load_log_history(state_path)
    train, eval_ = _split_streams(history)

    out_png = args.out_png or (args.save_dir / "training_curves.png")
    out_json = args.out_json or (args.save_dir / "training_metrics.json")
    label = args.run_label or args.save_dir.name

    _plot(train, eval_, out_png, label)
    out_json.write_text(json.dumps({"train": train, "eval": eval_}, indent=2))
    print(f"wrote {out_png}")
    print(f"wrote {out_json}")
    print(f"train series: {sorted(train.keys())}")
    print(f"eval series:  {sorted(eval_.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
