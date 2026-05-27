"""Dump verl 0.7.1's `RolloutConfig` dataclass schema.

V2l said "Please increase rollout.update_weights_bucket_megabytes(2048
MB)" but `RolloutConfig` rejects that key flat. The knob lives somewhere
nested. This probe lists every field on `RolloutConfig` (and its nested
dataclass-typed fields) so we can find the real path.
"""
from __future__ import annotations
import dataclasses
import inspect


def _walk(cls, prefix: str = "", seen: set | None = None, depth: int = 0) -> None:
    seen = seen or set()
    if cls in seen or depth > 4:
        return
    seen.add(cls)
    if not dataclasses.is_dataclass(cls):
        return
    for f in dataclasses.fields(cls):
        path = f"{prefix}.{f.name}" if prefix else f.name
        type_name = getattr(f.type, "__name__", str(f.type))
        default = "MISSING" if f.default is dataclasses.MISSING else repr(f.default)
        print(f"  {path}: {type_name} = {default[:80]}")
        # Recurse into nested dataclasses (handle string-form annotations).
        nested = f.type if isinstance(f.type, type) else None
        if nested and dataclasses.is_dataclass(nested):
            _walk(nested, path, seen, depth + 1)


def main() -> int:
    from verl.workers.config.rollout import RolloutConfig
    print("=== RolloutConfig fields ===")
    _walk(RolloutConfig)

    # Also grep for "bucket" anywhere in the verl source under workers/rollout
    import verl.workers.rollout as _r
    print("\n=== bucket-related symbols in verl.workers.rollout ===")
    for name in sorted(dir(_r)):
        if "bucket" in name.lower() or "weight" in name.lower():
            print(f"  {name}")

    # Look at bucketed_weight_transfer for the actual class
    try:
        from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer as bwt
        print(f"\n=== bucketed_weight_transfer module file: {bwt.__file__} ===")
        # Find any dataclass / config class in this module
        for name, obj in inspect.getmembers(bwt):
            if inspect.isclass(obj) and obj.__module__ == bwt.__name__:
                print(f"  class {name}: {obj}")
                if dataclasses.is_dataclass(obj):
                    print("    fields:")
                    _walk(obj, prefix=f"  {name}")
    except Exception as e:  # noqa: BLE001
        print(f"  (skipped: {type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
