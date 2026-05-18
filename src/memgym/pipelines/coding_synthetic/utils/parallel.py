"""Shared parallel_map utility for pipeline stage loops."""

import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, List, Optional, Set

from tqdm import tqdm


def _load_done_ids(output_path: Path) -> Set[str]:
    """Load instance IDs already written to output file (for resume)."""
    done = set()
    if not output_path.exists():
        return done
    try:
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and "instance_id" in data:
                        iid = data["instance_id"]
                        done.add(iid)
                        # Also add without prefix for cross-format matching
                        # Output has "memgym__X" but input items have "X"
                        if iid.startswith("memgym__"):
                            done.add(iid[len("memgym__"):])
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    return done


def parallel_map(
    fn: Callable[..., Any],
    items: List[Any],
    num_workers: int = 4,
    desc: str = "Processing",
    output_path: Optional[Path] = None,
    serialize_fn: Optional[Callable[[Any], str]] = None,
    show_progress: bool = True,
    filter_none: bool = True,
    max_retries: int = 2,
    item_id_fn: Optional[Callable[[Any], str]] = None,
) -> List[Any]:
    """Apply fn to each item, optionally in parallel with progress and file output.

    Args:
        fn: Function to apply to each item. Should return a result or None.
        items: List of items to process.
        num_workers: Number of parallel workers. 1 = sequential (for debugging).
        desc: Description for the progress bar.
        output_path: If provided, append serialized results to this file.
        serialize_fn: Function to serialize a result to a string line.
            Defaults to json.dumps if output_path is set.
        show_progress: Whether to show a tqdm progress bar.
        filter_none: If True, exclude None results from the output list.
        max_retries: Max retry attempts per item on failure (0 = no retries).
        item_id_fn: Function to extract an ID from an item for within-stage resume.
            If set and output_path exists, already-processed items are skipped.

    Returns:
        List of results (None values filtered out if filter_none=True).
    """
    if not items:
        return []

    # Within-stage resume: skip items already in output file
    if item_id_fn and output_path and output_path.exists():
        done_ids = _load_done_ids(output_path)
        if done_ids:
            original_count = len(items)
            items = [it for it in items if item_id_fn(it) not in done_ids]
            skipped = original_count - len(items)
            if skipped > 0:
                msg = f"  Resuming: skipped {skipped} already-processed items, {len(items)} remaining"
                print(msg)
            if not items:
                return []

    if serialize_fn is None and output_path is not None:
        serialize_fn = lambda r: json.dumps(r, default=str)

    write_lock = threading.Lock() if output_path else None
    results: List[Any] = []

    def _process_and_write(item: Any) -> Any:
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                result = fn(item)
                if result is not None and output_path and serialize_fn:
                    line = serialize_fn(result)
                    with write_lock:
                        with open(output_path, "a") as f:
                            f.write(line + "\n")
                return result
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    wait = min(2 ** (attempt + 1), 30)
                    time.sleep(wait)
                    continue
                raise last_err

    if num_workers <= 1:
        # Sequential mode
        iterator = tqdm(items, desc=desc) if show_progress else items
        for item in iterator:
            try:
                result = _process_and_write(item)
                if result is not None or not filter_none:
                    results.append(result)
            except Exception as e:
                tqdm.write(f"  Error processing item: {e}") if show_progress else print(f"  Error processing item: {e}")
                traceback.print_exc()
        return results

    # Parallel mode
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        future_to_idx = {
            pool.submit(_process_and_write, item): i
            for i, item in enumerate(items)
        }

        if show_progress:
            pbar = tqdm(total=len(items), desc=desc)

        # Collect results preserving order
        ordered_results = [None] * len(items)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                ordered_results[idx] = future.result()
            except Exception as e:
                msg = f"  Error processing item {idx}: {e}"
                if show_progress:
                    tqdm.write(msg)
                else:
                    print(msg)
                traceback.print_exc()
            finally:
                if show_progress:
                    pbar.update(1)

        if show_progress:
            pbar.close()

    if filter_none:
        return [r for r in ordered_results if r is not None]
    return ordered_results
