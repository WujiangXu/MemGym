"""Execute one synthesized pytest test per fact, inside a Docker container.

Lightweight alternative to the heavy SWE-bench patch-apply runner in
`gym/swe_bench/`. For each fact:

  1. Write the test to a temp file on the host.
  2. `docker run --rm -v <test>:/tmp/test_<fid>.py <image> pytest -x ...`
  3. Classify exit: 0 = passed; 1/non-zero with collection error in
     stderr = synthesis error (inconclusive); otherwise assertion fail.

This runner assumes Docker is installed locally and the SWE-smith
image is already pullable. For large sweeps, use `--workers` to
parallelize (bounded by Docker daemon concurrency).
"""
from __future__ import annotations

import concurrent.futures as cf
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import FactTestResult


_COLLECT_ERR_RE = re.compile(
    r"(ERROR collecting|ImportError|ModuleNotFoundError|SyntaxError|"
    r"IndentationError|NameError)",
    re.IGNORECASE,
)

# Docker / container plumbing failures — tool side, not data side.
_DOCKER_EXEC_ERR_RE = re.compile(
    r"(executable file not found|OCI runtime create failed|"
    r"docker: Error response from daemon|no such file or directory.*pytest|"
    # pytest usage-error (unknown flag, missing plugin) — test didn't run
    r"pytest.*unrecognized arguments|"
    r"__main__\.py: error:)",
    re.IGNORECASE,
)


def docker_available() -> bool:
    return shutil.which("docker") is not None


def run_one_test(
    fact_id: str,
    test_code: str,
    image: str,
    test_name: str = "",
    timeout_s: int = 30,
) -> FactTestResult:
    """Run a single synthesized pytest file in a Docker container."""
    if not docker_available():
        return FactTestResult(
            fact_id=fact_id,
            passed=False,
            test_synth_error="docker not available on host",
            test_name=test_name,
        )

    start = time.time()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"test_{fact_id}_", delete=False,
    ) as tf:
        tf.write(test_code)
        host_path = tf.name

    # Mount the test *inside* /testbed so pytest rootdir = the repo, which
    # makes repo-relative imports (e.g. `from pdfminer.layout import ...`)
    # resolve without extra sys.path gymnastics.
    container_path = f"/testbed/test_{fact_id}.py"
    # SWE-smith images ship pytest inside the `testbed` conda env. Activate
    # it via bash -lc so python + pytest are on PATH regardless of whether
    # the image's default ENTRYPOINT already does so. --timeout is a
    # pytest-timeout plugin flag that isn't always installed; we rely on
    # subprocess.run's timeout instead (timeout_s + 30s at the docker layer).
    inner = (
        "source /opt/miniconda3/bin/activate testbed 2>/dev/null || true; "
        f"python -m pytest -x --tb=short {container_path}"
    )
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "-v", f"{host_path}:{container_path}:ro",
        "--workdir", "/testbed",
        image,
        "bash", "-lc", inner,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s + 30,
        )
        runtime_ms = int((time.time() - start) * 1000)
        stderr = (proc.stderr or "")[:2000]
        stdout = (proc.stdout or "")[:2000]
        if proc.returncode == 0:
            return FactTestResult(
                fact_id=fact_id, passed=True, runtime_ms=runtime_ms,
                test_name=test_name, stderr=stderr,
            )
        # Distinguish synthesis errors from genuine assertion failures
        synth = None
        if _DOCKER_EXEC_ERR_RE.search(stderr) or _DOCKER_EXEC_ERR_RE.search(stdout):
            synth = "docker exec / image plumbing error — inconclusive"
        elif _COLLECT_ERR_RE.search(stderr) or _COLLECT_ERR_RE.search(stdout):
            synth = "collection / import / syntax error — test itself broken"
        return FactTestResult(
            fact_id=fact_id, passed=False, runtime_ms=runtime_ms,
            test_name=test_name, stderr=stderr + "\n---stdout---\n" + stdout,
            test_synth_error=synth,
        )
    except subprocess.TimeoutExpired:
        return FactTestResult(
            fact_id=fact_id, passed=False,
            runtime_ms=int((time.time() - start) * 1000),
            test_name=test_name,
            stderr=f"docker run timed out after {timeout_s + 30}s",
        )
    except Exception as e:
        return FactTestResult(
            fact_id=fact_id, passed=False,
            runtime_ms=int((time.time() - start) * 1000),
            test_name=test_name,
            stderr=f"docker run exception: {type(e).__name__}: {e}",
            test_synth_error="docker invocation failed",
        )
    finally:
        try:
            Path(host_path).unlink(missing_ok=True)
        except Exception:
            pass


def run_tests_parallel(
    jobs: List[Dict[str, Any]],
    workers: int = 8,
    timeout_s: int = 30,
) -> Dict[str, FactTestResult]:
    """Run a list of `{fact_id, test_code, image, test_name}` jobs.

    Returns {fact_id → FactTestResult}.
    """
    out: Dict[str, FactTestResult] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                run_one_test,
                j["fact_id"], j["test_code"], j["image"],
                j.get("test_name", ""), timeout_s,
            ): j["fact_id"]
            for j in jobs
        }
        for fut in cf.as_completed(futures):
            fid = futures[fut]
            try:
                out[fid] = fut.result()
            except Exception as e:
                out[fid] = FactTestResult(
                    fact_id=fid, passed=False,
                    stderr=f"future exception: {e}",
                    test_synth_error="executor failed",
                )
    return out
