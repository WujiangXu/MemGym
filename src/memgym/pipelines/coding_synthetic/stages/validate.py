"""
Stage 5 — Verify: Ablation-based verification of instance quality.

Two conditions:
  A (with memory): task_prompt + repo + memory_files → patch → Docker tests
  B (no memory):   task_prompt + repo → patch → Docker tests

Primary scoring: Docker test pass counts.
Fallback scoring: LLM judge when both conditions score 0 on Docker.

Acceptance: Condition A > Condition B (memory provides meaningful benefit).
"""

import json
import subprocess
from typing import Dict, List, Optional, Any, Tuple

import litellm
from tqdm import tqdm

from ..types.schemas import MemGymInstance, LLM_JUDGE_SCHEMA
from ..llm.client import LLMClient
from ..llm.prompts import PATCH_GENERATION_SYSTEM, LLM_JUDGE_PROMPT

# Models that use internal reasoning tokens and don't support temperature
_REASONING_PREFIXES = ("gpt-5", "o1", "o3")


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _llm_call(
    messages: List[Dict[str, str]], model: str, max_tokens: int = 16000
) -> str:
    """Call LLM via litellm, handling reasoning-model quirks."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if not any(model.startswith(p) for p in _REASONING_PREFIXES):
        kwargs["temperature"] = 0.7
    response = litellm.completion(**kwargs)
    return response.choices[0].message.content.strip()


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text


# ---------------------------------------------------------------------------
# Patch generation
# ---------------------------------------------------------------------------

def _generate_patch(
    task_prompt: str,
    repo_context: Optional[Dict[str, str]],
    memory_files: Optional[Dict[str, str]],
    referenced_files: List[str],
    model: str,
) -> str:
    """Generate a patch using the Verifier LLM.

    Args:
        task_prompt: The vague task description
        repo_context: Dict of repo files (if available)
        memory_files: Dict of memory files (if condition A)
        referenced_files: File paths referenced
        model: LLM model to use

    Returns:
        Generated patch string
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": PATCH_GENERATION_SYSTEM},
    ]

    # Add repo context if available
    if repo_context:
        repo_parts = []
        total = 0
        for fp, content in repo_context.items():
            truncated = content[:15000] if len(content) > 15000 else content
            repo_parts.append(f"=== {fp} ===\n{truncated}")
            total += len(truncated)
            if total > 50000:
                break
        repo_block = "\n\n".join(repo_parts)
        messages.append({
            "role": "user",
            "content": f"Here are the repository source files:\n\n{repo_block}",
        })
        messages.append({
            "role": "assistant",
            "content": "I've read the repository files. What's the issue?",
        })

    # Add memory files if condition A
    if memory_files:
        mem_parts = []
        for fname, content in memory_files.items():
            mem_parts.append(f"=== {fname} ===\n{content}")
        mem_block = "\n\n".join(mem_parts)
        messages.append({
            "role": "user",
            "content": (
                "I also have these relevant notes and documents:\n\n"
                + mem_block
            ),
        })
        messages.append({
            "role": "assistant",
            "content": "I've reviewed the notes. What needs to be fixed?",
        })

    # Add task prompt
    ref_files = ", ".join(referenced_files) if referenced_files else "(not specified)"
    messages.append({
        "role": "user",
        "content": (
            f"{task_prompt}\n\nReferenced files: {ref_files}\n\n"
            "Generate the unified diff patch that fixes this problem. "
            "Output ONLY the diff:"
        ),
    })

    try:
        patch = _llm_call(messages, model, max_tokens=16000)
        return _strip_markdown_fences(patch)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Docker test scoring
# ---------------------------------------------------------------------------

def score_patch_docker(
    patch: str,
    instance: MemGymInstance,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Apply generated patch to Docker container, run tests.

    Args:
        patch: Generated unified diff patch
        instance: MemGymInstance with image_name and test info
        timeout: Docker command timeout

    Returns:
        Dict with passes, failures, total, error
    """
    result = {"passes": 0, "failures": 0, "total": 0, "error": None}

    if not patch.strip() or not instance.image_name:
        result["error"] = "empty patch or no Docker image"
        return result

    tests = instance.FAIL_TO_PASS[:5]
    if not tests:
        result["error"] = "no FAIL_TO_PASS tests"
        return result

    result["total"] = len(tests)
    container_name = None

    try:
        container_name = (
            f"memgym_verify_{instance.instance_id[:20].replace('/', '_')}"
        )
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, timeout=10,
        )

        # Start container
        start = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", container_name,
                "--network", "none",
                "-m", "4g",
                instance.image_name,
                "sleep", str(timeout + 60),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if start.returncode != 0:
            result["error"] = f"container start failed: {start.stderr[:200]}"
            return result

        # Write patch to container
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", delete=False
        ) as f:
            f.write(patch)
            patch_path = f.name

        subprocess.run(
            ["docker", "cp", patch_path, f"{container_name}:/tmp/fix.patch"],
            capture_output=True, timeout=10,
        )
        os.unlink(patch_path)

        # Apply patch (ignore errors — patch may not apply cleanly)
        subprocess.run(
            [
                "docker", "exec", container_name,
                "bash", "-c",
                "cd /testbed && git apply /tmp/fix.patch 2>/dev/null || "
                "patch -p1 < /tmp/fix.patch 2>/dev/null || true",
            ],
            capture_output=True, timeout=30,
        )

        # Run tests
        test_cmd = " ".join(tests)
        test_result = subprocess.run(
            [
                "docker", "exec", container_name,
                "bash", "-c",
                f"cd /testbed && python -m pytest {test_cmd} -x -v 2>&1 | tail -30",
            ],
            capture_output=True, text=True, timeout=timeout,
        )

        output = test_result.stdout + test_result.stderr
        # Count passes/failures from pytest output
        for line in output.split("\n"):
            line_lower = line.lower().strip()
            if " passed" in line_lower:
                import re

                m = re.search(r"(\d+) passed", line_lower)
                if m:
                    result["passes"] = int(m.group(1))
            if " failed" in line_lower:
                import re

                m = re.search(r"(\d+) failed", line_lower)
                if m:
                    result["failures"] = int(m.group(1))

    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)[:200]
    finally:
        if container_name:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True, timeout=10,
            )

    return result


# ---------------------------------------------------------------------------
# LLM judge scoring (fallback)
# ---------------------------------------------------------------------------

def score_patch_llm(
    generated_patch: str,
    gold_fix: str,
    model: str,
) -> float:
    """LLM judge fallback: score generated vs gold patch.

    Args:
        generated_patch: The generated patch
        gold_fix: The gold standard fix
        model: LLM model to use for judging

    Returns:
        Score from 0.0 to 1.0
    """
    if not generated_patch.strip():
        return 0.0

    judge_prompt = LLM_JUDGE_PROMPT.format(
        generated=generated_patch[:3000],
        gold_fix=gold_fix[:3000],
    )

    try:
        response = _llm_call(
            [
                {
                    "role": "system",
                    "content": "You are a code evaluation judge. Output only valid JSON.",
                },
                {"role": "user", "content": judge_prompt},
            ],
            model,
            max_tokens=4000,
        )
        response = _strip_markdown_fences(response)
        result = json.loads(response)
        return float(result.get("score", 0.0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Verification conditions
# ---------------------------------------------------------------------------

def verify_instance(
    instance: MemGymInstance,
    verifier_model: str,
    repo_context: Optional[Dict[str, str]] = None,
    docker_available: bool = False,
) -> Dict[str, Any]:
    """Run ablation verification on a single instance.

    Condition A: task_prompt + repo + memory_files → patch → score
    Condition B: task_prompt + repo → patch → score (no memory)

    Args:
        instance: MemGymInstance to verify
        verifier_model: LLM model for patch generation + judging
        repo_context: Dict of repo files
        docker_available: Whether Docker is available for test execution

    Returns:
        Dict with scoring details for both conditions
    """
    result = {
        "instance_id": instance.instance_id,
        # Condition A (with memory)
        "cond_a_patch": "",
        "cond_a_docker_passes": 0,
        "cond_a_llm_score": 0.0,
        # Condition B (no memory)
        "cond_b_patch": "",
        "cond_b_docker_passes": 0,
        "cond_b_llm_score": 0.0,
        # Overall
        "accepted": False,
        "method": "none",
    }

    # Condition A: with memory
    cond_a_patch = _generate_patch(
        task_prompt=instance.task_prompt,
        repo_context=repo_context,
        memory_files=instance.memory_files,
        referenced_files=[],
        model=verifier_model,
    )
    result["cond_a_patch"] = cond_a_patch

    # Condition B: no memory
    cond_b_patch = _generate_patch(
        task_prompt=instance.task_prompt,
        repo_context=repo_context,
        memory_files=None,  # no memory files
        referenced_files=[],
        model=verifier_model,
    )
    result["cond_b_patch"] = cond_b_patch

    # Docker scoring
    if docker_available:
        docker_a = score_patch_docker(cond_a_patch, instance)
        docker_b = score_patch_docker(cond_b_patch, instance)
        result["cond_a_docker_passes"] = docker_a["passes"]
        result["cond_b_docker_passes"] = docker_b["passes"]
        result["cond_a_docker_detail"] = docker_a
        result["cond_b_docker_detail"] = docker_b

    # LLM judge scoring (always run for record keeping)
    result["cond_a_llm_score"] = score_patch_llm(
        cond_a_patch, instance.gold_fix, verifier_model
    )
    result["cond_b_llm_score"] = score_patch_llm(
        cond_b_patch, instance.gold_fix, verifier_model
    )

    # Acceptance criteria
    result["accepted"] = is_accepted(result)
    if result["accepted"]:
        if (
            result["cond_a_docker_passes"] > 0
            and result["cond_b_docker_passes"] == 0
        ):
            result["method"] = "docker"
        else:
            result["method"] = "llm_judge"

    return result


def is_accepted(result: Dict) -> bool:
    """Check if instance passes acceptance criteria.

    Args:
        result: Verification result dict

    Returns:
        True if instance is accepted
    """
    # Docker-based (when available)
    if result.get("cond_a_docker_passes", 0) > 0 and result.get(
        "cond_b_docker_passes", 0
    ) == 0:
        return True

    # LLM judge fallback
    a_score = result.get("cond_a_llm_score", 0.0)
    b_score = result.get("cond_b_llm_score", 0.0)
    if a_score >= 0.4 and a_score > b_score + 0.30:
        return True

    return False


# ---------------------------------------------------------------------------
# Batch verification
# ---------------------------------------------------------------------------

def verify_all_instances(
    instances: List[MemGymInstance],
    verifier_model: str = "claude-opus-4-20250514",
    repo_contexts: Optional[Dict[str, Dict[str, str]]] = None,
    docker_available: bool = False,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Verify all instances.

    Args:
        instances: List of MemGymInstance objects
        verifier_model: LLM model for verification
        repo_contexts: Dict mapping instance_id -> {file_path: content}
        docker_available: Whether Docker is available
        show_progress: Show progress bar

    Returns:
        List of verification result dicts
    """
    results = []
    iterator = tqdm(instances, desc="Verifying") if show_progress else instances

    for instance in iterator:
        repo_ctx = None
        if repo_contexts:
            repo_ctx = repo_contexts.get(instance.instance_id, {})
            if not repo_ctx:
                repo_ctx = repo_contexts.get(instance.base_instance, {})

        result = verify_instance(
            instance=instance,
            verifier_model=verifier_model,
            repo_context=repo_ctx,
            docker_available=docker_available,
        )
        results.append(result)

    # Summary
    accepted = sum(1 for r in results if r["accepted"])
    by_docker = sum(1 for r in results if r.get("method") == "docker")
    by_llm = sum(1 for r in results if r.get("method") == "llm_judge")

    avg_a_llm = (
        sum(r["cond_a_llm_score"] for r in results) / len(results)
        if results
        else 0
    )
    avg_b_llm = (
        sum(r["cond_b_llm_score"] for r in results) / len(results)
        if results
        else 0
    )

    print(
        f"Verified {len(results)} instances: "
        f"{accepted} accepted ({accepted / len(results) * 100:.0f}%)"
    )
    print(f"  By Docker: {by_docker}, By LLM judge: {by_llm}")
    print(
        f"  Avg LLM scores: with_memory={avg_a_llm:.2f}, "
        f"without_memory={avg_b_llm:.2f}, gap={avg_a_llm - avg_b_llm:.2f}"
    )

    return results
