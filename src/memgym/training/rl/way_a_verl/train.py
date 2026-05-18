"""CLI wrapper for Way A — agent-RL on SWE-Gym via VeRL.

Three responsibilities, all minimal:
    1. Load instances via the same helpers Way B uses
       (`_filter_by_subcategory`, `_select_diverse_by_repo`) so Way A
       and Way B see the same training distribution.
    2. Materialize them as a VeRL-shaped parquet (one row per instance,
       columns prompt/instance_id/ground_truth) since VeRL's PPO trainer
       reads parquet, not python dicts.
    3. Write a temporary tool-config YAML pointing at our `MemGymSWETool`
       and shell out to VeRL's `ppo_trainer.main` with our full-FT
       overlay.

Flags mirror Way B's `train_rl_online.py` for surface-level
consistency. The `--use-lora` flag is accepted but no-ops on this verl
0.7.1 pin (the optim dataclass rejects an `optim.lora` block, see
`config/way_a_qwen3_8b.yaml` for the parking note); friends with
big-GPU clusters run full-FT, which is what this scaffold trains.
`--num-gpus 8` sets `trainer.n_gpus_per_node`.

Why this is a thin wrapper rather than re-implementing the GRPO step:
VeRL handles the FSDP-trainer + vLLM-rollout co-location via Ray
WorkerGroups. That's the integration we adopt, and it's exactly what
made the bare `train_rl_online.py --mode agent` an unimplementable
NotImplementedError before this revision.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("way_a_verl.train")


def _instances_to_parquet(instances: List[Dict[str, Any]], out_path: Path) -> None:
    """Write VeRL-shaped parquet — one row per instance.

    VeRL's `RLHFDataset` reads `prompt_key` from the row plus any extra
    columns passed through to the `AgentLoop` via `kwargs`. We pack
    `instance_id` and `ground_truth` so `MemGymAgentLoop.run` can
    forward them to `MemGymSWETool.create()`.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for inst in instances:
        instance_id = inst["instance_id"]
        ground_truth = inst.get("patch") or inst.get("ground_truth") or ""
        # tools_kwargs is the verl 0.7.1 hand-off into our tool's create():
        # ToolAgentLoop._call_tool reads tools_kwargs[tool_name]["create_kwargs"]
        # and passes it as a kwarg, so MemGymSWETool.create can resolve the
        # dataset instance id and docker image without re-reading the dataset.
        tools_kwargs = {
            "MemGymSWETool": {
                "create_kwargs": {
                    "instance_id": instance_id,
                    "image_name": inst.get("image_name") or "",
                    "ground_truth": ground_truth,
                    "repo": inst.get("repo", ""),
                }
            }
        }
        rows.append({
            "prompt": [{
                "role": "user",
                "content": (
                    "You are a SWE-Bench coding agent. Solve the issue "
                    "described in the working directory. Use bash commands "
                    "between ```bash ... ``` blocks. Submit when done."
                ),
            }],
            "instance_id": instance_id,
            "ground_truth": ground_truth,
            # `data_source` is verl 0.7.1's canonical dispatch key on the
            # naive reward manager (`reward_loop/reward_manager/naive.py`
            # reads `data_item.non_tensor_batch["data_source"]`). With one
            # custom reward fn the value is cosmetic but the field must
            # exist; V2o crashed here.
            "data_source": "swe-gym",
            # `reward_model` is the next required nested field in verl
            # 0.7.1's naive reward manager (line 43:
            # `data_item.non_tensor_batch["reward_model"]["ground_truth"]`).
            # V2p crashed here. Our compute_reward reads ground_truth from
            # extra_info, but verl unpacks reward_model.ground_truth into
            # the kwargs it passes us — keep both for shape parity.
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            # extra_info is the verl-canonical channel for the reward fn
            # to read instance metadata. ``tools_kwargs`` MUST live inside
            # extra_info — verl 0.7.1's ``ToolAgentLoop._call_tool`` looks
            # it up via ``extra_info.get("tools_kwargs", {})``. A previous
            # revision wrote it as a top-level parquet column; the tool's
            # ``create()`` then ran with empty kwargs (no instance_id, no
            # image_name), silently breaking every rollout. Audit H1.
            "extra_info": {
                "instance_id": instance_id,
                "ground_truth": ground_truth,
                "tools_kwargs": tools_kwargs,
            },
        })
    table = pa.Table.from_pylist(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)


def _write_tool_config(tmp_dir: Path, dataset: str, docker_image_source: str,
                       max_steps: int) -> Path:
    """Materialize verl 0.7.1 tool-config YAML pointing at MemGymSWETool.

    Schema matches ``initialize_tools_from_config`` (verl 0.7.1
    ``tools.utils.tool_registry``): ``class_name`` is the fully-qualified
    Python class path, ``config.type`` selects the ``ToolType`` branch
    (``native`` here — ``mcp`` is for Model Context Protocol tools).
    """
    cfg = {
        "tools": [
            {
                "class_name": "memgym.training.rl.way_a_verl.tool.MemGymSWETool",
                "config": {
                    "type": "native",
                    "dataset": dataset,
                    "docker_image_source": docker_image_source,
                    "max_steps": max_steps,
                },
                "tool_schema": {
                    "type": "function",
                    "function": {
                        "name": "MemGymSWETool",
                        "description": "Run a bash action in the SWE-Bench instance.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string"},
                            },
                            "required": ["action"],
                        },
                    },
                },
            }
        ]
    }
    import yaml
    out = tmp_dir / "tool_config.yaml"
    with out.open("w") as f:
        yaml.safe_dump(cfg, f)
    return out


def _write_agent_loop_config(tmp_dir: Path) -> Path:
    """Register MemGymToolAgentLoop with verl's actor-side agent loop registry.

    verl 0.7.1's `AgentLoopWorker.__init__` (agent_loop.py:432-437) reads
    `rollout.agent.agent_loop_config_path` and inserts each entry into
    `_agent_loop_registry` *on the Ray actor*. This is the only mechanism
    that crosses the actor boundary — module-import side-effects from the
    launcher process don't reach the actors. Without this YAML the actor
    only sees verl's built-in `single_turn_agent` and `tool_agent` and
    `_run_agent_loop` asserts on lookup of `memgym_tool`.

    Format: a sequence of mappings; each must have `name` (the registry
    key) and `_target_` (Hydra fqdn used by `instantiate(config=...)` —
    importing the target module is what populates the @register decorator
    on the actor side as well).
    """
    cfg = [
        {
            "name": "memgym_tool",
            "_target_": (
                "memgym.training.rl.way_a_verl.memgym_agent_loop."
                "MemGymToolAgentLoop"
            ),
        },
    ]
    import yaml
    out = tmp_dir / "agent_loop_config.yaml"
    with out.open("w") as f:
        yaml.safe_dump(cfg, f)
    return out


def _load_instances(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Same helpers Way B uses — distribution parity matters for H.4'."""
    import random

    from datasets import load_dataset
    from memgym.gym.swe_bench.evaluate import DATASET_MAP
    from memgym.training.scripts.eval_memory_ab import (
        _filter_by_subcategory,
        _select_diverse_by_repo,
    )

    # Resolve project aliases (`swe-gym`, `swe-smith`, `lite`, `verified`,
    # `full`) to their Hub paths. Pass-through anything that's already a
    # Hub-shaped `org/name` string.
    dataset_path = DATASET_MAP.get(args.dataset, args.dataset)
    dataset = load_dataset(dataset_path, split=args.split)
    instances = list(dataset)
    if args.subcategory_include:
        allowed = [s.strip() for s in args.subcategory_include.split(",") if s.strip()]
        instances = _filter_by_subcategory(instances, allowed)
    if args.diverse_by_repo and args.n_instances < len(instances):
        instances = _select_diverse_by_repo(instances, args.n_instances)
    elif args.n_instances < len(instances):
        random.Random(args.seed).shuffle(instances)
        instances = instances[: args.n_instances]
    return instances


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=None,
        help="Path to way_a_qwen3_8b.yaml. Default: bundled config.",
    )
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--dataset", default="swe-gym")
    parser.add_argument("--docker-image-source", default="swe-gym")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-instances", type=int, default=200)
    parser.add_argument("--rollouts-per-instance", type=int, default=8)
    parser.add_argument(
        "--allow-degenerate-grpo", action="store_true",
        help=(
            "Allow ``--rollouts-per-instance < 4``. GRPO advantage is ±std of "
            "the rollout group; with N<4 the std collapses on tied rewards "
            "and the gradient becomes degenerate. Smoke runs that just want "
            "to exercise the pipeline can pass this; production runs should "
            "not."
        ),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=75)
    parser.add_argument("--subcategory-include", default=None)
    parser.add_argument("--diverse-by-repo", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument("--use-lora", dest="use_lora", action="store_true", default=True)
    lora_group.add_argument("--full-finetune", dest="use_lora", action="store_false")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Forwarded to trainer.n_gpus_per_node in the VeRL config.")
    parser.add_argument(
        "--model", default=None,
        help=(
            "Override actor_rollout_ref.model.path. Default: YAML's "
            "Qwen/Qwen3-8B-Base. Pass a smaller base (e.g. "
            "Qwen/Qwen3-1.7B-Base) to validate the pipeline on A100-40GB "
            "without hitting the 8B + FSDP + vLLM memory ceiling."
        ),
    )
    parser.add_argument(
        "--summarize-max-messages", type=int, default=None,
        help=(
            "Override the message-count trigger for self-summarize. Default "
            "100 matches trajectory-collection trigger. Plumbed via env var "
            "MEMGYM_SUMMARIZE_MAX_MESSAGES because verl 0.7.1's MultiTurnConfig "
            "is a strict dataclass that rejects unknown YAML keys. Smoke runs "
            "on A100-40GB at max_prompt_length=8192 should pass a smaller "
            "value (e.g. 20) so the self-summarize path actually fires "
            "inside the 75-turn cap."
        ),
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Audit M3: GRPO advantage is ±std of the rollout group; with N<4 the
    # std collapses on tied rewards and the gradient becomes degenerate.
    # Smokes that just want to exercise the pipeline can opt in via
    # --allow-degenerate-grpo; production runs should not.
    if args.rollouts_per_instance < 4 and not args.allow_degenerate_grpo:
        logger.error(
            "--rollouts-per-instance=%d is degenerate for GRPO (need >=4 to "
            "produce stable advantages on tied rewards). Pass "
            "--allow-degenerate-grpo to override for smoke runs.",
            args.rollouts_per_instance,
        )
        return 2

    # Resolve the bundled config when --config is omitted.
    if args.config is None:
        args.config = str(
            Path(__file__).parent / "config" / "way_a_qwen3_8b.yaml"
        )

    instances = _load_instances(args)
    if not instances:
        logger.error("No instances after filtering — check --dataset / --subcategory-include")
        return 2
    logger.info("Loaded %d instances", len(instances))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Audit H3: write all verl artifacts under save_dir rather than a
    # TemporaryDirectory. Two reasons: (1) Ray actors read tool_config /
    # agent_loop_config at rollout time — if ppo_main raises before the
    # actors load them, a TemporaryDirectory disappears together with
    # the launcher process and we lose the post-mortem evidence;
    # (2) friends repro'ing a smoke want the parquet + YAMLs reachable
    # for inspection alongside the checkpoint.
    artifacts_dir = save_dir / "verl_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_parquet = artifacts_dir / "train.parquet"
    _instances_to_parquet(instances, train_parquet)
    tool_cfg_path = _write_tool_config(
        artifacts_dir, args.dataset, args.docker_image_source, args.max_steps,
    )
    agent_loop_cfg_path = _write_agent_loop_config(artifacts_dir)

    # Hand to verl via Hydra-style overrides. We don't import
    # verl.trainer at module top so the rest of the package stays
    # installable without verl.
    import os as _os

    import verl as _verl_pkg  # type: ignore
    from hydra import compose, initialize_config_dir  # type: ignore
    from hydra.core.global_hydra import GlobalHydra  # type: ignore
    from omegaconf import OmegaConf  # type: ignore
    from verl.trainer.main_ppo import main as ppo_main  # type: ignore

    # Side-effect import: registers
    #   - BackticksToolParser under ToolParser._registry["backticks"]
    #   - MemGymToolAgentLoop under verl's _agent_loop_registry["memgym_tool"]
    # both before AgentLoopWorker calls hydra.utils.instantiate on the
    # agent_loop_config keyed by ``rollout.agent.default_agent_loop``.
    from memgym.training.rl.way_a_verl import agent_loop as _mem_agent_loop  # noqa: F401
    from memgym.training.rl.way_a_verl import memgym_agent_loop as _mem_loop  # noqa: F401

    # verl's `ppo_trainer.yaml` uses Hydra `defaults:` to pull in the
    # legacy_reward_impl / reward / actor / rollout component configs.
    # `OmegaConf.load()` doesn't run Hydra's defaults composition, so
    # we have to use the Compose API to build a complete cfg before
    # overlaying our profile YAML. Otherwise main_ppo trips on
    # `config.reward_model.num_workers` not existing.
    verl_cfg_dir = _os.path.join(_os.path.dirname(_verl_pkg.__file__),
                                 "trainer", "config")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=verl_cfg_dir):
        cfg = compose(config_name="ppo_trainer")
    # Overlay our profile YAML on top of verl's resolved defaults.
    # struct=False so we can ADD keys verl doesn't ship in its
    # default schema (`actor.optim.lora`, `rollout.multi_turn`,
    # `rollout.extra_body`, custom_reward_function). With struct=True,
    # OmegaConf rejects unknown keys.
    OmegaConf.set_struct(cfg, False)
    overlay = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, overlay)
    cfg.data.train_files = str(train_parquet)
    cfg.data.val_files = str(train_parquet)  # smoke uses train; friends override
    cfg.trainer.default_local_dir = str(save_dir)
    cfg.trainer.total_epochs = args.epochs
    cfg.trainer.n_gpus_per_node = args.num_gpus
    if args.model is not None:
        cfg.actor_rollout_ref.model.path = args.model
        logger.info("Overriding actor_rollout_ref.model.path = %s", args.model)
    cfg.actor_rollout_ref.rollout.n = args.rollouts_per_instance
    cfg.actor_rollout_ref.rollout.multi_turn.tool_config_path = str(tool_cfg_path)
    cfg.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = args.max_steps
    # Tell ToolAgentLoop to use our backticks parser. The "format"
    # key is what verl looks up via ToolParser.get_tool_parser(name).
    cfg.actor_rollout_ref.rollout.multi_turn.format = "backticks"

    # Use the self-summarizing agent loop. ``memgym_tool`` is registered
    # by ``memgym.training.rl.way_a_verl.memgym_agent_loop`` (imported
    # above as a side effect). It subclasses verl's stock ToolAgentLoop
    # and inserts a self-summarize step when ``len(messages)`` exceeds
    # ``multi_turn.summarize_max_messages`` (mirrors the trajectory-
    # collection condenser in ``llm_summarizing.py``). The same trainable
    # Qwen3-8B serves as both reasoning policy AND summarizer; summary
    # tokens carry RL gradient (response_mask=1) so GRPO learns the
    # compaction policy end-to-end. Friends who want vanilla verl tool
    # behavior can override this back to "tool_agent".
    if "agent" not in cfg.actor_rollout_ref.rollout:
        cfg.actor_rollout_ref.rollout.agent = OmegaConf.create({})
    cfg.actor_rollout_ref.rollout.agent.default_agent_loop = "memgym_tool"
    # Tell each Ray AgentLoopWorker actor to load our agent loop YAML.
    # Without this, only verl's built-in `single_turn_agent` and
    # `tool_agent` are in the actor's registry and `memgym_tool` lookup
    # fails (V4w crash). Setting this knob makes verl call
    # `hydra.utils.instantiate(config={'_target_': ...})` which imports
    # the module fqdn on the actor side, triggering @register("memgym_tool").
    cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(
        agent_loop_cfg_path
    )

    # Audit H2: verl 0.7.1's MultiTurnConfig is a strict dataclass and
    # we can't stash summarize knobs under multi_turn. Earlier revision
    # plumbed them via launcher os.environ + PPO_RAY_RUNTIME_ENV mutation;
    # both are bypassed by Ray-actor isolation. The launcher's os.environ
    # does NOT cross into Ray actor processes, AND verl's
    # ``get_ppo_ray_runtime_env`` (verl/trainer/constants_ppo.py:38-54)
    # actively pops any key from ``runtime_env.env_vars`` that's already
    # set in launcher os.environ, on the (incorrect for Ray actors)
    # assumption that env vars inherit. Net effect: actors saw nothing.
    #
    # Fix: inject directly into ``cfg.ray_kwargs.ray_init.runtime_env.
    # env_vars``, which verl's launcher passes verbatim to ``ray.init``.
    # That dict is what Ray ships to each actor's process environment.
    if args.summarize_max_messages is not None:
        if "ray_kwargs" not in cfg:
            cfg.ray_kwargs = OmegaConf.create({})
        if "ray_init" not in cfg.ray_kwargs:
            cfg.ray_kwargs.ray_init = OmegaConf.create({})
        if "runtime_env" not in cfg.ray_kwargs.ray_init:
            cfg.ray_kwargs.ray_init.runtime_env = OmegaConf.create({})
        if "env_vars" not in cfg.ray_kwargs.ray_init.runtime_env:
            cfg.ray_kwargs.ray_init.runtime_env.env_vars = OmegaConf.create({})
        cfg.ray_kwargs.ray_init.runtime_env.env_vars["MEMGYM_SUMMARIZE_MAX_MESSAGES"] = str(
            args.summarize_max_messages
        )
        logger.info(
            "Overriding MEMGYM_SUMMARIZE_MAX_MESSAGES = %d (smoke override; "
            "propagated to Ray actors via cfg.ray_kwargs.ray_init.runtime_env)",
            args.summarize_max_messages,
        )

    # verl 0.7.1's `FSDPOptimizerConfig` is a strict dataclass and
    # rejects any unknown kwarg — including `lora`. The legacy
    # `optim.lora` block (kept in the YAML for older verl) must be
    # popped before Hydra `instantiate()` runs validate_config, or
    # the trainer crashes with "got an unexpected keyword argument
    # 'lora'" at startup. Friends with full-cluster GPUs run
    # full-FT anyway; LoRA support on this verl pin is parked.
    if "lora" in cfg.actor_rollout_ref.actor.optim:
        del cfg.actor_rollout_ref.actor.optim.lora
    if args.use_lora:
        logger.warning(
            "verl 0.7.1 does not accept LoRA via actor.optim.lora; "
            "running full fine-tune. Re-launch with --full-finetune "
            "to silence this warning.",
        )

    # All Way A runs on this verl pin are full-FT (see optim block
    # parking note above). Keep the field stable so checkpoint dirs
    # land at one path even if friends flip --use-lora by habit.
    cfg.trainer.experiment_name = "qwen3_8b_full_ft"

    # verl 0.7.1's `get_custom_reward_fn` calls `load_module(path)` which
    # is `importlib.util.spec_from_file_location` — strict file path with
    # .py extension, NOT a Python dotted module name. The YAML carries
    # the dotted form for readability; resolve to an absolute file path
    # here so it works regardless of where the package is installed.
    from memgym.training.rl.way_a_verl import reward as _reward_mod
    cfg.custom_reward_function.path = _reward_mod.__file__

    logger.info(
        "Launching VeRL PPO trainer: profile=full_ft, n_gpus=%d, n_instances=%d, n_rollouts=%d",
        args.num_gpus, len(instances), args.rollouts_per_instance,
    )
    ppo_main(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
