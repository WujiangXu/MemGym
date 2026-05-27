# Scripts

Developer and operations tooling — **not** part of the installed `memgym`
package and **not** user-facing tutorials (those live in [`../examples/`](../examples/)).

| Path | Purpose |
|------|---------|
| `setup_swebench.sh` | One-time SWE-bench environment setup. |
| `stage_hf_dataset.py` | Stage a local dataset for upload to the Hugging Face Hub. |
| `push_handoff_to_hf.sh` | Push handoff artifacts to the Hub. |
| `ops/` | Cluster/GPU operations (see below). |

## `ops/`

Infrastructure helpers for running training/eval on a cluster. These are
environment-specific and meant to be run by hand, not imported.

| Path | Purpose |
|------|---------|
| `ops/kill_gpu_procs.py` | Kill stray GPU processes. |
| `ops/kill_stale_workers.py` | Kill stale distributed workers. |
| `ops/kill_stale_torchrun.py` | Kill stale `torchrun` launchers. |
| `ops/cleanup_stale_torchelastic.py` | Clean up stale TorchElastic rendezvous state. |
| `ops/dump_attn_config.py` | Dump the resolved attention config (debug). |
| `ops/dump_rollout_config.py` | Dump the resolved rollout config (debug). |
| `ops/run_50_sequential.sh` | Submit the 50-instance sequential eval jobs to a remote host. |
