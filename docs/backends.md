# LLM backends

MemGym calls every LLM through [litellm](https://github.com/BerriAI/litellm),
so any provider with an OpenAI-compatible or LiteLLM-supported endpoint works.
This page documents the three configurations the paper and audit have
exercised end-to-end.

## 1. Direct OpenAI / OpenAI-compatible

```bash
export OPENAI_API_KEY=sk-...
# (Optional) override the endpoint for a local server / proxy / non-OpenAI vendor.
export OPENAI_API_BASE=https://api.openai.com/v1
```

Model strings: `openai/gpt-4o-mini`, `openai/gpt-4o`, etc. — pass to
`--model` / `--policy_model` / `--worker-model` etc.

## 2. OpenAI-compatible passthrough (e.g. internal Llama / Metagen gateway)

For an internal gateway that re-exposes OpenAI-compatible endpoints with
a tenant key, set both the key and the base URL:

```bash
export OPENAI_API_KEY="$YOUR_GATEWAY_KEY"
export OPENAI_API_BASE='https://api.example.com/v1'   # gateway-specific
export OPENAI_BASE_URL="$OPENAI_API_BASE"

# Two switches the GPT-5 family in particular needs through MemGym:
export LITELLM_DROP_PARAMS=True       # GPT-5 rejects temperature=0.0
export MSWEA_COST_TRACKING=ignore_errors  # mini-swe-agent cost map noise
```

> Never commit the gateway key. Keep it in a gitignored `.env`
> (the repo's `.gitignore` already excludes `.env` / `.env.local`),
> or pass per-command from your shell.

Model string format depends on the gateway's catalog — pass the gateway-
specific name to MemGym CLIs (e.g. `--model openai/<gateway-model-id>`).

## 3. AWS Bedrock (for the paper's Claude reasoners)

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-west-2          # or whichever region serves your model
# Optional: explicit profile
# export AWS_PROFILE=memgym
```

Then pass a Bedrock model id, e.g.
`--policy_model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`. Make
sure `boto3` is installed (it ships with `[webarena]`).

---

## Track-specific notes

| Track | Backend gotchas |
|---|---|
| SWE-Gym local-env diagnostic | needs `MSWEA_COST_TRACKING=ignore_errors` to silence mini-swe-agent's cost-map fetch warnings when offline |
| SWE-Gym official re-eval | `docker run` against `swebench/sweb.eval.*` images — make sure `registry-1.docker.io` is reachable (or use a local mirror) |
| MemGym-DR / pre-screen | GPT-5 family rejects `temperature=0.0` from `LITELLM_DROP_PARAMS=True` is mandatory |
| tau2-bench + OpenHands | their pinned `litellm` versions conflict — use separate venvs (`tau2` needs `litellm<1.82.7,>=1.80.15`; OpenHands pins `litellm==1.84.1`) |
| HippoRAG (`[hipporag]`) | pins `litellm==1.73.1`, `vllm==0.6.6.post1`, `transformers==4.45.2`; install in a venv that does NOT have `[tau2]`/`[openhands]`/`[rl-way-a]` |

For local-env Podman setups (Docker-compatible API on Meta-style devservers):

```bash
systemctl --user enable --now podman.socket
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock
```

Official SWE-bench eval then reaches the harness; image pulls still require
`registry-1.docker.io` access.
