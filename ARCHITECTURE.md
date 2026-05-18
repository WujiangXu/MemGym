# MemGym Architecture

## Overview

MemGym is a modular framework for testing memory abilities of AI agents. It implements **Memory-Reasoning Separation** architecture.

```
+---------------------------------------------------------------------+
|                           MemGym                                     |
|  +-------------------+ +----------------+ +-------------------+     |
|  | Tau2BenchMemoryEnv| |  SWEMemoryEnv  | | WebArenaMemoryEnv |     |
|  +-------+-----------+ +-------+--------+ +---------+---------+     |
|          |                     |                     |               |
|  +-------v---------------------v---------------------v--------+      |
|  |                    Shared Components                       |      |
|  |  +--------------+  +--------------+  +------------------+  |      |
|  |  | MemoryModels |  | TokenTracker |  | TrajectoryRecorder|  |      |
|  |  | - PassThrough|  |              |  |                   |  |      |
|  |  | - Summarize  |  |              |  |                   |  |      |
|  |  +--------------+  +--------------+  +------------------+  |      |
|  +------------------------------------------------------------+      |
+----------------------------------------------------------------------+
```

## Memory-Reasoning Separation

```
Environment Observation
         |
         v
+------------------------+
|   Memory Model         |  Decides what to remember
|   - Store observation  |  Returns: MemoryAction
|   - Summarize context  |
+--------+---------------+
         | Context
         v
+------------------------+
|   Reasoning Model      |  Makes task decisions
|   - Receives context   |  Returns: ReasoningAction
+--------+---------------+
         | Action
         v
    Environment Step
```

## Core Components

### Memory Models

| Model | Description |
|-------|-------------|
| `PassThroughMemory` | Stores all observations (baseline) |
| `SummarizationMemory` | LLM-based compression when over token limit |

```python
from memgym.memory import get_memory_model

memory = get_memory_model("summarization", max_tokens=2000)
memory.process_observation("User said hello")
context = memory.get_context()
```

### Reasoning Models

| Model | Benchmark |
|-------|-----------|
| `Tau2BenchAgent` | tau2-bench dialogue (under `gym/tau2_bench/`) |
| `SWEAgentTracker` | SWE-bench coding |
| `WebArenaAgent` | WebArena-Infinity browser GUI |

### Memory-Aware Agent Wrapper (tau2-bench)

`MemoryManagerAdapter` (under `gym/tau2_bench/memory_adapter.py`) wraps tau2-bench
agents to add memory filtering:

- Inherits from `LLMSoloAgent` for orchestrator compatibility
- Intercepts `generate_next_message()` to apply context filtering via the
  modern `BaseMemoryManager.manage_context()` interface
- Symmetric: a parallel `UserMemoryManagerAdapter` wraps `UserSimulator` so
  the user-side memory can be ablated independently
- Maintains full history while providing filtered context to the agent
- Tracks compression stats per role (agent vs user)

### Environment Wrappers

```python
# Tau2-bench (Dialogue) — see gym/tau2_bench/
env = Tau2BenchMemoryEnv(
    domain="airline",
    task_id="book_flight_1",
    solo_mode=False,
    agent_llm="gpt-4o-mini",
)

# SWE-bench (Coding)
env = SWEMemoryEnv(
    dataset="princeton-nlp/SWE-bench_Lite",
    use_context_management=True
)
```

## Trajectory Structure

```json
{
  "episodes": [{
    "steps": [{
      "observation": "...",
      "action": "...",
      "memory_action": {"action_type": "summarize", "metadata": {...}},
      "context": {"content": "...", "metadata": {"has_summary": true}},
      "reasoning_action": {"action": "...", "metadata": {...}}
    }]
  }]
}
```

## File Structure

```
MemGym/
├── src/memgym/
│   ├── envs/           # Base env interfaces + lazy re-exports
│   ├── memory/         # Memory managers (PassThrough, LLMSummarizing, ...)
│   ├── agents/         # Base agent interface + non-gym agents
│   ├── runners/        # SWERunner, Tau2BenchRunner, ...
│   ├── gym/            # Per-track gyms (each self-contained)
│   │   ├── swe_bench/
│   │   ├── webarena/
│   │   └── tau2_bench/  # env, agent, memory_adapter, evaluate, install
│   ├── adapter/        # TrajectoryRecorder + utilities
│   └── utility/        # Token tracking, etc.
├── scripts/            # Evaluation + debug scripts
└── tests/              # Test suite
```

## Custom Memory Models

```python
from memgym.memory.base import BaseMemoryModel, register_memory_model

class MyMemory(BaseMemoryModel):
    def process_observation(self, obs, metadata=None):
        # Handle new observation
        return MemoryAction(action_type="store", metadata={})

    def get_context(self):
        return Context(content="...", metadata={"has_summary": False})

    def should_use_filtered_context(self):
        return self._token_count > self.config["max_tokens"]

register_memory_model("my_memory", MyMemory)
```

## Design Principles

1. **Memory-Reasoning Separation**: Memory constructs context; reasoning makes decisions
2. **Gym-Compatible Interface**: Standard reset()/step() API
3. **Observable Behavior**: All memory/reasoning actions tracked in trajectory
4. **Type Compatibility**: Wrappers inherit from base classes for isinstance() checks
