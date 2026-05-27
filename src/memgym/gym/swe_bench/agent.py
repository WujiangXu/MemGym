"""
Memory-aware wrapper for mini-swe-agent.

Wraps mini-swe-agent's DefaultAgent to add memory filtering capability.
Compatible with mini-swe-agent v2.2.4+.
"""

import re
from typing import Any, Dict, List, Optional

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded


def _extract_thought(content: str) -> str:
    """Extract THOUGHT text from agent response (before triple-backtick action block)."""
    # mini-swe-agent format: THOUGHT: ...\n```bash\n...\n```
    match = re.search(r'(?:THOUGHT:\s*)(.*?)(?=```|$)', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: everything before first code block
    idx = content.find('```')
    if idx > 0:
        return content[:idx].strip()
    return content.strip()[:500]


def _extract_action(content: str) -> str:
    """Extract bash action from triple-backtick block in agent response."""
    match = re.search(r'```(?:bash)?\s*\n(.*?)\n```', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


class MemoryAwareSWEAgent(DefaultAgent):
    """
    Wraps mini-swe-agent's DefaultAgent with memory filtering.

    Only difference: Filters message history before LLM queries.
    Everything else: Identical to mini-swe-agent.

    Architecture:
    - Maintains full message history in self.messages
    - Before LLM query, asks memory manager: manage_context()
    - Uses filtered context (view) for LLM query
    - On context window errors, triggers request_condensation() fallback
    - Tracks filtering statistics
    """

    def __init__(self, model, env, memory_manager, verbose=False, memory_gate=None, **kwargs):
        """
        Initialize memory-aware SWE agent.

        Args:
            model: LLM model (from minisweagent.models)
            env: Environment (from minisweagent.environments)
            memory_manager: MemGym memory manager (BaseMemoryManager)
            verbose: Print step-by-step progress
            memory_gate: Optional gate object exposing ``.gate(...)``. When
                set, every compaction event is scored before the compressed
                context reaches the LLM; rejections trigger one bounded
                re-compaction retry and fall back soft on second rejection.
            **kwargs: Additional config for DefaultAgent
        """
        super().__init__(model, env, **kwargs)
        self.memory_manager = memory_manager
        self.memory_gate = memory_gate
        self.verbose = verbose
        self._processed_msg_count = 0  # Track how many messages we've processed

        # Tracking
        self._wrapper_stats = {
            "total_queries": 0,
            "times_filtered": 0,
            "times_pass_through": 0,
            "compression_ratios": [],
            "gate_events": [],  # one dict per compaction when memory_gate is set
            # High-water LLM-context tokens from every memory pass.
            # `max_original_tokens` is the pre-filter full conversation size
            # (what the LLM would see without memory); `max_filtered_tokens`
            # is the post-filter size actually sent to the model. Both
            # support the A/B token-cost criterion.
            "max_original_tokens": 0,
            "max_filtered_tokens": 0,
        }
        self._training_steps: List[Dict] = []
        self._pending_step: Optional[Dict] = None  # Step awaiting observation from next query
        self._task_description: str = ""

    def run(self, task: str = "", **kwargs) -> dict:
        """
        Run step() until agent is finished. Returns dictionary with exit_status, submission keys.

        Override to add memory filtering reset and verbose logging.
        Matches DefaultAgent.run() API from mini-swe-agent v2.2.4+.
        """
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self._processed_msg_count = 0  # Reset for new episode
        self.memory_manager.reset()  # Reset memory for new episode
        self._training_steps = []  # Reset training trajectory
        self._pending_step = None
        self._task_description = task

        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )

        step_num = 0
        while True:
            try:
                step_num += 1
                if self.verbose:
                    print(f"\n--- Step {step_num} ---")

                self.step()

                if self.verbose:
                    self._print_step_info()

            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
                if self.verbose:
                    exit_status = "Unknown"
                    if e.messages:
                        exit_status = e.messages[0].get("extra", {}).get("exit_status", "Unknown")
                    print(f"\nAgent finished: {exit_status}")

            except Exception as e:
                self.handle_uncaught_exception(e)
                raise

            if self.messages[-1].get("role") == "exit":
                break

        return self.messages[-1].get("extra", {})

    def _print_step_info(self):
        """Print verbose step information."""
        # Show last assistant message (truncated)
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                content = str(msg.get("content", ""))[:200]
                print(f"Assistant: {content}...")
                break
        # Show last observation (truncated) — can be role="tool" or role="user"
        for msg in reversed(self.messages):
            role = msg.get("role", "")
            if role in ("tool", "user"):
                content = str(msg.get("content", ""))[:200]
                if "returncode" in content or "output" in content:
                    print(f"Observation: {content}...")
                    break
        print(f"Memory stats: {self._wrapper_stats['total_queries']} queries, "
              f"{self._wrapper_stats['times_filtered']} filtered")

    def query(self) -> dict:
        """
        Query the model with memory filtering.

        Override of DefaultAgent.query() to add memory filtering.
        On context window errors, triggers request_condensation() fallback
        (equivalent to OpenHands CondensationRequestAction).
        """
        # Check limits (same as mini-swe-agent v2.2.4)
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        self.n_calls += 1

        self._wrapper_stats["total_queries"] += 1

        # Get current observation (latest messages since last processing)
        new_messages = self.messages[self._processed_msg_count:]
        current_obs = new_messages[-1] if new_messages else None
        self._processed_msg_count = len(self.messages)

        # Use memory manager to filter context
        history = self.messages[:-1] if self.messages else []
        filtered_context = self.memory_manager.manage_context(
            original_context=history,
            current_observation=current_obs,
            metadata={"step": self._wrapper_stats["total_queries"]}
        )

        # Get the view (condensed or not) — all strategies return messages directly
        messages_to_send = filtered_context.content
        was_compacted = filtered_context.metadata.get("was_compacted", False)

        # Repair broken tool_call/tool_result pairs after condensation
        if was_compacted and isinstance(messages_to_send, list):
            from memgym.memory.base import repair_tool_call_pairs
            messages_to_send = repair_tool_call_pairs(messages_to_send)

        # World-model gate: score every compaction before the LLM sees the
        # compressed context. On reject, request harder condensation for
        # the NEXT step (via memory_manager.request_condensation()) and
        # fail-soft within THIS step — shipping the rejected compression
        # plus a log entry. Re-running `manage_context` in-place would
        # double-append `current_observation` on SummarizationMemory, so
        # retrying within the step is out. The plan's "bound retries at 1"
        # is satisfied: one attempt per step, no loops, always-forward.
        # `proposed_response` is the same LLM call the agent was about to
        # make; reusing it saves the redundant call below.
        gate_response: Optional[Dict[str, Any]] = None
        if was_compacted and self.memory_gate is not None:
            gate_result = self.memory_gate.gate(
                agent_model=self.model,
                full_history=self.messages,
                compressed_history=messages_to_send,
                step=self._wrapper_stats["total_queries"],
            )
            event = {
                "step": self._wrapper_stats["total_queries"],
                "decision": gate_result["decision"],
                "prob_safe": gate_result["prob_safe"],
                "latency_ms": gate_result["latency_ms"],
                "fail_soft": False,
            }
            gate_response = gate_result["proposed_response"]
            if not gate_result["decision"]:
                event["fail_soft"] = True
                if hasattr(self.memory_manager, "request_condensation"):
                    self.memory_manager.request_condensation()
                if self.verbose:
                    print(
                        f"  [Gate] reject p_safe={gate_result['prob_safe']:.3f} "
                        f"< {self.memory_gate.threshold:.3f} — fail-soft; "
                        f"harder condensation queued for next step."
                    )
            self._wrapper_stats["gate_events"].append(event)

        if was_compacted:
            # Calculate compression
            original_tokens = self.memory_manager.count_tokens(self.messages)
            filtered_tokens = self.memory_manager.count_tokens(messages_to_send)
            if filtered_tokens > 0:
                ratio = original_tokens / filtered_tokens
                self._wrapper_stats["compression_ratios"].append(ratio)

            if self.verbose:
                print(f"  [Memory] Filtering: {len(self.messages)} -> {len(messages_to_send)} messages")

            self._wrapper_stats["times_filtered"] += 1
        else:
            self._wrapper_stats["times_pass_through"] += 1

        # Query LLM with context window error fallback. When the gate ran
        # on this compaction its `proposed_response` is the same call we'd
        # make here, so we reuse it rather than doubling agent-model cost.
        try:
            if gate_response is not None:
                response = gate_response
            else:
                response = self.model.query(messages_to_send)
        except Exception as e:
            error_str = str(e).lower()
            is_context_error = (
                "context" in error_str
                or "token" in error_str
                or "prompt is too long" in error_str
                or "context length exceeded" in error_str
                or "please reduce the length" in error_str
            )

            if is_context_error and hasattr(self.memory_manager, 'request_condensation'):
                # Equivalent to OpenHands CondensationRequestAction
                if self.verbose:
                    print(f"  [Memory] Context window error, requesting condensation...")
                self.memory_manager.request_condensation()

                # Re-run manage_context with forced condensation
                filtered_context = self.memory_manager.manage_context(
                    original_context=history,
                    current_observation=current_obs,
                    metadata={"step": self._wrapper_stats["total_queries"]}
                )
                messages_to_send = filtered_context.content

                if self.verbose:
                    print(f"  [Memory] Retrying with {len(messages_to_send)} messages")

                response = self.model.query(messages_to_send)
                self._wrapper_stats["times_filtered"] += 1
            else:
                raise

        # Track cost and add response to history (same as mini-swe-agent v2.2.4)
        self.cost += response.get("extra", {}).get("cost", 0.0)
        self.add_messages(response)

        # Collect training step (observation is deferred — filled in next query)
        # Fill in observation for the PREVIOUS step from current_obs
        if self._pending_step is not None and current_obs and isinstance(current_obs, dict):
            self._pending_step["observation"] = str(current_obs.get("content", ""))[:500]
            self._training_steps.append(self._pending_step)

        content = str(response.get("content", ""))

        # Always record per-step memory stats (not just when compacted)
        orig_toks = int(filtered_context.metadata.get("original_tokens", 0) or 0)
        filt_toks = int(filtered_context.metadata.get("tokens", 0) or 0)
        memory_info = {
            "original_msgs": len(self.messages),
            "filtered_msgs": len(messages_to_send),
            "original_tokens": orig_toks,
            "filtered_tokens": filt_toks,
            "compression_ratio": filtered_context.metadata.get("compression_ratio", 1.0),
            "was_compacted": was_compacted,
        }
        if orig_toks > self._wrapper_stats["max_original_tokens"]:
            self._wrapper_stats["max_original_tokens"] = orig_toks
        if filt_toks > self._wrapper_stats["max_filtered_tokens"]:
            self._wrapper_stats["max_filtered_tokens"] = filt_toks

        # Add condensation-specific info when available
        cond_event = filtered_context.metadata.get("condensation_event")
        if was_compacted and cond_event:
            memory_info["summary"] = cond_event.get("summary_text")
            memory_info["view_size"] = len(messages_to_send)
            memory_info["forgotten"] = cond_event.get("forgotten_count", 0)
            memory_info["summarizer_prompt_tokens"] = cond_event.get("summarizer_prompt_tokens", 0)
            memory_info["summarizer_completion_tokens"] = cond_event.get("summarizer_completion_tokens", 0)

        # Store current step as pending (observation not yet available)
        self._pending_step = {
            "step": self._wrapper_stats["total_queries"],
            "thought": _extract_thought(content),
            "action": _extract_action(content),
            "observation": "",  # Will be filled in next query
            "memory": memory_info,
        }

        return response

    def get_training_trajectory(self) -> Dict[str, Any]:
        """Get step-based training trajectory for CoT training."""
        # Flush last pending step (its observation is the final state)
        if self._pending_step is not None:
            # Use last non-empty user/tool message as observation
            for msg in reversed(self.messages):
                content = str(msg.get("content", ""))
                if msg.get("role") in ("user", "tool") and content.strip():
                    self._pending_step["observation"] = content[:500]
                    break
            self._training_steps.append(self._pending_step)
            self._pending_step = None
        return {
            "problem": self._task_description[:2000],
            "model": getattr(getattr(self.model, 'config', None), 'model_name', None) or str(self.model),
            "memory_strategy": self.memory_manager.__class__.__name__,
            "num_steps": len(self._training_steps),
            "steps": self._training_steps,
        }

    def get_wrapper_stats(self) -> Dict[str, Any]:
        """Get memory wrapper statistics."""
        stats = self._wrapper_stats.copy()

        if stats["compression_ratios"]:
            stats["avg_compression_ratio"] = sum(stats["compression_ratios"]) / len(stats["compression_ratios"])
            stats["max_compression_ratio"] = max(stats["compression_ratios"])
        else:
            stats["avg_compression_ratio"] = 1.0
            stats["max_compression_ratio"] = 1.0

        stats["memory_manager_type"] = self.memory_manager.__class__.__name__

        return stats
