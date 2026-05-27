"""
AgenticMemorySystem: Memory management with LLM-based evolution.

Based on A-mem paper: https://arxiv.org/pdf/2502.12110

Key features:
1. Embedding-based memory retrieval
2. LLM-powered metadata generation (keywords, context, tags)
3. Memory evolution: strengthen links, update neighbor metadata
4. Zettelkasten-style knowledge linking
"""

import json
from typing import Any, Dict, List, Optional, Tuple

# Support both package and direct imports
try:
    from .note import MemoryNote
    from .retriever import EmbeddingRetriever
    from .llm_controller import LLMController
except ImportError:
    from note import MemoryNote
    from retriever import EmbeddingRetriever
    from llm_controller import LLMController


# JSON schema for evolution decisions
EVOLUTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {
                "should_evolve": {"type": "boolean"},
                "actions": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "suggested_connections": {
                    "type": "array",
                    "items": {"type": "integer"}
                },
                "tags_to_update": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "new_context_neighborhood": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "new_tags_neighborhood": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "required": [
                "should_evolve", "actions", "suggested_connections",
                "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}


# JSON schema for query generation
QUERY_GENERATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The retrieval query to find relevant memories"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this query is relevant"
                }
            },
            "required": ["query", "reasoning"],
            "additionalProperties": False
        },
        "strict": True
    }
}


QUERY_GENERATION_PROMPT = """You are helping an AI agent retrieve relevant memories from its memory database.

Current task/goal:
{task_description}

Recent observations (last {recent_count} turns):
{recent_observations}

Based on the current task and recent context, generate a retrieval query that will help find the most relevant past memories.
The query should focus on:
1. Key concepts and entities mentioned
2. Similar situations or patterns from the past
3. Relevant background information needed

Return a JSON object with:
- query: A concise but comprehensive search query (1-3 sentences)
- reasoning: Brief explanation of your query choice
"""


EVOLUTION_PROMPT = """You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
Make decisions about its evolution.

The new memory context:
{context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

Based on this information, determine:
1. Should this memory be evolved? Consider its relationships with other memories.
2. What specific actions should be taken (strengthen, update_neighbor)?
   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
The number of neighbors is {neighbor_number}.
Return your decision in JSON format with the following structure:
{{
    "should_evolve": True or False,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": ["neighbor_memory_ids"],
    "tags_to_update": ["tag_1",..."tag_n"],
    "new_context_neighborhood": ["new context",...,"new context"],
    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
}}
"""


class AgenticMemorySystem:
    """
    Memory management system with LLM-based evolution.

    This system implements the A-mem approach:
    1. Each memory is stored with LLM-generated metadata (keywords, context, tags)
    2. When a new memory is added, the system finds related memories
    3. The LLM decides whether to:
       - Strengthen connections (add links between memories)
       - Update neighbor metadata (improve tags/context of related memories)
    4. Periodically consolidates memories to rebuild the retriever
    5. Threshold-based context compression with query generation

    Config options:
        model_name: Embedding model for retrieval (default: all-MiniLM-L6-v2)
        llm_model: LLM for metadata and evolution (default: gpt-4o-mini)
        api_base: LiteLLM API base URL
        api_key: API key
        evo_threshold: Consolidation threshold (default: 100)
        enable_evolution: Whether to run evolution (default: True)
        retrieve_k: Number of neighbors for evolution (default: 5)
        max_context_tokens: Token threshold for context compression (default: 8000)
        recent_turns: Number of recent turns to keep in hybrid mode (default: 3)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        evo_threshold: int = 100,
        enable_evolution: bool = True,
        retrieve_k: int = 5,
        max_context_tokens: int = 8000,
        recent_turns: int = 3,
        verbose: bool = False,
        defer_evolution: bool = False,
    ):
        """
        Initialize the agentic memory system.

        Args:
            model_name: SentenceTransformer model for embeddings
            llm_model: LLM model name (LiteLLM format)
            api_base: API base URL for local models
            api_key: API key
            evo_threshold: Number of evolutions before consolidation
            enable_evolution: Whether to run memory evolution
            retrieve_k: Number of neighbors for evolution decisions
            max_context_tokens: Token threshold for context compression
            recent_turns: Number of recent turns to keep in hybrid mode
            verbose: Print debug information
        """
        self.model_name = model_name
        self.llm_model = llm_model
        self.enable_evolution = enable_evolution
        self.retrieve_k = retrieve_k
        self.max_context_tokens = max_context_tokens
        self.recent_turns = recent_turns
        self.verbose = verbose

        # Core components
        self.memories: Dict[str, MemoryNote] = {}
        self.retriever = EmbeddingRetriever(model_name)
        self.llm_controller = LLMController(
            model=llm_model,
            api_base=api_base,
            api_key=api_key
        )

        # Evolution tracking
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

        # Deferred (batched) evolution: when True, add_note skips the per-note
        # evolution LLM call; the caller must invoke flush_evolution() before
        # retrieval to run one batched pass over the full graph.
        self.defer_evolution = defer_evolution
        self._pending_evolution_ids: List[str] = []

        # Ordered observation tracking for recent turns
        self._observation_order: List[str] = []  # List of note IDs in order
        self._task_description: str = ""  # Current task description for query generation

    def add_note(
        self,
        content: str,
        time: Optional[str] = None,
        auto_generate_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        Add a new memory note.

        Args:
            content: Memory content
            time: Optional timestamp
            auto_generate_metadata: Whether to generate metadata via LLM
            **kwargs: Additional MemoryNote arguments

        Returns:
            Note ID
        """
        # Create note with LLM-generated metadata
        note = MemoryNote(
            content=content,
            timestamp=time,
            llm_controller=self.llm_controller if auto_generate_metadata else None,
            auto_generate_metadata=auto_generate_metadata,
            **kwargs
        )

        # Process memory evolution.
        # Two modes:
        #   (1) Online — call LLM per note (original A-MEM behavior).
        #   (2) Deferred — queue the note; a single batched LLM call at
        #       flush_evolution() inspects the whole graph at once, which
        #       both saves calls and gives the LLM richer context.
        if self.defer_evolution:
            self._pending_evolution_ids.append(note.id)
            evo_label = False
        elif self.enable_evolution and len(self.memories) > 0:
            evo_label, note = self._process_memory(note)
            if evo_label:
                self.evo_cnt += 1
                if self.evo_cnt % self.evo_threshold == 0:
                    self._consolidate_memories()
        else:
            evo_label = False

        # Store memory
        self.memories[note.id] = note

        # Track observation order
        self._observation_order.append(note.id)

        # Update retriever
        doc = self._format_document(note)
        self.retriever.add_documents([doc])

        return note.id

    def set_task_description(self, task: str) -> None:
        """Set the current task description for query generation."""
        self._task_description = task

    def get_recent_observations(self, n: int = None) -> List[MemoryNote]:
        """
        Get the N most recent observations in order.

        Args:
            n: Number of recent observations (default: self.recent_turns)

        Returns:
            List of recent MemoryNotes in chronological order
        """
        n = n or self.recent_turns
        recent_ids = self._observation_order[-n:] if self._observation_order else []
        return [self.memories[nid] for nid in recent_ids if nid in self.memories]

    def get_recent_observations_text(self, n: int = None) -> str:
        """Get recent observations as formatted text."""
        recent = self.get_recent_observations(n)
        if not recent:
            return "(No recent observations)"

        text = ""
        for i, note in enumerate(recent):
            text += f"[{i+1}] {note.content}\n"
        return text

    def _generate_retrieval_query(self) -> str:
        """
        Use LLM to generate an optimal retrieval query based on current context.

        Returns:
            Generated query string
        """
        recent_text = self.get_recent_observations_text()

        prompt = QUERY_GENERATION_PROMPT.format(
            task_description=self._task_description or "(No specific task)",
            recent_count=self.recent_turns,
            recent_observations=recent_text
        )

        if self.verbose:
            print(f"Query generation prompt:\n{prompt}")

        response = self.llm_controller.get_completion(
            prompt,
            response_format=QUERY_GENERATION_SCHEMA
        )

        try:
            response_cleaned = response.strip()
            if not response_cleaned.startswith("{"):
                start_idx = response_cleaned.find("{")
                if start_idx != -1:
                    response_cleaned = response_cleaned[start_idx:]
            if not response_cleaned.endswith("}"):
                end_idx = response_cleaned.rfind("}")
                if end_idx != -1:
                    response_cleaned = response_cleaned[:end_idx + 1]

            result = json.loads(response_cleaned)
            query = result.get("query", "")

            if self.verbose:
                print(f"Generated query: {query}")
                print(f"Reasoning: {result.get('reasoning', '')}")

            return query

        except json.JSONDecodeError as e:
            print(f"Query generation JSON error: {e}")
            # Fallback: use recent observation content as query
            recent = self.get_recent_observations(1)
            return recent[0].content if recent else ""

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        Simple approximation: ~4 chars per token for English.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def get_full_context(self) -> str:
        """
        Get full context from all observations in order.

        Returns:
            Formatted string of all observations
        """
        if not self._observation_order:
            return ""

        text = ""
        for note_id in self._observation_order:
            if note_id in self.memories:
                note = self.memories[note_id]
                text += f"{note.content}\n"
        return text

    def get_context_with_threshold(self, current_tokens: int = None) -> Tuple[str, Dict[str, Any]]:
        """
        Get context using threshold-based compression.

        If context < threshold: return full history (pass-through)
        If context >= threshold: use LLM to generate query, retrieve, combine with recent

        Args:
            current_tokens: Current token count (if None, estimates from stored memories)

        Returns:
            Tuple of (context_string, metadata_dict)
        """
        # Estimate current tokens if not provided
        if current_tokens is None:
            full_context = self.get_full_context()
            current_tokens = self.estimate_tokens(full_context)
        else:
            full_context = None

        metadata = {
            "current_tokens": current_tokens,
            "threshold": self.max_context_tokens,
            "num_memories": len(self.memories)
        }

        # Below threshold: pass-through (full history)
        if current_tokens < self.max_context_tokens:
            if full_context is None:
                full_context = self.get_full_context()

            metadata["strategy"] = "pass_through"
            metadata["compressed"] = False
            return full_context, metadata

        # Above threshold: compress using retrieval
        # 1. Generate query using LLM
        query = self._generate_retrieval_query()

        # 2. Calculate how many tokens we can use for retrieved content
        recent_text = self.get_recent_observations_text()
        recent_tokens = self.estimate_tokens(recent_text)
        available_for_retrieval = self.max_context_tokens - recent_tokens - 500  # buffer

        # 3. Retrieve relevant memories
        retrieved_str = self.find_related_memories_raw(query, k=self.retrieve_k)

        # 4. Trim retrieved content if needed
        retrieved_tokens = self.estimate_tokens(retrieved_str)
        if retrieved_tokens > available_for_retrieval:
            # Truncate to fit
            char_limit = available_for_retrieval * 4
            retrieved_str = retrieved_str[:char_limit] + "\n...(truncated)"

        # 5. Combine: retrieved memories + recent observations
        combined_context = f"""=== Retrieved Relevant Memories ===
{retrieved_str}

=== Recent Observations ===
{recent_text}"""

        metadata["strategy"] = "retrieval_compression"
        metadata["compressed"] = True
        metadata["query"] = query
        metadata["retrieved_tokens"] = self.estimate_tokens(retrieved_str)
        metadata["recent_tokens"] = recent_tokens

        return combined_context, metadata

    def _format_document(self, note: MemoryNote) -> str:
        """Format a memory note as a retrieval document."""
        return (
            f"content:{note.content} "
            f"context:{note.context} "
            f"keywords: {', '.join(note.keywords)} "
            f"tags: {', '.join(note.tags)}"
        )

    def _process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """
        Process a memory note for evolution.

        Uses LLM to decide:
        1. Should this memory be connected to others?
        2. Should neighbor metadata be updated?

        Args:
            note: New memory note

        Returns:
            Tuple of (did_evolve, updated_note)
        """
        # Find related memories
        neighbor_str, indices = self.find_related_memories(note.content, k=self.retrieve_k)

        if not indices:
            return False, note

        # Ask LLM about evolution
        prompt = EVOLUTION_PROMPT.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbor_str,
            neighbor_number=len(indices)
        )

        if self.verbose:
            print(f"Evolution prompt:\n{prompt}")

        response = self.llm_controller.get_completion(prompt, response_format=EVOLUTION_SCHEMA)

        try:
            # Parse response
            response_cleaned = response.strip()
            if not response_cleaned.startswith("{"):
                start_idx = response_cleaned.find("{")
                if start_idx != -1:
                    response_cleaned = response_cleaned[start_idx:]
            if not response_cleaned.endswith("}"):
                end_idx = response_cleaned.rfind("}")
                if end_idx != -1:
                    response_cleaned = response_cleaned[:end_idx + 1]

            response_json = json.loads(response_cleaned)

            if self.verbose:
                print(f"Evolution response: {response_json}")

        except json.JSONDecodeError as e:
            print(f"JSON parsing error in evolution: {e}")
            return False, note

        should_evolve = response_json.get("should_evolve", False)

        if should_evolve:
            actions = response_json.get("actions", [])
            notes_list = list(self.memories.values())
            notes_ids = list(self.memories.keys())

            for action in actions:
                if action == "strengthen":
                    # Add connections to related memories. note.links holds
                    # integer indices into the retriever's ordered memory
                    # list (see find_related_memories_raw); the prompt at
                    # `_evolution_prompt` asks for "neighbor_memory_ids" but
                    # some LLMs return strings (UUIDs / labels). Filter to
                    # int-in-range to match the contract enforced by the
                    # batched path at line ~798.
                    suggested = response_json.get("suggested_connections", [])
                    new_tags = response_json.get("tags_to_update", [])
                    n_existing = len(self.memories)
                    note.links.extend(
                        s for s in suggested
                        if isinstance(s, int) and 0 <= s < n_existing
                    )
                    if new_tags:
                        note.tags = new_tags

                elif action == "update_neighbor":
                    # Update neighbor metadata
                    new_contexts = response_json.get("new_context_neighborhood", [])
                    new_tags_list = response_json.get("new_tags_neighborhood", [])

                    for i in range(min(len(indices), len(new_tags_list))):
                        idx = indices[i]
                        if idx < len(notes_list):
                            neighbor = notes_list[idx]

                            # Update tags
                            if i < len(new_tags_list):
                                neighbor.tags = new_tags_list[i]

                            # Update context
                            if i < len(new_contexts):
                                neighbor.context = new_contexts[i]

                            # Save updated neighbor
                            self.memories[notes_ids[idx]] = neighbor

        return should_evolve, note

    def _consolidate_memories(self) -> None:
        """
        Consolidate memories: rebuild retriever with updated metadata.

        Called periodically after evolution threshold is reached.
        """
        if self.verbose:
            print(f"Consolidating memories (count: {len(self.memories)})")

        # Reset and rebuild retriever
        self.retriever = EmbeddingRetriever(self.model_name)

        for memory in self.memories.values():
            doc = self._format_document(memory)
            self.retriever.add_documents([doc])

    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[int]]:
        """
        Find related memories using embedding retrieval.

        Args:
            query: Query text
            k: Number of memories to retrieve

        Returns:
            Tuple of (formatted_string, indices)
        """
        if not self.memories:
            return "", []

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())

        memory_str = ""
        for i in indices:
            if i < len(all_memories):
                m = all_memories[i]
                memory_str += (
                    f"memory index:{i}\t"
                    f"talk start time:{m.timestamp}\t"
                    f"memory content: {m.content}\t"
                    f"memory context: {m.context}\t"
                    f"memory keywords: {m.keywords}\t"
                    f"memory tags: {m.tags}\n"
                )

        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """
        Find related memories with linked neighbors.

        Returns memories plus their connected neighbors (Zettelkasten links).

        Args:
            query: Query text
            k: Number of memories to retrieve

        Returns:
            Formatted string with memories and neighbors
        """
        if not self.memories:
            return ""

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())

        memory_str = ""
        for i in indices:
            if i >= len(all_memories):
                continue

            m = all_memories[i]
            memory_str += (
                f"talk start time:{m.timestamp}"
                f"memory content: {m.content}"
                f"memory context: {m.context}"
                f"memory keywords: {m.keywords}"
                f"memory tags: {m.tags}\n"
            )

            # Include linked neighbors. note.links is *supposed* to hold
            # int indices (see comment at the batched-evolution write site),
            # but the per-note evolution path historically extended links
            # with whatever the LLM returned. Skip anything that isn't a
            # valid in-range int rather than raising TypeError mid-eval.
            j = 0
            for neighbor_idx in m.links:
                if isinstance(neighbor_idx, int) and 0 <= neighbor_idx < len(all_memories):
                    n = all_memories[neighbor_idx]
                    memory_str += (
                        f"talk start time:{n.timestamp}"
                        f"memory content: {n.content}"
                        f"memory context: {n.context}"
                        f"memory keywords: {n.keywords}"
                        f"memory tags: {n.tags}\n"
                    )
                    j += 1
                    if j >= k:
                        break

        return memory_str

    def get_memory(self, note_id: str) -> Optional[MemoryNote]:
        """Get a memory by ID."""
        return self.memories.get(note_id)

    # ------------------------------------------------------------------
    # Deferred (batched) evolution
    # ------------------------------------------------------------------

    # Schema for the batched evolution call: returns pair-wise links
    # between note indices plus optional tag updates. The LLM sees all
    # notes at once, which gives richer linking context than per-note
    # evolution — one call instead of N, with strictly more information.
    _BATCH_EVOLUTION_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {
                "type": "object",
                "properties": {
                    "links": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "integer"},
                                "to": {"type": "integer"},
                                "reason": {"type": "string"},
                            },
                            "required": ["from", "to", "reason"],
                            "additionalProperties": False,
                        },
                    },
                    "tag_updates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "note_index": {"type": "integer"},
                                "tags": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["note_index", "tags"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["links", "tag_updates"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    _BATCH_EVOLUTION_PROMPT = (
        "You are managing a research memory system for a multi-hop "
        "investigation. Below are all memory notes collected during this "
        "research session, each with metadata. Identify cross-note connections "
        "that will help chain facts across hops.\n\n"
        "For each pair of notes that share named entities, methods, or datasets, "
        "emit a link with a short reason. These cross-hop connections are how "
        "the agent chains facts to answer the multi-hop research question. Only "
        "emit links with a clear shared entity or bridge fact — do not link "
        "purely because topics are similar.\n\n"
        "Optionally, update tags on individual notes to surface shared entities "
        "that will aid retrieval.\n\n"
        "Notes:\n{notes_listing}\n\n"
        "Return JSON:\n"
        "{{\n"
        "  \"links\": [{{\"from\": <int>, \"to\": <int>, \"reason\": \"...\"}}],\n"
        "  \"tag_updates\": [{{\"note_index\": <int>, \"tags\": [\"...\"]}}]\n"
        "}}"
    )

    def flush_evolution(self) -> bool:
        """Run one batched LLM evolution pass over all pending notes.

        Processes notes accumulated while ``defer_evolution`` was True.
        One LLM call sees the full graph — richer context than per-note
        evolution and cheaper by a factor of N.

        Returns True if an LLM call was made (i.e. something was pending).
        Idempotent: subsequent calls with no new pending notes are no-ops.
        """
        if not self.defer_evolution or not self._pending_evolution_ids:
            return False
        if not self.enable_evolution:
            self._pending_evolution_ids.clear()
            return False

        note_ids = list(self.memories.keys())
        notes_list = [self.memories[nid] for nid in note_ids]
        if len(notes_list) < 2:
            # Nothing to link against.
            self._pending_evolution_ids.clear()
            return False

        # Build a compact listing of all notes with indices the LLM can
        # reference. Truncate content to keep the prompt bounded.
        lines = []
        for i, n in enumerate(notes_list):
            content_preview = (n.content or "")[:400].replace("\n", " ")
            lines.append(
                f"[{i}] context: {n.context}\n"
                f"    keywords: {', '.join(n.keywords) if n.keywords else '(none)'}\n"
                f"    tags: {', '.join(n.tags) if n.tags else '(none)'}\n"
                f"    content: {content_preview}"
            )
        notes_listing = "\n\n".join(lines)
        prompt = self._BATCH_EVOLUTION_PROMPT.format(notes_listing=notes_listing)

        if self.verbose:
            print(f"Batched evolution over {len(notes_list)} notes")

        response = self.llm_controller.get_completion(
            prompt, response_format=self._BATCH_EVOLUTION_SCHEMA
        )

        try:
            response_json = json.loads(response.strip())
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"Batched evolution JSON parse error: {e}")
            self._pending_evolution_ids.clear()
            return False

        # Apply link updates. A-MEM stores note.links as integer indices
        # into the retriever's ordered memory list (see find_related_memories
        # at `neighbor_idx < len(all_memories)`), so we must push ints, not
        # UUID strings.
        for link in response_json.get("links", []):
            i, j = link.get("from"), link.get("to")
            if not isinstance(i, int) or not isinstance(j, int):
                continue
            if 0 <= i < len(note_ids) and 0 <= j < len(note_ids) and i != j:
                src = self.memories[note_ids[i]]
                if j not in src.links:
                    src.links.append(j)

        # Apply tag updates.
        for update in response_json.get("tag_updates", []):
            idx = update.get("note_index")
            new_tags = update.get("tags") or []
            if isinstance(idx, int) and 0 <= idx < len(note_ids) and new_tags:
                self.memories[note_ids[idx]].tags = new_tags

        self.evo_cnt += 1
        self._pending_evolution_ids.clear()
        return True

    def reset(self) -> None:
        """Reset the memory system."""
        self.memories = {}
        self.retriever = EmbeddingRetriever(self.model_name)
        self.evo_cnt = 0
        self._observation_order = []
        self._pending_evolution_ids = []
        self._task_description = ""

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "num_memories": len(self.memories),
            "evolution_count": self.evo_cnt,
            "evo_threshold": self.evo_threshold,
            "model_name": self.model_name,
            "llm_model": self.llm_model,
            "enable_evolution": self.enable_evolution,
            "retrieve_k": self.retrieve_k,
            "max_context_tokens": self.max_context_tokens,
            "recent_turns": self.recent_turns
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export system state to dict."""
        return {
            "config": {
                "model_name": self.model_name,
                "llm_model": self.llm_model,
                "evo_threshold": self.evo_threshold,
                "enable_evolution": self.enable_evolution,
                "retrieve_k": self.retrieve_k,
                "max_context_tokens": self.max_context_tokens,
                "recent_turns": self.recent_turns
            },
            "memories": {
                note_id: note.to_dict()
                for note_id, note in self.memories.items()
            },
            "observation_order": self._observation_order,
            "task_description": self._task_description,
            "evo_cnt": self.evo_cnt
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgenticMemorySystem":
        """Load system from dict."""
        config = data.get("config", {})
        system = cls(**config)

        # Load memories
        for note_id, note_data in data.get("memories", {}).items():
            note = MemoryNote.from_dict(note_data)
            system.memories[note_id] = note
            doc = system._format_document(note)
            system.retriever.add_documents([doc])

        system._observation_order = data.get("observation_order", [])
        system._task_description = data.get("task_description", "")
        system.evo_cnt = data.get("evo_cnt", 0)
        return system
