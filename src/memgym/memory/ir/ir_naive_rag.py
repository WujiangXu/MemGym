"""Naive RAG memory for IR evaluation.

Chunks incoming documents and retrieves top-k chunks by cosine similarity
at question time. No summarization, no reasoning — pure vector retrieval.

Wraps EmbeddingRetriever from the A-MEM module.
"""

from typing import Any, Dict, List, Optional

from ..base import BaseMemoryManager, FilteredContext, register_memory_model


class IRNaiveRAGMemory(BaseMemoryManager):
    """Vector-DB retrieval: embed chunks, cosine-similarity retrieve top-k.

    On each turn, incoming documents are chunked and embedded.
    At answer time, retrieve_for_question() returns the top-k most
    similar chunks to the question.
    """

    def __init__(
        self,
        max_tokens: int = 100000,
        chunk_size: int = 512,
        top_k: int = 10,
        model_name: str = "all-MiniLM-L6-v2",
        question: str = "",
        **kwargs,
    ):
        super().__init__(max_tokens=max_tokens, **kwargs)
        self._question = question
        self.chunk_size = chunk_size
        self.top_k = top_k
        try:
            from ..external.amem.retriever import EmbeddingRetriever
        except ImportError as e:
            raise ImportError(
                "IRNaiveRAGMemory requires the A-MEM optional dependencies "
                "(sentence-transformers, scikit-learn). Install them with:\n"
                "  pip install -r requirements-amem.txt\n"
                f"Original error: {e}"
            ) from e
        self.retriever = EmbeddingRetriever(model_name=model_name)
        self._all_observations: List[str] = []

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        """Chunk and embed the new observation."""
        if current_observation:
            text = str(current_observation)
            self._all_observations.append(text)
            chunks = self._chunk(text)
            if chunks:
                self.retriever.add_documents(chunks)

        tokens = self.count_tokens(self._all_observations)
        return FilteredContext(
            content=self._all_observations.copy(),
            metadata={
                "tokens": tokens,
                "original_tokens": tokens,
                "was_compacted": False,
                "strategy": "ir_naive_rag",
                "num_chunks": len(self.retriever),
            },
        )

    def retrieve_for_question(self, question: str, top_k: Optional[int] = None) -> str:
        """Retrieve top-k chunks most similar to the question."""
        k = top_k or self.top_k
        indices = self.retriever.search(question, k=k)
        if not indices:
            return "(no relevant chunks found)"
        chunks = [self.retriever.corpus[i] for i in indices]
        return "\n\n---\n\n".join(chunks)

    def get_notes_text(self) -> str:
        """Return empty — retrieval is on-demand via retrieve_for_question."""
        return ""

    def reset(self) -> None:
        self.retriever.reset()
        self._all_observations.clear()
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update({
            "num_chunks": len(self.retriever),
            "chunk_size": self.chunk_size,
            "top_k": self.top_k,
        })
        return stats

    def _chunk(self, text: str) -> List[str]:
        """Split text into roughly chunk_size-token pieces."""
        words = text.split()
        # Approximate: 1 token ~ 0.75 words
        words_per_chunk = max(1, int(self.chunk_size * 0.75))
        chunks = []
        for i in range(0, len(words), words_per_chunk):
            chunk = " ".join(words[i : i + words_per_chunk])
            if chunk.strip():
                chunks.append(chunk)
        return chunks


register_memory_model("ir_naive_rag", IRNaiveRAGMemory)
