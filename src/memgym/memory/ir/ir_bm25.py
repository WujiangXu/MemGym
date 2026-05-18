"""BM25 keyword retrieval memory for IR evaluation.

Chunks incoming documents and retrieves top-k chunks by BM25 score
at question time. Complements Naive RAG (semantic) with keyword matching.
"""

from typing import Any, Dict, List, Optional

from ..base import BaseMemoryManager, FilteredContext, register_memory_model

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


class IRBM25Memory(BaseMemoryManager):
    """BM25 keyword retrieval: tokenize chunks, score by term overlap.

    On each turn, incoming documents are chunked and tokenized.
    At answer time, retrieve_for_question() returns the top-k BM25-scored
    chunks for the question.
    """

    def __init__(
        self,
        max_tokens: int = 100000,
        chunk_size: int = 512,
        top_k: int = 10,
        question: str = "",
        **kwargs,
    ):
        if BM25Okapi is None:
            raise ImportError("rank_bm25 is required: pip install rank-bm25")
        super().__init__(max_tokens=max_tokens, **kwargs)
        self._question = question
        self.chunk_size = chunk_size
        self.top_k = top_k
        self._chunks: List[str] = []
        self._tokenized: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._all_observations: List[str] = []

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        """Chunk and index the new observation."""
        if current_observation:
            text = str(current_observation)
            self._all_observations.append(text)
            new_chunks = self._chunk(text)
            if new_chunks:
                self._chunks.extend(new_chunks)
                self._tokenized.extend(
                    [self._tokenize(c) for c in new_chunks]
                )
                self._bm25 = BM25Okapi(self._tokenized)

        tokens = self.count_tokens(self._all_observations)
        return FilteredContext(
            content=self._all_observations.copy(),
            metadata={
                "tokens": tokens,
                "original_tokens": tokens,
                "was_compacted": False,
                "strategy": "ir_bm25",
                "num_chunks": len(self._chunks),
            },
        )

    def retrieve_for_question(self, question: str, top_k: Optional[int] = None) -> str:
        """Retrieve top-k chunks by BM25 score."""
        if not self._bm25 or not self._chunks:
            return "(no relevant chunks found)"
        k = top_k or self.top_k
        query_tokens = self._tokenize(question)
        scores = self._bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = [self._chunks[i] for i in top_indices if scores[i] > 0]
        if not results:
            return "(no relevant chunks found)"
        return "\n\n---\n\n".join(results)

    def get_notes_text(self) -> str:
        """Return empty — retrieval is on-demand via retrieve_for_question."""
        return ""

    def reset(self) -> None:
        self._chunks.clear()
        self._tokenized.clear()
        self._bm25 = None
        self._all_observations.clear()
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update({
            "num_chunks": len(self._chunks),
            "chunk_size": self.chunk_size,
            "top_k": self.top_k,
        })
        return stats

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def _chunk(self, text: str) -> List[str]:
        """Split text into roughly chunk_size-token pieces."""
        words = text.split()
        words_per_chunk = max(1, int(self.chunk_size * 0.75))
        chunks = []
        for i in range(0, len(words), words_per_chunk):
            chunk = " ".join(words[i : i + words_per_chunk])
            if chunk.strip():
                chunks.append(chunk)
        return chunks


register_memory_model("ir_bm25", IRBM25Memory)
