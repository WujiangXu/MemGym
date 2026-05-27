"""
Retriever: Embedding-based memory retrieval.

Provides semantic search over memories using SentenceTransformers.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from .note import MemoryNote


class EmbeddingRetriever:
    """
    Simple embedding-based retrieval using SentenceTransformers.

    This retriever:
    1. Encodes documents using a sentence transformer model
    2. Uses cosine similarity for retrieval
    3. Supports incremental document addition

    Config options:
        model_name: SentenceTransformer model (default: all-MiniLM-L6-v2)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding retriever.

        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.corpus: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.document_ids: Dict[str, int] = {}  # Map document content to index

    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the retriever.

        Args:
            documents: List of text documents to add
        """
        if not documents:
            return

        if not self.corpus:
            # First batch of documents
            self.corpus = documents
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            # Append to existing
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)

            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])

            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx

    def add_document(self, document: str) -> bool:
        """
        Add a single document to the retriever.

        Args:
            document: Text content to add

        Returns:
            True if document was added, False if already exists
        """
        if document in self.document_ids:
            return False

        self.add_documents([document])
        return True

    def search(self, query: str, k: int = 5) -> List[int]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of document indices sorted by relevance
        """
        if not self.corpus or self.embeddings is None:
            return []

        # Encode query
        query_embedding = self.model.encode([query])[0]

        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Get top k indices
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return top_k_indices.tolist()

    def reset(self) -> None:
        """Reset the retriever, clearing all documents."""
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}

    def save(self, cache_file: str, embeddings_file: str) -> None:
        """
        Save retriever state to disk.

        Args:
            cache_file: Path for corpus and metadata
            embeddings_file: Path for embeddings (.npy)
        """
        if self.embeddings is not None:
            np.save(embeddings_file, self.embeddings)

        state = {
            "corpus": self.corpus,
            "document_ids": self.document_ids,
            "model_name": self.model_name
        }
        with open(cache_file, "wb") as f:
            pickle.dump(state, f)

    def load(self, cache_file: str, embeddings_file: str) -> "EmbeddingRetriever":
        """
        Load retriever state from disk.

        Args:
            cache_file: Path for corpus and metadata
            embeddings_file: Path for embeddings (.npy)

        Returns:
            Self for chaining
        """
        cache_path = Path(cache_file)
        embeddings_path = Path(embeddings_file)

        if embeddings_path.exists():
            self.embeddings = np.load(str(embeddings_path))

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                state = pickle.load(f)
                self.corpus = state["corpus"]
                self.document_ids = state.get("document_ids", {})

        return self

    @classmethod
    def from_memories(
        cls,
        memories: Dict[str, "MemoryNote"],
        model_name: str = "all-MiniLM-L6-v2"
    ) -> "EmbeddingRetriever":
        """
        Create retriever from existing memories.

        Combines memory content with metadata for better retrieval.

        Args:
            memories: Dict mapping memory ID to MemoryNote
            model_name: SentenceTransformer model name

        Returns:
            Initialized retriever with all memories
        """
        all_docs = []
        for m in memories.values():
            # Combine content with metadata for richer retrieval
            metadata_text = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)

        retriever = cls(model_name)
        retriever.add_documents(all_docs)
        return retriever

    def __len__(self) -> int:
        """Return number of documents in retriever."""
        return len(self.corpus)
