#!/usr/bin/env python
"""
Test A-mem integration with MemGym.

Tests:
- Core components (MemoryNote, LLMController, EmbeddingRetriever, AgenticMemorySystem)
- Base AMemMemoryModel with threshold-based compression
- Environment-specific models (SWEAMemModel)

Prerequisites:
    uv pip install -r requirements-amem.txt
    # or
    uv pip install sentence-transformers scikit-learn rank-bm25 nltk litellm

Usage:
    python scripts/test_amem.py
"""

import sys
from pathlib import Path

# Add src to path
SRC_PATH = Path(__file__).parent.parent.parent / "src"

def test_imports():
    """Test that all components can be imported."""
    print("Testing imports...")

    from note import MemoryNote
    print("  - MemoryNote: OK")

    from llm_controller import LLMController
    print("  - LLMController: OK")

    from retriever import EmbeddingRetriever
    print("  - EmbeddingRetriever: OK")

    from system import AgenticMemorySystem
    print("  - AgenticMemorySystem: OK")

    # Test model import (requires base classes)
    from base import BaseMemoryModel, Context, MemoryAction
    from model import AMemMemoryModel
    print("  - AMemMemoryModel: OK")

    print("\nAll imports successful!")
    return True

def test_retriever():
    """Test embedding retriever."""
    print("\nTesting EmbeddingRetriever...")

    from retriever import EmbeddingRetriever

    retriever = EmbeddingRetriever()
    retriever.add_documents([
        "The cat sat on the mat",
        "Dogs are loyal pets",
        "Machine learning is a subset of AI"
    ])

    results = retriever.search("feline animal", k=2)
    print(f"  Search 'feline animal': indices {results}")
    assert 0 in results, "Expected 'cat' document in results"
    print("  - Retriever: OK")

def test_llm_controller():
    """Test LLM controller (requires API key)."""
    print("\nTesting LLMController...")

    from llm_controller import LLMController

    controller = LLMController(model="gpt-4o-mini")
    print(f"  - Controller created with model: {controller.model}")
    print("  - LLMController: OK (API call skipped)")

def test_memory_note():
    """Test MemoryNote without LLM."""
    print("\nTesting MemoryNote...")

    from note import MemoryNote

    # Create without LLM metadata generation
    note = MemoryNote(
        content="Test memory content",
        keywords=["test", "memory"],
        context="Testing context",
        tags=["test"],
        auto_generate_metadata=False
    )

    print(f"  - Note ID: {note.id[:8]}...")
    print(f"  - Content: {note.content}")
    print(f"  - Keywords: {note.keywords}")
    assert note.content == "Test memory content"
    print("  - MemoryNote: OK")

def test_system_without_llm():
    """Test AgenticMemorySystem without LLM calls."""
    print("\nTesting AgenticMemorySystem (evolution disabled)...")

    from system import AgenticMemorySystem

    # Create system with evolution disabled to avoid LLM calls
    system = AgenticMemorySystem(
        llm_model="gpt-4o-mini",
        enable_evolution=False
    )

    # Add notes without LLM metadata generation
    note_id = system.add_note(
        "First test memory",
        auto_generate_metadata=False,
        keywords=["test", "first"],
        context="Test context",
        tags=["test"]
    )
    print(f"  - Added note: {note_id[:8]}...")

    note_id2 = system.add_note(
        "Second test memory about cats",
        auto_generate_metadata=False,
        keywords=["test", "cats"],
        context="Cat context",
        tags=["animals"]
    )
    print(f"  - Added note: {note_id2[:8]}...")

    # Test retrieval
    related, indices = system.find_related_memories("cats and animals", k=2)
    print(f"  - Found {len(indices)} related memories")

    stats = system.get_stats()
    print(f"  - Total memories: {stats['num_memories']}")

    assert stats['num_memories'] == 2
    print("  - AgenticMemorySystem: OK")

def test_amem_model():
    """Test AMemMemoryModel interface."""
    print("\nTesting AMemMemoryModel...")

    from base import BaseMemoryModel
    from model import AMemMemoryModel

    # Create model with evolution disabled
    model = AMemMemoryModel({
        "llm_model": "gpt-4o-mini",
        "enable_evolution": False,
        "max_context_tokens": 1000,  # Low threshold for testing
        "recent_turns": 2
    })

    # Process observations (will skip LLM metadata if no API key)
    # For testing, we'll access the system directly
    model._memory_system.add_note(
        "User asked about flights",
        auto_generate_metadata=False,
        keywords=["flights", "booking"],
        context="Travel inquiry",
        tags=["travel"]
    )
    model._observation_count = 1
    model._last_observation = "User asked about flights"
    model._current_tokens = 100

    # Get context (should be pass-through since below threshold)
    context = model.get_context()
    print(f"  - Context retrieved: {len(context.content)} chars")
    print(f"  - Strategy: {context.metadata.get('strategy')}")
    print(f"  - Compressed: {context.metadata.get('compressed')}")

    # Check stats
    stats = model.get_stats()
    print(f"  - Num memories: {stats['num_memories']}")
    print(f"  - Max context tokens: {stats['max_context_tokens']}")

    print("  - AMemMemoryModel: OK")

def test_threshold_compression():
    """Test threshold-based context compression."""
    print("\nTesting threshold-based compression...")

    from system import AgenticMemorySystem

    # Create system with low threshold
    system = AgenticMemorySystem(
        llm_model="gpt-4o-mini",
        enable_evolution=False,
        max_context_tokens=500,
        recent_turns=2
    )

    # Add several memories
    for i in range(5):
        system.add_note(
            f"Memory {i}: " + "x" * 200,  # ~200 chars each
            auto_generate_metadata=False,
            keywords=[f"test{i}"],
            context=f"Context {i}",
            tags=["test"]
        )

    # Test below threshold
    context_low, meta_low = system.get_context_with_threshold(current_tokens=100)
    print(f"  - Below threshold strategy: {meta_low['strategy']}")
    assert meta_low['strategy'] == 'pass_through'
    assert meta_low['compressed'] == False

    # Test above threshold (no LLM, will use fallback query)
    context_high, meta_high = system.get_context_with_threshold(current_tokens=1000)
    print(f"  - Above threshold strategy: {meta_high['strategy']}")
    # Note: Without LLM API key, query generation will use fallback
    assert meta_high['compressed'] == True

    print("  - Threshold compression: OK")

def test_swe_model():
    """Test SWEAMemModel for code tasks."""
    print("\nTesting SWEAMemModel...")

    from swe_model import SWEAMemModel

    model = SWEAMemModel({
        "llm_model": "gpt-4o-mini",
        "enable_evolution": False
    })

    # Process code operations
    model._memory_system.add_note(
        "[FILE READ: src/main.py]\ndef hello(): pass",
        auto_generate_metadata=False,
        keywords=["main.py", "function"],
        context="File read",
        tags=["read"]
    )
    model._observation_count = 1
    model._step_count = 1
    model._operation_counts["read"] = 1
    model._files_touched["src/main.py"] = 1

    model._memory_system.add_note(
        "[ERROR]\nSyntaxError: invalid syntax",
        auto_generate_metadata=False,
        keywords=["error", "syntax"],
        context="Error encountered",
        tags=["error"]
    )
    model._observation_count = 2
    model._step_count = 2
    model._operation_counts["error"] = 1
    model._errors_encountered.append("SyntaxError: invalid syntax")

    # Get summary
    summary = model.get_code_summary()
    print(f"  - Step count: {summary['step_count']}")
    print(f"  - Operation counts: {summary['operation_counts']}")
    print(f"  - Files touched: {summary['files_touched']}")
    print(f"  - Errors: {summary['errors_encountered']}")

    # Check stats
    stats = model.get_stats()
    print(f"  - Track files: {stats['track_files']}")

    print("  - SWEAMemModel: OK")

def main():
    print("=" * 60)
    print("A-mem Integration Test")
    print("=" * 60)

    try:
        test_imports()
        test_retriever()
        test_llm_controller()
        test_memory_note()
        test_system_without_llm()
        test_amem_model()
        test_threshold_compression()
        test_swe_model()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nTo test with LLM calls, set OPENAI_API_KEY and run:")
        print("  python -c \"from memory.amem import AgenticMemorySystem; ...")
        print("\nUsage in evaluation scripts:")
        print("  python scripts/evaluate_swe.py --dataset princeton-nlp/SWE-bench_Lite --agent-llm gpt-4o-mini --memory-model swe-amem")

    except ImportError as e:
        print(f"\nImport error: {e}")
        print("\nPlease install dependencies:")
        print("  uv pip install -r requirements-amem.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
