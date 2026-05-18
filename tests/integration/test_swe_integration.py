#!/usr/bin/env python
"""
Integration test for SWE-bench Memory Environment.

This test verifies that SWEMemoryEnv:
1. Correctly loads tasks from HuggingFace datasets
2. Tracks memory/tokens throughout episodes
3. Records trajectories with memory actions
4. Supports both baseline and context-managed modes

Note: Full evaluation requires Docker. This test focuses on data loading
and memory tracking without requiring Docker.
"""

import sys
from pathlib import Path

# Add src to path
SRC_PATH = Path(__file__).parent.parent.parent / "src"

print("=" * 70)
print("SWE-bench Memory Environment Integration Test")
print("=" * 70)

def test_imports():
    """Test that all SWE-bench components can be imported."""
    print("\n1. Testing imports...")
    try:
        from memgym.envs import SWEMemoryEnv, SWETask
        from memgym.memory import PassThroughMemoryModel, SummarizationMemoryModel
        from memgym.agents import SWEReasoningModel, get_reasoning_model
        from memgym.utility import TokenTracker
        from memgym.adapter import TrajectoryRecorder
        print("   OK - All imports successful")
        return True
    except Exception as e:
        print(f"   FAIL - Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_swe_task_dataclass():
    """Test SWETask dataclass functionality."""
    print("\n2. Testing SWETask dataclass...")
    try:
        from memgym.envs import SWETask

        # Create from dict (simulating HuggingFace format)
        task_data = {
            "instance_id": "test__repo-12345",
            "repo": "test/repo",
            "base_commit": "abc123",
            "problem_statement": "Fix the bug in function X",
            "hints_text": "Look at file Y",
            "created_at": "2024-01-01",
            "patch": "--- a/file.py\n+++ b/file.py\n...",
            "test_patch": "--- a/test.py\n+++ b/test.py\n...",
            "version": "1.0",
            "environment_setup_commit": "def456"
        }

        task = SWETask.from_dict(task_data)

        assert task.instance_id == "test__repo-12345"
        assert task.repo == "test/repo"
        assert task.problem_statement == "Fix the bug in function X"

        print(f"   OK - SWETask created successfully")
        print(f"   - instance_id: {task.instance_id}")
        print(f"   - repo: {task.repo}")
        return True
    except Exception as e:
        print(f"   FAIL - SWETask error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_swe_reasoning_model():
    """Test SWE reasoning model for patch tracking."""
    print("\n3. Testing SWEReasoningModel...")
    try:
        from memgym.agents import SWEReasoningModel

        model = SWEReasoningModel()

        # Simulate patch submission
        action1 = model.set_action_from_external(
            action="Patch submitted:\n--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n-old\n+new",
            context="Issue: fix the bug"
        )

        stats = model.get_stats()
        print(f"   OK - SWEReasoningModel works")
        print(f"   - Patches submitted: {stats['patches_submitted']}")
        print(f"   - Total patch size: {stats['total_patch_size']}")

        # Test reset
        model.reset()
        stats_after_reset = model.get_stats()
        assert stats_after_reset['patches_submitted'] == 0

        print(f"   - Reset works correctly")
        return True
    except Exception as e:
        print(f"   FAIL - SWEReasoningModel error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_env_creation_without_dataset():
    """Test SWEMemoryEnv creation without loading dataset."""
    print("\n4. Testing SWEMemoryEnv creation (no dataset)...")
    try:
        from memgym.envs import SWEMemoryEnv

        # Create env with a dummy instance_id (won't load dataset until reset)
        env = SWEMemoryEnv(
            instance_id="dummy__repo-12345",
            dataset="princeton-nlp/SWE-bench_Lite",
            use_context_management=False,
            use_docker=False  # Disable Docker for testing
        )

        print(f"   OK - SWEMemoryEnv created")
        print(f"   - instance_id: {env.instance_id}")
        print(f"   - dataset: {env.dataset}")
        print(f"   - context_management: {env._use_context_management}")
        print(f"   - Has memory_model: {env.memory_model is not None}")
        print(f"   - Has reasoning_model: {env.reasoning_model is not None}")
        return True
    except Exception as e:
        print(f"   FAIL - SWEMemoryEnv creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test loading SWE-bench dataset from HuggingFace."""
    print("\n5. Testing HuggingFace dataset loading...")
    try:
        # Check if datasets package is available
        try:
            from datasets import load_dataset
        except ImportError:
            print("   SKIP - datasets package not installed")
            print("   Install with: pip install datasets")
            return True  # Not a failure, just skip

        from memgym.envs import SWEMemoryEnv

        # Create env and load dataset
        env = SWEMemoryEnv(
            dataset="princeton-nlp/SWE-bench_Lite",
            split="test",
            use_docker=False
        )

        # Load task list
        task_ids = env._get_task_list()

        print(f"   OK - Dataset loaded successfully")
        print(f"   - Total tasks: {len(task_ids)}")
        if task_ids:
            print(f"   - First task: {task_ids[0]}")
            print(f"   - Last task: {task_ids[-1]}")
        return True
    except Exception as e:
        print(f"   FAIL - Dataset loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reset_and_step_mock():
    """Test reset and step with mock data (no Docker evaluation)."""
    print("\n6. Testing reset/step with mock data...")
    try:
        from memgym.envs import SWEMemoryEnv, SWETask
        from memgym.memory import PassThroughMemoryModel

        # Create env
        env = SWEMemoryEnv(
            dataset="princeton-nlp/SWE-bench_Lite",
            use_docker=False,  # No Docker evaluation
            max_steps=3
        )

        # Manually inject a mock task to avoid needing HuggingFace
        env._task = SWETask(
            instance_id="mock__test-001",
            repo="mock/test",
            base_commit="abc123",
            problem_statement="Fix the function to return correct value",
            hints_text="Check line 42",
            created_at="2024-01-01",
            patch="--- a/file.py\n+++ b/file.py",
            test_patch="",
            version="1.0",
            environment_setup_commit="def456"
        )
        env.instance_id = "mock__test-001"

        # Reset models manually (since we skipped normal reset)
        env._memory_model.reset()
        env._reasoning_model.reset()
        env._token_tracker.reset()
        env._current_step = 0
        env._episode_observations = []
        env._done = False

        # Start trajectory recording
        env._trajectory_recorder.start_episode(metadata={
            "instance_id": "mock__test-001",
            "repo": "mock/test",
            "test_mode": True
        })

        # Build observation
        obs = env._build_observation()

        print(f"   - Observation built:")
        print(f"     - instance_id: {obs['instance_id']}")
        print(f"     - repo: {obs['repo']}")
        print(f"     - problem: {obs['problem_statement'][:50]}...")

        # Test step with a mock patch
        mock_patch = """--- a/file.py
+++ b/file.py
@@ -40,5 +40,5 @@ def buggy_function():
-    return wrong_value
+    return correct_value
"""
        action = {"patch": mock_patch}
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"   - Step executed:")
        print(f"     - Reward: {reward}")
        print(f"     - Terminated: {terminated}")
        print(f"     - Memory stats: {info['memory_info']['memory_stats']}")

        # Get trajectory
        trajectory = env.get_trajectory()
        print(f"   - Trajectory recorded: {trajectory['num_episodes']} episode(s)")

        if trajectory['num_episodes'] > 0:
            episode = trajectory['episodes'][0]
            if episode.get('steps'):
                step = episode['steps'][0]
                print(f"     - Has memory_action: {step.get('memory_action') is not None}")
                print(f"     - Has context: {step.get('context') is not None}")
                print(f"     - Has reasoning_action: {step.get('reasoning_action') is not None}")

        print(f"   OK - Reset/step works correctly")
        return True
    except Exception as e:
        print(f"   FAIL - Reset/step error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_management_comparison():
    """Compare baseline vs context-managed memory."""
    print("\n7. Testing baseline vs context-managed...")
    try:
        from memgym.envs import SWEMemoryEnv, SWETask
        from memgym.memory import PassThroughMemoryModel
        from memgym.memory import SummarizationMemoryModel

        # Create baseline env
        baseline_env = SWEMemoryEnv(
            dataset="princeton-nlp/SWE-bench_Lite",
            use_context_management=False,
            use_docker=False
        )

        # Create context-managed env
        context_env = SWEMemoryEnv(
            dataset="princeton-nlp/SWE-bench_Lite",
            use_context_management=True,
            max_context_tokens=100,  # Low threshold for testing
            use_docker=False
        )

        print(f"   Baseline memory model: {type(baseline_env.memory_model).__name__}")
        print(f"   Context-managed memory model: {type(context_env.memory_model).__name__}")

        assert isinstance(baseline_env.memory_model, PassThroughMemoryModel)
        assert isinstance(context_env.memory_model, SummarizationMemoryModel)

        print(f"   OK - Memory model types correct")
        return True
    except Exception as e:
        print(f"   FAIL - Comparison error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_episode_with_dataset():
    """Test full episode with real HuggingFace data (no Docker)."""
    print("\n8. Testing full episode with dataset (no Docker)...")
    try:
        # Check if datasets package is available
        try:
            from datasets import load_dataset
        except ImportError:
            print("   SKIP - datasets package not installed")
            return True

        from memgym.envs import SWEMemoryEnv

        # Create env
        env = SWEMemoryEnv(
            dataset="princeton-nlp/SWE-bench_Lite",
            split="test",
            use_docker=False,
            use_context_management=False,
            max_steps=2
        )

        # Reset with first task
        task_ids = env._get_task_list()
        if not task_ids:
            print("   SKIP - No tasks available")
            return True

        first_task_id = task_ids[0]
        obs, info = env.reset(options={"instance_id": first_task_id})

        print(f"   - Loaded task: {first_task_id}")
        print(f"   - Repo: {obs['repo']}")
        print(f"   - Problem length: {len(obs['problem_statement'])} chars")

        # Submit a dummy patch
        action = {"patch": "--- a/dummy.py\n+++ b/dummy.py\n@@ -1 +1 @@\n-old\n+new"}
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"   - Step result: reward={reward}, terminated={terminated}")
        print(f"   - Memory tokens: {info['memory_info']['tokens']}")

        # Get trajectory
        trajectory = env.get_trajectory()
        print(f"   - Trajectory episodes: {trajectory['num_episodes']}")

        print(f"   OK - Full episode test passed")
        return True
    except Exception as e:
        print(f"   FAIL - Full episode error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    results = []

    results.append(("Imports", test_imports()))
    results.append(("SWETask dataclass", test_swe_task_dataclass()))
    results.append(("SWEReasoningModel", test_swe_reasoning_model()))
    results.append(("Env creation", test_env_creation_without_dataset()))
    results.append(("Dataset loading", test_dataset_loading()))
    results.append(("Reset/Step mock", test_reset_and_step_mock()))
    results.append(("Context management", test_context_management_comparison()))
    results.append(("Full episode", test_full_episode_with_dataset()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll SWE-bench integration tests passed!")
        return 0
    else:
        print("\nSome tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
