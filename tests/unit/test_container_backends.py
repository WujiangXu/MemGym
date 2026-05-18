"""Tests for the container backend abstraction."""

import pytest

from memgym.gym.swe_bench.container import BACKENDS, get_backend
from memgym.gym.swe_bench.bubblewrap_env import BubblewrapEnvironment

class TestBackendRegistry:
    """Test backend registration and discovery."""

    def test_all_backends_registered(self):
        expected = {"docker", "singularity", "bubblewrap", "swerex_docker", "local"}
        assert set(BACKENDS.keys()) == expected

    def test_get_backend_valid(self):
        backend = get_backend("local")
        assert backend.is_available()

    def test_get_backend_unknown(self):
        with pytest.raises(ValueError, match="Unknown container backend"):
            get_backend("nonexistent")

    def test_local_always_available(self):
        from memgym.gym.swe_bench.container import LocalBackend
        assert LocalBackend().is_available()

class TestBubblewrapEnvironment:
    """Test BubblewrapEnvironment command construction."""

    def test_basic_cmd_construction(self):
        env = BubblewrapEnvironment(rootfs="/tmp/test_rootfs")
        cmd = env._build_bwrap_cmd("echo hello", "/testbed")
        assert cmd[0] == "bwrap"
        assert "--ro-bind" in cmd
        assert "--unshare-all" in cmd
        assert "--die-with-parent" in cmd
        assert "bash" in cmd
        assert "echo hello" in cmd

    def test_env_vars_passed(self):
        env = BubblewrapEnvironment(
            rootfs="/tmp/test",
            env_vars={"FOO": "bar", "BAZ": "qux"},
        )
        cmd = env._build_bwrap_cmd("true", "/testbed")
        # Check env vars are set
        idx_foo = cmd.index("FOO")
        assert cmd[idx_foo - 1] == "--setenv"
        assert cmd[idx_foo + 1] == "bar"

    def test_custom_cwd(self):
        env = BubblewrapEnvironment(rootfs="/tmp/test", cwd="/workspace")
        cmd = env._build_bwrap_cmd("ls", "/workspace")
        assert "--chdir" in cmd
        chdir_idx = cmd.index("--chdir")
        assert cmd[chdir_idx + 1] == "/workspace"

    def test_execute_dict_action(self):
        """Test that execute handles dict action format."""
        env = BubblewrapEnvironment(rootfs="/tmp/test")
        # We can't actually run bwrap in tests, but we test the interface
        assert hasattr(env, "execute")
        assert hasattr(env, "close")

    def test_close_is_noop(self):
        env = BubblewrapEnvironment(rootfs="/tmp/test")
        env.close()  # Should not raise

class TestBubblewrapBackend:
    """Test BubblewrapBackend factory."""

    def test_requires_rootfs(self):
        from memgym.gym.swe_bench.container import BubblewrapBackend
        backend = BubblewrapBackend()
        with pytest.raises(ValueError, match="requires --bwrap-rootfs"):
            backend.create_env(
                image="test:latest",
                cwd="/testbed",
                env_vars={},
                timeout=60,
                verbose=False,
            )

    def test_with_rootfs(self):
        from memgym.gym.swe_bench.container import BubblewrapBackend
        backend = BubblewrapBackend(rootfs_dir="/tmp/test_rootfs")
        env = backend.create_env(
            image="test:latest",
            cwd="/testbed",
            env_vars={"PAGER": "cat"},
            timeout=60,
            verbose=False,
            rootfs="/tmp/test_rootfs",
        )
        assert isinstance(env, BubblewrapEnvironment)
