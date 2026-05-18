"""Container backend abstraction for SWE-bench agent environments.

Supports Docker (default), Singularity/Apptainer (HPC), Bubblewrap (lightweight
Linux sandbox), and SWE-ReX Docker (advanced orchestration).

For HPC without Docker: use --env-type singularity/bubblewrap with --skip-eval.
Evaluate patches later on a Docker machine via /reeval.
"""

import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class ContainerBackend(ABC):
    """Abstract interface for running SWE-bench agent in a container."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        ...

    @abstractmethod
    def create_env(
        self,
        image: str,
        cwd: str,
        env_vars: Dict[str, str],
        timeout: int,
        verbose: bool,
        **kwargs,
    ):
        """Create an environment instance for the given image."""
        ...


class DockerBackend(ContainerBackend):
    """Standard Docker environment via mini-swe-agent."""

    def is_available(self) -> bool:
        return shutil.which("docker") is not None

    def create_env(self, image, cwd, env_vars, timeout, verbose, **kwargs):
        import subprocess
        from minisweagent.environments.docker import DockerEnvironment

        if verbose:
            print(f"  Docker image: {image}")

        # Pre-pull image if not already local.
        # mini-swe-agent's DockerEnvironment has pull_timeout=120s which is too
        # short for large SWE-Gym images (3-11 GB). We do a separate pull with
        # a generous timeout so docker run only needs to start the container.
        try:
            check = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True, timeout=10
            )
            if check.returncode != 0:
                if verbose:
                    print(f"  Pulling image (not cached locally)...")
                subprocess.run(
                    ["docker", "pull", image],
                    capture_output=True, text=True,
                    timeout=600,  # 10 minutes for large images
                )
        except subprocess.TimeoutExpired:
            if verbose:
                print(f"  Warning: image pull timed out after 600s")
        except Exception as e:
            if verbose:
                print(f"  Warning: pre-pull failed: {e}")

        return DockerEnvironment(image=image, timeout=timeout, cwd=cwd, env=env_vars)


class SingularityBackend(ContainerBackend):
    """Singularity/Apptainer environment for HPC clusters."""

    def is_available(self) -> bool:
        return (
            shutil.which("singularity") is not None
            or shutil.which("apptainer") is not None
        )

    def create_env(self, image, cwd, env_vars, timeout, verbose, **kwargs):
        from minisweagent.environments.singularity import SingularityEnvironment

        sing_image = f"docker://{image}"
        if verbose:
            print(f"  Singularity image: {sing_image}")
        return SingularityEnvironment(
            image=sing_image, timeout=timeout, cwd=cwd, env=env_vars
        )


class BubblewrapBackend(ContainerBackend):
    """Lightweight unprivileged sandbox using bwrap (bubblewrap).

    Requires: bwrap installed, Docker image pre-exported to rootfs directory.

    To prepare rootfs from a Docker image::

        docker create --name tmp <image>
        docker export tmp | tar -xf - -C /scratch/rootfs/<image_name>
        docker rm tmp
    """

    def __init__(
        self,
        rootfs_dir: Optional[str] = None,
        executable: Optional[str] = None,
        wrapper_args: Optional[list[str]] = None,
    ):
        self.rootfs_dir = rootfs_dir
        self.executable = executable
        self.wrapper_args = list(wrapper_args or [])

    def is_available(self) -> bool:
        executable = self.executable or "bwrap"
        return shutil.which(executable) is not None

    def create_env(self, image, cwd, env_vars, timeout, verbose, **kwargs):
        from memgym.gym.swe_bench.bubblewrap_env import BubblewrapEnvironment

        rootfs = kwargs.get("rootfs") or self.rootfs_dir
        executable = kwargs.get("executable") or self.executable
        wrapper_args = kwargs.get("wrapper_args")
        if wrapper_args is None:
            wrapper_args = self.wrapper_args
        if not rootfs:
            raise ValueError(
                "BubblewrapBackend requires --bwrap-rootfs (path to pre-exported rootfs dir). "
                "To prepare: docker create --name tmp <image> && "
                "docker export tmp | tar -xf - -C /scratch/rootfs/<name> && "
                "docker rm tmp"
            )
        if verbose:
            print(f"  Bubblewrap rootfs: {rootfs}")
        return BubblewrapEnvironment(
            rootfs=rootfs,
            cwd=cwd,
            env_vars=env_vars,
            timeout=timeout,
            executable=executable,
            wrapper_args=wrapper_args,
        )


class SwerexDockerBackend(ContainerBackend):
    """SWE-ReX Docker environment with advanced orchestration."""

    def is_available(self) -> bool:
        try:
            import swerex  # noqa: F401
            return shutil.which("docker") is not None
        except ImportError:
            return False

    def create_env(self, image, cwd, env_vars, timeout, verbose, **kwargs):
        # Import the adapter that's defined in env.py
        from memgym.gym.swe_bench.env import SwerexEnvironmentAdapter

        if verbose:
            print(f"  SWE-ReX Docker image: {image}")
        return SwerexEnvironmentAdapter(
            image=image,
            timeout=timeout,
            cwd=cwd,
            env_vars=env_vars,
            verbose=verbose,
        )


class LocalBackend(ContainerBackend):
    """Local environment with no isolation (for debugging)."""

    def is_available(self) -> bool:
        return True

    def create_env(self, image, cwd, env_vars, timeout, verbose, **kwargs):
        from minisweagent.environments.local import LocalEnvironment

        instance_id = kwargs.get("instance_id", "unknown")
        tmp_dir = Path(__file__).parent.parent.parent.parent / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        workspace = tempfile.mkdtemp(
            prefix=f"swe_{instance_id}_", dir=str(tmp_dir)
        )
        if verbose:
            print(f"  Workspace: {workspace}")
        return LocalEnvironment(timeout=timeout, cwd=workspace, env=env_vars)


BACKENDS: Dict[str, type] = {
    "docker": DockerBackend,
    "singularity": SingularityBackend,
    "bubblewrap": BubblewrapBackend,
    "swerex_docker": SwerexDockerBackend,
    "local": LocalBackend,
}


def get_backend(name: str, **kwargs) -> ContainerBackend:
    """Get a container backend by name.

    Args:
        name: Backend name (docker, singularity, bubblewrap, swerex_docker, local)
        **kwargs: Backend-specific options (e.g., rootfs_dir for bubblewrap)

    Returns:
        ContainerBackend instance

    Raises:
        ValueError: If backend name is unknown
        RuntimeError: If backend is not available on this system
    """
    backend_cls = BACKENDS.get(name)
    if backend_cls is None:
        raise ValueError(
            f"Unknown container backend: {name}. "
            f"Available: {list(BACKENDS.keys())}"
        )
    backend = backend_cls(**{k: v for k, v in kwargs.items() if k in backend_cls.__init__.__code__.co_varnames})
    if not backend.is_available():
        available = [k for k, v in BACKENDS.items() if v().is_available()]
        raise RuntimeError(
            f"{name} backend not available on this system. "
            f"Available backends: {available}"
        )
    return backend
