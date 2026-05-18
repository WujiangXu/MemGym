"""Bubblewrap-based environment for SWE-bench agent.

Runs agent commands in a lightweight Linux namespace sandbox using bwrap.
Requires: bwrap binary, pre-exported rootfs directory from Docker image.

To prepare rootfs from a Docker image::

    docker create --name tmp <image>
    docker export tmp | tar -xf - -C /scratch/rootfs/<image_name>
    docker rm tmp

This environment is compatible with mini-swe-agent's environment interface:
- execute(command) -> dict with "output" and "returncode"
- close() -> cleanup
"""

import os
import shlex
import subprocess
from typing import Any, Dict, Optional


class BubblewrapEnvironment:
    """Mini-swe-agent compatible environment using bwrap sandboxing.

    Each command runs in a fresh bwrap namespace. The rootfs is mounted
    read-only, with /testbed (or the configured cwd) bind-mounted as
    writable so the agent can modify files.
    """

    def __init__(
        self,
        rootfs: str,
        cwd: str = "/testbed",
        env_vars: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        executable: Optional[str] = None,
        wrapper_args: Optional[list[str]] = None,
    ):
        self.rootfs = rootfs
        self.cwd = cwd
        self.env_vars = env_vars or {}
        self.timeout = timeout
        self.executable = executable or os.getenv("MSWEA_BUBBLEWRAP_EXECUTABLE", "bwrap")
        self.wrapper_args = self._normalize_wrapper_args(wrapper_args)

    @staticmethod
    def _normalize_wrapper_args(wrapper_args: Optional[list[str]]) -> list[str]:
        """Flatten shell-style wrapper arg fragments into argv tokens."""
        flattened: list[str] = []
        for fragment in wrapper_args or []:
            flattened.extend(shlex.split(fragment))
        return flattened

    def execute(self, action: Any, cwd: str = "", *, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute a command inside the bwrap sandbox.

        Args:
            action: Command string or dict with "command" key
            cwd: Override working directory (default: self.cwd)
            timeout: Override timeout in seconds

        Returns:
            Dict with "output" (stdout+stderr) and "returncode"
        """
        if isinstance(action, dict):
            command = action.get("command", "")
        else:
            command = str(action)

        effective_timeout = timeout or self.timeout
        effective_cwd = cwd or self.cwd

        cmd = self._build_bwrap_cmd(command, effective_cwd)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            return {
                "output": result.stdout + result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "output": f"Command timed out after {effective_timeout}s",
                "returncode": -1,
            }

    def _build_bwrap_cmd(self, command: str, cwd: str) -> list:
        """Build the bwrap command line for sandboxed execution."""
        bwrap = [self.executable]
        # Bind rootfs as read-only root filesystem
        bwrap += ["--ro-bind", self.rootfs, "/"]
        # Bind the working directory as writable
        testbed_path = f"{self.rootfs}{cwd}"
        bwrap += ["--bind", testbed_path, cwd]
        # Writable /tmp
        bwrap += ["--tmpfs", "/tmp"]
        # Proc and dev filesystems
        bwrap += ["--proc", "/proc", "--dev", "/dev"]
        # Unshare all namespaces, die with parent
        bwrap += ["--unshare-all", "--die-with-parent"]
        # Additional wrapper args are appended to the default sandbox config.
        bwrap += self.wrapper_args
        # Working directory
        bwrap += ["--chdir", cwd]
        # Environment variables
        for k, v in self.env_vars.items():
            bwrap += ["--setenv", k, v]
        # The actual command
        bwrap += ["bash", "-c", command]
        return bwrap

    def close(self):
        """Cleanup. bwrap is per-command, so nothing to clean up."""
        pass
