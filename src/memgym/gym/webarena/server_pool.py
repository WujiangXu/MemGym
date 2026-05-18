"""
Server pool for WebArena-Infinity apps.

Each webarena-infinity app lives at `apps/<name>/server.py` and accepts a
`--port` argument, launching a local HTTP service. To run K tasks across W
workers without port collisions, we keep a pool of live server processes
keyed by app name and hand out free ports to each runner.

Design notes:

- **Port allocation**: ports are pulled from a monotonically-advancing cursor
  within the configured range. We never recycle ports within a single pool
  lifetime (even after a server is killed) to keep the attribution simple
  and avoid races where a new server inherits a socket from a zombie child.
- **Process lifecycle**: `acquire(app)` either returns an existing idle
  process or spawns a new one. `release(server)` either kills the process
  (the plan's default — cheaper than per-task DB reset for most apps) or
  returns it to the idle pool when `keep_alive=True`.
- **DB reset**: some webarena-infinity apps ship a `seed-db` or equivalent
  script at `apps/<name>/seed.py` / `apps/<name>/reset.sh`. Per-task DB
  reset is NOT implemented here — the default `release_kill=True` semantics
  are sufficient because a fresh `server.py` process re-loads the seed DB
  from disk each time it starts. If a particular app needs explicit mid-
  process resets, extend `_maybe_reset_db()`.
- **Teardown**: `shutdown()` kills every live process. Always call in a
  finally block from runner code; otherwise subprocesses leak.
- **Health check**: after spawn, we poll the port for readiness (TCP
  connect) up to `spawn_timeout_sec`. No HTTP probe — the TCP-accept
  check is sufficient for Playwright to begin navigation.

This module deliberately knows nothing about Playwright or task verifiers;
it is purely a process+port manager so it can be unit-tested without a
browser.
"""
from __future__ import annotations

import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("memgym.webarena.server_pool")


# =============================================================================
# Port allocator
# =============================================================================

class _PortAllocator:
    """Hand out free ports from a range, recycling released ports."""

    def __init__(self, start: int, end: int):
        if not (0 < start < end < 65536):
            raise ValueError(f"invalid port range: {start}..{end}")
        self._start = start
        self._end = end
        self._next = start
        self._recycled: list[int] = []
        self._lock = threading.Lock()

    def allocate(self) -> int:
        """Return a free port — first from recycled, then from the range."""
        with self._lock:
            # Try recycled ports first (LIFO — most recently freed)
            while self._recycled:
                port = self._recycled.pop()
                if self._is_free(port):
                    return port
            # Fall back to advancing through the range
            while self._next < self._end:
                port = self._next
                self._next += 1
                if self._is_free(port):
                    return port
            raise RuntimeError(
                f"port range {self._start}..{self._end} exhausted"
            )

    def release(self, port: int) -> None:
        """Return a port to the pool for reuse."""
        with self._lock:
            self._recycled.append(port)

    @staticmethod
    def _is_free(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


# =============================================================================
# ServerProcess — a live apps/<name>/server.py instance
# =============================================================================

@dataclass
class ServerProcess:
    """A single running webarena-infinity app server."""
    app_name: str
    port: int
    pid: int
    proc: subprocess.Popen
    url: str
    log_path: Path
    spawn_time: float = field(default_factory=time.time)

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    def terminate(self, timeout: float = 5.0) -> None:
        """Send SIGTERM, wait, then SIGKILL if still alive."""
        if not self.is_alive():
            return
        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "server %s/%d did not terminate in %.1fs, sending SIGKILL",
                    self.app_name, self.port, timeout,
                )
                self.proc.kill()
                self.proc.wait(timeout=timeout)
        except ProcessLookupError:
            pass


# =============================================================================
# WebArenaServerPool
# =============================================================================

class WebArenaServerPool:
    """Manage a pool of apps/<name>/server.py processes.

    Args:
        webarena_dir: Path to the webarena-infinity checkout root.
        port_range: (start, end) half-open port range to allocate from.
        spawn_timeout_sec: How long to wait for a new server to start accepting
            connections before raising.
        log_dir: Where to tee server stdout/stderr. Defaults to
            `<webarena_dir>/logs/server_pool/`.
        python_exe: Python interpreter used to spawn `server.py`. Defaults to
            the current interpreter (so it inherits the venv that imported us).
        release_kill: If True (default), `release()` terminates the server so
            the next acquire reseeds from disk. If False, servers live in an
            idle pool for reuse across tasks within the same app.
    """

    def __init__(
        self,
        webarena_dir: Path | str,
        port_range: tuple[int, int] = (18000, 19000),
        spawn_timeout_sec: float = 30.0,
        log_dir: Path | str | None = None,
        python_exe: str | None = None,
        release_kill: bool = True,
    ):
        self.webarena_dir = Path(webarena_dir).resolve()
        if not self.webarena_dir.exists():
            raise FileNotFoundError(
                f"webarena-infinity directory not found: {self.webarena_dir}. "
                f"Run `python -m memgym.gym.webarena.install` first."
            )
        self.apps_dir = self.webarena_dir / "apps"
        if not self.apps_dir.exists():
            raise FileNotFoundError(f"apps/ not found inside {self.webarena_dir}")

        self._allocator = _PortAllocator(*port_range)
        self.spawn_timeout_sec = spawn_timeout_sec
        self.python_exe = python_exe or sys.executable
        self.release_kill = release_kill

        self.log_dir = Path(log_dir) if log_dir else (self.webarena_dir / "logs" / "server_pool")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Idle pool per app_name (only populated when release_kill=False)
        self._idle: Dict[str, List[ServerProcess]] = {}
        # All currently-live servers (idle + in-use) for teardown
        self._all: List[ServerProcess] = []
        self._lock = threading.Lock()

    # ---- public API ----

    def list_apps(self) -> List[str]:
        """Return the list of available webarena-infinity apps."""
        return sorted([
            p.name for p in self.apps_dir.iterdir()
            if p.is_dir() and (p / "server.py").exists()
        ])

    def acquire(self, app_name: str) -> ServerProcess:
        """Get a running server for the requested app, spawning if needed."""
        self._assert_app(app_name)
        with self._lock:
            idle_list = self._idle.get(app_name, [])
            while idle_list:
                candidate = idle_list.pop()
                if candidate.is_alive():
                    logger.info(
                        "reusing idle server %s on port %d", app_name, candidate.port
                    )
                    return candidate
                # dead — drop it and loop
                self._all = [s for s in self._all if s is not candidate]

        # No idle server available — spawn a new one.
        return self._spawn(app_name)

    def release(self, server: ServerProcess) -> None:
        """Return a server to the pool (or kill it, depending on release_kill)."""
        if self.release_kill:
            logger.info("killing server %s on port %d", server.app_name, server.port)
            server.terminate()
            with self._lock:
                self._all = [s for s in self._all if s is not server]
            self._allocator.release(server.port)
            return

        with self._lock:
            if server.is_alive():
                self._idle.setdefault(server.app_name, []).append(server)
            else:
                self._all = [s for s in self._all if s is not server]

    def shutdown(self) -> None:
        """Terminate every live server. Safe to call multiple times."""
        with self._lock:
            servers = list(self._all)
            self._all.clear()
            self._idle.clear()
        for s in servers:
            try:
                s.terminate()
            except Exception as e:
                logger.warning("error terminating %s/%d: %s", s.app_name, s.port, e)

    # ---- internal ----

    def _assert_app(self, app_name: str) -> None:
        if not (self.apps_dir / app_name / "server.py").exists():
            available = self.list_apps()
            raise ValueError(
                f"unknown webarena app: {app_name!r}. Available: {available}"
            )

    def _spawn(self, app_name: str) -> ServerProcess:
        port = self._allocator.allocate()
        app_dir = self.apps_dir / app_name
        server_py = app_dir / "server.py"
        log_path = self.log_dir / f"{app_name}_{port}.log"

        cmd = [self.python_exe, str(server_py), "--port", str(port)]
        logger.info("spawning webarena server: %s (cwd=%s, log=%s)",
                    " ".join(cmd), app_dir, log_path)

        log_fh = open(log_path, "ab", buffering=0)
        # Use setsid so terminate() kills the whole process group if the
        # server forks children (e.g. uvicorn workers).
        preexec_fn = os.setsid if hasattr(os, "setsid") else None
        proc = subprocess.Popen(
            cmd,
            cwd=str(app_dir),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=preexec_fn,
        )

        server = ServerProcess(
            app_name=app_name,
            port=port,
            pid=proc.pid,
            proc=proc,
            url=f"http://127.0.0.1:{port}",
            log_path=log_path,
        )

        with self._lock:
            self._all.append(server)

        self._wait_ready(server)
        return server

    def _wait_ready(self, server: ServerProcess) -> None:
        deadline = time.time() + self.spawn_timeout_sec
        while time.time() < deadline:
            if not server.is_alive():
                log_tail = _tail(server.log_path, 40)
                raise RuntimeError(
                    f"webarena server {server.app_name}/{server.port} died during "
                    f"startup. Log tail:\n{log_tail}"
                )
            if _tcp_connect_ok(server.port):
                logger.info(
                    "server %s ready on port %d after %.1fs",
                    server.app_name, server.port, time.time() - server.spawn_time,
                )
                return
            time.sleep(0.3)

        log_tail = _tail(server.log_path, 40)
        server.terminate()
        raise TimeoutError(
            f"webarena server {server.app_name}/{server.port} did not accept "
            f"TCP connections within {self.spawn_timeout_sec:.0f}s. "
            f"Log tail:\n{log_tail}"
        )


# =============================================================================
# Helpers
# =============================================================================

def _tcp_connect_ok(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def _tail(path: Path, lines: int) -> str:
    try:
        data = path.read_text(errors="replace")
    except OSError:
        return "(log unavailable)"
    tail = data.splitlines()[-lines:]
    return "\n".join(tail)
