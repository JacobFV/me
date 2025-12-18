"""
Interactive process reader with refresh rate polling.

Allows the agent to track and read output from interactive CLI processes
like SSH sessions, sub-agents, or long-running commands.

Key concept: The refresh rate is fundamental to the agent's perception.
When any CLI command blocks longer than 5x the refresh rate, the agent
starts receiving incremental output updates. This makes streaming
perception automatic rather than requiring special tools.
"""

from __future__ import annotations

import asyncio
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable


class ProcessStatus(Enum):
    """Status of a tracked process."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class TrackedProcess:
    """A tracked interactive process."""

    id: str
    command: str
    args: list[str]
    process: asyncio.subprocess.Process | None = None
    output_buffer: str = ""
    error_buffer: str = ""
    read_position: int = 0  # Position of last read
    status: ProcessStatus = ProcessStatus.UNKNOWN
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: datetime | None = None
    exit_code: int | None = None


class ProcessReader:
    """
    Manages interactive process reading with configurable refresh rate.

    The refresh rate determines how often the agent polls for new output
    from tracked processes. This enables reading interactive CLI processes
    like SSH sessions or sub-agents.
    """

    def __init__(self, refresh_rate_ms: int = 100):
        self.refresh_rate_ms = refresh_rate_ms
        self._processes: dict[str, TrackedProcess] = {}
        self._lock = asyncio.Lock()

    async def start_process(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        """
        Start a new interactive process and track it.

        Returns the process ID for later reading.
        """
        process_id = str(uuid.uuid4())[:8]
        args = args or []

        # Build environment
        import os

        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        # Start the process with pipes for stdout/stderr
        proc = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=process_env,
        )

        tracked = TrackedProcess(
            id=process_id,
            command=command,
            args=args,
            process=proc,
            status=ProcessStatus.RUNNING,
        )

        async with self._lock:
            self._processes[process_id] = tracked

        # Start background task to read output
        asyncio.create_task(self._read_output_loop(process_id))

        return process_id

    async def _read_output_loop(self, process_id: str) -> None:
        """Background task to continuously read process output."""
        tracked = self._processes.get(process_id)
        if not tracked or not tracked.process:
            return

        proc = tracked.process

        async def read_stream(stream, buffer_attr: str):
            """Read from a stream and append to buffer."""
            if stream is None:
                return
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        stream.read(4096),
                        timeout=self.refresh_rate_ms / 1000.0,
                    )
                    if not chunk:
                        break
                    async with self._lock:
                        current = getattr(tracked, buffer_attr)
                        setattr(tracked, buffer_attr, current + chunk.decode("utf-8", errors="replace"))
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break

        # Read both stdout and stderr concurrently
        await asyncio.gather(
            read_stream(proc.stdout, "output_buffer"),
            read_stream(proc.stderr, "error_buffer"),
            return_exceptions=True,
        )

        # Wait for process to complete
        exit_code = await proc.wait()

        async with self._lock:
            tracked.status = ProcessStatus.COMPLETED if exit_code == 0 else ProcessStatus.FAILED
            tracked.exit_code = exit_code
            tracked.ended_at = datetime.now()

    async def read(self, process_id: str, include_errors: bool = True) -> str:
        """
        Read new output from a process since the last read.

        Returns only the output that hasn't been read yet.
        """
        async with self._lock:
            tracked = self._processes.get(process_id)
            if not tracked:
                return f"(process {process_id} not found)"

            # Get new output since last read
            new_output = tracked.output_buffer[tracked.read_position:]
            tracked.read_position = len(tracked.output_buffer)

            if include_errors and tracked.error_buffer:
                new_output += f"\n[stderr]\n{tracked.error_buffer}"
                tracked.error_buffer = ""

            return new_output

    async def read_all(self, process_id: str) -> str:
        """Read all output from a process (including already-read output)."""
        async with self._lock:
            tracked = self._processes.get(process_id)
            if not tracked:
                return f"(process {process_id} not found)"

            return tracked.output_buffer

    async def write_to_process(self, process_id: str, data: str) -> bool:
        """Write data to a process's stdin."""
        async with self._lock:
            tracked = self._processes.get(process_id)
            if not tracked or not tracked.process or tracked.process.stdin is None:
                return False

            try:
                tracked.process.stdin.write(data.encode("utf-8"))
                await tracked.process.stdin.drain()
                return True
            except Exception:
                return False

    async def kill(self, process_id: str) -> bool:
        """Kill a tracked process."""
        async with self._lock:
            tracked = self._processes.get(process_id)
            if not tracked or not tracked.process:
                return False

            try:
                tracked.process.kill()
                tracked.status = ProcessStatus.FAILED
                tracked.ended_at = datetime.now()
                return True
            except Exception:
                return False

    def list_processes(self) -> list[dict[str, Any]]:
        """List all tracked processes."""
        return [
            {
                "id": p.id,
                "command": p.command,
                "args": p.args,
                "status": p.status.value,
                "started_at": p.started_at.isoformat(),
                "ended_at": p.ended_at.isoformat() if p.ended_at else None,
                "exit_code": p.exit_code,
            }
            for p in self._processes.values()
        ]

    def get_status(self, process_id: str) -> ProcessStatus:
        """Get the status of a tracked process."""
        tracked = self._processes.get(process_id)
        if not tracked:
            return ProcessStatus.UNKNOWN
        return tracked.status

    async def wait_for_output(
        self,
        process_id: str,
        pattern: str | None = None,
        timeout_ms: int = 30000,
    ) -> str:
        """
        Wait for output from a process, optionally matching a pattern.

        This uses the refresh rate for polling.
        """
        import re

        start = datetime.now()
        accumulated = ""

        while True:
            elapsed = (datetime.now() - start).total_seconds() * 1000
            if elapsed > timeout_ms:
                return accumulated or "(timeout waiting for output)"

            new_output = await self.read(process_id)
            if new_output:
                accumulated += new_output
                if pattern is None or re.search(pattern, accumulated):
                    return accumulated

            # Sleep for refresh rate
            await asyncio.sleep(self.refresh_rate_ms / 1000.0)

            # Check if process ended
            status = self.get_status(process_id)
            if status in (ProcessStatus.COMPLETED, ProcessStatus.FAILED):
                # Get any remaining output
                remaining = await self.read(process_id)
                accumulated += remaining
                return accumulated

    async def run_with_streaming(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Run a command with automatic streaming based on refresh rate.

        This is the core perception-aware execution model:
        - If command completes within 5x refresh_rate: yields single final result
        - If command takes longer: yields incremental output at refresh_rate intervals

        Yields dicts with:
            - type: "output" | "complete" | "error"
            - content: the output text
            - elapsed_ms: time since start
            - process_id: for long-running processes
        """
        process_id = await self.start_process(command, args, cwd, env)
        start = datetime.now()
        streaming_threshold_ms = 5 * self.refresh_rate_ms
        started_streaming = False
        accumulated = ""

        while True:
            elapsed_ms = (datetime.now() - start).total_seconds() * 1000

            # Check process status
            status = self.get_status(process_id)
            new_output = await self.read(process_id)

            if new_output:
                accumulated += new_output

            # Process completed
            if status in (ProcessStatus.COMPLETED, ProcessStatus.FAILED):
                # Get any final output
                final = await self.read(process_id)
                accumulated += final

                tracked = self._processes.get(process_id)
                exit_code = tracked.exit_code if tracked else None

                yield {
                    "type": "complete" if status == ProcessStatus.COMPLETED else "error",
                    "content": accumulated,
                    "elapsed_ms": elapsed_ms,
                    "process_id": process_id,
                    "exit_code": exit_code,
                }
                return

            # Check if we should start streaming
            if elapsed_ms > streaming_threshold_ms:
                if not started_streaming:
                    started_streaming = True
                    # Yield initial streaming notification
                    yield {
                        "type": "streaming_started",
                        "content": accumulated,
                        "elapsed_ms": elapsed_ms,
                        "process_id": process_id,
                        "message": f"Command running > {streaming_threshold_ms}ms, streaming output...",
                    }
                elif new_output:
                    # Yield incremental output
                    yield {
                        "type": "output",
                        "content": new_output,
                        "elapsed_ms": elapsed_ms,
                        "process_id": process_id,
                    }

            # Wait for next refresh cycle
            await asyncio.sleep(self.refresh_rate_ms / 1000.0)

    @property
    def streaming_threshold_ms(self) -> int:
        """The threshold after which commands start streaming (5x refresh rate)."""
        return 5 * self.refresh_rate_ms
