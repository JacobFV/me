"""
File operation interceptor for body VFS.

This module provides hooks that intercept file operations (Read, Edit, Write)
and route them through the body VFS when they target body files.

The interceptor sits between the Claude SDK tools and the actual filesystem,
transparently handling body file operations.

## How it works

1. Agent calls Read/Edit/Write with a path
2. Interceptor checks if path is within agent's body directory
3. If body path → route through BodyVFS (validates, updates in-memory)
4. If not body path → pass through to actual filesystem

## Integration

The interceptor is used by wrapping the standard file tools:

```python
from me.agent.file_intercept import FileInterceptor

interceptor = FileInterceptor(body_vfs)

# In tool definitions:
@tool("Read", ...)
async def read(args):
    path = args["file_path"]
    if interceptor.should_intercept(path):
        return interceptor.read(path)
    else:
        return actual_read(path)
```
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from me.agent.body_vfs import BodyVFS


class InterceptResult:
    """Result of an intercepted operation."""

    def __init__(
        self,
        success: bool,
        content: Optional[str] = None,
        error: Optional[str] = None,
        bytes_written: int = 0,
    ):
        self.success = success
        self.content = content
        self.error = error
        self.bytes_written = bytes_written

    def to_tool_result(self) -> dict[str, Any]:
        """Convert to Claude SDK tool result format."""
        if self.success:
            return {
                "content": [{
                    "type": "text",
                    "text": self.content or f"(wrote {self.bytes_written} bytes)"
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"ERROR: {self.error}"
                }],
                "isError": True,
            }


class FileInterceptor:
    """
    Intercepts file operations for body VFS routing.

    This class determines whether a file operation should be handled
    by the body VFS or passed through to the actual filesystem.
    """

    def __init__(self, body_vfs: "BodyVFS"):
        self.body_vfs = body_vfs

    def should_intercept(self, path: str | Path) -> bool:
        """Check if this path should be intercepted by body VFS."""
        return self.body_vfs.is_body_path(path)

    def read(self, path: str | Path) -> InterceptResult:
        """
        Intercept a read operation.

        For body files, returns the current in-memory state.
        """
        try:
            path = Path(path).resolve()
            rel_path = self.body_vfs.relative_path(path)

            content = self.body_vfs.read_text(rel_path)

            # Format like Read tool output (with line numbers)
            lines = content.split("\n")
            formatted_lines = []
            for i, line in enumerate(lines, 1):
                formatted_lines.append(f"{i:>6}→{line}")

            return InterceptResult(
                success=True,
                content="\n".join(formatted_lines),
            )

        except FileNotFoundError as e:
            return InterceptResult(success=False, error=str(e))
        except Exception as e:
            return InterceptResult(success=False, error=f"Read failed: {e}")

    def write(self, path: str | Path, content: str) -> InterceptResult:
        """
        Intercept a write operation.

        For body files, validates and updates in-memory state.
        """
        try:
            path = Path(path).resolve()
            rel_path = self.body_vfs.relative_path(path)

            result = self.body_vfs.write(rel_path, content.encode("utf-8"))

            if result.success:
                return InterceptResult(
                    success=True,
                    content=f"Updated {rel_path}",
                    bytes_written=result.bytes_written,
                )
            else:
                return InterceptResult(
                    success=False,
                    error=result.error,
                )

        except Exception as e:
            return InterceptResult(success=False, error=f"Write failed: {e}")

    def edit(
        self,
        path: str | Path,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> InterceptResult:
        """
        Intercept an edit operation.

        For body files, applies the edit to in-memory state.
        """
        try:
            path = Path(path).resolve()
            rel_path = self.body_vfs.relative_path(path)

            # Read current content
            current = self.body_vfs.read_text(rel_path)

            # Check if old_string exists
            if old_string not in current:
                return InterceptResult(
                    success=False,
                    error=f"String not found in {rel_path}: {old_string[:50]}..."
                )

            # Check uniqueness (unless replace_all)
            if not replace_all and current.count(old_string) > 1:
                return InterceptResult(
                    success=False,
                    error=f"String appears {current.count(old_string)} times in {rel_path}. "
                          f"Use replace_all=True or provide more context."
                )

            # Apply edit
            if replace_all:
                new_content = current.replace(old_string, new_string)
            else:
                new_content = current.replace(old_string, new_string, 1)

            # Write back
            result = self.body_vfs.write(rel_path, new_content.encode("utf-8"))

            if result.success:
                return InterceptResult(
                    success=True,
                    content=f"Edited {rel_path}",
                    bytes_written=result.bytes_written,
                )
            else:
                return InterceptResult(
                    success=False,
                    error=result.error,
                )

        except FileNotFoundError:
            return InterceptResult(
                success=False,
                error=f"File not found: {path}"
            )
        except Exception as e:
            return InterceptResult(success=False, error=f"Edit failed: {e}")

    def list_directory(self, path: str | Path) -> InterceptResult:
        """
        Intercept a directory listing operation.
        """
        try:
            path = Path(path).resolve()
            rel_path = self.body_vfs.relative_path(path)

            entries = self.body_vfs.list_dir(rel_path)

            # Format like ls output
            lines = []
            for entry in entries:
                type_char = "d" if entry["type"] == "directory" else "-"
                size = entry.get("size", 0)
                name = entry["name"]
                lines.append(f"{type_char} {size:>8} {name}")

            return InterceptResult(
                success=True,
                content="\n".join(lines) if lines else "(empty directory)",
            )

        except FileNotFoundError as e:
            return InterceptResult(success=False, error=str(e))
        except Exception as e:
            return InterceptResult(success=False, error=f"List failed: {e}")


class ToolWrapper:
    """
    Wraps standard file tools to route body paths through the interceptor.

    This is used to create intercepted versions of Read, Write, Edit tools.
    """

    def __init__(self, interceptor: FileInterceptor):
        self.interceptor = interceptor

    def wrap_read(self, original_read):
        """Wrap the Read tool."""
        async def wrapped_read(args: dict[str, Any]) -> dict[str, Any]:
            path = args.get("file_path", "")

            if self.interceptor.should_intercept(path):
                result = self.interceptor.read(path)
                return result.to_tool_result()

            # Pass through to original
            return await original_read(args)

        return wrapped_read

    def wrap_write(self, original_write):
        """Wrap the Write tool."""
        async def wrapped_write(args: dict[str, Any]) -> dict[str, Any]:
            path = args.get("file_path", "")
            content = args.get("content", "")

            if self.interceptor.should_intercept(path):
                result = self.interceptor.write(path, content)
                return result.to_tool_result()

            # Pass through to original
            return await original_write(args)

        return wrapped_write

    def wrap_edit(self, original_edit):
        """Wrap the Edit tool."""
        async def wrapped_edit(args: dict[str, Any]) -> dict[str, Any]:
            path = args.get("file_path", "")
            old_string = args.get("old_string", "")
            new_string = args.get("new_string", "")
            replace_all = args.get("replace_all", False)

            if self.interceptor.should_intercept(path):
                result = self.interceptor.edit(path, old_string, new_string, replace_all)
                return result.to_tool_result()

            # Pass through to original
            return await original_edit(args)

        return wrapped_edit
