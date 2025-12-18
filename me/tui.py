"""
Textual-based TUI for the agent interactive session.

Provides:
- Multi-line input with cursor movement
- Scrollable output with markdown rendering
- Tool call details in expandable panels
- Agent status display
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Static,
    Input,
    RichLog,
    Label,
    Collapsible,
    Rule,
)
from textual.message import Message
from textual import work
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.console import Group

from me.agent.core import Agent, AgentConfig


class ToolCallWidget(Static):
    """Widget displaying a tool call with details."""

    def __init__(self, tool_name: str, tool_input: dict[str, Any] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.tool_input = tool_input or {}

    def compose(self) -> ComposeResult:
        # Format tool input for display
        if self.tool_input:
            details = []
            for key, value in self.tool_input.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                details.append(f"  {key}: {value}")
            detail_text = "\n".join(details)
        else:
            detail_text = "  (no parameters)"

        yield Collapsible(
            Static(detail_text, classes="tool-details"),
            title=f"[bold cyan]{self.tool_name}[/]",
            collapsed=True,
        )


class AgentOutput(Static):
    """Widget for displaying agent output."""

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._content_parts: list[str] = []

    def add_text(self, text: str):
        """Add text output from the agent."""
        self._content_parts.append(text)
        self._update_content()

    def add_tool_call(self, tool_name: str, tool_input: dict[str, Any] | None = None):
        """Add a tool call display."""
        if tool_input:
            details = []
            for key, value in list(tool_input.items())[:5]:  # Limit to 5 params
                if isinstance(value, str):
                    if len(value) > 80:
                        value = value[:80] + "..."
                    value = value.replace("\n", "\\n")
                details.append(f"{key}={value}")
            params = ", ".join(details)
            self._content_parts.append(f"\n[dim][Tool: {tool_name}({params})][/dim]\n")
        else:
            self._content_parts.append(f"\n[dim][Tool: {tool_name}][/dim]\n")
        self._update_content()

    def clear(self):
        """Clear the output."""
        self._content_parts = []
        self._update_content()

    def _update_content(self):
        """Render all content."""
        self.update("".join(self._content_parts))


class ChatMessage(Static):
    """A single chat message (user or agent)."""

    def __init__(self, role: str, content: str, **kwargs):
        # Format the message directly
        if role == "user":
            text = f"[bold green]you:[/] {content}"
        else:
            text = f"[bold blue]agent:[/]\n{content}"
        super().__init__(text, **kwargs)
        self.role = role
        self.message_content = content


class ChatHistory(ScrollableContainer):
    """Scrollable container for chat history."""

    def add_user_message(self, content: str):
        """Add a user message."""
        msg = ChatMessage("user", content)
        self.mount(msg)
        self.scroll_end(animate=False)

    def add_agent_message(self, content: str):
        """Add an agent message."""
        msg = ChatMessage("agent", content)
        self.mount(msg)
        self.scroll_end(animate=False)

    def add_tool_call(self, tool_name: str, tool_input: dict[str, Any] | None = None):
        """Add a tool call indicator."""
        if tool_input:
            details = []
            for key, value in list(tool_input.items())[:3]:
                if isinstance(value, str):
                    if len(value) > 50:
                        value = value[:50] + "..."
                    value = value.replace("\n", "\\n")
                details.append(f"{key}={value}")
            params = ", ".join(details)
            text = f"[dim italic]  > {tool_name}({params})[/]"
        else:
            text = f"[dim italic]  > {tool_name}[/]"
        self.mount(Static(text, classes="tool-indicator"))
        self.scroll_end(animate=False)

    def add_separator(self):
        """Add a visual separator."""
        self.mount(Rule(style="dim"))
        self.scroll_end(animate=False)


class StatusBar(Static):
    """Status bar showing agent state."""

    def __init__(self, **kwargs):
        super().__init__("Loading...", **kwargs)
        self.agent_id = ""
        self.step = 0
        self.status = "ready"

    def update_status(self, agent_id: str = "", step: int = 0, status: str = "ready"):
        self.agent_id = agent_id or self.agent_id
        self.step = step or self.step
        self.status = status
        self._update_content()

    def _update_content(self):
        status_color = {
            "ready": "green",
            "thinking": "yellow",
            "running": "cyan",
            "error": "red",
        }.get(self.status, "white")

        self.update(
            f"[bold]Agent:[/] {self.agent_id} | "
            f"[bold]Step:[/] {self.step} | "
            f"[bold]Status:[/] [{status_color}]{self.status}[/]"
        )


class AgentTUI(App):
    """Textual TUI for agent interactive sessions."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        grid-rows: auto 1fr auto auto;
    }

    #status-bar {
        dock: top;
        height: 1;
        background: $surface;
        padding: 0 1;
    }

    #chat-history {
        height: 100%;
        border: solid $primary;
        padding: 1;
    }

    .user-message {
        margin-bottom: 1;
    }

    .agent-message {
        margin-bottom: 1;
    }

    .tool-indicator {
        margin-left: 2;
    }

    #input-container {
        height: auto;
        padding: 1;
    }

    #prompt-input {
        width: 100%;
    }

    #help-text {
        height: 1;
        color: $text-muted;
        text-align: center;
    }

    Collapsible {
        margin: 0;
        padding-left: 2;
    }

    .tool-details {
        color: $text-muted;
        padding-left: 2;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+d", "quit", "Quit"),
        Binding("escape", "clear_input", "Clear"),
        Binding("ctrl+l", "clear_history", "Clear History"),
    ]

    def __init__(self, agent: Agent, initial_prompt: str | None = None):
        super().__init__()
        self.agent = agent
        self.initial_prompt = initial_prompt
        self._current_response: list[str] = []
        self._processing = False

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status-bar")
        yield ChatHistory(id="chat-history")
        yield Container(
            Input(placeholder="Type a message (Enter to send, /help for commands)", id="prompt-input"),
            id="input-container"
        )
        yield Static(
            "[dim]Enter: send | Ctrl+C: quit | /help for commands[/]",
            id="help-text"
        )

    def on_mount(self):
        """Initialize on mount."""
        status = self.query_one("#status-bar", StatusBar)
        status.update_status(
            agent_id=self.agent.config.agent_id,
            step=self.agent.body.get_current_step_number(),
            status="ready"
        )

        # Focus the input
        self.query_one("#prompt-input", Input).focus()

        # Send initial prompt if provided
        if self.initial_prompt:
            self.call_later(self._send_initial_prompt)

    async def _send_initial_prompt(self):
        """Send the initial prompt after mount."""
        await self._process_input(self.initial_prompt)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input field."""
        if not self._processing:
            text = event.value.strip()
            if text:
                event.input.clear()
                await self._process_input(text)

    async def _process_input(self, text: str):
        """Process user input."""
        if self._processing:
            return

        chat = self.query_one("#chat-history", ChatHistory)
        status = self.query_one("#status-bar", StatusBar)

        # Handle special commands
        if text.lower() in ("exit", "quit", "/exit", "/quit"):
            self.exit()
            return

        if text.startswith("/"):
            await self._handle_command(text)
            return

        # Add user message to chat
        chat.add_user_message(text)

        # Update status
        self._processing = True
        status.update_status(status="thinking")

        # Run agent
        self._current_response = []
        await self._run_agent(text)

        # Finalize
        if self._current_response:
            response_text = "".join(self._current_response)
            chat.add_agent_message(response_text)

        chat.add_separator()
        status.update_status(
            step=self.agent.body.get_current_step_number(),
            status="ready"
        )
        self._processing = False

    async def _handle_command(self, command: str):
        """Handle slash commands."""
        chat = self.query_one("#chat-history", ChatHistory)
        cmd = command.lower().strip()

        if cmd == "/help":
            chat.add_agent_message(
                "**Commands:**\n"
                "- `/body` - Show agent body state\n"
                "- `/memory` - Show memory stats\n"
                "- `/step` - Show current step number\n"
                "- `/clear` - Clear chat history\n"
                "- `/help` - Show this help\n"
                "- `/exit` or `/quit` - Exit\n"
            )
        elif cmd == "/body":
            import json
            body_dict = self.agent.body.to_dict()
            # Summarize for display
            summary = {
                "identity": body_dict.get("identity", {}).get("name"),
                "step": body_dict.get("current_step"),
                "location": body_dict.get("embodiment", {}).get("location"),
                "mood": body_dict.get("embodiment", {}).get("mood"),
                "goals": len(body_dict.get("working_set", {}).get("goals", [])),
                "memory": body_dict.get("memory"),
            }
            chat.add_agent_message(f"```json\n{json.dumps(summary, indent=2)}\n```")
        elif cmd == "/memory":
            stats = self.agent.memory.stats()
            chat.add_agent_message(
                f"**Memory Stats:**\n"
                f"- Total memories: {stats['total_memories']}\n"
                f"- Total runs: {stats['total_runs']}\n"
            )
        elif cmd == "/step":
            step = self.agent.body.get_current_step_number()
            chat.add_agent_message(f"Current step: {step}")
        elif cmd == "/clear":
            chat.remove_children()
            chat.add_agent_message("Chat history cleared.")
        else:
            chat.add_agent_message(f"Unknown command: {command}\nType `/help` for available commands.")

        chat.add_separator()

    @work(exclusive=True)
    async def _run_agent(self, prompt: str):
        """Run the agent asynchronously."""
        from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, ResultMessage

        chat = self.query_one("#chat-history", ChatHistory)
        status = self.query_one("#status-bar", StatusBar)

        try:
            async for message in self.agent.run(prompt):
                if isinstance(message, AssistantMessage):
                    status.update_status(status="running")
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            self._current_response.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            # Show tool call with parameters
                            chat.add_tool_call(block.name, block.input)

                elif isinstance(message, ResultMessage):
                    pass  # Handled in _process_input

        except Exception as e:
            status.update_status(status="error")
            chat.add_agent_message(f"[red]Error: {e}[/]")

    def action_quit(self):
        """Quit the application."""
        self.exit()

    def action_clear_input(self):
        """Clear the input field."""
        self.query_one("#prompt-input", Input).clear()

    def action_clear_history(self):
        """Clear chat history."""
        chat = self.query_one("#chat-history", ChatHistory)
        chat.remove_children()


async def run_tui(agent: Agent, initial_prompt: str | None = None):
    """Run the TUI application."""
    app = AgentTUI(agent, initial_prompt)
    await app.run_async()
