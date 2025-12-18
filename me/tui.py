"""
Textual-based TUI for the agent interactive session.

Provides:
- Multi-line input with cursor movement
- Scrollable output with markdown rendering
- Right sidebar with agent stats
- Tool call tracking
- Slash command autocomplete
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
import time

# Set up file logging for TUI debugging
_log_dir = Path.home() / ".me" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
_log_file = _log_dir / "tui.log"
logging.basicConfig(
    filename=str(_log_file),
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tui")

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Static,
    Input,
    Label,
    Rule,
    OptionList,
)
from textual.widgets.option_list import Option
from textual.reactive import reactive
from rich.markdown import Markdown
from rich.text import Text

from me.agent.core import Agent, AgentConfig


# Available slash commands
SLASH_COMMANDS = {
    "/help": "Show available commands",
    "/body": "Show agent body state",
    "/memory": "Show memory stats",
    "/goals": "Show current goals",
    "/queue": "Focus message queue",
    "/clear": "Clear chat history",
    "/exit": "Exit the TUI",
}


class ChatMessage(Static):
    """A single chat message (user or agent)."""

    def __init__(self, role: str, content: str, timestamp: datetime | None = None, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.message_content = content
        self.timestamp = timestamp or datetime.now()
        self.add_class(f"{role}-message")

    def on_mount(self):
        """Render the message with proper markdown support."""
        time_str = self.timestamp.strftime("%H:%M")

        if self.role == "user":
            text = Text()
            text.append(f"{time_str} ", style="#484f58")
            text.append("› ", style="#58a6ff")
            text.append(self.message_content, style="#c9d1d9")
            self.update(text)


class ChatHistory(ScrollableContainer):
    """Scrollable container for chat history."""

    def add_user_message(self, content: str):
        """Add a user message."""
        msg = ChatMessage("user", content)
        self.mount(msg)
        self.scroll_end(animate=False)

    def add_agent_message(self, content: str):
        """Add an agent message."""
        from rich.console import Group
        logger.debug(f"add_agent_message called with content length: {len(content)}")
        logger.debug(f"content preview: {content[:200] if content else '(empty)'}")
        timestamp = datetime.now().strftime("%H:%M")
        # Combine header and markdown into single renderable
        header = Text()
        header.append(f"{timestamp} ", style="#484f58")
        header.append("◆", style="#a371f7")
        group = Group(header, Markdown(content))
        widget = Static(group, classes="agent-message")
        self.mount(widget)
        logger.debug("agent message widget mounted")
        self.scroll_end(animate=False)

    def add_tool_call(self, tool_name: str, tool_input: dict[str, Any] | None = None):
        """Add a tool call indicator."""
        if tool_input:
            details = []
            for key, value in list(tool_input.items())[:2]:
                if isinstance(value, str):
                    if len(value) > 40:
                        value = value[:40] + "…"
                    value = value.replace("\n", "↵")
                details.append(f"{key}={value}")
            params = ", ".join(details)
            if len(tool_input) > 2:
                params += ", …"
            text = f"[#30363d]   ╰─[/] [#8b949e]{tool_name}[/][#484f58]({params})[/]"
        else:
            text = f"[#30363d]   ╰─[/] [#8b949e]{tool_name}[/]"
        self.mount(Static(text, classes="tool-indicator"))
        self.scroll_end(animate=False)

    def add_separator(self):
        """Add a visual separator."""
        self.mount(Static("[#21262d]───[/]", classes="separator"))
        self.scroll_end(animate=False)

    def start_streaming_message(self) -> "StreamingMessage":
        """Start a new streaming agent message."""
        msg = StreamingMessage()
        self.mount(msg)
        self.scroll_end(animate=False)
        return msg


class StreamingMessage(Static):
    """A streaming agent message that updates as content arrives."""

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._content_parts: list[str] = []
        self._timestamp = datetime.now().strftime("%H:%M")
        self.add_class("agent-message")

    def append_text(self, text: str):
        """Append text to the message and refresh display."""
        self._content_parts.append(text)
        self._refresh_display()

    def _refresh_display(self):
        """Update the displayed content."""
        from rich.console import Group
        content = "".join(self._content_parts)
        header = Text()
        header.append(f"{self._timestamp} ", style="#484f58")
        header.append("◆", style="#a371f7")
        if content:
            group = Group(header, Markdown(content))
        else:
            # Show typing indicator when no content yet
            header.append(" …", style="#484f58 italic")
            group = header
        self.update(group)
        # Scroll parent to show new content
        if self.parent:
            self.parent.scroll_end(animate=False)

    def get_content(self) -> str:
        """Get the full content."""
        return "".join(self._content_parts)


class StatsPanel(Static):
    """Right panel showing agent stats and info."""

    def __init__(self, agent: Agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.start_time = time.time()
        self.tool_calls = 0
        self.messages = 0

    def on_mount(self):
        """Start periodic updates."""
        self.set_interval(1.0, self._refresh_stats)
        self._refresh_stats()

    def increment_tool_calls(self):
        self.tool_calls += 1
        self._refresh_stats()

    def increment_messages(self):
        self.messages += 1
        self._refresh_stats()

    def _refresh_stats(self):
        """Update the stats display."""
        # Session duration
        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)
        hours, mins = divmod(mins, 60)
        if hours:
            duration = f"{hours}h {mins}m"
        elif mins:
            duration = f"{mins}m {secs}s"
        else:
            duration = f"{secs}s"

        # Agent info
        body = self.agent.body
        identity = body.identity
        step = body.get_current_step_number()

        # Memory stats
        try:
            mem_stats = self.agent.memory.stats()
            memories = mem_stats.get('total_memories', 0)
        except Exception:
            memories = "?"

        # Goals
        goals = body.working_set.goals[:3] if body.working_set.goals else []
        goals_text = "\n".join(f"  [#484f58]•[/] [#c9d1d9]{g[:22]}{'…' if len(g) > 22 else ''}[/]" for g in goals)
        if not goals_text:
            goals_text = "  [#484f58]none[/]"
        if len(body.working_set.goals) > 3:
            goals_text += f"\n  [#484f58]+{len(body.working_set.goals) - 3} more[/]"

        # Build stats display
        mood = body.embodiment.mood or "neutral"
        stats = f"""[#8b949e bold]Agent[/]
[#484f58]id[/]     [#c9d1d9]{identity.agent_id[:8]}[/]
[#484f58]step[/]   [#c9d1d9]{step}[/]

[#8b949e bold]Session[/]
[#484f58]time[/]   [#c9d1d9]{duration}[/]
[#484f58]msgs[/]   [#c9d1d9]{self.messages}[/]
[#484f58]tools[/]  [#c9d1d9]{self.tool_calls}[/]

[#8b949e bold]Memory[/]
[#484f58]stored[/] [#c9d1d9]{memories}[/]

[#8b949e bold]Goals[/]
{goals_text}

[#8b949e bold]Mood[/]
  [#c9d1d9]{mood}[/]
"""
        self.update(stats)


class SlashCommandList(OptionList):
    """Dropdown for slash command autocomplete."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._filter = ""

    def filter_commands(self, prefix: str):
        """Filter commands by prefix and update the list."""
        self._filter = prefix
        self.clear_options()
        for cmd, desc in SLASH_COMMANDS.items():
            if cmd.startswith(prefix):
                self.add_option(Option(f"{cmd}  [dim]{desc}[/]", id=cmd))
        if self.option_count > 0:
            self.display = True
            self.highlighted = 0
        else:
            self.display = False

    def get_selected_command(self) -> str | None:
        """Get the currently highlighted command."""
        if self.highlighted is not None and self.option_count > 0:
            option = self.get_option_at_index(self.highlighted)
            return option.id
        return None


class StatusIndicator(Static):
    """Minimal status indicator in the input area."""

    status = reactive("ready")

    def watch_status(self, status: str):
        icons = {
            "ready": "[#3fb950]●[/]",
            "thinking": "[#d29922]◐[/]",
            "running": "[#58a6ff]↻[/]",
            "error": "[#f85149]✗[/]",
        }
        self.update(icons.get(status, "[#484f58]○[/]"))


class QueuePanel(Static):
    """Panel showing queued messages."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._queue: list[str] = []
        self._selected_index: int = -1
        self._focused = False

    def add_message(self, text: str) -> int:
        """Add a message to the queue. Returns queue position."""
        self._queue.append(text)
        self._refresh_display()
        return len(self._queue)

    def pop_message(self) -> str | None:
        """Remove and return the first message from queue."""
        if self._queue:
            msg = self._queue.pop(0)
            if self._selected_index >= len(self._queue):
                self._selected_index = len(self._queue) - 1
            self._refresh_display()
            return msg
        return None

    def remove_selected(self) -> str | None:
        """Remove the selected message from queue."""
        if self._queue and 0 <= self._selected_index < len(self._queue):
            msg = self._queue.pop(self._selected_index)
            if self._selected_index >= len(self._queue):
                self._selected_index = max(0, len(self._queue) - 1)
            self._refresh_display()
            return msg
        return None

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def count(self) -> int:
        return len(self._queue)

    def set_focused(self, focused: bool):
        self._focused = focused
        if focused and self._queue and self._selected_index < 0:
            self._selected_index = 0
        self._refresh_display()

    def move_selection(self, delta: int):
        if self._queue:
            self._selected_index = max(0, min(len(self._queue) - 1, self._selected_index + delta))
            self._refresh_display()

    def _refresh_display(self):
        if not self._queue:
            self.update("[#484f58]Queue empty[/]")
            return

        lines = [f"[#8b949e bold]Queue[/] [#484f58]({len(self._queue)})[/]"]
        for i, msg in enumerate(self._queue):
            truncated = msg[:18] + "…" if len(msg) > 18 else msg
            if self._focused and i == self._selected_index:
                lines.append(f"[#58a6ff]› {truncated}[/]")
            else:
                lines.append(f"[#484f58]  {truncated}[/]")
        self.update("\n".join(lines))


class AgentTUI(App):
    """Textual TUI for agent interactive sessions."""

    CSS = """
    /* Layout */
    Screen {
        layout: grid;
        grid-size: 1;
        grid-rows: 1fr auto;
        background: #0d1117;
    }

    #main-container {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 26;
    }

    #chat-panel {
        height: 100%;
    }

    #chat-history {
        height: 100%;
        padding: 1 2;
        scrollbar-size: 1 1;
        scrollbar-color: #30363d;
    }

    /* Right Sidebar */
    #right-sidebar {
        height: 100%;
        background: #161b22;
        border-left: solid #30363d;
        layout: grid;
        grid-size: 1;
        grid-rows: 1fr auto;
    }

    #stats-panel {
        padding: 1 2;
    }

    #queue-panel {
        padding: 1 2;
        border-top: solid #30363d;
        min-height: 5;
        max-height: 12;
    }

    /* Bottom Bar */
    #bottom-bar {
        dock: bottom;
        height: auto;
        background: #161b22;
        border-top: solid #30363d;
    }

    #slash-commands {
        display: none;
        height: auto;
        max-height: 8;
        margin: 0 2;
        background: #21262d;
        border: solid #30363d;
    }

    #slash-commands > .option-list--option {
        padding: 0 1;
        color: #c9d1d9;
    }

    #slash-commands > .option-list--option-highlighted {
        background: #388bfd33;
        color: #58a6ff;
    }

    #input-row {
        height: 3;
        padding: 0 1;
        layout: horizontal;
    }

    #status-indicator {
        width: 3;
        height: 3;
        content-align: center middle;
        color: #8b949e;
    }

    #prompt-input {
        width: 1fr;
        background: #0d1117;
        border: solid #30363d;
        padding: 0 1;
        color: #c9d1d9;
    }

    #prompt-input:focus {
        border: solid #58a6ff;
    }

    #prompt-input > .input--placeholder {
        color: #484f58;
    }

    #help-bar {
        height: 1;
        color: #484f58;
        text-align: center;
        padding: 0 1;
    }

    /* Messages */
    .user-message {
        margin: 1 0 0 0;
        color: #c9d1d9;
    }

    .agent-message {
        margin: 1 0 0 0;
        padding: 0 0 0 3;
        color: #c9d1d9;
    }

    .tool-indicator {
        color: #484f58;
    }

    .separator {
        margin: 1 0;
        text-align: center;
        color: #21262d;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+d", "quit", "Quit", show=False),
        Binding("escape", "dismiss_autocomplete", "Dismiss", show=False),
        Binding("ctrl+l", "clear_history", "Clear", show=False),
        Binding("tab", "select_autocomplete", "Select", show=False),
        Binding("down", "autocomplete_down", "Down", show=False),
        Binding("up", "autocomplete_up", "Up", show=False),
    ]


    def __init__(self, agent: Agent, initial_prompt: str | None = None):
        super().__init__()
        self.agent = agent
        self.initial_prompt = initial_prompt
        self._current_response: list[str] = []
        self._processing = False
        self._queue_focused = False

    def compose(self) -> ComposeResult:
        with Container(id="main-container"):
            with Vertical(id="chat-panel"):
                yield ChatHistory(id="chat-history")
            with Vertical(id="right-sidebar"):
                yield StatsPanel(self.agent, id="stats-panel")
                yield QueuePanel(id="queue-panel")
        with Vertical(id="bottom-bar"):
            yield SlashCommandList(id="slash-commands")
            with Horizontal(id="input-row"):
                yield StatusIndicator(id="status-indicator")
                yield Input(placeholder="Message… (/ for commands)", id="prompt-input")
            yield Static("[dim]enter[/] send/queue  [dim]ctrl+s[/] priority  [dim]/[/] commands", id="help-bar")

    def on_mount(self):
        """Initialize on mount."""
        self.query_one("#prompt-input", Input).focus()
        self.query_one("#status-indicator", StatusIndicator).status = "ready"

        if self.initial_prompt:
            self.call_later(self._send_initial_prompt)

    async def on_key(self, event) -> None:
        """Handle special key combinations."""
        logger.debug(f"on_key: {event.key!r}")

        # Cmd+Enter or Ctrl+Enter to interrupt and send immediately
        # On Mac terminals, this often comes through as escape sequences
        # We'll use Ctrl+S as a reliable alternative (common "send" shortcut)
        if event.key in ("ctrl+s",):
            logger.info("Interrupt send triggered")
            event.prevent_default()
            event.stop()
            await self._interrupt_and_send()
            return

        # Queue focus mode key handling
        if self._queue_focused:
            queue = self.query_one("#queue-panel", QueuePanel)

            if event.key == "up":
                queue.move_selection(-1)
                event.prevent_default()
                event.stop()
            elif event.key == "down":
                queue.move_selection(1)
                event.prevent_default()
                event.stop()
            elif event.key in ("delete", "backspace"):
                removed = queue.remove_selected()
                if removed:
                    self.notify(f"Removed: {removed[:20]}…" if len(removed) > 20 else f"Removed: {removed}", timeout=1)
                if queue.is_empty():
                    self._exit_queue_focus()
                event.prevent_default()
                event.stop()
            elif event.key == "escape":
                self._exit_queue_focus()
                event.prevent_default()
                event.stop()

    async def _interrupt_and_send(self):
        """Interrupt current processing and send input immediately."""
        input_field = self.query_one("#prompt-input", Input)
        text = input_field.value.strip()

        if not text:
            self.notify("Nothing to send", timeout=1)
            return

        input_field.clear()

        # If processing, this will queue but we'll show it's prioritized
        if self._processing:
            queue = self.query_one("#queue-panel", QueuePanel)
            # Insert at front of queue (priority)
            queue._queue.insert(0, text)
            queue._refresh_display()
            self.notify("Priority queued (will send next)", timeout=1.5)
            logger.info(f"Priority queued: {text[:50]}")
        else:
            # Send immediately
            logger.info(f"Interrupt sending: {text[:50]}")
            if text.startswith("/"):
                await self._handle_command(text)
            else:
                await self._process_input(text)

    async def _send_initial_prompt(self):
        """Send the initial prompt after mount."""
        await self._process_input(self.initial_prompt)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes for slash command autocomplete."""
        text = event.value
        slash_list = self.query_one("#slash-commands", SlashCommandList)

        if text.startswith("/") and not self._processing:
            slash_list.filter_commands(text)
        else:
            slash_list.display = False

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle slash command selection from autocomplete."""
        if event.option_list.id == "slash-commands":
            command = event.option.id
            input_field = self.query_one("#prompt-input", Input)
            input_field.value = command
            input_field.cursor_position = len(command)
            event.option_list.display = False
            input_field.focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key - send immediately or queue if agent is running."""
        logger.info("on_input_submitted called")
        slash_list = self.query_one("#slash-commands", SlashCommandList)
        slash_list.display = False

        # Exit queue focus mode when submitting
        if self._queue_focused:
            self._exit_queue_focus()

        text = event.value.strip()
        if not text:
            logger.debug("Empty text, returning")
            return

        event.input.clear()

        # If agent is running, queue the message
        if self._processing:
            queue = self.query_one("#queue-panel", QueuePanel)
            pos = queue.add_message(text)
            logger.info(f"Queued message at position {pos}: {text[:50]}")
            self.notify(f"Queued #{pos}", timeout=1)
            return

        logger.info(f"Sending message: {text[:50]}")

        # Process immediately
        if text.startswith("/"):
            await self._handle_command(text)
        else:
            await self._process_input(text)

    async def _process_input(self, text: str):
        """Process user input (called for non-slash messages only)."""
        logger.info(f"_process_input called with: {text[:50]}")
        if self._processing:
            logger.warning("Already processing, returning early")
            return

        chat = self.query_one("#chat-history", ChatHistory)
        status = self.query_one("#status-indicator", StatusIndicator)
        stats = self.query_one("#stats-panel", StatsPanel)

        # Add user message to chat
        chat.add_user_message(text)
        stats.increment_messages()

        # Update status
        self._processing = True
        status.status = "thinking"

        # Run agent
        self._current_response = []
        logger.info("Calling _run_agent...")
        await self._run_agent(text)
        logger.info(f"_run_agent completed. Response parts: {len(self._current_response)}")

        # Finalize (streaming message already displayed the content)
        if self._current_response:
            logger.info(f"Agent response complete: {len(''.join(self._current_response))} chars")
        else:
            logger.warning("No response content collected!")

        chat.add_separator()
        status.status = "ready"
        self._processing = False

        # Process next queued message if any
        await self._process_queue()

    async def _process_queue(self):
        """Process queued messages one by one."""
        queue = self.query_one("#queue-panel", QueuePanel)
        if not queue.is_empty() and not self._processing:
            next_msg = queue.pop_message()
            if next_msg:
                logger.info(f"Processing queued message: {next_msg[:50]}")
                if next_msg.startswith("/"):
                    await self._handle_command(next_msg)
                else:
                    await self._process_input(next_msg)

    def _enter_queue_focus(self):
        """Enter queue focus mode."""
        self._queue_focused = True
        queue = self.query_one("#queue-panel", QueuePanel)
        queue.set_focused(True)
        self.query_one("#prompt-input", Input).placeholder = "↑↓ navigate, del remove, esc exit"

    def _exit_queue_focus(self):
        """Exit queue focus mode."""
        self._queue_focused = False
        queue = self.query_one("#queue-panel", QueuePanel)
        queue.set_focused(False)
        self.query_one("#prompt-input", Input).placeholder = "Message… (/ for commands)"
        self.query_one("#prompt-input", Input).focus()

    async def _handle_command(self, command: str):
        """Handle slash commands."""
        chat = self.query_one("#chat-history", ChatHistory)
        cmd = command.lower().strip()

        # Show command in chat
        chat.add_user_message(command)

        if cmd == "/help":
            help_text = "**Commands:**\n"
            for c, desc in SLASH_COMMANDS.items():
                help_text += f"- `{c}` — {desc}\n"
            chat.add_agent_message(help_text)
        elif cmd == "/body":
            import json
            body_dict = self.agent.body.to_dict()
            summary = {
                "identity": body_dict.get("identity", {}).get("name"),
                "step": body_dict.get("current_step"),
                "location": body_dict.get("embodiment", {}).get("location"),
                "mood": body_dict.get("embodiment", {}).get("mood"),
            }
            chat.add_agent_message(f"```json\n{json.dumps(summary, indent=2)}\n```")
        elif cmd == "/memory":
            stats = self.agent.memory.stats()
            chat.add_agent_message(
                f"**Memory Stats:**\n"
                f"- Total memories: {stats['total_memories']}\n"
                f"- Total runs: {stats['total_runs']}\n"
            )
        elif cmd == "/goals":
            goals = self.agent.body.working_set.goals
            if goals:
                goals_text = "**Current Goals:**\n" + "\n".join(f"- {g}" for g in goals)
            else:
                goals_text = "No active goals."
            chat.add_agent_message(goals_text)
        elif cmd == "/queue":
            queue = self.query_one("#queue-panel", QueuePanel)
            if queue.is_empty():
                chat.add_agent_message("Queue is empty.")
            else:
                self._enter_queue_focus()
                chat.add_agent_message(f"Focused on queue ({queue.count()} messages). Use ↑↓ to navigate, Delete to remove, Esc to exit.")
            return  # Don't add separator, user is in queue mode
        elif cmd == "/clear":
            chat.remove_children()
            chat.add_agent_message("Chat cleared.")
        elif cmd in ("/exit", "/quit"):
            self.exit()
            return
        else:
            chat.add_agent_message(f"Unknown command: `{command}`\nType `/help` for available commands.")

        chat.add_separator()

    async def _run_agent(self, prompt: str):
        """Run the agent asynchronously with streaming output."""
        from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, ResultMessage

        chat = self.query_one("#chat-history", ChatHistory)
        status = self.query_one("#status-indicator", StatusIndicator)
        stats = self.query_one("#stats-panel", StatsPanel)

        streaming_msg: StreamingMessage | None = None

        try:
            message_count = 0
            logger.info(f"Starting agent.run with prompt: {prompt[:50]}")
            async for message in self.agent.run(prompt):
                message_count += 1
                logger.debug(f"Received message {message_count}: {type(message).__name__}")
                if isinstance(message, AssistantMessage):
                    status.status = "running"
                    logger.debug(f"AssistantMessage has {len(message.content)} content blocks")
                    for block in message.content:
                        logger.debug(f"  Block type: {type(block).__name__}")
                        if isinstance(block, TextBlock):
                            logger.debug(f"  TextBlock: {block.text[:100] if block.text else '(empty)'}")
                            # Start streaming message on first text
                            if streaming_msg is None:
                                streaming_msg = chat.start_streaming_message()
                            streaming_msg.append_text(block.text)
                            self._current_response.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            logger.debug(f"  ToolUseBlock: {block.name}")
                            chat.add_tool_call(block.name, block.input)
                            stats.increment_tool_calls()

                elif isinstance(message, ResultMessage):
                    logger.debug("ResultMessage received")

            logger.info(f"agent.run completed. Total messages: {message_count}")

            # Handle cases where no response was shown
            if message_count == 0:
                logger.warning("No messages received from agent")
                chat.add_agent_message("*No response from agent*")
            elif streaming_msg is None:
                logger.warning(f"Agent sent {message_count} messages but no text content")
                chat.add_agent_message(f"*Agent sent {message_count} messages but no text content*")

        except Exception as e:
            import traceback
            logger.error(f"Exception in _run_agent: {e}")
            logger.error(traceback.format_exc())
            status.status = "error"
            chat.add_agent_message(f"**Error:** {e}\n```\n{traceback.format_exc()}\n```")

    def action_quit(self):
        """Quit the application."""
        self.exit()

    def action_dismiss_autocomplete(self):
        """Dismiss autocomplete and clear input."""
        slash_list = self.query_one("#slash-commands", SlashCommandList)
        if slash_list.display:
            slash_list.display = False
        else:
            self.query_one("#prompt-input", Input).clear()

    def action_clear_history(self):
        """Clear chat history."""
        chat = self.query_one("#chat-history", ChatHistory)
        chat.remove_children()

    def action_select_autocomplete(self):
        """Select the highlighted autocomplete option."""
        slash_list = self.query_one("#slash-commands", SlashCommandList)
        if slash_list.display and slash_list.option_count > 0:
            command = slash_list.get_selected_command()
            if command:
                input_field = self.query_one("#prompt-input", Input)
                input_field.value = command
                input_field.cursor_position = len(command)
                slash_list.display = False

    def action_autocomplete_down(self):
        """Move down in autocomplete list."""
        slash_list = self.query_one("#slash-commands", SlashCommandList)
        if slash_list.display and slash_list.option_count > 0:
            slash_list.action_cursor_down()

    def action_autocomplete_up(self):
        """Move up in autocomplete list."""
        slash_list = self.query_one("#slash-commands", SlashCommandList)
        if slash_list.display and slash_list.option_count > 0:
            slash_list.action_cursor_up()


async def run_tui(agent: Agent, initial_prompt: str | None = None):
    """Run the TUI application."""
    app = AgentTUI(agent, initial_prompt)
    await app.run_async()
