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

    /* Stats Panel */
    #stats-panel {
        height: 100%;
        padding: 1 2;
        background: #161b22;
        border-left: solid #30363d;
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

    def compose(self) -> ComposeResult:
        with Container(id="main-container"):
            with Vertical(id="chat-panel"):
                yield ChatHistory(id="chat-history")
            yield StatsPanel(self.agent, id="stats-panel")
        with Vertical(id="bottom-bar"):
            yield SlashCommandList(id="slash-commands")
            with Horizontal(id="input-row"):
                yield StatusIndicator(id="status-indicator")
                yield Input(placeholder="Message… (/ for commands)", id="prompt-input")
            yield Static("[dim]enter[/] send  [dim]ctrl+c[/] quit  [dim]/[/] commands", id="help-bar")

    def on_mount(self):
        """Initialize on mount."""
        self.query_one("#prompt-input", Input).focus()
        self.query_one("#status-indicator", StatusIndicator).status = "ready"

        if self.initial_prompt:
            self.call_later(self._send_initial_prompt)

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
        """Handle Enter key - send message immediately."""
        logger.info("on_input_submitted called")
        slash_list = self.query_one("#slash-commands", SlashCommandList)
        slash_list.display = False

        text = event.value.strip()
        if not text:
            logger.debug("Empty text, returning")
            return

        event.input.clear()
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

        # Finalize
        if self._current_response:
            response_text = "".join(self._current_response)
            logger.info(f"Adding agent message with {len(response_text)} chars")
            chat.add_agent_message(response_text)
        else:
            logger.warning("No response content collected!")

        chat.add_separator()
        status.status = "ready"
        self._processing = False

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
        """Run the agent asynchronously."""
        from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, ResultMessage

        chat = self.query_one("#chat-history", ChatHistory)
        status = self.query_one("#status-indicator", StatusIndicator)
        stats = self.query_one("#stats-panel", StatsPanel)

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
                            self._current_response.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            logger.debug(f"  ToolUseBlock: {block.name}")
                            chat.add_tool_call(block.name, block.input)
                            stats.increment_tool_calls()

                elif isinstance(message, ResultMessage):
                    logger.debug("ResultMessage received")

            logger.info(f"agent.run completed. Total messages: {message_count}")

            # Debug: show if no messages received
            if message_count == 0:
                logger.warning("No messages received from agent")
                chat.add_agent_message("*No response from agent*")
            elif not self._current_response:
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
