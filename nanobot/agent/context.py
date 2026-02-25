"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""
    
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
    
    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        parts = [self._get_identity()]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills (Reference Guides)

Skills are NOT tools — they are reference documents that teach you HOW to accomplish tasks using your actual tools (exec, web_search, web_fetch, read_file, etc.).
Do NOT try to call a skill by name. Instead, read the SKILL.md file to learn the approach, then use the appropriate tool.
For example, the "weather" skill teaches you to use `exec` with `curl wttr.in` or `web_search` — it is NOT a callable tool.

{skills_summary}""")

        return "\n\n---\n\n".join(parts)
    
    @staticmethod
    def _market_session() -> tuple[str, str]:
        """Return (ET datetime string, market session status)."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        et = datetime.now(ZoneInfo("America/New_York"))
        et_str = et.strftime("%Y-%m-%d %H:%M (%A)")
        weekday = et.weekday()  # 0=Mon, 6=Sun
        h, m = et.hour, et.minute
        t = h * 60 + m  # minutes since midnight
        if weekday >= 5:
            session = "Closed (Weekend)"
        elif t < 240:       # before 4:00
            session = "Closed"
        elif t < 570:       # 4:00 – 9:30
            session = "Pre-Market"
        elif t < 960:       # 9:30 – 16:00
            session = "Market Hours"
        elif t < 1200:      # 16:00 – 20:00
            session = "After-Hours"
        else:
            session = "Closed"
        return et_str, session

    def _get_identity(self) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        et_str, market_session = self._market_session()

        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant. You have access to tools that allow you to:
- Read, write, and edit files
- Execute shell commands
- Search the web and fetch web pages
- Send messages to users on chat channels
- Spawn subagents for complex background tasks

## Current Time
{et_str} (US Eastern)
US Market: {market_session}

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md (write important facts here)
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable)
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

## nanobot Guidelines
- State intent before tool calls, but NEVER predict or claim results before receiving them.
- Before modifying a file, read it first. Do not assume files or directories exist.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.
- Ask for clarification when the request is ambiguous.

## Response Rules

- **Be concise for simple Q&A.** 2-3 sentences for simple questions. No filler, no preamble.
- **But present data in full.** When a tool returns a numbered list or multiple items, you MUST output every single item. Never truncate, summarize, or cherry-pick from tool results.
- **Never show your reasoning.** No "The user asked...", "I should...", "I will...", "Let me...". Just give the answer directly.
- **No recap sections.** No "Summary of Actions", "Next Steps", "Here's what I did". The user can see what you did.
- **Include key data in your reply.** Numbers, prices, temperatures — the actual useful information, not meta-commentary about it.
- **When saving files, state the path.** e.g. "已保存到 ~/workspace/analysis.md"
- **Consolidate sources.** Merge duplicate info into one clean answer. Don't list every source separately.
- **Match the user's language.** If the user writes in Chinese, reply in Chinese.

## Tool Resilience

- **Never give up after one failure.** If a tool fails, try an alternative before telling the user you can't help.
- **Fallback chain:** web_search ↔ exec (curl) ↔ web_fetch. Try at least two approaches.
- **Do NOT memorize tool failures.** Temporary errors (API timeouts, network issues) must never be written to MEMORY.md. Only store lasting, useful facts.
- **Always try a tool** for real-time questions (weather, news, prices). Never say "I cannot access real-time data" without trying first.

Reply directly with text for conversations. Only use the 'message' tool to send to a specific chat channel."""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)
    
    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []
        
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
        
        return "\n\n".join(parts) if parts else ""
    
    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        return [
            {"role": "system", "content": self.build_system_prompt(skill_names)},
            *history,
            {"role": "user", "content": self._build_runtime_context(channel, chat_id)},
            {"role": "user", "content": self._build_user_content(current_message, media)},
        ]

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text
        
        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        
        if not images:
            return text
        return images + [{"type": "text", "text": text}]
    
    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages
    
    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        messages.append(msg)
        return messages
