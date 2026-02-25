"""Cron tool for scheduling reminders and tasks."""

import json
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule


class CronTool(Tool):
    """Tool to schedule reminders and recurring tasks."""
    
    def __init__(self, cron_service: CronService):
        self._cron = cron_service
        self._channel = ""
        self._chat_id = ""
    
    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current session context for delivery."""
        self._channel = channel
        self._chat_id = chat_id
    
    @property
    def name(self) -> str:
        return "cron"
    
    @property
    def description(self) -> str:
        return (
            "Schedule reminders, recurring tasks, or direct tool calls. Actions: add, list, remove. "
            "Use tool_name for direct tool execution (no LLM cost per run)."
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove"],
                    "description": "Action to perform"
                },
                "message": {
                    "type": "string",
                    "description": "Reminder message (for add)"
                },
                "every_seconds": {
                    "type": "integer",
                    "description": "Interval in seconds (for recurring tasks)"
                },
                "cron_expr": {
                    "type": "string",
                    "description": "Cron expression like '0 9 * * *' (for scheduled tasks)"
                },
                "tz": {
                    "type": "string",
                    "description": "IANA timezone for cron expressions (e.g. 'America/Vancouver')"
                },
                "at": {
                    "type": "string",
                    "description": "ISO datetime for one-time execution (e.g. '2026-02-12T10:30:00')"
                },
                "job_id": {
                    "type": "string",
                    "description": "Job ID (for remove)"
                },
                "tool_name": {
                    "type": "string",
                    "description": "Workspace tool to execute directly, bypassing LLM (e.g. 'alert_check'). Use for recurring checks."
                },
                "tool_args": {
                    "type": "string",
                    "description": "JSON arguments for the tool (e.g. '{\"tickers\":[\"AAPL\"],\"conditions\":{\"rsi_above\":70}}')"
                },
                "silent_marker": {
                    "type": "string",
                    "description": "If tool output contains this string, suppress delivery (e.g. 'NO_ALERTS_TRIGGERED')"
                }
            },
            "required": ["action"]
        }
    
    async def execute(
        self,
        action: str,
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        tz: str | None = None,
        at: str | None = None,
        job_id: str | None = None,
        tool_name: str | None = None,
        tool_args: str | None = None,
        silent_marker: str | None = None,
        **kwargs: Any
    ) -> str:
        if action == "add":
            return self._add_job(message, every_seconds, cron_expr, tz, at,
                                 tool_name, tool_args, silent_marker)
        elif action == "list":
            return self._list_jobs()
        elif action == "remove":
            return self._remove_job(job_id)
        return f"Unknown action: {action}"

    def _add_job(
        self,
        message: str,
        every_seconds: int | None,
        cron_expr: str | None,
        tz: str | None,
        at: str | None,
        tool_name: str | None = None,
        tool_args: str | None = None,
        silent_marker: str | None = None,
    ) -> str:
        if not message and not tool_name:
            return "Error: either message or tool_name is required for add"
        if tool_args:
            try:
                json.loads(tool_args)
            except json.JSONDecodeError:
                return "Error: tool_args must be valid JSON"
        if not self._channel or not self._chat_id:
            return "Error: no session context (channel/chat_id)"
        if tz and not cron_expr:
            return "Error: tz can only be used with cron_expr"
        if tz:
            from zoneinfo import ZoneInfo
            try:
                ZoneInfo(tz)
            except (KeyError, Exception):
                return f"Error: unknown timezone '{tz}'"
        
        # Build schedule
        delete_after = False
        if every_seconds:
            schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)
        elif cron_expr:
            schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
        elif at:
            from datetime import datetime
            dt = datetime.fromisoformat(at)
            at_ms = int(dt.timestamp() * 1000)
            schedule = CronSchedule(kind="at", at_ms=at_ms)
            delete_after = True
        else:
            return "Error: either every_seconds, cron_expr, or at is required"
        
        job_name = tool_name or message[:30]
        job = self._cron.add_job(
            name=job_name,
            schedule=schedule,
            message=message,
            deliver=True,
            channel=self._channel,
            to=self._chat_id,
            delete_after_run=delete_after,
            tool_name=tool_name,
            tool_args=tool_args,
            silent_marker=silent_marker,
        )
        kind_label = "tool_call" if tool_name else "reminder"
        return f"Created {kind_label} job '{job.name}' (id: {job.id})"
    
    def _list_jobs(self) -> str:
        jobs = self._cron.list_jobs()
        if not jobs:
            return "No scheduled jobs."
        lines = []
        for j in jobs:
            if j.payload.kind == "tool_call":
                lines.append(f"- [tool_call] {j.payload.tool_name} (id: {j.id}, {j.schedule.kind})")
            else:
                lines.append(f"- {j.name} (id: {j.id}, {j.schedule.kind})")
        return "Scheduled jobs:\n" + "\n".join(lines)
    
    def _remove_job(self, job_id: str | None) -> str:
        if not job_id:
            return "Error: job_id is required for remove"
        if self._cron.remove_job(job_id):
            return f"Removed job {job_id}"
        return f"Job {job_id} not found"
