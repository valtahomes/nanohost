"""Notification service — polls ~/.nanobot/notifications/ and delivers to active channels.

Background tools (e.g. deep_research) write JSON notification files.
This service picks them up and sends to the user's active channel.

Notification file format:  ~/.nanobot/notifications/{uuid}.json
  {"message": "Research complete: NVDA ..."}

Channel routing:
  The service tracks the last active channel/chat_id from inbound messages.
  Notifications are delivered to the most recently active channel.
"""

import asyncio
import json
import os
from pathlib import Path

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus

NOTIF_DIR = Path.home() / ".nanobot" / "notifications"
POLL_INTERVAL_S = 5


class NotificationService:
    """Polls notification directory and sends messages to active channels."""

    def __init__(self, bus: MessageBus, interval_s: int = POLL_INTERVAL_S):
        self.bus = bus
        self.interval_s = interval_s
        self._running = False
        self._task: asyncio.Task | None = None
        # Last active channel/chat_id (updated by agent loop)
        self._last_channel: str | None = None
        self._last_chat_id: str | None = None

    def update_last_channel(self, channel: str, chat_id: str) -> None:
        """Called by agent loop on each inbound message to track active channel."""
        self._last_channel = channel
        self._last_chat_id = chat_id

    async def start(self) -> None:
        """Start the notification polling service."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Notification service started (polling every {self.interval_s}s)")

    def stop(self) -> None:
        """Stop the notification service."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self._check_notifications()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Notification service error: {e}")

    async def _check_notifications(self) -> None:
        """Scan notification directory and deliver any pending notifications."""
        if not NOTIF_DIR.is_dir():
            return

        if not self._last_channel or not self._last_chat_id:
            # No active channel yet — can't deliver notifications
            return

        for path in sorted(NOTIF_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                message = data.get("message", "")
                if not message:
                    path.unlink(missing_ok=True)
                    continue

                # Deliver to last active channel
                await self.bus.publish_outbound(OutboundMessage(
                    channel=self._last_channel,
                    chat_id=self._last_chat_id,
                    content=message,
                ))
                logger.info(f"Notification delivered: {path.name} -> {self._last_channel}:{self._last_chat_id}")

                # Remove processed notification
                path.unlink(missing_ok=True)

            except json.JSONDecodeError:
                logger.warning(f"Invalid notification file: {path}")
                path.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Failed to deliver notification {path.name}: {e}")
                # Don't delete — retry on next poll
