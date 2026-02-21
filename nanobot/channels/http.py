"""HTTP channel with SSE streaming for desktop app integration."""

import asyncio
import json
from typing import Any

from aiohttp import web
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import HTTPConfig


class HTTPChannel(BaseChannel):
    """
    HTTP channel exposing a local REST API with SSE streaming.

    POST /chat  — accepts {session_id, message}, returns SSE event stream
    GET /health — returns {"status": "ok"}

    Per-request routing: _pending[session_id] = asyncio.Queue
    ChannelManager calls send(msg) → routes to the correct request's queue.
    """

    name = "http"

    def __init__(self, config: HTTPConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: HTTPConfig = config
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._pending: dict[str, asyncio.Queue[OutboundMessage | None]] = {}

    async def start(self) -> None:
        self._running = True

        app = web.Application(middlewares=[self._cors_middleware])
        app.router.add_post("/chat", self._handle_chat)
        app.router.add_get("/health", self._handle_health)
        app.router.add_route("OPTIONS", "/chat", self._handle_options)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(
            self._runner, self.config.host, self.config.port
        )
        await self._site.start()
        logger.info(
            f"HTTP channel listening on {self.config.host}:{self.config.port}"
        )

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        for q in self._pending.values():
            await q.put(None)
        self._pending.clear()

        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        logger.info("HTTP channel stopped")

    async def send(self, msg: OutboundMessage) -> str | None:
        """Route outbound message to the pending request queue.

        Always returns None so ChannelManager doesn't track progress IDs.
        For SSE, each progress/message is a separate event — no need to
        "edit" a previous message. The msg.progress flag is preserved and
        used in _handle_chat to determine the SSE event type.
        """
        q = self._pending.get(msg.chat_id)
        if q:
            await q.put(msg)
            return None
        else:
            logger.warning(f"HTTP: no pending request for chat_id={msg.chat_id}")
            return None

    async def edit(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Edit = send a progress SSE event (naturally handled by stream)."""
        q = self._pending.get(chat_id)
        if q:
            await q.put(
                OutboundMessage(
                    channel="http",
                    chat_id=chat_id,
                    content=content,
                    progress=True,
                )
            )

    # ── HTTP Handlers ──────────────────────────────────

    async def _handle_health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _handle_options(self, _request: web.Request) -> web.Response:
        return web.Response(
            status=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            },
        )

    async def _handle_chat(self, request: web.Request) -> web.StreamResponse:
        """POST /chat — accepts {session_id, message}, returns SSE stream."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        session_id = body.get("session_id", "").strip()
        message = body.get("message", "").strip()

        if not session_id:
            return web.json_response(
                {"error": "session_id is required"}, status=400
            )
        if not message:
            return web.json_response(
                {"error": "message is required"}, status=400
            )

        # Create per-request response queue
        q: asyncio.Queue[OutboundMessage | None] = asyncio.Queue()
        self._pending[session_id] = q

        # Prepare SSE response
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )
        await response.prepare(request)

        # Publish inbound message to bus
        # chat_id = session_id so nanobot session key = "http:{session_id}"
        await self._handle_message(
            sender_id="desktop",
            chat_id=session_id,
            content=message,
        )

        # Stream responses from queue
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=300)
                except asyncio.TimeoutError:
                    await self._write_sse(
                        response, "error", {"error": "Request timed out"}
                    )
                    break

                if msg is None:
                    # Channel stopping
                    break

                if msg.progress:
                    await self._write_sse(
                        response, "progress", {"content": msg.content}
                    )
                else:
                    # Final response
                    await self._write_sse(
                        response, "message", {"content": msg.content}
                    )
                    await self._write_sse(response, "done", {})
                    break
        except (ConnectionResetError, ConnectionAbortedError):
            logger.debug(f"HTTP: client disconnected for {session_id}")
        finally:
            self._pending.pop(session_id, None)

        return response

    # ── Middleware ──────────────────────────────────────

    @web.middleware
    async def _cors_middleware(self, request: web.Request, handler):
        if request.method == "OPTIONS":
            return await self._handle_options(request)
        resp = await handler(request)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        return resp

    # ── Helpers ────────────────────────────────────────

    @staticmethod
    async def _write_sse(
        response: web.StreamResponse, event: str, data: dict
    ) -> None:
        payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        await response.write(payload.encode("utf-8"))
