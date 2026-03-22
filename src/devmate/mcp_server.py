"""MCP search server exposed over Streamable HTTP."""

from __future__ import annotations

import contextlib
from typing import Any, Literal

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.cors import CORSMiddleware

from devmate.config import load_config
from devmate.logging_config import configure_logging
from devmate.search_service import TavilySearchService


def create_mcp_app(config_path: str | None = None):
    """Create the MCP Streamable HTTP application."""

    config = load_config(config_path)
    configure_logging(config.app.log_level)
    search_service = TavilySearchService(config.search)
    mcp = FastMCP(
        "DevMate Search",
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[
                "localhost:*",
                "127.0.0.1:*",
                "search-mcp:*",
            ],
            allowed_origins=[
                "http://localhost:*",
                "http://127.0.0.1:*",
                "http://search-mcp:*",
            ],
        ),
    )
    mcp.settings.streamable_http_path = config.mcp.streamable_http_path

    @mcp.tool()
    def search_web(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
    ) -> dict[str, Any]:
        """Search the public web using Tavily and return structured results."""

        sanitized_max_results = max(1, min(max_results, 20))
        return search_service.search(
            query=query,
            max_results=sanitized_max_results,
            topic=topic,
        )

    mcp_asgi = mcp.streamable_http_app()
    cors_wrapped = CORSMiddleware(
        mcp_asgi,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE"],
        expose_headers=["Mcp-Session-Id"],
        allow_headers=["*"],
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        async with mcp.session_manager.run():
            yield

    app = FastAPI(title="DevMate MCP Search", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.mount("/", cors_wrapped)
    return app
