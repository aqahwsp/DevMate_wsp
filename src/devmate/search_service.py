"""Tavily-backed web search service for DevMate."""

from __future__ import annotations

import logging
from typing import Any

from tavily import TavilyClient

from devmate.config import SearchSettings, is_config_secret_set

LOGGER = logging.getLogger(__name__)


class TavilySearchService:
    """Small wrapper around Tavily search with an optional mock mode."""

    def __init__(self, settings: SearchSettings) -> None:
        self.settings = settings

    def search(
        self,
        query: str,
        max_results: int | None = None,
        topic: str = "general",
    ) -> dict[str, Any]:
        """Search the web with Tavily or return mock data for local tests."""

        result_count = max_results or self.settings.max_results
        if is_config_secret_set(self.settings.tavily_api_key):
            LOGGER.info("Executing Tavily search for query=%s", query)
            client = TavilyClient(api_key=self.settings.tavily_api_key)
            return client.search(
                query,
                topic=topic,
                max_results=result_count,
                include_answer=True,
                include_raw_content=False,
                search_depth="basic",
            )

        if self.settings.allow_mock_search:
            LOGGER.warning(
                "Using mock Tavily search results because no valid API key is "
                "configured"
            )
            return self._mock_search(query=query, max_results=result_count, topic=topic)

        raise RuntimeError(
            "Tavily API key is not configured and mock search is disabled"
        )

    def _mock_search(
        self,
        query: str,
        max_results: int,
        topic: str,
    ) -> dict[str, Any]:
        """Return deterministic mock search results for local smoke tests."""

        lowercase_query = query.lower()
        if "hiking" in lowercase_query or "徒步" in lowercase_query:
            results = [
                {
                    "title": "Hiking directory best practices",
                    "url": "https://example.com/hiking-best-practices",
                    "content": (
                        "Use searchable route cards, a map section, difficulty "
                        "filters, and responsive layouts for outdoor discovery sites."
                    ),
                    "score": 0.98,
                },
                {
                    "title": "Modern JS mapping libraries overview",
                    "url": "https://example.com/js-maps",
                    "content": (
                        "Leaflet and MapLibre are lightweight options for map-based "
                        "interfaces, especially when you need embeddable route views."
                    ),
                    "score": 0.91,
                },
            ]
        elif "fastapi" in lowercase_query:
            results = [
                {
                    "title": "FastAPI project layout",
                    "url": "https://example.com/fastapi-layout",
                    "content": (
                        "Use app, routers, services, schemas, and models packages, "
                        "plus a health check endpoint and Docker support."
                    ),
                    "score": 0.96,
                }
            ]
        else:
            results = [
                {
                    "title": "General software project scaffold guidance",
                    "url": "https://example.com/project-scaffold",
                    "content": (
                        "Plan the file tree first, prefer small modules, validate the "
                        "generated code, and write a concise README."
                    ),
                    "score": 0.89,
                }
            ]

        return {
            "query": query,
            "topic": topic,
            "answer": (
                "Mock Tavily response: use local knowledge first, then scaffold files "
                "with clear responsibilities and verification steps."
            ),
            "results": results[:max_results],
            "response_time": 0.01,
        }


def format_search_response(response: dict[str, Any]) -> str:
    """Format Tavily results into compact markdown for the agent."""

    answer = response.get("answer") or ""
    results = response.get("results", [])
    lines = []
    if answer:
        lines.append(f"Summary: {answer}")
    for item in results:
        lines.append(
            "- {title} | {url} | {content}".format(
                title=item.get("title", "Untitled"),
                url=item.get("url", ""),
                content=item.get("content", ""),
            )
        )
    if not lines:
        return "No web search results were returned."
    return "\n".join(lines)
