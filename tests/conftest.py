from __future__ import annotations

import sys
import types
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class _FakeAgent:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self.tools = kwargs.get("tools", [])
        self.backend = kwargs.get("backend")

    async def ainvoke(self, payload, config=None):  # pragma: no cover - fallback
        return {"messages": [{"role": "assistant", "content": ""}]}

    async def astream(self, payload, config=None, stream_mode=None, subgraphs=None):
        if False:  # pragma: no cover - async generator shim
            yield None
        return


class _FakeFilesystemBackend:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir


@dataclass(slots=True)
class _FakeDocument:
    page_content: str
    metadata: dict[str, object] | None = None


class _FakeEmbeddings:
    pass


class _FakeChatModel:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def ainvoke(self, prompt):  # pragma: no cover - not used in tests
        return types.SimpleNamespace(content="")


class _FakeOpenAIEmbeddings:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeChroma:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.documents: list[_FakeDocument] = []

    def add_documents(self, documents):
        self.documents.extend(documents)

    def similarity_search(self, query: str, k: int = 4):
        return self.documents[:k]


class _FakeTextSplitter:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def split_documents(self, documents):
        return documents


class _FakeMultiServerMCPClient:
    def __init__(self, servers) -> None:
        self.servers = servers

    async def get_tools(self):
        return []


class _FakeTransportSecuritySettings:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeFastMCP:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.settings = types.SimpleNamespace(streamable_http_path="/mcp")
        self._tools = []
        self.session_manager = types.SimpleNamespace(run=self._run)

    @asynccontextmanager
    async def _run(self):
        yield

    def tool(self):
        def decorator(func):
            self._tools.append(func)
            return func

        return decorator

    def streamable_http_app(self):
        async def app(scope, receive, send):  # pragma: no cover - ASGI shim
            return None

        return app


class _FakeTavilyClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, *args, **kwargs):
        return {"results": [], "answer": "", "response_time": 0.0}



def _install_module(name: str, module: types.ModuleType) -> None:
    sys.modules[name] = module



def _tool_decorator(func=None, **kwargs):
    def decorator(inner):
        inner.name = getattr(inner, "__name__", "tool")
        return inner

    if func is None:
        return decorator
    return decorator(func)



def _install_test_doubles() -> None:
    deepagents_module = types.ModuleType("deepagents")
    deepagents_module.create_deep_agent = lambda **kwargs: _FakeAgent(**kwargs)
    _install_module("deepagents", deepagents_module)

    backends_module = types.ModuleType("deepagents.backends")
    backends_module.FilesystemBackend = _FakeFilesystemBackend
    _install_module("deepagents.backends", backends_module)

    langchain_core_module = types.ModuleType("langchain_core")
    _install_module("langchain_core", langchain_core_module)

    tools_module = types.ModuleType("langchain_core.tools")
    tools_module.tool = _tool_decorator
    _install_module("langchain_core.tools", tools_module)

    documents_module = types.ModuleType("langchain_core.documents")
    documents_module.Document = _FakeDocument
    _install_module("langchain_core.documents", documents_module)

    embeddings_module = types.ModuleType("langchain_core.embeddings")
    embeddings_module.Embeddings = _FakeEmbeddings
    _install_module("langchain_core.embeddings", embeddings_module)

    mcp_client_module = types.ModuleType("langchain_mcp_adapters.client")
    mcp_client_module.MultiServerMCPClient = _FakeMultiServerMCPClient
    _install_module("langchain_mcp_adapters.client", mcp_client_module)

    langchain_openai_module = types.ModuleType("langchain_openai")
    langchain_openai_module.ChatOpenAI = _FakeChatModel
    langchain_openai_module.OpenAIEmbeddings = _FakeOpenAIEmbeddings
    _install_module("langchain_openai", langchain_openai_module)

    langchain_deepseek_module = types.ModuleType("langchain_deepseek")
    langchain_deepseek_module.ChatDeepSeek = _FakeChatModel
    _install_module("langchain_deepseek", langchain_deepseek_module)

    langchain_chroma_module = types.ModuleType("langchain_chroma")
    langchain_chroma_module.Chroma = _FakeChroma
    _install_module("langchain_chroma", langchain_chroma_module)

    splitters_module = types.ModuleType("langchain_text_splitters")
    splitters_module.RecursiveCharacterTextSplitter = _FakeTextSplitter
    _install_module("langchain_text_splitters", splitters_module)

    tavily_module = types.ModuleType("tavily")
    tavily_module.TavilyClient = _FakeTavilyClient
    _install_module("tavily", tavily_module)

    mcp_module = types.ModuleType("mcp")
    _install_module("mcp", mcp_module)

    mcp_server_module = types.ModuleType("mcp.server")
    _install_module("mcp.server", mcp_server_module)

    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = _FakeFastMCP
    _install_module("mcp.server.fastmcp", fastmcp_module)

    transport_module = types.ModuleType("mcp.server.transport_security")
    transport_module.TransportSecuritySettings = _FakeTransportSecuritySettings
    _install_module("mcp.server.transport_security", transport_module)


_install_test_doubles()
