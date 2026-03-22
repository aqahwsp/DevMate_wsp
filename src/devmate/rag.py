"""Local knowledge-base ingestion and retrieval."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from devmate.config import AppConfig, is_config_secret_set
from devmate.embeddings import HashEmbeddings

LOGGER = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".md", ".txt"}


class KnowledgeBase:
    """Manage local document ingestion and semantic retrieval."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._vector_store: Chroma | None = None

    def ensure_directories(self) -> None:
        """Create required directories if they do not exist."""

        self.config.docs_dir.mkdir(parents=True, exist_ok=True)
        self.config.persist_dir.mkdir(parents=True, exist_ok=True)
        self.config.research_cache_dir.mkdir(parents=True, exist_ok=True)

    def build_embeddings(self) -> Embeddings:
        """Create the configured embeddings implementation."""

        provider = self.config.model.embedding_provider.lower()
        if provider == "openai" and is_config_secret_set(self.config.model.api_key):
            return OpenAIEmbeddings(
                model=self.config.model.embedding_model_name,
                api_key=self.config.model.api_key,
                base_url=self.config.model.ai_base_url,
            )
        LOGGER.warning(
            "Falling back to HashEmbeddings because OpenAI embeddings are not "
            "configured"
        )
        return HashEmbeddings()

    def build_vector_store(self) -> Chroma:
        """Create or reuse the configured Chroma vector store."""

        if self._vector_store is None:
            self.ensure_directories()
            self._vector_store = Chroma(
                collection_name=self.config.rag.collection_name,
                persist_directory=str(self.config.persist_dir),
                embedding_function=self.build_embeddings(),
            )
        return self._vector_store

    def _build_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create the configured text splitter."""

        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.rag.chunk_size,
            chunk_overlap=self.config.rag.chunk_overlap,
        )

    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Chunk documents using the configured text splitter."""

        return self._build_splitter().split_documents(documents)

    def load_documents(self) -> list[Document]:
        """Load markdown and text files from the docs directory."""

        self.ensure_directories()
        documents: list[Document] = []
        for path in sorted(self.config.docs_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            content = path.read_text(encoding="utf-8")
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(path.relative_to(self.config.project_root)),
                    },
                )
            )
        return documents

    def ingest(self, rebuild: bool = True) -> int:
        """Chunk local documents and store them in Chroma."""

        if rebuild and self.config.persist_dir.exists():
            shutil.rmtree(self.config.persist_dir)
            self._vector_store = None

        documents = self.load_documents()
        if not documents:
            LOGGER.warning(
                "No knowledge-base documents were found under %s",
                self.config.docs_dir,
            )
            return 0

        chunks = self._chunk_documents(documents)
        vector_store = self.build_vector_store()
        vector_store.add_documents(chunks)
        LOGGER.info("Indexed %s document chunks into the knowledge base", len(chunks))
        return len(chunks)

    def search(self, query: str, k: int | None = None) -> list[Document]:
        """Run a similarity search against the local knowledge base."""

        vector_store = self.build_vector_store()
        result_count = k or self.config.rag.top_k
        return vector_store.similarity_search(query, k=result_count)

    def format_search_results(self, query: str, k: int | None = None) -> str:
        """Return a compact text representation of retrieval results."""

        documents = self.search(query=query, k=k)
        if not documents:
            return "No relevant local knowledge-base documents were found."

        lines = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "unknown")
            excerpt = document.page_content.strip().replace("\n", " ")
            excerpt = excerpt[:400]
            lines.append(f"[{index}] {source}: {excerpt}")
        return "\n".join(lines)

    def add_text_document(self, *, content: str, source: str) -> int:
        """Add a text document to the vector store incrementally."""

        normalized = content.strip()
        if not normalized:
            return 0
        documents = [Document(page_content=normalized, metadata={"source": source})]
        chunks = self._chunk_documents(documents)
        self.build_vector_store().add_documents(chunks)
        LOGGER.info("Added %s cached knowledge chunk(s) from %s", len(chunks), source)
        return len(chunks)

    def cache_research_knowledge(
        self,
        *,
        run_id: str,
        round_index: int,
        prompt: str,
        content: str,
    ) -> Path | None:
        """Persist Researcher findings as a local TXT document and index it."""

        normalized = content.strip()
        if not normalized:
            return None

        self.ensure_directories()
        slug = self._slugify(prompt)
        filename = f"research_{run_id}_round_{round_index}_{slug}.txt"
        path = self.config.research_cache_dir / filename
        body = (
            f"Run ID: {run_id}\n"
            f"Research Round: {round_index}\n"
            f"Original Prompt: {prompt.strip()}\n\n"
            f"{normalized}\n"
        )
        path.write_text(body, encoding="utf-8")
        source = path.relative_to(self.config.project_root).as_posix()
        self.add_text_document(content=body, source=source)
        return path

    def _slugify(self, value: str, max_length: int = 48) -> str:
        """Create a filesystem-friendly slug."""

        filtered = [
            character.lower() if character.isalnum() else "-"
            for character in value.strip()
        ]
        slug = "".join(filtered).strip("-")
        while "--" in slug:
            slug = slug.replace("--", "-")
        slug = slug[:max_length].strip("-")
        return slug or "prompt"
