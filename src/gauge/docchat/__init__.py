"""Insurance document chatbot: PDF upload, retrieval, and Q&A.

Public surface kept intentionally small. The orchestrator is
`DocumentChatService`; everything else (extractor, chunker, retrieval
index, LLM client) is composable and pluggable.
"""

from gauge.docchat.chunker import chunk_pages
from gauge.docchat.extractor import extract_pages
from gauge.docchat.index import RetrievalIndex, TfidfRetrievalIndex
from gauge.docchat.llm import EchoLLM, LLMClient, auto_select_llm
from gauge.docchat.schemas import (
    ChatRequest,
    ChatResponse,
    Chunk,
    Citation,
    DocumentMeta,
    UploadResponse,
)
from gauge.docchat.service import DocumentChatService
from gauge.docchat.store import InMemoryDocumentStore

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "Chunk",
    "Citation",
    "DocumentChatService",
    "DocumentMeta",
    "EchoLLM",
    "InMemoryDocumentStore",
    "LLMClient",
    "RetrievalIndex",
    "TfidfRetrievalIndex",
    "UploadResponse",
    "auto_select_llm",
    "chunk_pages",
    "extract_pages",
]
