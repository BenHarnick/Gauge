"""Unit tests for the LLM client implementations and auto-selection."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from health_app.docchat.llm import AnthropicLLM, EchoLLM, OpenAILLM, auto_select_llm
from health_app.docchat.schemas import Chunk

pytestmark = pytest.mark.unit


def test_echo_handles_empty_context() -> None:
    answer = EchoLLM().answer("anything", [])
    assert "couldn't find anything" in answer.lower()


def test_echo_includes_page_citations_and_snippets() -> None:
    chunks = [
        Chunk(
            document_id="d",
            chunk_index=0,
            text="Annual deductible is $1,000 individual.",
            page_numbers=[1],
        ),
        Chunk(
            document_id="d",
            chunk_index=1,
            text="Specialist copay is $50.",
            page_numbers=[2, 3],
        ),
    ]
    answer = EchoLLM().answer("what is the deductible?", chunks)
    assert "$1,000" in answer
    assert "p. 1" in answer
    assert "p. 2, 3" in answer
    assert "ANTHROPIC_API_KEY" in answer


# ---------------------------------------------------------------------------
# AnthropicLLM error paths
# ---------------------------------------------------------------------------


def test_anthropic_llm_raises_when_sdk_not_installed(monkeypatch) -> None:
    """AnthropicLLM raises RuntimeError when the anthropic package is absent."""
    monkeypatch.setitem(sys.modules, "anthropic", None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="anthropic"):
        AnthropicLLM()


def test_anthropic_llm_raises_when_api_key_missing(monkeypatch) -> None:
    """AnthropicLLM raises RuntimeError when ANTHROPIC_API_KEY is not set."""
    mock_pkg = MagicMock()
    monkeypatch.setitem(sys.modules, "anthropic", mock_pkg)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM()


# ---------------------------------------------------------------------------
# OpenAILLM error paths
# ---------------------------------------------------------------------------


def test_openai_llm_raises_when_sdk_not_installed(monkeypatch) -> None:
    """OpenAILLM raises RuntimeError when the openai package is absent."""
    monkeypatch.setitem(sys.modules, "openai", None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="openai"):
        OpenAILLM()


def test_openai_llm_raises_when_api_key_missing(monkeypatch) -> None:
    """OpenAILLM raises RuntimeError when OPENAI_API_KEY is not set."""
    mock_pkg = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", mock_pkg)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAILLM()


# ---------------------------------------------------------------------------
# auto_select_llm
# ---------------------------------------------------------------------------


def test_auto_select_llm_returns_echo_when_no_keys_set(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = auto_select_llm()
    assert isinstance(result, EchoLLM)


def test_auto_select_llm_falls_through_to_echo_when_anthropic_sdk_absent(
    monkeypatch,
) -> None:
    """ANTHROPIC_API_KEY set but SDK not installed -> falls through to EchoLLM."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # anthropic is not installed in the test environment, so AnthropicLLM()
    # raises RuntimeError, and auto_select_llm should catch it and continue.
    monkeypatch.setitem(sys.modules, "anthropic", None)  # type: ignore[arg-type]
    result = auto_select_llm()
    assert isinstance(result, EchoLLM)


def test_auto_select_llm_falls_through_to_echo_when_openai_sdk_absent(
    monkeypatch,
) -> None:
    """OPENAI_API_KEY set but SDK not installed -> falls through to EchoLLM."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setitem(sys.modules, "openai", None)  # type: ignore[arg-type]
    result = auto_select_llm()
    assert isinstance(result, EchoLLM)
