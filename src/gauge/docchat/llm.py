"""LLM client protocol and concrete implementations.

The default `EchoLLM` returns retrieved context verbatim, which is
useful for a no-API-key prototype and makes the rest of the pipeline
testable in isolation. Real LLM backends (Anthropic, OpenAI) live behind
the same protocol so the upstream service code is provider-agnostic.

`auto_select_llm` picks the strongest available backend at runtime:
1. Anthropic, if the SDK is importable and `ANTHROPIC_API_KEY` is set.
2. OpenAI, if the SDK is importable and `OPENAI_API_KEY` is set.
3. EchoLLM otherwise.
"""

from __future__ import annotations

import os
from typing import Protocol

from gauge.docchat.schemas import Chunk

SYSTEM_PROMPT = (
    "You are a careful assistant answering questions about a US health "
    "insurance plan document. Answer using ONLY the provided excerpts. "
    "If the answer isn't in the excerpts, say so plainly. Quote dollar "
    "amounts, percentages, and definitions verbatim when relevant. "
    "Reference the source by page number when you cite a fact."
)


class LLMClient(Protocol):
    """Anything that can answer a question given retrieved chunks."""

    @property
    def name(self) -> str:
        """Identifier reported to the client for observability."""
        ...

    def answer(self, question: str, contexts: list[Chunk]) -> str:
        """Return a plain-text answer grounded in the supplied chunks.

        Parameters
        ----------
        question : str
            The user's question.
        contexts : list[Chunk]
            Retrieved chunks from the retrieval index.

        Returns
        -------
        str
            Plain-text answer.
        """
        ...


class EchoLLM:
    """Returns the retrieved chunks formatted as a plain-English summary.

    Not a real LLM. Useful for a working prototype before an API key is
    wired up, and for tests that need deterministic output.
    """

    name = "echo"

    def answer(self, question: str, contexts: list[Chunk]) -> str:
        """Format retrieved chunks as a plain-English response.

        Parameters
        ----------
        question : str
            The user's question.
        contexts : list[Chunk]
            Retrieved chunks from the retrieval index.

        Returns
        -------
        str
            Formatted plain-text answer with page citations, or a prompt
            to rephrase if no chunks were provided.
        """
        if not contexts:
            return (
                "I couldn't find anything in this document that looks "
                "relevant to your question. Try rephrasing it or "
                "uploading a more specific section of the plan."
            )
        header = (
            f"Here is what the document says that seems relevant to "
            f"\"{question.strip()}\":"
        )
        body_lines = []
        for c in contexts:
            pages = ", ".join(str(p) for p in c.page_numbers) or "?"
            snippet = c.text.strip().replace("\n", " ")
            body_lines.append(f"\n[p. {pages}] {snippet}")
        suffix = (
            "\n\nNote: this prototype is returning the most relevant "
            "passages directly. Plug in an LLM backend (set "
            "ANTHROPIC_API_KEY or OPENAI_API_KEY and reinstall with "
            "the matching extra) to get a synthesised answer."
        )
        return header + "".join(body_lines) + suffix


class AnthropicLLM:
    """Real LLM via Anthropic's API.

    Imported lazily so the module doesn't pull in `anthropic` for users
    who haven't installed the optional dependency.
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        """Initialise the Anthropic client.

        Parameters
        ----------
        model : str, optional
            Anthropic model identifier. Default is
            ``"claude-haiku-4-5-20251001"``.

        Raises
        ------
        RuntimeError
            If the ``anthropic`` package is not installed or
            ``ANTHROPIC_API_KEY`` is not set.
        """
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Install the 'anthropic' extra to use AnthropicLLM."
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
        from anthropic import Anthropic

        self._client = Anthropic()
        self._model = model

    def answer(self, question: str, contexts: list[Chunk]) -> str:
        """Answer a question using the Anthropic Messages API.

        Parameters
        ----------
        question : str
            The user's question.
        contexts : list[Chunk]
            Retrieved chunks to ground the answer in.

        Returns
        -------
        str
            Synthesised plain-text answer, or a canned "not found" response
            when ``contexts`` is empty.
        """
        if not contexts:
            return (
                "I couldn't find anything in this document that looks "
                "relevant to your question."
            )
        context_block = "\n\n".join(
            f"[Excerpt {i + 1}, page(s) {', '.join(str(p) for p in c.page_numbers)}]\n{c.text}"
            for i, c in enumerate(contexts)
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Excerpts from the plan document:\n{context_block}"
                    ),
                }
            ],
        )
        # The response.content is a list of content blocks; take the text.
        parts = [block.text for block in response.content if hasattr(block, "text")]
        return "".join(parts).strip()


class OpenAILLM:
    """Real LLM via OpenAI's Responses API."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        """Initialise the OpenAI client.

        Parameters
        ----------
        model : str, optional
            OpenAI model identifier. Default is ``"gpt-4o-mini"``.

        Raises
        ------
        RuntimeError
            If the ``openai`` package is not installed or
            ``OPENAI_API_KEY`` is not set.
        """
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Install the 'openai' extra to use OpenAILLM."
            ) from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model

    def answer(self, question: str, contexts: list[Chunk]) -> str:
        """Answer a question using the OpenAI Chat Completions API.

        Parameters
        ----------
        question : str
            The user's question.
        contexts : list[Chunk]
            Retrieved chunks to ground the answer in.

        Returns
        -------
        str
            Synthesised plain-text answer, or a canned "not found" response
            when ``contexts`` is empty.
        """
        if not contexts:
            return (
                "I couldn't find anything in this document that looks "
                "relevant to your question."
            )
        context_block = "\n\n".join(
            f"[Excerpt {i + 1}, page(s) {', '.join(str(p) for p in c.page_numbers)}]\n{c.text}"
            for i, c in enumerate(contexts)
        )
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=600,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Excerpts from the plan document:\n{context_block}"
                    ),
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()


def auto_select_llm() -> LLMClient:
    """Pick the strongest LLM backend available at runtime.

    Selection order:

    1. :class:`AnthropicLLM` if ``ANTHROPIC_API_KEY`` is set.
    2. :class:`OpenAILLM` if ``OPENAI_API_KEY`` is set.
    3. :class:`EchoLLM` as a no-API-key fallback.

    Returns
    -------
    LLMClient
        A ready-to-use LLM client conforming to the :class:`LLMClient`
        protocol.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM()
        except RuntimeError:
            pass
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAILLM()
        except RuntimeError:
            pass
    return EchoLLM()
