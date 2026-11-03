"""OpenAI-compatible chat client: OpenRouter (paid) or a vLLM server on Colab (free), same code."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Protocol

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMError(RuntimeError):
    """Unusable LLM call: API error or empty response."""


class BudgetExceeded(LLMError):
    """The run's spending cap is reached: no further call is sent."""


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class LLM(Protocol):
    def chat(self, messages: list[dict]) -> Completion: ...


class Budget:
    """Spending cap shared by a run's threads. A call's cost is only known afterwards, so the worst-case
    overshoot is the cost of the calls already in flight (one per worker)."""

    def __init__(self, max_usd: float | None) -> None:
        self.max_usd = max_usd
        self.spent_usd = 0.0
        self._lock = threading.Lock()

    def check(self) -> None:
        if self.max_usd is not None and self.spent_usd >= self.max_usd:
            raise BudgetExceeded(f"Budget of ${self.max_usd} reached (${self.spent_usd:.4f} spent).")

    def charge(self, usd: float) -> None:
        with self._lock:
            self.spent_usd += usd


class ChatLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 1024, budget: Budget | None = None, client: Any = None) -> None:
        from openai import OpenAI

        self.base_url = base_url or os.getenv("SQL_AGENT_BASE_URL") or OPENROUTER_BASE_URL
        self.model = model or os.getenv("SQL_AGENT_MODEL", "qwen/qwen-2.5-7b-instruct")
        self._client = client or OpenAI(base_url=self.base_url, api_key=_api_key(self.base_url), max_retries=4)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.budget = budget or Budget(None)

    @property
    def is_paid(self) -> bool:
        return self.base_url.rstrip("/") == OPENROUTER_BASE_URL

    def chat(self, messages: list[dict]) -> Completion:
        import openai

        self.budget.check()
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
        except openai.APIError as exc:
            raise LLMError(f"API error ({self.model}): {exc}") from exc
        if not resp.choices or resp.choices[0].message.content is None:
            raise LLMError(f"Empty response from {self.model}.")
        usage = resp.usage
        extra = (usage.model_extra or {}) if usage else {}
        cost = float(extra.get("cost") or 0.0)  # OpenRouter reports the actual cost; vLLM reports none
        self.budget.charge(cost)
        return Completion(
            text=resp.choices[0].message.content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            cost_usd=cost,
        )


def _api_key(base_url: str) -> str:
    if base_url.rstrip("/") != OPENROUTER_BASE_URL:
        return os.getenv("SQL_AGENT_API_KEY", "EMPTY")  # vLLM needs no key by default
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY is missing: set it in .env.")
    return key
