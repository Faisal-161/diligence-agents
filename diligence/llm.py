"""Language-model layer. Every agent call goes through `LLM.run`, which returns a validated Pydantic
object plus a `Usage` record (tokens, cost, latency), so cost is tracked per agent.

- `ClaudeLLM` forces a tool call whose input schema is the expected Pydantic model, and retries once
  with the validation error if the output doesn't fit.
- `FakeLLM` is offline and deterministic (see `fake.py`); used for tests, CI and the free demo.
"""

from __future__ import annotations

import json
import os
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings
from .models import Usage

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    def run(self, agent: str, system: str, prompt: str, schema: type[T], context: dict) -> tuple[T, Usage]: ...


def _cost(settings: Settings, tin: int, tout: int) -> float:
    return round(tin / 1e6 * settings.price_input_per_m + tout / 1e6 * settings.price_output_per_m, 6)


class ClaudeLLM:
    def __init__(self, settings: Settings):
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY to use Claude (DILIGENCE_LLM=claude).")
        self.client, self.settings = anthropic.Anthropic(), settings

    def run(self, agent: str, system: str, prompt: str, schema: type[T], context: dict) -> tuple[T, Usage]:
        t0, tin, tout = time.perf_counter(), 0, 0
        tool = {
            "name": "respond",
            "description": f"Return the {schema.__name__}.",
            "input_schema": schema.model_json_schema(),
        }
        messages: list[dict] = [{"role": "user", "content": prompt}]
        for attempt in range(2):
            resp = self.client.messages.create(
                model=self.settings.model,
                max_tokens=4000,
                temperature=0,
                system=system,
                tools=[tool],
                tool_choice={"type": "tool", "name": "respond"},
                messages=messages,
            )
            tin, tout = tin + resp.usage.input_tokens, tout + resp.usage.output_tokens
            use = next(b for b in resp.content if b.type == "tool_use")
            try:
                out = schema.model_validate(use.input)
                break
            except ValidationError as exc:
                if attempt == 1:
                    raise
                retry = {
                    "type": "tool_result",
                    "tool_use_id": use.id,
                    "is_error": True,
                    "content": f"Invalid output, fix and retry: {exc}",
                }
                messages += [{"role": "assistant", "content": resp.content}, {"role": "user", "content": [retry]}]
        ms = (time.perf_counter() - t0) * 1000
        return out, Usage(
            agent=agent,
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=_cost(self.settings, tin, tout),
            latency_ms=round(ms, 1),
        )


class FakeLLM:
    """Deterministic stand-in. Token counts are estimated (4 characters per token) so cost
    comparisons between architectures stay meaningful offline."""

    def __init__(self, settings: Settings):
        from . import fake

        self.settings, self.impl = settings, fake

    def run(self, agent: str, system: str, prompt: str, schema: type[T], context: dict) -> tuple[T, Usage]:
        t0 = time.perf_counter()
        out = getattr(self.impl, agent)(**context)
        if not isinstance(out, schema):
            out = schema.model_validate(out)
        tin = (len(system) + len(prompt)) // 4
        tout = len(json.dumps(out.model_dump(mode="json"))) // 4
        ms = (time.perf_counter() - t0) * 1000
        return out, Usage(
            agent=agent,
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=_cost(self.settings, tin, tout),
            latency_ms=round(ms, 2),
        )


def get_llm(settings: Settings) -> LLM:
    return ClaudeLLM(settings) if settings.llm == "claude" else FakeLLM(settings)
