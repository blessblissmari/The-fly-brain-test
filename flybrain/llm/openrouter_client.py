"""OpenRouter chat completion client (Round 6 — secondary backend).

Mirrors `flybrain.llm.yandex_client.YandexClient` so that the bench runner
can swap backends via `--backend openrouter` without touching the rest of
the pipeline.

Key differences from YandexClient:

* Uses the OpenAI-compatible `/api/v1/chat/completions` endpoint at
  https://openrouter.ai (so we can talk to a wide set of free + paid
  models behind one API).
* Supports **two API keys** with automatic rotation on HTTP 429 (free-tier
  rate limit is ~200 req/day per key; round-6 N=5 mini-bench needs ~300
  calls, so we rotate keys to spread load).
* `cost_rub = 0` for free-tier models — round 6 hard requirement is 0 ₽
  spend. We still record token usage for reporting purposes.
* Tier mapping defaults to the same model for LITE and PRO since round-6
  smoke-tested only `google/gemma-3-27b-it:free` as a working free model;
  others (llama-3.3-70b, qwen3-coder, gpt-oss-20b) are upstream-throttled
  as of 2026-05-02. See `docs/round6_openrouter_free.md`.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

from flybrain.llm.base import LLMClient, LLMResponse, Message, ModelTier
from flybrain.llm.budget import BudgetTracker
from flybrain.llm.cache import SQLiteCache, cache_key

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# Default free-tier model fallback chain (round 6, 2026-05-02 smoke).
# Rationale: OpenRouter free-tier shares ONE upstream quota across all
# keys per model, so key-rotation alone doesn't help when a provider is
# 429-throttled. Model-fallback works around this — when one provider's
# quota is exhausted we try a different upstream provider.
DEFAULT_LITE_MODELS: tuple[str, ...] = (
    "openai/gpt-oss-120b:free",  # OpenInference, ~8s latency, 131K ctx
    "google/gemma-3-27b-it:free",  # Google AI Studio, ~1s latency, 131K ctx
    "minimax/minimax-m2.5:free",  # MiniMax, ~3s latency, 196K ctx
    "z-ai/glm-4.5-air:free",  # Z.AI, 131K ctx
    "openai/gpt-oss-20b:free",  # OpenInference, 131K ctx
    "meta-llama/llama-3.3-70b-instruct:free",  # Venice, 65K ctx
    "qwen/qwen3-coder:free",  # Venice, 262K ctx
)


@dataclass(slots=True)
class OpenRouterConfig:
    """Static config for the OpenRouter client.

    `api_keys` holds 1+ keys; the client rotates on 429 errors.
    `lite_models` / `pro_models` are fallback chains: when one returns 429
    we try the next.
    """

    api_keys: list[str]
    lite_models: list[str] = field(default_factory=lambda: list(DEFAULT_LITE_MODELS))
    pro_models: list[str] = field(default_factory=lambda: list(DEFAULT_LITE_MODELS))
    timeout_s: float = 120.0
    max_retries: int = 3
    initial_backoff_s: float = 2.0
    referrer: str = "https://github.com/blessblissmari/The-fly-brain-test"
    title: str = "FlyBrain Optimizer (Round 6)"

    @classmethod
    def from_env(cls) -> OpenRouterConfig:
        keys: list[str] = []
        primary = os.environ.get("OPENROUTER_API_KEY", "").strip()
        secondary = os.environ.get("OPENROUTER_API_KEY_2", "").strip()
        if primary:
            keys.append(primary)
        if secondary and secondary != primary:
            keys.append(secondary)
        if not keys:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; cannot init OpenRouter client. "
                "Set OPENROUTER_API_KEY (and optionally OPENROUTER_API_KEY_2 for "
                "rate-limit rotation) in your environment."
            )

        def _split(env_var: str, default: tuple[str, ...]) -> list[str]:
            raw = os.environ.get(env_var, "").strip()
            if not raw:
                return list(default)
            return [m.strip() for m in raw.split(",") if m.strip()]

        return cls(
            api_keys=keys,
            lite_models=_split("OPENROUTER_LITE_MODELS", DEFAULT_LITE_MODELS),
            pro_models=_split("OPENROUTER_PRO_MODELS", DEFAULT_LITE_MODELS),
        )

    # Backwards-compat: some callers expect `.lite_model` / `.pro_model`.
    @property
    def lite_model(self) -> str:
        return self.lite_models[0] if self.lite_models else ""

    @property
    def pro_model(self) -> str:
        return self.pro_models[0] if self.pro_models else ""


@dataclass(slots=True)
class OpenRouterClient(LLMClient):
    """Async OpenRouter chat client with key-rotation on 429.

    Pricing semantics: `cost_rub` is reported as 0 for `*:free` models and
    the budget tracker is not decremented. For paid models (out of round-6
    scope) cost is approximated from token usage at rate 0.0 — callers
    must supply their own pricing if they want hard accounting.
    """

    config: OpenRouterConfig
    cache: SQLiteCache | None = None
    budget: BudgetTracker | None = None
    _client: Any = field(default=None, init=False, repr=False)
    _key_idx: int = field(default=0, init=False, repr=False)
    _model_idx: int = field(default=0, init=False, repr=False)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as e:  # pragma: no cover - runtime dep
                raise RuntimeError(
                    "httpx is not installed; install with `uv pip install httpx`"
                ) from e
            self._client = httpx.AsyncClient(
                base_url=OPENROUTER_BASE_URL,
                timeout=self.config.timeout_s,
            )
        return self._client

    def _models_for(self, tier: ModelTier) -> list[str]:
        models = self.config.lite_models if tier == ModelTier.LITE else self.config.pro_models
        return list(models) if models else list(DEFAULT_LITE_MODELS)

    def _is_free(self, model: str) -> bool:
        return model.endswith(":free")

    def _headers(self, key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": self.config.referrer,
            "X-Title": self.config.title,
            "Content-Type": "application/json",
        }

    async def _try_one(
        self,
        *,
        client: Any,
        model: str,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
    ) -> tuple[dict | None, str | None]:
        """Try a single (model, key) pair with retries on 429 across keys.

        Returns ``(data, None)`` on success or ``(None, error_text)`` if the
        provider is exhausted on this model.
        """
        body = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        n_keys = len(self.config.api_keys)
        last_text = ""
        for k_off in range(n_keys):
            api_key = self.config.api_keys[(self._key_idx + k_off) % n_keys]
            try:
                resp = await client.post(
                    "/chat/completions",
                    json=body,
                    headers=self._headers(api_key),
                )
            except Exception as e:
                last_text = f"network error: {e}"
                continue
            if resp.status_code == 429:
                last_text = (resp.text or "")[:500]
                continue
            if resp.status_code >= 400:
                last_text = (resp.text or "")[:500]
                # Non-rate-limit error: probably bad model name → break out
                # of the key loop and try a different model in the chain.
                if resp.status_code in (400, 401, 403, 404):
                    break
                continue
            try:
                data = resp.json()
            except Exception as e:
                last_text = f"json decode failed: {e}"
                continue
            # Some providers return 200 with an `error` payload (e.g. when
            # the upstream rate-limited but OpenRouter returned a body).
            err = data.get("error")
            if err:
                last_text = str(err)[:500]
                continue
            return data, None
        return None, last_text

    async def complete(
        self,
        messages: list[Message],
        *,
        tier: ModelTier = ModelTier.LITE,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        models = self._models_for(tier)
        n_models = len(models)

        # Cache key against PRIMARY model so identical prompts hit the same
        # entry regardless of which fallback served the response.
        primary = models[0]
        key = cache_key(primary, temperature, messages)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        client = self._get_client()
        backoff = self.config.initial_backoff_s
        last_err = ""
        data: dict | None = None
        served_model = primary

        t0 = time.perf_counter()
        for _attempt in range(self.config.max_retries):
            for m_off in range(n_models):
                model = models[(self._model_idx + m_off) % n_models]
                data, err = await self._try_one(
                    client=client,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if data is not None:
                    served_model = model
                    # Promote the working model so subsequent calls try it
                    # first (locality-of-reference for upstream quotas).
                    self._model_idx = (self._model_idx + m_off) % n_models
                    break
                last_err = err or last_err
            if data is not None:
                break
            # All models 429'd: backoff and retry.
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

        latency_ms = int((time.perf_counter() - t0) * 1000)

        if data is None:
            raise RuntimeError(
                f"OpenRouter exhausted after {self.config.max_retries} rounds "
                f"across {n_models} models x {len(self.config.api_keys)} keys; "
                f"last error: {last_err[:300]}"
            )

        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):  # pragma: no cover
            content = ""
        usage = data.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens", 0) or 0)
        tokens_out = int(usage.get("completion_tokens", 0) or 0)
        if tokens_in == 0 and tokens_out == 0:
            tokens_in = sum(max(1, len(m.content) // 4) for m in messages)
            tokens_out = max(1, len(content) // 4)

        # Round-6 hard requirement: 0 ₽ spend. Free-tier models cost
        # nothing; we still record token usage for reporting.
        cost = 0.0
        if self.budget is not None:
            self.budget.record(tokens_in=tokens_in, tokens_out=tokens_out, cost_rub=cost)

        response = LLMResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_rub=cost,
            model=served_model,
            cached=False,
            raw=None,
        )
        if self.cache is not None:
            self.cache.put(key, response)
        return response

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # pragma: no cover
                pass
            self._client = None
