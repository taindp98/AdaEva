"""OpenAI chat-completions client for LLM-driven algorithm design.

Features:
    - Loads `OPENAI_API_KEY` from environment (typically via python-dotenv),
      falling back to `OPENROUTER_API_KEY` so existing runner guards and
      deployments keep working unchanged.
    - Uses the OpenAI chat-completions endpoint on
      `https://api.openai.com/v1` (override via `OPENAI_BASE_URL`).
    - Adapts the request body for reasoning models (gpt-5*, o1/o3/o4*), which
      require `max_completion_tokens` instead of `max_tokens` and reject any
      `temperature` other than the default.
    - Surfaces HTTP/auth errors with a clear message.

Example:
    >>> from utils.llm import OpenRouterClient
    >>> client = OpenRouterClient(model='gpt-4.1-nano-2025-04-14')
    >>> text = client.chat([{'role': 'user', 'content': 'Say hi.'}])  # doctest: +SKIP
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import requests


DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"
# OpenAI enforces per-minute request AND token (TPM) caps. A busy evolutionary
# run saturates TPM easily, and an un-retried 429 is swallowed by the samplers
# as an empty solution -- the run then "succeeds" while doing no real work.
DEFAULT_OPENAI_MAX_RETRIES = 6
DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Google AI Studio (Gemini) OpenAI-compatible endpoint + default model.
DEFAULT_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_GOOGLE_MODEL = "gemini-3.1-flash-lite"
# Gemini "thinking" (reasoning) token budget. 0 = thinking OFF (default), a positive
# int caps thinking tokens, -1 = dynamic/auto. Overridable via GOOGLE_THINKING_BUDGET.
DEFAULT_GOOGLE_THINKING_BUDGET = 0

# Reasoning models take `max_completion_tokens` instead of `max_tokens` and
# only accept the default temperature; anything else is a 400.
_REASONING_MODEL_RE = re.compile(r"^(?:gpt-5|o1|o3|o4)", re.IGNORECASE)


def _is_reasoning_model(model: str) -> bool:
    """True if `model` is an OpenAI reasoning model with restricted params."""
    return bool(_REASONING_MODEL_RE.match(str(model).split("/")[-1]))

DEFAULT_OLLAMA_HOST = "localhost:11434"
DEFAULT_OLLAMA_MODEL = "codellama"
DEFAULT_OLLAMA_MAX_RETRIES = 3

DEFAULT_MISTRAL_MODEL = "mistral-large-latest"
DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
# Mistral devstral tier allows 0.83 requests/second (~1 every 1.2s). Default to
# a slightly larger spacing so concurrent sampler threads stay safely under the
# cap. Override per-deployment via `MISTRAL_MIN_INTERVAL` (seconds) or the
# `min_interval` constructor arg.
DEFAULT_MISTRAL_MIN_INTERVAL = 1.25
DEFAULT_MISTRAL_MAX_RETRIES = 5

DEFAULT_VLLM_MODEL = "mistralai/Devstral-Small-2-24B-Instruct-2512"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
# vLLM prompts/completions grow over a run, so a timeout that early
# generations met comfortably can start firing late in the run.
DEFAULT_VLLM_MAX_RETRIES = 3


class vLLMClient:
    """Thin HTTPS wrapper around a local vLLM chat-completions endpoint.
    
    Compatible with OpenAI's API format. Does not implement strict rate-limiting
    because vLLM handles high concurrency automatically.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        max_tokens: int = 2048,
        max_retries: Optional[int] = None,
    ):
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL") or DEFAULT_VLLM_BASE_URL
        self.base_url = self.base_url.rstrip("/")
        self.model = model or os.environ.get("VLLM_MODEL") or DEFAULT_VLLM_MODEL
        self.timeout = timeout
        self.max_tokens = max_tokens
        if max_retries is None:
            env_val = os.environ.get("VLLM_MAX_RETRIES")
            max_retries = int(env_val) if env_val else DEFAULT_VLLM_MAX_RETRIES
        self.max_retries = max(0, int(max_retries))

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.8,
        max_tokens: Optional[int] = None,
    ) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
        }
        resp = _post_with_retry(
            f"{self.base_url}/chat/completions",
            headers=headers, json_body=body, timeout=self.timeout,
            max_retries=self.max_retries, tag="vllm",
        )
        if resp.status_code >= 400:
            raise requests.HTTPError(
                f"vLLM returned {resp.status_code}: {resp.text[:400]}"
            )
        data = resp.json()
        self.last_usage = _norm_usage(data.get("usage"))
        msg = data["choices"][0]["message"]
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            joined = "".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
            if joined.strip():
                return joined
        raise RuntimeError(f"vLLM response had no usable text: {str(msg)[:400]}")


class OpenRouterClient:
    """Thin HTTPS wrapper around OpenAI's chat-completions endpoint.

    The name is kept for backwards compatibility: every runner imports and
    constructs `OpenRouterClient`, so pointing this class at OpenAI switches
    the whole project over without touching the runners.

    Arguments:
        api_key: OpenAI API key. If None, reads `OPENAI_API_KEY` and then
            `OPENROUTER_API_KEY` from the process environment (the latter so
            the runners' existing pre-flight guard keeps working).
        model: Model id (e.g. "gpt-4.1-nano-2025-04-14", "gpt-4o-mini"). If
            None, reads `OPENAI_MODEL` / `OPENROUTER_MODEL` or falls back to
            `DEFAULT_MODEL`. A legacy "provider/model" id is reduced to its
            last segment, since OpenAI expects a bare model name.
        timeout: Per-request HTTP timeout in seconds.
        base_url: Override for the API base URL; defaults to `OPENAI_BASE_URL`
            or `DEFAULT_BASE_URL`.
        x_title: Retained for call-site compatibility. OpenAI ignores the
            OpenRouter attribution headers, so this is unused.

    Example:
        >>> client = OpenRouterClient()  # doctest: +SKIP
        >>> client.chat([{'role': 'user', 'content': 'ping'}])  # doctest: +SKIP
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        base_url: Optional[str] = None,
        x_title: str = "llm4ad",
        max_retries: Optional[int] = None,
    ):
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env at the "
                "repo root and fill in your key, then `load_dotenv()` before constructing the client."
            )
        model = (
            model
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("OPENROUTER_MODEL")
            or DEFAULT_MODEL
        )
        # Tolerate legacy OpenRouter-style ids ("openai/gpt-4o-mini").
        self.model = str(model).split("/")[-1]
        self.timeout = timeout
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.x_title = x_title
        # Extra top-level fields merged into every request body (subclasses set this,
        # e.g. GoogleClient's Gemini thinking config). Empty for plain OpenAI.
        self._extra_body: dict = {}
        if max_retries is None:
            env_val = os.environ.get("OPENAI_MAX_RETRIES")
            max_retries = int(env_val) if env_val else DEFAULT_OPENAI_MAX_RETRIES
        self.max_retries = max(0, int(max_retries))

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat-completion request and return the assistant message text.

        Arguments:
            messages: A list of {"role": "...", "content": "..."} dicts.
            temperature: Sampling temperature. EoH typically uses 0.7-1.0
                for diversity across children. Ignored for reasoning models,
                which only accept their default temperature.
            max_tokens: Cap on the response length. Sent as
                `max_completion_tokens` for reasoning models.

        Returns:
            The assistant's response string.

        Raises:
            requests.HTTPError: On non-2xx responses.
        """
        body = {"model": self.model, "messages": messages}
        if _is_reasoning_model(self.model):
            # gpt-5* / o-series: `max_tokens` is rejected outright, and any
            # explicit `temperature` other than the default is a 400. The cap
            # also has to cover the (invisible) reasoning tokens, so scale it
            # up or a short answer comes back empty with finish_reason=length.
            body["max_completion_tokens"] = max(max_tokens * 4, 4096)
        else:
            body["temperature"] = temperature
            body["max_tokens"] = max_tokens
        if self._extra_body:
            body.update(self._extra_body)
        resp = _post_with_retry(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json_body=body, timeout=self.timeout,
            max_retries=self.max_retries, tag="openai", reset_headers=True,
        )
        if resp.status_code >= 400:
            raise requests.HTTPError(
                f"OpenAI returned {resp.status_code}: {resp.text[:400]}"
            )
        data = resp.json()
        self.last_usage = _norm_usage(data.get("usage"))
        msg = data["choices"][0]["message"]
        # Some reasoning models return content=None and put text in
        # `reasoning` or `reasoning_content`. Fall back through them.
        for key in ("content", "reasoning_content", "reasoning"):
            v = msg.get(key)
            if isinstance(v, str) and v.strip():
                return v
            if isinstance(v, list):  # some providers return content blocks
                joined = "".join(b.get("text", "") for b in v if isinstance(b, dict))
                if joined.strip():
                    return joined
        raise RuntimeError(f"OpenAI response had no usable text: {str(msg)[:400]}")


class GoogleClient(OpenRouterClient):
    """Google AI Studio (Gemini) via its OpenAI-compatible endpoint.

    Gemini exposes an OpenAI chat-completions API at
    ``https://generativelanguage.googleapis.com/v1beta/openai`` with a
    ``Bearer <GOOGLE_API_KEY>`` header and the standard {model, messages,
    temperature, max_tokens} body — byte-identical to what ``OpenRouterClient``
    already sends. So this subclass only swaps the key/base_url defaults and
    inherits ``chat()`` (retries, usage parsing, content fallbacks) unchanged.

    Arguments mirror ``OpenRouterClient``; the api_key defaults to
    ``GOOGLE_API_KEY`` and the model id keeps its bare name (e.g.
    "google/gemini-3.1-flash-lite" -> "gemini-3.1-flash-lite", which the
    endpoint accepts).

    ``thinking_budget`` controls Gemini's reasoning ("thinking") token budget,
    passed through the OpenAI-compatible endpoint as
    ``extra_body.google.thinking_config.thinking_budget``: ``0`` disables
    thinking (default), a positive int caps thinking tokens, ``-1`` is
    dynamic/auto. Defaults to ``GOOGLE_THINKING_BUDGET`` env or
    ``DEFAULT_GOOGLE_THINKING_BUDGET`` (0).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        base_url: Optional[str] = None,
        x_title: str = "llm4ad",
        max_retries: Optional[int] = None,
        thinking_budget: Optional[int] = None,
    ):
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Add it to the repo-root .env and "
                "`load_dotenv()` before constructing GoogleClient."
            )
        model = model or os.environ.get("GOOGLE_MODEL") or DEFAULT_GOOGLE_MODEL
        base_url = base_url or os.environ.get("GOOGLE_BASE_URL") or DEFAULT_GOOGLE_BASE_URL
        super().__init__(api_key=api_key, model=model, timeout=timeout,
                         base_url=base_url, x_title=x_title, max_retries=max_retries)
        # Gemini thinking budget: explicit arg > GOOGLE_THINKING_BUDGET env > default (0=off).
        if thinking_budget is None:
            env_tb = os.environ.get("GOOGLE_THINKING_BUDGET")
            thinking_budget = int(env_tb) if env_tb not in (None, "") else DEFAULT_GOOGLE_THINKING_BUDGET
        self.thinking_budget = int(thinking_budget)
        print(f"[GoogleClient] Gemini thinking_budget={self.thinking_budget}", flush=True)
        # Passed through the OpenAI-compatible endpoint to Gemini's native thinking_config.
        # thinking_budget: 0 = no thinking, N>0 = token cap, -1 = dynamic/auto.
        self._extra_body = {
            "extra_body": {"google": {"thinking_config": {"thinking_budget": self.thinking_budget}}}
        }

    
_ZERO_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cached_tokens": 0,
    "reasoning_tokens": 0,
    "cost": 0.0,
}






def _parse_reset(v) -> float:
    """Seconds from a rate-limit header. Accepts plain seconds ("1.5") and
    OpenAI's duration form ("1m6.404s", "59.473s", "120ms"). 0.0 if unparseable."""
    if not v:
        return 0.0
    txt = str(v).strip()
    try:
        return float(txt)
    except ValueError:
        pass
    m = re.fullmatch(
        r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?"
        r"(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?", txt)
    if not m or not any(m.groups()):
        return 0.0
    h, mi, sec, ms = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + sec + ms / 1000.0


# Transient HTTP statuses worth retrying: rate limits and upstream hiccups.
_RETRY_STATUSES = (408, 409, 429, 500, 502, 503, 504)


def _post_with_retry(url, *, headers, json_body, timeout, max_retries, tag,
                     reset_headers=False):
    """POST with retries on transient failures, returning the final response.

    Retries connection errors and read timeouts as well as `_RETRY_STATUSES`.
    This matters because every sampler in this repo turns an exception into an
    empty candidate: an un-retried blip is not a loud failure but a silently
    degraded run (a Qwen/vLLM run lost 74 of 200 offspring this way when late
    generations outgrew a 120s timeout).

    `reset_headers` additionally honours OpenAI's ``x-ratelimit-reset-*``
    headers, whose ``Retry-After`` is often far too optimistic on a 429.
    """
    attempt = 0
    while True:
        try:
            resp = requests.post(url, headers=headers, json=json_body,
                                 timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt >= max_retries:
                raise
            attempt += 1
            delay = min(5.0 * attempt, 30.0)
            print(f"  [{tag}] {type(exc).__name__}; retry {attempt}/{max_retries} "
                  f"after {delay:.0f}s (timeout={timeout}s)", flush=True)
            time.sleep(delay)
            continue
        if resp.status_code in _RETRY_STATUSES and attempt < max_retries:
            delay = 0.0
            if reset_headers:
                delay = max(
                    _parse_reset(resp.headers.get("x-ratelimit-reset-requests")),
                    _parse_reset(resp.headers.get("x-ratelimit-reset-tokens")),
                )
            if delay <= 0:
                delay = _parse_reset(resp.headers.get("Retry-After"))
            if delay <= 0:
                delay = min(2.0 * (2 ** attempt), 60.0)
            attempt += 1
            delay = min(delay + 0.5, 90.0)
            print(f"  [{tag}] HTTP {resp.status_code}; retry {attempt}/{max_retries} "
                  f"after {delay:.1f}s", flush=True)
            time.sleep(delay)
            continue
        return resp


def _norm_usage(u) -> dict:
    """Normalise a provider ``usage`` block to a common token/cost shape.

    Returns ``{prompt_tokens, completion_tokens, cached_tokens,
    reasoning_tokens, cost}``. Handles OpenAI-style
    (``prompt_tokens``/``completion_tokens`` plus the ``*_tokens_details``
    sub-objects, used by vLLM / OpenRouter / Mistral) and Ollama
    (``prompt_eval_count``/``eval_count``). ``cost`` is taken from the provider
    when it reports one (OpenRouter does; OpenAI does not) and is otherwise
    left at 0.0 when the provider does not report one; cost is derived in
    post-processing from the token counts.
    """
    if not isinstance(u, dict):
        return dict(_ZERO_USAGE)
    pt = u.get("prompt_tokens", u.get("prompt_eval_count", 0)) or 0
    ct = u.get("completion_tokens", u.get("eval_count", 0)) or 0
    pd = u.get("prompt_tokens_details") or {}
    cd = u.get("completion_tokens_details") or {}
    # Accept either the raw provider shape (counts nested under *_details) or
    # an already-normalised flat dict -- cache hits re-normalise what we wrote.
    cached = u.get("cached_tokens")
    if cached is None:
        cached = (pd.get("cached_tokens", 0) or 0) if isinstance(pd, dict) else 0
    reasoning = u.get("reasoning_tokens")
    if reasoning is None:
        reasoning = (cd.get("reasoning_tokens", 0) or 0) if isinstance(cd, dict) else 0
    cost = u.get("cost")  # provider-reported (OpenRouter); absent on OpenAI
    try:
        return {
            "prompt_tokens": int(pt),
            "completion_tokens": int(ct),
            "cached_tokens": int(cached),
            "reasoning_tokens": int(reasoning),
            "cost": float(cost) if cost is not None else 0.0,
        }
    except Exception:
        return dict(_ZERO_USAGE)


class CachedLLM:
    """Wraps an `OpenRouterClient` with a deterministic on-disk JSON cache.

    Cache key = sha256(model + temperature + max_tokens + JSON(messages)).
    Hits return instantly without network use; misses call the provider and
    persist the response. Re-runs of a notebook are free and reproducible.
    """

    def __init__(
        self,
        client: "OpenRouterClient | OllamaClient | MistralClient",
        cache_dir: str | Path,
        prompt_log: str | Path | None = None,
    ):
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        # Token accounting for the dual-budget analysis. Every chat() call is counted
        # (cache hits included): a cache hit replays the
        # tokens stored at first compute, so cumulative tokens reflect the un-cached
        # workload. `last_usage` is the most recent call's usage, for per-prompt logging.
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        self.total_reasoning_tokens = 0
        self.last_usage = dict(_ZERO_USAGE)
        # Per-thread usage of the most recent chat() call on THIS thread. With
        # threaded sampling (num_threads>1) the shared `last_usage` races -- a
        # concurrent call overwrites it between another thread's call and its
        # read -- so loggers must use `take_last_usage()` instead.
        self._tls = threading.local()
        # Optional per-call prompt+token log (JSONL). Runners that own their
        # sampling loop write their own richer record (op/gen/parents); this is
        # for runners that delegate sampling to a vendored framework, where
        # CachedLLM is the only per-call seam we control. Written AFTER each
        # call so the record carries that call's token usage.
        self._prompt_log = Path(prompt_log) if prompt_log else None
        self._prompt_log_lock = threading.Lock()
        self._prompt_call_idx = 0

    def _key(
        self, messages: list[dict], temperature: float, max_tokens: int, salt: str
    ) -> str:
        payload = json.dumps(
            {
                "m": self.client.model,
                "t": temperature,
                "n": max_tokens,
                "msgs": messages,
                "salt": salt,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.8,
        max_tokens: Optional[int] = None,
        cache_salt: str = "",
    ) -> str:
        if max_tokens is None:
            max_tokens = getattr(self.client, "max_tokens", 2048)
            
        key = self._key(messages, temperature, max_tokens, cache_salt)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            rec = json.loads(path.read_text())
            usage = _norm_usage(rec.get("usage"))   # tokens stored at first compute ({} for old caches)
            self._account(usage, billed=False)       # replay: no money spent
            self._log_prompt(messages, rec["response"], usage, cache_hit=True)
            return rec["response"]
        text = self.client.chat(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        _ = cache_salt  # salt is hash-only; not sent to the provider
        usage = _norm_usage(getattr(self.client, "last_usage", None))
        path.write_text(json.dumps(
            {"model": self.client.model, "response": text, "usage": usage}))
        self.misses += 1
        self._account(usage, billed=True)
        self._log_prompt(messages, text, usage, cache_hit=False)
        return text

    def _account(self, usage: dict, billed: bool) -> None:
        """Count one chat() call (hit or miss) toward the cumulative totals.

        `billed` is True only for cache misses -- the calls that actually hit
        the provider and cost money.
        """
        self.last_usage = usage
        self._tls.usage = usage
        self.total_calls += 1
        self.total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.total_cached_tokens += int(usage.get("cached_tokens", 0) or 0)
        self.total_reasoning_tokens += int(usage.get("reasoning_tokens", 0) or 0)

    def _log_prompt(self, messages, response, usage: dict, cache_hit: bool) -> None:
        """Append one JSON record per call to the prompt log, if enabled.

        Never raises: logging must not be able to abort a run.
        """
        if self._prompt_log is None:
            return
        try:
            with self._prompt_log_lock:
                idx = self._prompt_call_idx
                self._prompt_call_idx += 1
                rec = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "call_idx": idx,
                    "model": getattr(self.client, "model", ""),
                    "cache_hit": bool(cache_hit),
                    "prompt": messages,
                    "response": response,
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "cached_tokens": int(usage.get("cached_tokens", 0) or 0),
                    "reasoning_tokens": int(usage.get("reasoning_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                }
                with open(self._prompt_log, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def take_last_usage(self) -> dict:
        """Usage of the most recent chat() call made *on the calling thread*.

        Race-free under threaded sampling, unlike the shared `last_usage`.
        Returns a zeroed dict if this thread has not made a call yet.
        """
        return dict(getattr(self._tls, "usage", None) or _ZERO_USAGE)



class OllamaClient:
    """Thin HTTPS wrapper around the Ollama /api/chat endpoint.

    Arguments:
        host: Ollama server host:port. If None, reads `OLLAMA_HOST` from the
            environment or falls back to `DEFAULT_OLLAMA_HOST`.
        model: Ollama model name. If None, reads `OLLAMA_MODEL` or falls back
            to `DEFAULT_OLLAMA_MODEL`.
        timeout: Per-request HTTP timeout in seconds.

    Example:
        >>> client = OllamaClient(model="codellama")  # doctest: +SKIP
        >>> client.chat([{'role': 'user', 'content': 'ping'}])  # doctest: +SKIP
    """

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        max_retries: Optional[int] = None,
    ):
        self.host = (
            host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
        ).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        self.timeout = timeout
        if max_retries is None:
            env_val = os.environ.get("OLLAMA_MAX_RETRIES")
            max_retries = int(env_val) if env_val else DEFAULT_OLLAMA_MAX_RETRIES
        self.max_retries = max(0, int(max_retries))

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat request to Ollama and return the assistant message text.

        Arguments:
            messages: A list of {"role": "...", "content": "..."} dicts.
            temperature: Sampling temperature.
            max_tokens: Cap on the response length (mapped to `num_predict`).

        Returns:
            The assistant's response string.

        Raises:
            requests.HTTPError: On non-2xx responses.
            RuntimeError: If the response contains no usable text.
        """
        resp = _post_with_retry(
            f"http://{self.host}/api/chat",
            headers={"Content-Type": "application/json"},
            json_body={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=self.timeout,
            max_retries=self.max_retries, tag="ollama",
        )
        if resp.status_code >= 400:
            raise requests.HTTPError(
                f"Ollama returned {resp.status_code}: {resp.text[:400]}"
            )
        data = resp.json()
        self.last_usage = _norm_usage(data.get("usage") or data)
        if "error" in data:
            raise RuntimeError(f"Ollama API error: {data['error']}")
        content = data.get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Ollama response had no usable text: {str(data)[:400]}")
        return content


class MistralClient:
    """Thin HTTPS wrapper around Mistral's chat-completions endpoint.

    Uses raw `requests` against `https://api.mistral.ai/v1/chat/completions`
    (no SDK dependency) so it mirrors `OpenRouterClient` / `OllamaClient` and
    drops into `CachedLLM` and the `MistralLLM4AD` adapter unchanged. The
    endpoint is OpenAI-compatible: the response is parsed from
    `choices[0].message.content`.

    Rate limiting: Mistral's lower tiers cap requests/second (e.g. devstral =
    0.83 req/s). Because the EoH drivers sample offspring on several threads,
    this client enforces a process-wide minimum spacing between requests via a
    shared lock (`min_interval` seconds) AND retries on HTTP 429, honoring the
    `Retry-After` header. Together these keep all sampler threads safely under
    the cap without per-driver coordination.

    Arguments:
        api_key: Mistral API key. If None, reads `MISTRAL_API_KEY` from the
            process environment.
        model: Model id (e.g. "mistral-large-latest", "devstral-small-latest").
            If None, reads `MISTRAL_MODEL` or falls back to
            `DEFAULT_MISTRAL_MODEL`.
        timeout: Per-request HTTP timeout in seconds.
        base_url: Override for the Mistral base URL.
        min_interval: Minimum seconds between request *starts*, enforced across
            all threads sharing this client. If None, reads
            `MISTRAL_MIN_INTERVAL` or falls back to
            `DEFAULT_MISTRAL_MIN_INTERVAL`.
        max_retries: Number of times to retry on HTTP 429 before raising.

    Example:
        >>> client = MistralClient(model="mistral-large-latest")  # doctest: +SKIP
        >>> client.chat([{'role': 'user', 'content': 'ping'}])    # doctest: +SKIP
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        base_url: str = DEFAULT_MISTRAL_BASE_URL,
        min_interval: Optional[float] = None,
        max_retries: int = DEFAULT_MISTRAL_MAX_RETRIES,
    ):
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MISTRAL_API_KEY is not set. Copy .env.example to .env at the "
                "repo root and fill in your key, then `load_dotenv()` before "
                "constructing the client."
            )
        self.model = model or os.environ.get("MISTRAL_MODEL") or DEFAULT_MISTRAL_MODEL
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        if min_interval is None:
            env_val = os.environ.get("MISTRAL_MIN_INTERVAL")
            min_interval = float(env_val) if env_val else DEFAULT_MISTRAL_MIN_INTERVAL
        self.min_interval = max(0.0, float(min_interval))
        self.max_retries = max(0, int(max_retries))
        # Shared throttle state: serialises the spacing decision across threads.
        self._rate_lock = threading.Lock()
        self._next_allowed = 0.0  # monotonic time when the next request may start

    def _throttle(self) -> None:
        """Block until at least `min_interval` has elapsed since the previous
        request start, then reserve the next slot. Thread-safe."""
        if self.min_interval <= 0:
            return
        while True:
            with self._rate_lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    # Reserve this slot; next request waits a full interval more.
                    self._next_allowed = now + self.min_interval
                    return
                wait = self._next_allowed - now
            time.sleep(wait)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat-completion request and return the assistant message text.

        Proactively throttles to `min_interval` between requests and retries on
        HTTP 429 (honoring `Retry-After`) up to `max_retries` times.

        Arguments:
            messages: A list of {"role": "...", "content": "..."} dicts.
            temperature: Sampling temperature.
            max_tokens: Cap on the response length.

        Returns:
            The assistant's response string.

        Raises:
            requests.HTTPError: On non-2xx responses (other than retried 429s).
            RuntimeError: If the response contains no usable text.
        """
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._throttle()
        resp = _post_with_retry(
            f"{self.base_url}/chat/completions",
            headers=headers, json_body=body, timeout=self.timeout,
            max_retries=self.max_retries, tag="mistral",
        )
        if resp.status_code >= 400:
            raise requests.HTTPError(
                f"Mistral returned {resp.status_code}: {resp.text[:400]}"
            )
        data = resp.json()
        self.last_usage = _norm_usage(data.get("usage"))
        msg = data["choices"][0]["message"]
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):  # some models return content blocks
            joined = "".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
            if joined.strip():
                return joined
        raise RuntimeError(f"Mistral response had no usable text: {str(msg)[:400]}")


class OllamaLLM4AD:
    """Adapter exposing a cached Ollama client as an `llm4ad.base.LLM`.

    Mirrors `OpenRouterLLM4AD` — accepts a `CachedLLM` wrapping an
    `OllamaClient`. Each call to `draw_sample` mixes a monotonic counter
    into the cache salt so repeated identical prompts produce distinct
    cached responses.
    """

    def __new__(cls, cached_llm: "CachedLLM", **kwargs):
        from llm4ad.base import LLM as _LLM  # deferred import

        Dyn = type("OllamaLLM4AD_", (_LLM,), {})
        inst = object.__new__(Dyn)
        _LLM.__init__(inst, **kwargs)
        inst._cached = cached_llm
        inst._call_counter = 0
        inst.draw_sample = lambda prompt, *a, **kw: _draw_sample(inst, prompt)
        return inst


class vLLMLLM4AD:
    """Adapter exposing a cached vLLM client as an `llm4ad.base.LLM`."""

    def __new__(cls, cached_llm: "CachedLLM", **kwargs):
        from llm4ad.base import LLM as _LLM  # deferred import

        Dyn = type("vLLMLLM4AD_", (_LLM,), {})
        inst = object.__new__(Dyn)
        _LLM.__init__(inst, **kwargs)
        inst._cached = cached_llm
        inst._call_counter = 0
        inst.draw_sample = lambda prompt, *a, **kw: _draw_sample(inst, prompt)
        return inst


class OpenRouterLLM4AD:
    """Adapter exposing our cached OpenAI client as an `llm4ad.base.LLM`.

    Subclasses `llm4ad.base.LLM` at construction time (the import is deferred
    so utils doesn't hard-depend on LLM4AD being installed). Each call to
    `draw_sample` mixes a monotonic counter into the cache salt so repeated
    identical prompts (e.g. EoH's `i1` population init) produce distinct
    cached responses without altering what the LLM actually sees.
    """

    def __new__(cls, cached_llm: "CachedLLM", **kwargs):
        from llm4ad.base import LLM as _LLM  # deferred import

        # Build a one-off subclass that inherits from llm4ad's LLM base.
        Dyn = type("OpenRouterLLM4AD_", (_LLM,), {})
        inst = object.__new__(Dyn)
        _LLM.__init__(inst, **kwargs)
        inst._cached = cached_llm
        inst._call_counter = 0
        inst.draw_sample = lambda prompt, *a, **kw: _draw_sample(inst, prompt)
        return inst


class MistralLLM4AD:
    """Adapter exposing a cached Mistral client as an `llm4ad.base.LLM`.

    Mirrors `OpenRouterLLM4AD` / `OllamaLLM4AD` — accepts a `CachedLLM` wrapping
    a `MistralClient`. Each call to `draw_sample` mixes a monotonic counter into
    the cache salt so repeated identical prompts produce distinct cached
    responses. The `llm4ad` import is deferred so utils need not hard-depend on
    LLM4AD being installed.
    """

    def __new__(cls, cached_llm: "CachedLLM", **kwargs):
        from llm4ad.base import LLM as _LLM  # deferred import

        Dyn = type("MistralLLM4AD_", (_LLM,), {})
        inst = object.__new__(Dyn)
        _LLM.__init__(inst, **kwargs)
        inst._cached = cached_llm
        inst._call_counter = 0
        inst.draw_sample = lambda prompt, *a, **kw: _draw_sample(inst, prompt)
        return inst


class GoogleLLM4AD:
    """Adapter exposing a cached Google (Gemini) client as an `llm4ad.base.LLM`.

    Mirrors `OpenRouterLLM4AD` — accepts a `CachedLLM` wrapping a `GoogleClient`.
    Each call to `draw_sample` mixes a monotonic counter into the cache salt so
    repeated identical prompts produce distinct cached responses.
    """

    def __new__(cls, cached_llm: "CachedLLM", **kwargs):
        from llm4ad.base import LLM as _LLM  # deferred import

        Dyn = type("GoogleLLM4AD_", (_LLM,), {})
        inst = object.__new__(Dyn)
        _LLM.__init__(inst, **kwargs)
        inst._cached = cached_llm
        inst._call_counter = 0
        inst.draw_sample = lambda prompt, *a, **kw: _draw_sample(inst, prompt)
        return inst


def _draw_sample(inst, prompt) -> str:
    inst._call_counter += 1
    salt = f"call-{inst._call_counter}"
    msg = (
        prompt
        if isinstance(prompt, list)
        else [{"role": "user", "content": str(prompt)}]
    )
    return inst._cached.chat(msg, temperature=0.9, cache_salt=salt)


_DESC_RE = re.compile(r"\{([^{}]+)\}", re.DOTALL)
_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def parse_eoh_response(text: str) -> tuple[str, str]:
    """Extract the algorithm description and Python source from an EoH response.

    Features:
        - EoH prompts ask the model to put the description in boxed `{...}`
          and the code in a fenced ```python ... ``` block. This helper
          pulls both out and tolerates minor format drift (code without the
          `python` language hint; description without surrounding text).

    Arguments:
        text: Raw assistant message string.

    Returns:
        (description, source) tuple. Either string may be empty if the
        corresponding pattern was not found.

    Example:
        >>> msg = '''Idea: {a fast greedy}
        ... ```python
        ... def f(x): return x
        ... ```'''
        >>> desc, src = parse_eoh_response(msg)
        >>> desc
        'a fast greedy'
        >>> 'def f' in src
        True
    """
    if not isinstance(text, str):
        return "", ""
    desc_match = _DESC_RE.search(text)
    description = desc_match.group(1).strip() if desc_match else ""
    code_match = _CODE_RE.search(text)
    source = code_match.group(1).strip() if code_match else ""
    if not source:
        # Fallback: try to grab anything that looks like a def block.
        m = re.search(r"(import .*?\ndef .*)", text, re.DOTALL)
        source = m.group(1).strip() if m else ""
    return description, source
