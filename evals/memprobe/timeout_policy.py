"""Observe native HTTP requests; no AST patching or deadline replacement."""
from __future__ import annotations

import json
import time
from contextvars import ContextVar

import httpx

from kore_next.memory_protocol import RELATION_PROMPT

HTTP_OBSERVATION = ContextVar("memprobe_http_observation", default=None)
OriginalAsyncClient = httpx.AsyncClient


def is_relation_request(messages):
    return (len(messages) == 1 and messages[0].get("role") == "user"
            and str(messages[0].get("content", "")).startswith(RELATION_PROMPT))


class ObservedHTTPClient(OriginalAsyncClient):
    def __init__(self, *args, **kwargs):
        observation = HTTP_OBSERVATION.get()
        if observation is not None:
            hooks = {k: list(v) for k, v in (kwargs.pop("event_hooks", {}) or {}).items()}

            async def request_hook(request):
                body = json.loads(request.content)
                observation.update(request_started_seconds=time.perf_counter() - observation["clock"],
                                   effective_timeout=request.extensions.get("timeout"),
                                   wire_model=body.get("model"), stream=bool(body.get("stream")),
                                   wire_parameters={k: body[k] for k in ("temperature", "reasoning_effort", "reasoning") if k in body})

            async def response_hook(response):
                observation.update(response_headers_seconds=time.perf_counter() - observation["clock"],
                                   http_status=response.status_code)
                if not observation.get("stream") and response.is_success:
                    await response.aread()
                    try:
                        observation["reported_model"] = response.json().get("model")
                    except (ValueError, AttributeError):
                        pass

            hooks.setdefault("request", []).append(request_hook)
            hooks.setdefault("response", []).append(response_hook)
            kwargs["event_hooks"] = hooks
        super().__init__(*args, **kwargs)


def install_transport_observer():
    # Isolated process only; no change to request bodies, retries or timeouts.
    httpx.AsyncClient = ObservedHTTPClient


def public_policy():
    return {"profile": "native_product_policy", "source": "MemoryService.policy and Application._memory_relation_model",
            "relation_total_default_seconds": 600, "relation_http_read_default_seconds": 300,
            "other_calls_default_http_seconds": 120, "overrides": "isolated copy of saved memory_policy",
            "retry_policy": "unchanged product transient retries; no quality retries",
            "product_source_modified": False, "ast_rewriting": False}
