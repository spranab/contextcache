"""Production middleware for ContextCache API — auth, rate limiting, metrics."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class APIKeyStore:
    """Simple API key validation with optional per-key metadata."""

    def __init__(self):
        self._keys: dict[str, dict] = {}

    def add_key(self, key: str, name: str = "", rate_limit: int = 60):
        self._keys[key] = {"name": name, "rate_limit": rate_limit}

    def validate(self, key: str) -> dict | None:
        return self._keys.get(key)

    def remove_key(self, key: str):
        self._keys.pop(key, None)

    @classmethod
    def from_json(cls, path: str | Path) -> "APIKeyStore":
        store = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("keys", []):
            store.add_key(
                entry["key"],
                name=entry.get("name", ""),
                rate_limit=entry.get("rate_limit", 60),
            )
        return store

    @classmethod
    def from_env_key(cls, key: str) -> "APIKeyStore":
        store = cls()
        store.add_key(key, name="env_key", rate_limit=120)
        return store

    @property
    def num_keys(self) -> int:
        return len(self._keys)


class SlidingWindowRateLimiter:
    """Per-key sliding window rate limiter (requests per minute)."""

    def __init__(self, default_rpm: int = 60):
        self.default_rpm = default_rpm
        self._windows: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str, limit: int | None = None) -> tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, remaining)."""
        rpm = limit or self.default_rpm
        now = time.time()
        window = self._windows[key]

        # Prune old entries (older than 60s)
        cutoff = now - 60
        window[:] = [t for t in window if t > cutoff]

        if len(window) >= rpm:
            return False, 0

        window.append(now)
        return True, rpm - len(window)


class MetricsCollector:
    """Per-key request metrics."""

    def __init__(self):
        self._metrics: dict[str, dict] = defaultdict(
            lambda: {
                "total_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_latency_ms": 0.0,
                "errors": 0,
            }
        )

    def record_request(
        self, key: str, latency_ms: float, cache_hit: bool = False, error: bool = False
    ):
        m = self._metrics[key]
        m["total_requests"] += 1
        m["total_latency_ms"] += latency_ms
        if cache_hit:
            m["cache_hits"] += 1
        else:
            m["cache_misses"] += 1
        if error:
            m["errors"] += 1

    def get_metrics(self, key: str | None = None) -> dict:
        if key:
            m = self._metrics.get(key)
            if not m:
                return {}
            avg = m["total_latency_ms"] / max(m["total_requests"], 1)
            return {**m, "avg_latency_ms": round(avg, 1)}

        # Global summary
        total = {"total_requests": 0, "cache_hits": 0, "total_latency_ms": 0.0, "errors": 0}
        for m in self._metrics.values():
            total["total_requests"] += m["total_requests"]
            total["cache_hits"] += m["cache_hits"]
            total["total_latency_ms"] += m["total_latency_ms"]
            total["errors"] += m["errors"]

        avg = total["total_latency_ms"] / max(total["total_requests"], 1)
        return {
            **total,
            "avg_latency_ms": round(avg, 1),
            "num_keys": len(self._metrics),
            "per_key": {
                k: {**v, "avg_latency_ms": round(v["total_latency_ms"] / max(v["total_requests"], 1), 1)}
                for k, v in self._metrics.items()
            },
        }


# Skip paths that don't need auth
_SKIP_PATHS = {"/health", "/mode", "/docs", "/openapi.json", "/redoc"}


class AuthRateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware combining API key auth + rate limiting.

    Skips auth for health/docs endpoints and static file paths.
    """

    def __init__(
        self,
        app,
        key_store: APIKeyStore,
        rate_limiter: SlidingWindowRateLimiter,
        metrics: MetricsCollector,
    ):
        super().__init__(app)
        self.key_store = key_store
        self.rate_limiter = rate_limiter
        self.metrics = metrics

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for health, docs, and static files
        if path in _SKIP_PATHS or path.startswith("/static"):
            return await call_next(request)

        # Extract API key
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if not api_key:
            return Response(
                content=json.dumps({"detail": "Missing API key. Set X-API-Key header."}),
                status_code=401,
                media_type="application/json",
            )

        key_info = self.key_store.validate(api_key)
        if key_info is None:
            return Response(
                content=json.dumps({"detail": "Invalid API key."}),
                status_code=403,
                media_type="application/json",
            )

        # Rate limit check
        allowed, remaining = self.rate_limiter.check(
            api_key, limit=key_info.get("rate_limit")
        )
        if not allowed:
            self.metrics.record_request(api_key, 0, error=True)
            return Response(
                content=json.dumps({"detail": "Rate limit exceeded. Try again later."}),
                status_code=429,
                media_type="application/json",
                headers={"X-RateLimit-Remaining": "0"},
            )

        # Process request
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - t0) * 1000

        is_error = response.status_code >= 400
        self.metrics.record_request(api_key, latency_ms, error=is_error)

        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
