"""Managed llama-server engine for GGUF inference with built-in prompt caching.

Wraps llama-server.exe as a managed subprocess, providing:
- Automatic server lifecycle management (start/stop/health check)
- OpenAI-compatible completion API
- Built-in prompt prefix caching (same prefix → cached KV)
- Configurable threads, context size, batch size
- Tool routing with forced prefix and stop tokens

This is the production inference backend for contextcache.
The server binary (llama-server) is shipped with the project.
"""

import json
import os
import platform
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests


class LlamaServerEngine:
    """Managed llama-server subprocess with HTTP API for inference."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 16384,
        n_threads: int = 4,
        n_batch: int = 2048,
        n_slots: int = 1,
        port: int = 8190,
        host: str = "127.0.0.1",
        server_binary: Optional[str] = None,
        cache_ram_mb: int = 8192,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.port = port
        self.host = host
        self.base_url = f"http://{host}:{port}"
        self.verbose = verbose
        self._process: Optional[subprocess.Popen] = None

        # Find server binary
        self.server_binary = server_binary or self._find_server_binary()

        # Build command
        cmd = [
            self.server_binary,
            "-m", model_path,
            "-c", str(n_ctx),
            "-t", str(n_threads),
            "-b", str(n_batch),
            "--port", str(port),
            "--host", host,
            "-np", str(n_slots),
            "--cache-ram", str(cache_ram_mb),
        ]

        # Start server
        stderr_target = None if verbose else subprocess.DEVNULL
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL if not verbose else None,
            stderr=stderr_target,
        )

        # Wait for server to be ready
        if not self._wait_ready(timeout=120):
            self.stop()
            raise RuntimeError(
                f"llama-server failed to start within 120s. "
                f"Binary: {self.server_binary}, Model: {model_path}"
            )

    def _find_server_binary(self) -> str:
        """Find llama-server executable."""
        search_paths = []

        env_path = os.environ.get("LLAMA_SERVER_BIN")
        if env_path:
            search_paths.append(env_path)

        project_root = Path(__file__).parent.parent
        exe = "llama-server.exe" if platform.system() == "Windows" else "llama-server"

        search_paths.extend([
            str(project_root / "bin" / exe),
            str(Path.home() / "llama-cpp-b8185" / exe),
        ])

        for path in search_paths:
            if os.path.exists(path):
                return path

        raise RuntimeError(
            f"Could not find {exe}. Searched: {search_paths}. "
            "Set LLAMA_SERVER_BIN environment variable."
        )

    def _wait_ready(self, timeout: int = 120) -> bool:
        """Wait for server health endpoint to respond."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    return True
            except requests.ConnectionError:
                pass

            # Check if process died
            if self._process and self._process.poll() is not None:
                return False

            time.sleep(0.5)
        return False

    @property
    def is_alive(self) -> bool:
        """Check if server process is running and healthy."""
        if not self._process or self._process.poll() is not None:
            return False
        try:
            r = requests.get(f"{self.base_url}/health", timeout=2)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    # ─── Inference API ───────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
    ) -> dict:
        """Generate completion for a prompt.

        Returns dict with keys: text, prompt_tokens, completion_tokens,
        prompt_eval_ms, eval_ms, total_ms.
        """
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        t0 = time.perf_counter()
        r = requests.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            timeout=120,
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()

        data = r.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        timings = data.get("timings", {})

        return {
            "text": choice["text"],
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_eval_ms": timings.get("prompt_ms", 0),
            "eval_ms": timings.get("predicted_ms", 0),
            "prompt_tok_s": timings.get("prompt_per_second", 0),
            "eval_tok_s": timings.get("predicted_per_second", 0),
            "wall_ms": wall_ms,
        }

    def route_tool(
        self,
        system_prompt: str,
        tools_text: str,
        user_query: str,
        max_tokens: int = 15,
    ) -> dict:
        """Route a user query to the best tool. Returns tool name + timing.

        Uses Qwen3.5 chat template with forced tool call prefix.
        The prompt is constructed to force the model to output a tool name.
        """
        prompt = (
            f"<|im_start|>system\n{system_prompt}\n\n"
            f"# Tools\n\nYou have access to the following functions:\n\n"
            f"<tools>\n{tools_text}\n</tools><|im_end|>\n"
            f"<|im_start|>user\n/no_think\n{user_query}<|im_end|>\n"
            f'<|im_start|>assistant\n<tool_call>\n{{"name": "'
        )

        result = self.complete(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=['"', "</tool_call>", "<|im_end|>", "<|endoftext|>"],
        )

        # Extract tool name (model generates the name, we prepend {"name": ")
        tool_name = result["text"].strip().rstrip('"').strip()

        return {
            "tool": tool_name,
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "prompt_eval_ms": result["prompt_eval_ms"],
            "eval_ms": result["eval_ms"],
            "wall_ms": result["wall_ms"],
            "prompt_tok_s": result["prompt_tok_s"],
            "eval_tok_s": result["eval_tok_s"],
        }

    # ─── Lifecycle ───────────────────────────────────────────────────

    def stop(self):
        """Stop the server process."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

    def close(self):
        """Alias for stop()."""
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()
