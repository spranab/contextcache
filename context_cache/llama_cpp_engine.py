"""Minimal ctypes wrapper around llama.cpp DLL for GGUF inference with KV cache control.

This module provides direct access to llama.cpp's C API via ctypes, supporting:
- Model loading from GGUF files
- Tokenization
- Forward pass (prefill) with batched decoding
- KV cache save/restore (both in-memory and to disk)
- Greedy or temperature-based sampling
- Token-to-text detokenization

Designed for the contextcache project's tool routing use case:
prefill tool schemas once, save KV state, restore per query.
"""

import ctypes
import ctypes.util
import os
import platform
import numpy as np
from pathlib import Path
from typing import Optional

# Prevent OpenMP conflict between llama.cpp and numpy/MKL
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# ─── Locate DLL ──────────────────────────────────────────────────────────

_LIB = None
_GGML_LIB = None
_BACKENDS_LOADED = False

def _find_lib_dir() -> str:
    """Find the directory containing llama.cpp libraries."""
    search_dirs = []

    env_path = os.environ.get("LLAMA_CPP_LIB")
    if env_path:
        search_dirs.append(os.path.dirname(os.path.abspath(env_path)))

    project_root = Path(__file__).parent.parent
    if platform.system() == "Windows":
        search_dirs.append(str(project_root / "bin"))
    else:
        search_dirs.append(str(project_root / "lib"))

    if platform.system() == "Windows":
        search_dirs.append(str(Path.home() / "llama-cpp-b8185"))

    for d in search_dirs:
        llama_name = "llama.dll" if platform.system() == "Windows" else "libllama.so"
        if os.path.exists(os.path.join(d, llama_name)):
            return d

    raise RuntimeError(
        f"Could not find llama.dll. Searched: {search_dirs}. "
        "Set LLAMA_CPP_LIB environment variable to the DLL path."
    )


def _load_libs() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    """Load llama.dll and ggml.dll, register CPU backends."""
    global _LIB, _GGML_LIB, _BACKENDS_LOADED

    if _LIB is not None:
        return _LIB, _GGML_LIB

    lib_dir = _find_lib_dir()

    # Add to DLL search path so dependencies are found
    if platform.system() == "Windows":
        os.add_dll_directory(lib_dir)

    # Load ggml.dll first (has backend registration)
    ggml_name = "ggml.dll" if platform.system() == "Windows" else "libggml.so"
    _GGML_LIB = ctypes.CDLL(os.path.join(lib_dir, ggml_name))

    # Register all CPU backends (ggml-cpu-*.dll)
    if not _BACKENDS_LOADED:
        _GGML_LIB.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
        _GGML_LIB.ggml_backend_load_all_from_path.restype = None
        _GGML_LIB.ggml_backend_load_all_from_path(lib_dir.encode("utf-8"))

        _GGML_LIB.ggml_backend_dev_count.argtypes = []
        _GGML_LIB.ggml_backend_dev_count.restype = ctypes.c_size_t
        n_backends = _GGML_LIB.ggml_backend_dev_count()
        _BACKENDS_LOADED = True

    # Load llama.dll
    llama_name = "llama.dll" if platform.system() == "Windows" else "libllama.so"
    _LIB = ctypes.CDLL(os.path.join(lib_dir, llama_name))

    return _LIB, _GGML_LIB


# ─── C Type Definitions ─────────────────────────────────────────────────

llama_model_p = ctypes.c_void_p
llama_context_p = ctypes.c_void_p
llama_vocab_p = ctypes.c_void_p
llama_sampler_p = ctypes.c_void_p
llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32


class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.c_void_p),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),                    # ggml_backend_dev_t *
        ("tensor_buft_overrides", ctypes.c_void_p),      # const llama_model_tensor_buft_override *
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),                   # enum llama_split_mode
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_direct_io", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
    ]


class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),            # enum llama_rope_scaling_type
        ("pooling_type", ctypes.c_int32),                  # enum llama_pooling_type
        ("attention_type", ctypes.c_int32),                # enum llama_attention_type
        ("flash_attn_type", ctypes.c_int32),               # enum llama_flash_attn_type
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),                      # ggml_backend_sched_eval_callback
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),                        # enum ggml_type
        ("type_v", ctypes.c_int32),                        # enum ggml_type
        ("abort_callback", ctypes.c_void_p),               # ggml_abort_callback
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.c_void_p),                     # llama_sampler_seq_config *
        ("n_samplers", ctypes.c_size_t),
    ]


class llama_sampler_chain_params(ctypes.Structure):
    _fields_ = [
        ("no_perf", ctypes.c_bool),
    ]


# ─── LlamaCppEngine ─────────────────────────────────────────────────────

class LlamaCppEngine:
    """High-level interface to llama.cpp for GGUF inference with KV cache control."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 16384,
        n_threads: int = 4,
        n_batch: int = 2048,
        verbose: bool = False,
    ):
        self.lib, self.ggml_lib = _load_libs()
        self._setup_functions()

        # Init backend
        self.lib.llama_backend_init()

        # Load model
        model_params = self.lib.llama_model_default_params()
        model_params.n_gpu_layers = 0  # CPU only
        if not verbose:
            model_params.progress_callback = None

        self.model = self.lib.llama_load_model_from_file(
            model_path.encode("utf-8"),
            model_params,
        )
        if not self.model:
            raise RuntimeError(f"Failed to load model: {model_path}")

        # Create context
        ctx_params = self.lib.llama_context_default_params()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_batch
        ctx_params.n_threads = n_threads
        ctx_params.n_threads_batch = n_threads
        ctx_params.flash_attn_type = 0  # disabled
        ctx_params.offload_kqv = False

        self.ctx = self.lib.llama_new_context_with_model(self.model, ctx_params)
        if not self.ctx:
            raise RuntimeError("Failed to create context")

        self.n_ctx = n_ctx
        self.n_batch = n_batch

        # Get vocab
        self.vocab = self.lib.llama_model_get_vocab(self.model)
        self.n_vocab = self.lib.llama_vocab_n_tokens(self.vocab)
        self.eos_token = self.lib.llama_token_eos(self.vocab)
        self.bos_token = self.lib.llama_token_bos(self.vocab)

    def _setup_functions(self):
        """Set up ctypes function signatures."""
        lib = self.lib

        # Backend
        lib.llama_backend_init.argtypes = []
        lib.llama_backend_init.restype = None

        # Model params
        lib.llama_model_default_params.argtypes = []
        lib.llama_model_default_params.restype = llama_model_params

        # Context params
        lib.llama_context_default_params.argtypes = []
        lib.llama_context_default_params.restype = llama_context_params

        # Load model
        lib.llama_load_model_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
        lib.llama_load_model_from_file.restype = llama_model_p

        # New context
        lib.llama_new_context_with_model.argtypes = [llama_model_p, llama_context_params]
        lib.llama_new_context_with_model.restype = llama_context_p

        # Vocab
        lib.llama_model_get_vocab.argtypes = [llama_model_p]
        lib.llama_model_get_vocab.restype = llama_vocab_p

        lib.llama_vocab_n_tokens.argtypes = [llama_vocab_p]
        lib.llama_vocab_n_tokens.restype = ctypes.c_int32

        lib.llama_token_eos.argtypes = [llama_vocab_p]
        lib.llama_token_eos.restype = llama_token

        lib.llama_token_bos.argtypes = [llama_vocab_p]
        lib.llama_token_bos.restype = llama_token

        # Tokenize
        lib.llama_tokenize.argtypes = [
            llama_vocab_p,
            ctypes.c_char_p, ctypes.c_int32,
            ctypes.POINTER(llama_token), ctypes.c_int32,
            ctypes.c_bool, ctypes.c_bool,
        ]
        lib.llama_tokenize.restype = ctypes.c_int32

        # Batch
        lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        lib.llama_batch_init.restype = llama_batch

        lib.llama_batch_free.argtypes = [llama_batch]
        lib.llama_batch_free.restype = None

        # Decode
        lib.llama_decode.argtypes = [llama_context_p, llama_batch]
        lib.llama_decode.restype = ctypes.c_int32

        # Logits
        lib.llama_get_logits_ith.argtypes = [llama_context_p, ctypes.c_int32]
        lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

        # N ctx
        lib.llama_n_ctx.argtypes = [llama_context_p]
        lib.llama_n_ctx.restype = ctypes.c_uint32

        # State save/load (full context)
        lib.llama_state_get_size.argtypes = [llama_context_p]
        lib.llama_state_get_size.restype = ctypes.c_size_t

        lib.llama_state_get_data.argtypes = [
            llama_context_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ]
        lib.llama_state_get_data.restype = ctypes.c_size_t

        lib.llama_state_set_data.argtypes = [
            llama_context_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ]
        lib.llama_state_set_data.restype = ctypes.c_size_t

        # State save/load to file
        lib.llama_state_save_file.argtypes = [
            llama_context_p, ctypes.c_char_p,
            ctypes.POINTER(llama_token), ctypes.c_size_t,
        ]
        lib.llama_state_save_file.restype = ctypes.c_size_t

        lib.llama_state_load_file.argtypes = [
            llama_context_p, ctypes.c_char_p,
            ctypes.POINTER(llama_token), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.llama_state_load_file.restype = ctypes.c_bool

        # Seq-level state save/load
        lib.llama_state_seq_get_size.argtypes = [llama_context_p, llama_seq_id]
        lib.llama_state_seq_get_size.restype = ctypes.c_size_t

        lib.llama_state_seq_get_data.argtypes = [
            llama_context_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            llama_seq_id,
        ]
        lib.llama_state_seq_get_data.restype = ctypes.c_size_t

        lib.llama_state_seq_set_data.argtypes = [
            llama_context_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            llama_seq_id,
        ]
        lib.llama_state_seq_set_data.restype = ctypes.c_size_t

        # Sampler
        lib.llama_sampler_chain_default_params.argtypes = []
        lib.llama_sampler_chain_default_params.restype = llama_sampler_chain_params

        lib.llama_sampler_chain_init.argtypes = [llama_sampler_chain_params]
        lib.llama_sampler_chain_init.restype = llama_sampler_p

        lib.llama_sampler_init_greedy.argtypes = []
        lib.llama_sampler_init_greedy.restype = llama_sampler_p

        lib.llama_sampler_init_temp.argtypes = [ctypes.c_float]
        lib.llama_sampler_init_temp.restype = llama_sampler_p

        lib.llama_sampler_init_top_k.argtypes = [ctypes.c_int32]
        lib.llama_sampler_init_top_k.restype = llama_sampler_p

        lib.llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
        lib.llama_sampler_init_top_p.restype = llama_sampler_p

        lib.llama_sampler_chain_add.argtypes = [llama_sampler_p, llama_sampler_p]
        lib.llama_sampler_chain_add.restype = None

        lib.llama_sampler_sample.argtypes = [llama_sampler_p, llama_context_p, ctypes.c_int32]
        lib.llama_sampler_sample.restype = llama_token

        lib.llama_sampler_reset.argtypes = [llama_sampler_p]
        lib.llama_sampler_reset.restype = None

        lib.llama_sampler_free.argtypes = [llama_sampler_p]
        lib.llama_sampler_free.restype = None

        # Detokenize
        lib.llama_token_to_piece.argtypes = [
            llama_vocab_p, llama_token,
            ctypes.c_char_p, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_bool,
        ]
        lib.llama_token_to_piece.restype = ctypes.c_int32

        # Free
        lib.llama_free.argtypes = [llama_context_p]
        lib.llama_free.restype = None

        lib.llama_model_free.argtypes = [llama_model_p]
        lib.llama_model_free.restype = None

    # ─── Tokenization ────────────────────────────────────────────────

    def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
        """Tokenize text into token IDs."""
        text_bytes = text.encode("utf-8")
        max_tokens = len(text_bytes) + 1024
        tokens = (llama_token * max_tokens)()

        n = self.lib.llama_tokenize(
            self.vocab, text_bytes, len(text_bytes),
            tokens, max_tokens,
            add_bos, True,  # add_special, parse_special
        )
        if n < 0:
            # Need more space
            max_tokens = -n + 64
            tokens = (llama_token * max_tokens)()
            n = self.lib.llama_tokenize(
                self.vocab, text_bytes, len(text_bytes),
                tokens, max_tokens,
                add_bos, True,
            )
        return list(tokens[:n])

    def detokenize(self, tokens: list[int]) -> str:
        """Convert token IDs back to text."""
        pieces = []
        buf = ctypes.create_string_buffer(256)
        for tok in tokens:
            n = self.lib.llama_token_to_piece(
                self.vocab, tok, buf, 256, 0, False,
            )
            if n > 0:
                pieces.append(buf.raw[:n].decode("utf-8", errors="replace"))
        return "".join(pieces)

    # ─── Forward Pass ────────────────────────────────────────────────

    def prefill(self, tokens: list[int], start_pos: int = 0) -> None:
        """Run forward pass on tokens (fills KV cache). No output sampling."""
        n_tokens = len(tokens)
        for i in range(0, n_tokens, self.n_batch):
            chunk = tokens[i:i + self.n_batch]
            batch = self._make_batch(chunk, start_pos + i, logits_last=(i + len(chunk) >= n_tokens))
            rc = self.lib.llama_decode(self.ctx, batch)
            self.lib.llama_batch_free(batch)
            if rc != 0:
                raise RuntimeError(f"llama_decode failed with code {rc}")

    def _make_batch(self, tokens: list[int], start_pos: int, logits_last: bool = True) -> llama_batch:
        """Create a batch struct for decode."""
        n = len(tokens)
        batch = self.lib.llama_batch_init(n, 0, 1)
        batch.n_tokens = n

        for i, tok in enumerate(tokens):
            batch.token[i] = tok
            batch.pos[i] = start_pos + i
            batch.n_seq_id[i] = 1
            # Write to existing C-allocated seq_id array (don't replace pointer)
            batch.seq_id[i][0] = 0
            batch.logits[i] = 1 if (logits_last and i == n - 1) else 0

        return batch

    # ─── Sampling ────────────────────────────────────────────────────

    def sample_greedy(self, pos: int) -> int:
        """Sample next token greedily from logits at the last decoded position."""
        chain_params = self.lib.llama_sampler_chain_default_params()
        chain = self.lib.llama_sampler_chain_init(chain_params)
        greedy = self.lib.llama_sampler_init_greedy()
        self.lib.llama_sampler_chain_add(chain, greedy)

        token = self.lib.llama_sampler_sample(chain, self.ctx, -1)
        self.lib.llama_sampler_free(chain)
        return token

    def generate(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 32,
        stop_tokens: Optional[list[int]] = None,
        stop_strings: Optional[list[str]] = None,
        start_pos: int = 0,
    ) -> tuple[list[int], str]:
        """Generate tokens autoregressively. Returns (token_ids, text)."""
        if stop_tokens is None:
            stop_tokens = [self.eos_token]
        if stop_strings is None:
            stop_strings = []

        # Prefill prompt
        self.prefill(prompt_tokens, start_pos=start_pos)

        generated = []
        pos = start_pos + len(prompt_tokens)

        for _ in range(max_tokens):
            tok = self.sample_greedy(pos)
            if tok in stop_tokens:
                break
            generated.append(tok)

            # Check stop strings
            text_so_far = self.detokenize(generated)
            should_stop = False
            for ss in stop_strings:
                if ss in text_so_far:
                    # Trim text at stop string
                    text_so_far = text_so_far[:text_so_far.index(ss)]
                    should_stop = True
                    break
            if should_stop:
                return generated, text_so_far

            # Feed generated token back
            batch = self._make_batch([tok], pos, logits_last=True)
            rc = self.lib.llama_decode(self.ctx, batch)
            self.lib.llama_batch_free(batch)
            if rc != 0:
                raise RuntimeError(f"llama_decode failed during generation: {rc}")
            pos += 1

        return generated, self.detokenize(generated)

    # ─── KV Cache State Management ──────────────────────────────────

    def save_state(self) -> bytes:
        """Save full KV cache state to memory. Returns state bytes."""
        size = self.lib.llama_state_get_size(self.ctx)
        buf = (ctypes.c_uint8 * size)()
        written = self.lib.llama_state_get_data(self.ctx, buf, size)
        return bytes(buf[:written])

    def load_state(self, state_data: bytes) -> None:
        """Restore full KV cache state from memory."""
        size = len(state_data)
        buf = (ctypes.c_uint8 * size).from_buffer_copy(state_data)
        self.lib.llama_state_set_data(self.ctx, buf, size)

    def save_state_file(self, filepath: str, tokens: list[int]) -> None:
        """Save KV cache state to file."""
        token_arr = (llama_token * len(tokens))(*tokens)
        self.lib.llama_state_save_file(
            self.ctx, filepath.encode("utf-8"),
            token_arr, len(tokens),
        )

    def load_state_file(self, filepath: str, max_tokens: int = 131072) -> list[int]:
        """Load KV cache state from file. Returns the token list."""
        tokens = (llama_token * max_tokens)()
        n_tokens = ctypes.c_size_t(0)
        ok = self.lib.llama_state_load_file(
            self.ctx, filepath.encode("utf-8"),
            tokens, max_tokens,
            ctypes.byref(n_tokens),
        )
        if not ok:
            raise RuntimeError(f"Failed to load state from {filepath}")
        return list(tokens[:n_tokens.value])

    # ─── Cleanup ─────────────────────────────────────────────────────

    def close(self):
        """Free model and context."""
        if getattr(self, "ctx", None):
            self.lib.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            self.lib.llama_model_free(self.model)
            self.model = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
