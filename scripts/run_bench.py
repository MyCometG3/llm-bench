#!/usr/bin/env python3
"""Unified LLM benchmark runner for oMLX (thinking x effort x budget).

--thinking-budgets N [M ...] adds a top-level "thinking_budget": N (integer
>= 0) to thinking=on request bodies (oMLX engine-side <think> block monitor,
template-independent); thinking=off rows never send it. On thinking=on rows
the selected efforts and budgets are crossed, so each (effort, budget) pair
becomes its own condition and row key; 'none' adds a no-budget (natural
thinking length) condition, and an empty list never sends the field.

Each (model, thinking, effort, budget) condition runs ONE fully isolated load
cycle: unload loaded physical models -> [admin reload] -> sleep -> load
model -> throwaway warmup request (same condition, small budget) -> then
runs x prompts measurement requests. (One load cycle per condition,
not per run.)

If any setup step (model status, unload, reload, availability wait, load,
warmup) fails, the condition's rows are recorded with status "error"
and no HTTP measurement request is sent for that condition.

Output files under ./data/<stamp>/ (one set per run, named <kind>_<stamp>.*):
  manifest_<stamp>.json  experiment config + script/prompts hashes
                         (--resume verifies this before appending)
  rows_<stamp>.jsonl     one record per ATTEMPTED request (errors included);
                         the only place prompt/usage/raw responses are saved
  rows_<stamp>.csv       same rows, scalar fields only (no prompt/usage/raw)
  load_log_<stamp>.tsv   every unload/reload/reload_wait/load HTTP call with
                         measured wall time (millisecond timestamps)
  summary_<stamp>.md     per-condition ok-row stats + separate load table
                         (same ok predicate as --resume)
  <model>.setting.json   oMLX model_settings.json settings block for each
                         selected model (saved once at run start; path
                         overridable via $OMLX_SETTINGS_FILE, default
                         ~/.omlx/model_settings.json)

Row status:
   ok           complete SSE response ([DONE] seen, finish_reason present,
                no malformed/unknown chunks, at least one non-empty token,
                valid usage.completion_tokens)
   incomplete   stream ended but protocol incomplete (no [DONE], no
                finish_reason, malformed/unknown chunk, tokenless stream,
                invalid usage value)
   error        HTTP/timeout exception, or condition setup failed
--resume and the summary use the SAME done predicate (is_row_done): a row is
done iff error is null, status is "ok" (absent on legacy rows), and
finish_reason is present.

Metrics per ok run:
  ttft_s                   request start -> first content token
  first_any_token_s        request start -> first reasoning OR content token
  first_reasoning_token_s  request start -> first reasoning token
  total_s                  total request wall time
  content_tokens           server usage completion_tokens (always recorded;
                           for thinking models this includes thinking tokens)
  reasoning_chars          len(reasoning_content)
  content_chars            len(content)
  thinking_chars           thinking char count (see THINKING_MODES)
  answer_chars             len(content) NOT counting an echoed thinking block;
                           null when the thinking arrived as an unsplitable
                           content prefix (trace + answer in one blob)
  thinking_echo            1 when content is a verbatim copy of
                           reasoning_content: the engine flushed a still-open
                           think block, so the row carries no answer
  answer_unknown           1 when the answer length cannot be derived at all
                           (content-prefix routing): excluded from the
                           answer_chars mean and from Table 5 scoring
  truncated                1 when finish_reason is "length" OR
                           usage.completion_tokens reached max_tokens (the
                           server's finish_reason alone is not trusted: engines
                           report "stop" at the cap)
  no_answer                1 when answer_chars == 0
                           The last four are derived by annotate_row at report
                           time (never stored in the JSONL) and are written to
                           the CSV, so --resummarize reproduces them for older
                           runs. status=ok does NOT mean an answer was given.
  trace_flag / leak_rc /   1 only on thinking=off rows where thinking still
  leak_content             appeared (reasoning_content present / trace header
                           in content)
  load_s                   /v1/models/<m>/load response time for this
                           condition's load cycle (does NOT include reload
                           wait time; that is a separate load_log row)
  finish_reason            stop / length / other
  error                    protocol/HTTP error text, or null
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
PROMPTS_FILE = DATA_DIR / "prompts.json"
DEFAULT_SETTINGS_PATH = Path.home() / ".omlx" / "model_settings.json"

# Throwaway warmup budget: enough to pay the hidden in-server load cost
# on the first request after load without paying for a full generation.
WARMUP_MAX_TOKENS = 16

MODELS = [
    # Defaults are pinned HF repos with verified oMLX-format builds; see the
    # README "Obtaining models" section for exact repo + revision.
    "Ornith-1.5-35B-A3B-oQ4e-mtp",  # HF scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp
    "Qwen3.8-27B-oQ4e-mtp",         # HF Jundot/Qwen3.8-27B-oQ4e-mtp
    # "Ornith-1.0-35B-oQ4e",  # no oMLX-format HF repo; base:
    #                         # ornith-ai/Ornith-1.0-35B, MLX build:
    #                         # lmstudio-community/Ornith-1.0-35B-MLX-4bit
    # "Ornith-1.5-35B-A3B-MLX-oQ4e",
    # "Ornith-1.0-35B-oQ8e",
    # "Ornith-1.5-35B-A3B-oQ8e-mtp",
    # "Ornith-1.5-35B-A3B-MLX-oQ8e",
    # "Laguna-XS-2.1-oQ8e",
    # "Qwen3.8-27B-oQ6e-mtp",
    # "Laguna-S-2.1-oQ3e",
    # "Laguna-S-2.1-oQ4e",
]
THINKING = ["off", "on"]
# "xhigh" is a real Qwen3.8 template value but is not part of the default
# matrix; add it explicitly with --efforts to measure it.
EFFORTS = ["off", "low", "medium", "high", "xhigh"]
DEFAULT_EFFORTS = ["low", "medium", "high"]

# Thinking-output routing per model family (see NOTE below for how the
# table was determined).
# Thinking output routing + trace-header pattern per model.
#   "reasoning_content" : thinking text normally arrives in
#                         message.reasoning_content.
#   "content_prefix"    : thinking text normally arrives as a prefix of
#                         message.content (no reasoning_content field); the
#                         header pattern below is used to detect a trace in
#                         content.
# IMPORTANT: for 1.0-family (oQ8e/oQ4e) the field placement is NOT
# consistent per response - the same model+setting put thinking in
# reasoning_content for one prompt and in a content prefix for another
# (observed 2026-08-23, 5/5 "8*7" responses -> reasoning_content, 2/2
# counting-problem responses -> content prefix). So per-response
# auto-detection takes priority (see summarize_thinking), and the table
# value is only used as the fallback default.
# The per-row `thinking_mode` field reports what THAT response showed:
#   "reasoning_content" | "content_prefix" | "not_observed"
# "not_observed" = neither a trace header in content nor any reasoning_content
# (every thinking=off row lands here; it used to carry the table fallback and
# read as if the routing had been observed).
THINKING_MODES: dict[str, tuple[str, str]] = {
    "Qwen3.8-27B-oQ4e-mtp": ("reasoning_content", ""),
    "Qwen3.8-27B-oQ6e-mtp": ("reasoning_content", ""),
    "Ornith-1.0-35B-oQ4e": ("content_prefix", "Here's a thinking process"),
    "Ornith-1.0-35B-oQ8e": ("content_prefix", "Here's a thinking process"),
    "Ornith-1.5-35B-A3B-MLX-oQ4e": ("reasoning_content", ""),
    "Ornith-1.5-35B-A3B-MLX-oQ8e": ("reasoning_content", ""),
    "Ornith-1.5-35B-A3B-oQ4e-mtp": ("reasoning_content", ""),
    "Ornith-1.5-35B-A3B-oQ8e-mtp": ("reasoning_content", ""),
    "Laguna-XS-2.1-oQ4e": ("reasoning_content", ""),
    "Laguna-XS-2.1-oQ8e": ("reasoning_content", ""),
    "Laguna-S-2.1-oQ3e": ("reasoning_content", ""),
    "Laguna-S-2.1-oQ4e": ("reasoning_content", ""),
}
# Default for models not in the table; auto-detection still applies.
DEFAULT_THINKING_MODE = ("reasoning_content", "")

# Header patterns that mark a thinking trace embedded in `content`
# (content_prefix placement). Checked first; if found, thinking = the whole
# content (unsplitable) and mode is reported as content_prefix for that row.
TRACE_HEADERS = [
    "Here's a thinking process",
    "Here is a thinking process",
    "Let's think step by step",
    "Thinking process",
    "Step-by-step thinking",
]

# effort values -> request body. "off" never sends reasoning_effort on the
# thinking=off row (enable_thinking drives that); on thinking=on rows "off"
# sends reasoning_effort="off" (template alias -> low in some models; useful
# to observe what "off" actually does per model).
EFFORT_OFF_BODY = "off"

# Canonical effort ordering for summary tables (intensity order, matching
# plot_summary_charts.py; unknown effort values sort after, alphabetically).
EFFORT_ORDER = ["off", "low", "medium", "high", "xhigh"]


def _effort_sort_key(effort: str) -> tuple[int, str]:
    """Sort key placing efforts in intensity order (off, low, medium, high,
    xhigh), unknown values after them alphabetically."""
    return (
        EFFORT_ORDER.index(effort) if effort in EFFORT_ORDER else len(EFFORT_ORDER),
        effort,
    )

BASE_URL: str = DEFAULT_BASE_URL


def unique_values(values: list[str]) -> list[str]:
    """Return values once each while preserving their command-line order."""
    return list(dict.fromkeys(values))


def _budget_token(value: str) -> int | None:
    """Parse one --thinking-budgets item: 'none'/'-' = no budget, else int."""
    if value.lower() in ("none", "-"):
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid budget {value!r}: use an integer >= 0 or 'none'"
        )


def is_finite_number(value: object) -> bool:
    """Return whether value is a finite int or float, excluding bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def http_json(
    path: str,
    body: dict | None = None,
    timeout: float = 60.0,
    method: str | None = None,
) -> tuple[int, object]:
    """Send an HTTP request and return its status code and JSON-like body.

    Network exceptions are converted to status ``0`` and an error payload;
    HTTP errors retain their status code and are parsed when possible.
    """
    url = BASE_URL + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method or ("POST" if body is not None else "GET"),
        headers={"Content-Type": "application/json"},
    )

    def _read(response: Any) -> tuple[int, object]:
        """Read one HTTP response and decode JSON when available."""
        status = getattr(response, "status", None) or response.getcode()
        raw = response.read().decode()
        try:
            return status, json.loads(raw)
        except json.JSONDecodeError:
            return status, {"raw": raw[:2000]}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _read(resp)
    except urllib.error.HTTPError as e:
        try:
            return _read(e)
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:  # noqa: BLE001
        return 0, {"error": f"{type(e).__name__}: {e}"}


@dataclass
class LoadLogger:
    path: Path

    def log(
        self, model: str, action: str, status: int, payload: object, wall_s: float
    ) -> None:
        """Append one load-related HTTP event with its measured duration."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(
                f"{datetime.now().isoformat(timespec='milliseconds')}\t{model}\t{action}\t{status}\t{wall_s:.3f}\t{json.dumps(payload, ensure_ascii=False)[:200]}\n"
            )


# Error strings (lowercased) that are transient: a load call may fail while
# unload/reload traffic is in flight; a retry after the server settles
# succeeds. Non-transient errors (bad path, 4xx, ...) are not retried.
TRANSIENT_ERR_MARKERS = (
    "timeout",
    "timed out",
    "connection refused",
    "connection reset",
    "broken pipe",
    "temporarily unavailable",
)


def _is_transient_error(status: int, body: object) -> bool:
    """Return whether a client-side failure is safe to retry once."""
    if status != 0:
        return False
    err = str(body if isinstance(body, dict) else body)
    low = err.lower()
    return "timeout" in low or any(m in low for m in TRANSIENT_ERR_MARKERS)


def load_model(
    m: str, logger: LoadLogger, timeout: float = 600.0, retry_wait: int = 8
) -> float:
    """Load a model; retry once on a transient client-side failure (observed:
    oMLX load request is dropped when concurrent unload/reload traffic is in
    flight; a retry after the server settles succeeds)."""
    for attempt in (1, 2):
        t0 = time.perf_counter()
        status, body = http_json(
            f"/v1/models/{urllib.parse.quote(m, safe='')}/load", {}, timeout=timeout
        )
        wall = time.perf_counter() - t0
        logger.log(m, f"load#{attempt}", status, body, wall)
        if status == 200:
            return wall
        if not _is_transient_error(status, body) or attempt == 2:
            raise RuntimeError(f"load failed: status={status} body={body}")
        print(
            f"  ! load transient client failure after {wall:.0f}s ({body}); "
            f"retrying in {retry_wait}s",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(retry_wait)
    raise RuntimeError("unreachable")


def unload_model(m: str, logger: LoadLogger) -> None:
    """Unload one physical model ID.

    A 200 response means the model was unloaded and a 400 response means it
    was already unloaded. Any other status raises because isolation would be
    broken and the caller must fail the condition instead of measuring.
    """
    t0 = time.perf_counter()
    status, body = http_json(
        f"/v1/models/{urllib.parse.quote(m, safe='')}/unload", {}, timeout=120.0
    )
    wall = time.perf_counter() - t0
    logger.log(m, "unload", status, body, wall)
    if status not in (200, 400):
        raise RuntimeError(f"unload {m} failed: status={status} body={body}")


def loaded_models(timeout: float = 30.0) -> list[str]:
    """Return physical model IDs currently loaded or loading in oMLX.

    The status endpoint retains physical IDs and includes hidden/helper models,
    unlike the public model list, which may expose aliases or omit entries.
    """
    status, body = http_json("/v1/models/status", timeout=timeout)
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"/v1/models/status failed: {body}")
    models = body.get("models")
    if not isinstance(models, list) or not all(
        isinstance(model, dict) and isinstance(model.get("id"), str)
        for model in models
    ):
        raise RuntimeError(
            f"/v1/models/status bad schema (models not a list of {{id: str}}): {body}"
        )
    return [
        model["id"]
        for model in models
        if model.get("source_model_id") is None
        and (model.get("loaded") is True or model.get("is_loading") is True)
    ]


def wait_until_available(timeout_total: float = 60.0) -> bool:
    """Poll /v1/models until the oMLX server accepts requests again (needed
    after /admin/api/reload, which 500s briefly during re-discovery).
    Returns True once a full round-trip succeeds, False on timeout (the
    caller must fail the condition)."""
    deadline = time.time() + timeout_total
    while time.time() < deadline:
        try:
            status, _ = http_json("/v1/models", timeout=8)
            if status == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    print(
        f"  ! server still busy after {timeout_total:.0f}s; failing condition",
        file=sys.stderr,
        flush=True,
    )
    return False


def request_body(
    model: str,
    prompt: str,
    thinking: str,
    effort: str,
    max_tokens: int,
    thinking_budget: int | None = None,
) -> dict:
    """Build the OpenAI-compatible request body for one condition."""
    body: dict = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        # enable_thinking must be a JSON boolean: Qwen3.8 / 1.0-family
        # chat templates use Jinja identity tests (`is true` / `is false`),
        # which ignore the strings "True"/"False" (they fall into the
        # default-ON branch).
        "chat_template_kwargs": {},
    }
    if thinking == "off":
        body["chat_template_kwargs"]["enable_thinking"] = False
    else:
        body["chat_template_kwargs"]["enable_thinking"] = True
        if effort == "off":
            # thinking stays ON; "off" effort is an observable alias case
            # (see NOTE above: Qwen template rejects it and oMLX maps
            # "off" -> low via the alias table; others ignore the field).
            body["reasoning_effort"] = EFFORT_OFF_BODY
        else:
            body["reasoning_effort"] = effort
        if thinking_budget is not None:
            # Top-level oMLX field (integer >= 0): the engine caps thinking
            # by watching the generated <think> block, so it works on every
            # template. Only sent while thinking is ON (an OFF row writes an
            # empty think block via enable_thinking=false).
            body["thinking_budget"] = thinking_budget
    return body


@dataclass
class _StreamState:
    """Mutable accumulator + protocol validator for one SSE chat stream.

    Feed decoded SSE lines via feed_line(); [DONE], finish_reason, token
    timing and usage are accumulated, and the first malformed or unknown
    chunk sets protocol_error and stops further event processing. The
    request is complete iff the stream ended with [DONE], carried a
    non-empty finish_reason, had no malformed or unknown chunks, contained
    at least one non-empty reasoning/content value, and provided a valid
    completion_tokens value; otherwise protocol_error is set (the row must
    be treated as incomplete, not ok).
    """

    t0: float = field(default_factory=time.perf_counter)
    ttft: float | None = None
    first_any: float | None = None
    first_reasoning: float | None = None
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    protocol_error: str | None = None
    done_seen: bool = False
    reasoning_parts: list[str] = field(default_factory=list)
    content_parts: list[str] = field(default_factory=list)
    _data_buf: list[str] = field(default_factory=list)

    def _now(self) -> float:
        """Return elapsed monotonic time from the request start."""
        return time.perf_counter() - self.t0

    def feed_line(self, line: str) -> bool:
        """Consume one decoded/stripped SSE line. Returns True once the
        [DONE] event is processed and reading may stop early."""
        if line.startswith("data:"):
            self._data_buf.append(line[len("data:") :].strip())
            return False
        if line:
            # comment (":...") / event: / id: / retry: fields - ignore
            return False
        return self._flush_event()

    def flush_pending(self) -> None:
        """Process a trailing buffered event (stream ended without a final
        blank line)."""
        if any(self._data_buf):
            self._flush_event()

    def _flush_event(self) -> bool:
        """Join+process the buffered SSE event; report a terminal event."""
        if not self._data_buf:
            return False
        payload = "\n".join(self._data_buf)
        self._data_buf.clear()
        if self.protocol_error is not None:
            return False
        return self._process_event(payload)

    def _process_event(self, payload: str) -> bool:
        """Parse+validate one SSE data payload. Returns True on [DONE]."""
        if payload == "[DONE]":
            self.done_seen = True
            return True
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError as e:
            self.protocol_error = f"bad chunk: {e}"
            return False
        if not isinstance(chunk, dict):
            self.protocol_error = f"bad chunk (not a JSON object): {str(chunk)[:80]}"
            return False
        choices = chunk.get("choices")
        chunk_usage = chunk.get("usage")
        # An event must carry choices (possibly an empty usage-only list) or
        # a usage object; anything else is an unknown shape -> protocol error
        # (see class docstring: ok requires NO bad/unknown chunks).
        if choices is None and chunk_usage is None:
            self.protocol_error = "bad chunk (no choices and no usage)"
            return False
        if choices is not None:
            err = self._consume_choices(choices, chunk_usage is not None)
            if err is not None:
                self.protocol_error = err
                return False
        if chunk_usage is not None:
            err = self._consume_usage(chunk_usage)
            if err is not None:
                self.protocol_error = err
                return False
        return False

    def _consume_choices(self, choices: object, has_usage: bool) -> str | None:
        """Validate+consume the choices list; returns a protocol error or None."""
        if not isinstance(choices, list) or not all(
            isinstance(c, dict) for c in choices
        ):
            return "bad chunk (choices is not a list of objects)"
        if not choices and not has_usage:
            return "bad chunk (empty choices without usage)"
        for choice in choices:
            err = self._consume_choice(choice)
            if err is not None:
                return err
        return None

    def _consume_choice(self, choice: dict) -> str | None:
        """Validate+consume one choice (finish_reason + delta tokens)."""
        if "delta" not in choice and "finish_reason" not in choice:
            return "bad chunk (choice has no delta or finish_reason)"
        fr = choice.get("finish_reason")
        if fr is not None:
            if not isinstance(fr, str):
                return "bad chunk (finish_reason not a string)"
            if fr:
                self.finish_reason = fr
        delta = choice.get("delta")
        if delta is None:
            return None
        if not isinstance(delta, dict):
            return "bad chunk (delta is not an object)"
        rc = delta.get("reasoning_content")
        if rc is not None and not isinstance(rc, str):
            return "bad chunk (reasoning_content not a string)"
        c = delta.get("content")
        if c is not None and not isinstance(c, str):
            return "bad chunk (content not a string)"
        now = self._now() if rc or c else None
        if rc:
            if self.first_reasoning is None:
                self.first_reasoning = now
            if self.first_any is None:
                self.first_any = now
            self.reasoning_parts.append(rc)
        if c:
            if self.ttft is None:
                self.ttft = now
            if self.first_any is None:
                self.first_any = now
            self.content_parts.append(c)
        return None

    def _consume_usage(self, chunk_usage: object) -> str | None:
        """Validate+merge one usage object; returns a protocol error or None."""
        if not isinstance(chunk_usage, dict):
            return "bad chunk (usage is not an object)"
        self.usage.update(chunk_usage)
        return None

    def finalize(self) -> dict:
        """Apply end-of-stream completeness checks and build the metrics dict."""
        protocol_error = self.protocol_error
        if not self.done_seen:
            protocol_error = protocol_error or "stream ended without [DONE]"
        if self.finish_reason is None:
            protocol_error = protocol_error or "no finish_reason in stream"
        if self.first_any is None and self.first_reasoning is None:
            protocol_error = protocol_error or "no non-empty tokens in stream"
        ct_raw = self.usage.get("completion_tokens")
        if (
            not isinstance(ct_raw, (int, float))
            or isinstance(ct_raw, bool)
            or not is_finite_number(ct_raw)
            or ct_raw < 0
        ):
            protocol_error = protocol_error or (
                f"bad usage (completion_tokens not a finite number >= 0): {ct_raw!r}"
            )
        return {
            "ttft_s": self.ttft,
            "first_any_token_s": self.first_any,
            "first_reasoning_token_s": self.first_reasoning,
            "total_s": self._now(),
            "reasoning_content": "".join(self.reasoning_parts),
            "content": "".join(self.content_parts),
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "content_tokens": ct_raw,
            "protocol_error": protocol_error,
        }


def stream_chat(model: str, body: dict, timeout: float) -> dict:
    """Send one streaming chat request; returns parsed metrics. Raises on
    HTTP error.

    The result dict is built by _StreamState.finalize(); see there for the
    SSE completeness contract (ok vs incomplete rows)."""
    url = BASE_URL + "/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    state = _StreamState()
    # SSE spec: an event's data is the concatenation of its `data:` lines
    # (newline-joined), terminated by a blank line. feed_line buffers per
    # event so a JSON payload split over several `data:` lines still parses.
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if state.feed_line(raw.decode(errors="replace").strip()):
                break
    state.flush_pending()
    return state.finalize()


def run_stream_request(
    model: str,
    prompt: str,
    thinking: str,
    effort: str,
    max_tokens: int,
    timeout: float = 600.0,
    seed: int | None = None,
    thinking_budget: int | None = None,
) -> dict:
    """Run one measured streaming request with usage collection enabled."""
    body = request_body(
        model, prompt, thinking, effort, max_tokens, thinking_budget
    )
    if seed is not None:
        body["seed"] = seed
    body["stream_options"] = {"include_usage": True}
    return stream_chat(model, body, timeout)


def warmup(
    model: str,
    prompt: str,
    thinking: str,
    effort: str,
    timeout: float = 300.0,
    seed: int | None = None,
    thinking_budget: int | None = None,
) -> dict:
    """One throwaway request with the SAME request body as the measurement
    (prompt, thinking, effort, thinking budget, seed) but a small token
    budget; it absorbs the hidden in-server load cost that would otherwise
    pollute TTFT (observed 36s). Raises on failure - HTTP errors AND an
    incomplete SSE stream (protocol_error set) both count as failure; the
    caller must then fail the condition and send no measurement request."""
    body = request_body(
        model, prompt, thinking, effort, WARMUP_MAX_TOKENS, thinking_budget
    )
    if seed is not None:
        body["seed"] = seed
    body["stream_options"] = {"include_usage": True}
    res = stream_chat(model, body, timeout=timeout)
    if res["protocol_error"] is not None:
        raise RuntimeError(f"warmup stream incomplete: {res['protocol_error']}")
    # A tokenless "ok" stream (usage-only / empty deltas) is NOT a valid
    # warmup: the in-server load cost it is meant to absorb may have been
    # zero, or the response is broken. Fail the condition either way.
    if res["first_any_token_s"] is None:
        raise RuntimeError("warmup returned no non-empty tokens")
    return res


def summarize_thinking(
    result: dict, mode: str, header: str, thinking_off: bool
) -> tuple[int, int, int, int, int, int, str]:
    """Return (reasoning_chars, content_chars, thinking_chars, trace_flag,
    leak_rc, leak_content, mode).

    thinking_chars  = len(reasoning_content) when no trace header is inside
                      content (normal routing); = len(content) when a trace
                      header is inside content (content_prefix placement,
                      whole content unsplitable).
    trace_flag      = 1 when thinking is OFF and thinking still appeared
                      (a trace header in content AND/OR reasoning_content).
    leak_rc         = 1 when thinking is OFF and reasoning_content is present.
    leak_content    = 1 when thinking is OFF and a trace header is in content.
    mode            = the per-response routing actually observed
                      (content_prefix | reasoning_content), or "not_observed"
                      when the response carried neither a trace header nor
                      reasoning_content (all thinking=off rows).
    """
    reasoning = result.get("reasoning_content")
    reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
    content = result["content"]
    content_chars = len(content)
    has_rc = reasoning_chars > 0
    has_header = any(h in content for h in TRACE_HEADERS) or (
        header and header in content
    )
    if has_header:
        # 1.0-family trace placed inside content (whole content unsplitable:
        # content holds trace + answer and cannot be split reliably), so the
        # ENTIRE content is counted as thinking (header-first rule). In the
        # rare mixed case (header in content AND reasoning_content present)
        # reasoning_content is an additional copy of the thinking and is not
        # added on top.
        mode = "content_prefix"
        thinking_chars = content_chars
    elif has_rc:
        # thinking delivered in reasoning_content. NOTE: for 1.0-family
        # some responses put BOTH reasoning_content AND a short answer in
        # content (mixed case) - we count
        # reasoning_content here; the raw fields are preserved for audit.
        # Auto-detected routing: observed here even if the model's fallback
        # mode in THINKING_MODES is "content_prefix".
        mode = "reasoning_content"
        thinking_chars = reasoning_chars
    else:
        # Nothing observable: no trace header inside content AND no
        # reasoning_content. `mode` here is only the model's CONFIGURED
        # fallback, so returning it unchanged would report a routing this
        # response never showed - every thinking=off row used to carry its
        # model's fallback label as if it had been observed.
        mode = "not_observed"
        thinking_chars = 0
    leak_rc = 1 if (thinking_off and has_rc) else 0
    leak_content = 1 if (thinking_off and has_header) else 0
    return (
        reasoning_chars,
        content_chars,
        thinking_chars,
        1 if (leak_rc or leak_content) else 0,
        leak_rc,
        leak_content,
        mode,
    )


@dataclass
class Condition:
    model: str
    thinking: str
    effort: str
    budget: int | None = None


def condition_matrix(args: argparse.Namespace) -> list[Condition]:
    """Expand CLI selections into unique benchmark conditions.

    thinking=on rows cross the selected efforts with the selected thinking
    budgets (empty budgets -> one no-budget pass). thinking=off rows never
    carry a budget, and the auto-added effort=off alias-probe row stays
    budget-free on purpose (it observes the raw alias behavior)."""
    budgets: list[int | None] = list(args.thinking_budgets or []) or [None]
    out = []
    for m in args.models:
        for t in args.thinking:
            if t == "off":
                out.append(Condition(m, t, "off"))
                if "on" not in args.thinking:
                    out.append(Condition(m, "on", "off"))
            else:
                for e in args.efforts:
                    for b in budgets:
                        out.append(Condition(m, t, e, b))
    return out


ROW_REQUIRED_FIELDS = ("model", "thinking", "effort", "prompt_id", "run", "error")

# Numeric metric fields; each may be null but must otherwise be a finite
# number (bools rejected).
ROW_NUMERIC_FIELDS = (
    "thinking_budget",
    "load_s",
    "first_any_token_s",
    "first_reasoning_token_s",
    "ttft_s",
    "total_s",
    "reasoning_chars",
    "content_chars",
    "thinking_chars",
    "content_tokens",
)


def _row_numeric_error(rec: dict) -> str | None:
    """Return the first invalid numeric metric field in the row, or None."""
    for key in ROW_NUMERIC_FIELDS:
        value = rec.get(key)
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not is_finite_number(value)
        ):
            return f"{key} must be a finite number or null (got {value!r})"
    return None


def _row_optional_error(rec: dict) -> str | None:
    """Return the first invalid optional-field type in the row, or None."""
    usage = rec.get("usage")
    if usage is not None and not isinstance(usage, dict):
        return "usage must be an object or null"
    completion_tokens = (
        usage.get("completion_tokens") if isinstance(usage, dict) else None
    )
    if completion_tokens is not None and (
        not isinstance(completion_tokens, (int, float))
        or isinstance(completion_tokens, bool)
        or not is_finite_number(completion_tokens)
    ):
        return "usage.completion_tokens must be a finite number or null"
    for key in ("content_raw", "reasoning_raw"):
        if rec.get(key) is not None and not isinstance(rec[key], str):
            return f"{key} must be a string or null"
    return None


def _row_field_error(rec: dict) -> str | None:
    """Return the first invalid identity/status field type in the row, or None."""
    if not isinstance(rec["model"], str) or not rec["model"]:
        return "model must be a non-empty string"
    if not isinstance(rec["thinking"], str):
        return "thinking must be a string"
    if not isinstance(rec["effort"], str):
        return "effort must be a string"
    if not isinstance(rec["prompt_id"], str) or not rec["prompt_id"]:
        return "prompt_id must be a non-empty string"
    if (
        not isinstance(rec["run"], int)
        or isinstance(rec["run"], bool)
        or rec["run"] < 1
    ):
        return "run must be an integer >= 1"
    if (rec["status"] if "status" in rec else "ok") not in (
        "ok",
        "incomplete",
        "error",
    ):
        return "status must be one of ok/incomplete/error"
    if rec["error"] is not None and not isinstance(rec["error"], str):
        return "error must be a string or null"
    for key in ("finish_reason", "load_event_id"):
        value = rec.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            return f"{key} must be a non-empty string or null"
    return None


def _warn_row_skip(msg: str) -> None:
    """Print one skip warning for a malformed JSONL row."""
    print(
        f"  ! {msg}",
        file=sys.stderr,
        flush=True,
    )


def iter_row_records(results_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (line_number, dict) for structurally valid JSONL rows.

    The status field is optional for legacy rows. All other fields needed to
    identify an attempted row remain required, and malformed records are
    skipped with a warning so resume and report generation cannot crash.
    """
    if not results_path.exists():
        return
    with results_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                _warn_row_skip(
                    f"skipping corrupted JSONL line {i} "
                    f"({results_path.name[:40]}: bad JSON)"
                )
                continue
            if not isinstance(rec, dict):
                _warn_row_skip(f"skipping non-object JSONL line {i}")
                continue
            missing = [k for k in ROW_REQUIRED_FIELDS if k not in rec]
            if missing:
                _warn_row_skip(f"skipping JSONL line {i} missing keys {missing}")
                continue
            field_error = (
                _row_numeric_error(rec)
                or _row_optional_error(rec)
                or _row_field_error(rec)
            )
            if field_error is not None:
                _warn_row_skip(
                    f"skipping JSONL line {i} with invalid field types: {field_error}"
                )
                continue
            yield i, rec


def is_row_done(rec: dict) -> bool:
    """A row counts as done for --resume only if it is a complete, error-free
    measurement. (Legacy rows have no status field: ok iff error is null and
    finish_reason is present.)"""
    if not isinstance(rec, dict):
        return False
    if rec.get("error") is not None:
        return False
    if rec.get("status", "ok") != "ok":
        return False
    return isinstance(rec.get("finish_reason"), str) and bool(rec["finish_reason"])


def _is_thinking_echo(reasoning: str, content: str) -> bool:
    """True when ``content`` is an unsplit copy of the thinking stream.

    An engine that flushes a still-open think block at end of stream puts the
    whole thinking text into ``delta.content`` as well as into
    ``delta.reasoning_content``, so the row carries NO answer even though
    ``content_chars`` is large. Exact string equality is the conservative
    test; a partial echo would be a different failure mode."""
    return bool(content) and bool(reasoning) and content == reasoning


def _trace_in_content(model: str, content: str) -> bool:
    """True when ``content`` opens with a thinking trace (1.0-family routing).

    Same detection rule as summarize_thinking: the shared TRACE_HEADERS list
    plus this model's configured header. Derived from the raw text rather than
    from the stored ``thinking_mode``, because legacy rows carry the configured
    FALLBACK label on rows where nothing was ever observed (every
    thinking=off row), which would misclassify a perfectly good answer."""
    if not content:
        return False
    header = THINKING_MODES.get(model, DEFAULT_THINKING_MODE)[1]
    return any(h in content for h in TRACE_HEADERS) or bool(header) and header in content


def annotate_row(rec: dict, max_tokens: object = None) -> dict:
    """Add the derived answer-delivery fields the report tables need (in place).

    ``status: ok`` measures SSE protocol completeness only, so both a
    generation cut off by ``max_tokens`` and a row whose entire content is an
    echoed think block count as ok. These fields carry delivery information:

      thinking_echo   1 when content_raw is a verbatim copy of reasoning_raw
      answer_unknown  1 when the thinking arrived as an unsplitable prefix of
                      content (1.0-family routing): content holds trace +
                      answer with no reliable boundary, so the answer length is
                      UNKNOWN (answer_chars = null) rather than overstated
      answer_chars    len(content) excluding an echoed thinking block; null
                      when the answer cannot be separated
      truncated       1 when finish_reason is "length" OR completion_tokens
                      reached max_tokens (the server's finish_reason is not
                      trusted here: engines have been observed reporting "stop"
                      on rows that stopped exactly at the cap)
      no_answer       1 when answer_chars == 0

    Values are always derived rather than read back from the row, so
    --resummarize of an older JSONL reproduces what a fresh run would print."""
    reasoning = rec.get("reasoning_raw")
    content = rec.get("content_raw")
    reasoning = reasoning if isinstance(reasoning, str) else ""
    content = content if isinstance(content, str) else ""
    echo = _is_thinking_echo(reasoning, content)
    # Same detection rule as summarize_thinking: a trace header inside content
    # means content is trace + answer with no reliable boundary, whether or not
    # reasoning_content also holds a copy. Echo is checked first because that
    # case is a KNOWN no-answer, which is stronger than "unknown".
    unknown = bool(content) and _trace_in_content(rec.get("model", ""), content) and not echo
    completion = rec.get("content_tokens")
    truncatable = isinstance(max_tokens, (int, float)) and not isinstance(
        max_tokens, bool
    )
    counted = isinstance(completion, (int, float)) and not isinstance(completion, bool)
    truncated = rec.get("finish_reason") == "length" or bool(
        truncatable and counted and completion >= max_tokens
    )
    rec["thinking_echo"] = 1 if echo else 0
    rec["answer_unknown"] = 1 if unknown else 0
    rec["answer_chars"] = None if unknown else (0 if echo else len(content))
    rec["truncated"] = 1 if truncated else 0
    rec["no_answer"] = 1 if rec["answer_chars"] == 0 else 0
    return rec


def annotate_rows(rows: list[dict], max_tokens: object = None) -> list[dict]:
    """annotate_row over a list of row records."""
    return [annotate_row(r, max_tokens) for r in rows]


def result_files(data_dir: Path) -> list[Path]:
    """Return result JSONL files ordered by experiment timestamp.

    New runs are stored below ``data/<timestamp>/``. Direct ``data/rows_*.jsonl``
    files are retained here for backward-compatible resume support. Only
    timestamp-named directories are searched, so chart and other helper
    directories are not mistaken for benchmark runs.
    """
    paths = list(data_dir.glob("rows_*.jsonl"))
    for directory in data_dir.iterdir():
        if not directory.is_dir():
            continue
        try:
            datetime.strptime(directory.name, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        paths.extend(directory.glob("rows_*.jsonl"))

    def experiment_stamp(path: Path) -> str:
        if path.parent == data_dir:
            return path.stem.removeprefix("rows_")
        return path.parent.name

    return sorted(paths, key=lambda path: (experiment_stamp(path), str(path)))


def load_completed(results_path: Path) -> set[str]:
    """Set of condition+prompt+run keys that have a done row in the JSONL."""
    done: set[str] = set()
    if not results_path.exists():
        return done
    for _, rec in iter_row_records(results_path):
        if is_row_done(rec):
            done.add(
                f"{rec['model']}|{rec['thinking']}|{rec['effort']}"
                f"|{rec.get('thinking_budget')}|{rec['prompt_id']}|{rec['run']}"
            )
    return done


def load_events_from_jsonl(results_path: Path) -> dict[tuple, dict[str, float]]:
    """Read load events from result rows."""
    events: dict[tuple, dict[str, float]] = {}
    if not results_path.exists():
        return events
    for _, rec in iter_row_records(results_path):
        v = rec.get("load_s")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        condition = (
            rec["model"],
            rec["thinking"],
            rec["effort"],
            rec.get("thinking_budget"),
        )
        event_id = rec.get("load_event_id")
        if not isinstance(event_id, str) or not event_id:
            event_id = f"legacy-load-s:{float(v):.12g}"
        events.setdefault(condition, {})[event_id] = float(v)
    return events


# Upper edge of the "budget was honored" band, as a multiple of the budget.
# est tokens carry +-5-10% error (wall time x generation rate), so the band is
# deliberately wide; anything past it overshoots the budget and is treated as
# not enforced rather than as bound.
BUDGET_HONOR_UPPER = 1.3


def _est_think_tokens(rec: dict) -> float | None:
    """Estimated natural thinking tokens for one row (think wall time x rate)."""
    ttft = rec.get("ttft_s")
    first_reasoning = rec.get("first_reasoning_token_s")
    rate = (rec.get("usage") or {}).get("generation_tokens_per_second")
    if ttft is None or first_reasoning is None or rate is None:
        return None
    if not is_finite_number(rate) or rate <= 0:
        return None
    return max(ttft - first_reasoning, 0.0) * rate


def _survival_quantile(points: list[tuple[float, float]], target: float):
    """Interpolate the budget b where S(b) = target on sorted (b, S) points.

    S decreases in b (higher budget binds less). Returns ("value", b) when
    the crossing lies inside the measured range, ("below", edge) when target
    exceeds every measured S (crossing at smaller b), or ("above", edge)
    when target is below every measured S (crossing at larger b)."""
    pts = sorted(points)
    s_max = max(s for _, s in pts)
    for (b1, s1), (b2, s2) in zip(pts, pts[1:]):
        if (s1 - target) * (s2 - target) <= 0 and s1 != s2:
            b = b1 + (s1 - target) / (s1 - s2) * (b2 - b1)
            return "value", b
    if target > s_max:
        return "below", pts[0][0]
    return "above", pts[-1][0]


def _group_budget_rows(rows: list[dict]) -> dict[tuple, dict]:
    """Group ok thinking=on rows by (model, effort) for budget analysis.

    Each group holds `none` rows (uncensored natural thinking samples, with a
    count of how many of them hit max_tokens) and, per budget > 0, the row
    count plus est-token samples classified against the budget:

      honored    0.95*b <= est <= BUDGET_HONOR_UPPER*b  (cut AT the budget)
      overshoot  est > BUDGET_HONOR_UPPER*b  (budget ignored: thinking ran
                 past it, so the engine did not enforce it)
      under      est < 0.95*b  (natural thinking was shorter than the budget)

    A one-sided "est >= 0.95*b" test is NOT a binding test: a model that
    ignores the budget entirely overshoots it and would score as bound.
    Rows without a `thinking_budget` key (legacy) and budget-0 rows
    (natural length unobservable) are excluded."""
    groups: dict[tuple, dict] = {}
    for rec in rows:
        if not is_row_done(rec) or rec.get("thinking") != "on":
            continue
        if "thinking_budget" not in rec:
            continue  # legacy row: budget regime unknown
        budget = rec["thinking_budget"]
        est = _est_think_tokens(rec)
        if est is None:
            continue
        key = (rec["model"], rec["effort"])
        g = groups.setdefault(key, {"none": [], "none_censored": 0, "budgets": {}})
        if budget is None:
            g["none"].append(est)
            if rec.get("truncated"):
                g["none_censored"] += 1
        elif budget > 0:
            b = g["budgets"].setdefault(
                budget, {"n": 0, "est": [], "honored": 0, "overshoot": 0, "censored": 0}
            )
            b["n"] += 1
            b["est"].append(est)
            if est > BUDGET_HONOR_UPPER * budget:
                b["overshoot"] += 1
            elif est >= 0.95 * budget:
                b["honored"] += 1
            if rec.get("truncated"):
                b["censored"] += 1
    return groups


def _budget_under(cell: dict) -> int:
    """Rows in one budget cell whose thinking stopped short of the budget
    (natural length below b) - the only rows that prove the budget did NOT
    bind."""
    return cell["n"] - cell["honored"] - cell["overshoot"]


def _natural_quantile_lines(rows: list[dict]) -> list[str]:
    """Build Table 3: natural thinking length quantiles per (model, effort).

    Q10/Q50/Q90 are the budget values expected to bind ~90%/~50%/~10% of
    rows (the budget is an upper bound, so binding happens when the natural
    thinking length exceeds it). Two estimators:
    - direct: percentiles over `none`-condition rows when >= 3 exist
      (uncensored natural samples);
    - survival: the binding rate at each budget estimates S(b) =
      P(natural >= b); linear interpolation (anchored at the smallest
      non-bound estimate, S ~ 1.0) yields the quantiles, with out-of-range
      values reported as <= min / >= max budget.
    Skipped when neither is possible. The `n` column is the sample count the
    Q values come from (none rows for direct, budgeted rows for survival), not
    every row in the group. When the budgets are not enforced (median est far
    above the smallest budget and no scaling across budgets, or no row cut at
    any budget - e.g. engines that never bind thinking_budget) the group carries a "budget
    INEFFECTIVE" note on either branch, and a direct-branch note also flags
    how many none rows were truncated at max_tokens (their est is then a lower
    bound)."""
    groups = _group_budget_rows(rows)
    if not groups:
        return []

    def pct(sorted_vals: list[float], p: float) -> float:
        return sorted_vals[round(p * (len(sorted_vals) - 1))]

    lines = [
        "",
        "Table 3: natural thinking length quantiles (est tokens) per model x effort. "
        "Q10/Q50/Q90 = budget values expected to bind ~90%/~50%/~10% of rows "
        "(<= min / >= max = outside the measured budget range). n = samples the "
        "Q values are computed from.",
        "",
        "| model | effort | method | n | Q10 | Q50 | Q90 | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (model, effort), g in sorted(
        groups.items(), key=lambda kv: (kv[0][0], _effort_sort_key(kv[0][1]))
    ):
        budgets = sorted(g["budgets"])
        method, note = "-", ""
        q: list = ["-", "-", "-"]
        n_used = len(g["none"]) + sum(b["n"] for b in g["budgets"].values())
        # Budget-effectiveness guard, evaluated on every branch: a previous
        # version had it only on the survival path, so a group that happened
        # to have >= 3 none rows was never checked even when every budget
        # overshot it (exactly how an ignored budget goes undetected).
        ineffective = ""
        if budgets:
            total_n = sum(g["budgets"][b]["n"] for b in budgets)
            total_honored = sum(g["budgets"][b]["honored"] for b in budgets)
            b_min, b_max = budgets[0], budgets[-1]
            med_min = statistics.median(g["budgets"][b_min]["est"])
            med_max = statistics.median(g["budgets"][b_max]["est"])
            if total_n and total_honored == 0:
                ineffective = (
                    f"budget INEFFECTIVE (ignored by engine?): 0/{total_n} rows cut at "
                    f"any budget; median est {med_min:,.0f} tok at the smallest budget "
                    f"{b_min} = {med_min / b_min:.1f}x"
                )
            elif med_min > max(2 * b_min, 100) and med_max < 1.5 * med_min:
                ineffective = (
                    f"budget INEFFECTIVE (ignored by engine?): median est "
                    f"{med_min:,.0f} tok at budget {b_min} = {med_min / b_min:.1f}x, "
                    f"no scaling to {b_max}"
                )
        if len(g["none"]) >= 3:
            method = "direct (none rows)"
            n_used = len(g["none"])  # the samples the Q values actually use
            vals = sorted(g["none"])
            q = [pct(vals, 0.1), pct(vals, 0.5), pct(vals, 0.9)]
            if ineffective:
                note = ineffective
            if g["none_censored"]:
                note = (
                    f"{note} ; " if note else ""
                ) + f"{g['none_censored']}/{n_used} none rows truncated at max_tokens -> values are LOWER BOUNDS"
        elif len(budgets) >= 2:
            n_used = sum(g["budgets"][b]["n"] for b in budgets)
            if ineffective:
                note = ineffective + " (survival curve not computed)"
            else:
                method = "survival (budget binding)"
                points = [
                    (
                        b,
                        (g["budgets"][b]["n"] - _budget_under(g["budgets"][b]))
                        / g["budgets"][b]["n"],
                    )
                    for b in budgets
                ]
                non_bound = sorted(
                    est
                    for b in budgets
                    for est in g["budgets"][b]["est"]
                    if est < 0.95 * b
                )
                if non_bound:
                    points.append((non_bound[0], 1.0))
                q = []
                for target in (0.9, 0.5, 0.1):
                    kind, val = _survival_quantile(points, target)
                    q.append(val if kind == "value" else (kind, val))
        if method == "-":
            skip_note = note or "not enough budget points / none rows"
            lines.append(
                f"| {model} | {effort} | - | {n_used} | - | - | - | {skip_note} |"
            )
            continue
        cells = []
        for item in q:
            if isinstance(item, tuple):
                kind, edge = item
                cells.append(
                    f"<= {edge:,.0f}" if kind == "below" else f">= {edge:,.0f}"
                )
            else:
                cells.append(f"~{item:,.0f}")
        lines.append(
            f"| {model} | {effort} | {method} | {n_used} | {cells[0]} | {cells[1]} | {cells[2]} | {note} |"
        )
    unestimable = sum(
        1
        for r in rows
        if is_row_done(r)
        and r.get("thinking") == "on"
        and "thinking_budget" in r
        and _est_think_tokens(r) is None
    )
    lines += [
        "",
        "- n is the sample count the Q values are computed from (direct: `none` rows only; survival: budgeted rows), not all rows in the group.",
        "- Q values assume the prompt-mixture distribution of this run (per-prompt values differ); est tokens carry +-5-10% error and small-sample noise.",
    ] + (
        [
            f"- {unestimable} ok thinking=on row(s) could not be estimated at all (no reasoning-content timing and/or no answer timing: content-prefix routing or an unclosed think block) and are excluded from Tables 3/4."
        ]
        if unestimable
        else []
    )
    return lines


def _mean_med_cell(vals: list[float]) -> str:
    if not vals:
        return "-"
    return f"{statistics.mean(vals):,.0f}/{statistics.median(vals):,.0f}"


def _binding_table_lines(rows: list[dict]) -> list[str]:
    """Build Table 4: per-budget binding analysis (observed vs expected).

    est tok = estimated thinking tokens (think wall time x generation rate).
    honored = rows cut AT the budget (0.95b <= est <= BUDGET_HONOR_UPPER*b);
    overshoot = rows whose est ran past the budget, i.e. the budget was NOT
    enforced; under = the rest (natural thinking shorter than the budget).
    expected = share of `none` rows whose natural length is >= 0.95x budget
    (natural-length prediction, shown only when >= 3 none rows exist; rows
    truncated at max_tokens are lower bounds and are flagged). budget 0 rows
    are NOT shown here at all: _group_budget_rows skips them, because a forced
    cut leaves no natural length to compare against. The zero-budget anchor is
    visible in Table 1 and in the rows files only.
    """
    groups = _group_budget_rows(rows)
    if not groups:
        return []
    lines = [
        "",
        "Table 4: budget binding analysis. est tok = estimated thinking tokens "
        "(think wall time x generation rate). honored/overshoot/under partition "
        "the rows against the budget; overshoot means the engine did not enforce "
        "it. expected = share of none rows with natural length >= 0.95x budget "
        "(prediction; - when fewer than 3 none rows).",
        "",
        "| model | effort | budget | n | est tok mean/med | honored | overshoot | under | expected |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (model, effort), g in sorted(
        groups.items(), key=lambda kv: (kv[0][0], _effort_sort_key(kv[0][1]))
    ):
        none_est = sorted(g["none"])
        cells = []
        if none_est:
            cells.append(("-", len(none_est), _mean_med_cell(none_est), "-", "-", "-", "-"))
        for b in sorted(g["budgets"]):
            cell = g["budgets"][b]
            expected = "-"
            if len(none_est) >= 3:
                k = sum(1 for v in none_est if v >= 0.95 * b)
                expected = f"{k}/{len(none_est)} ({k / len(none_est):.0%})"
                if g["none_censored"]:
                    expected += f" [{g['none_censored']} censored]"
            honored = f"{cell['honored']}/{cell['n']}" if b else "-"
            overshoot = f"{cell['overshoot']}/{cell['n']}" if b else "-"
            under = f"{_budget_under(cell)}/{cell['n']}" if b else "-"
            cells.append(
                (
                    str(b),
                    cell["n"],
                    _mean_med_cell(cell["est"]),
                    honored,
                    overshoot,
                    under,
                    expected,
                )
            )
        for budget_s, n, mm, honored, overshoot, under, expected in cells:
            lines.append(
                f"| {model} | {effort} | {budget_s} | {n} | {mm} | {honored} "
                f"| {overshoot} | {under} | {expected} |"
            )
    return lines


def _check_key(check: str) -> str | None:
    """Last number in a check string, used as the spot-check answer key.

    Non-string values yield None (skipped) rather than raising: this runs
    inside summary generation, after the measurement is already paid for.
    validate_prompts rejects them at input time; this is the second line of
    defence for hand-edited or older prompt files."""
    if not isinstance(check, str):
        return None
    nums = re.findall(r"\d[\d,]*(?:\.\d+)?", check)
    return nums[-1] if nums else None


def _content_matches(content: str, key: str) -> bool:
    """Heuristic match: the key number appears in the content with digit
    boundaries (comma and plain forms both accepted)."""
    if not content:
        return False
    for candidate in {key, key.replace(",", "")}:
        if re.search(r"(?<![0-9])" + re.escape(candidate) + r"(?![0-9])", content):
            return True
    return False


def _check_table_lines(rows: list[dict], prompts: dict | None) -> list[str]:
    """Build Table 5: answer spot-check against the prompts' `check` values.

    Heuristic: the last number in each check string must appear in the row's
    answer with digit boundaries. Not a full verification.

    Only rows that actually delivered an answer are scored: an echoed thinking
    block (annotate_row) quotes the number in the model's own reasoning, so a
    contains-check on raw content passes on rows that never answered."""
    checks = {
        p.get("id"): p.get("check")
        for p in (prompts or {}).get("prompts", [])
        if isinstance(p, dict) and p.get("check")
    }
    if not checks:
        return []
    groups: dict[tuple, dict] = {}
    for rec in rows:
        if not is_row_done(rec):
            continue
        key = _check_key(checks.get(rec.get("prompt_id"), ""))
        if key is None:
            continue
        cell = groups.setdefault(
            (
                rec["model"],
                rec["effort"],
                rec.get("thinking"),
                rec.get("thinking_budget"),
                rec.get("prompt_id"),
            ),
            {"n": 0, "answered": 0, "ok": 0, "no_answer": 0, "unsplitable": 0, "key": key},
        )
        cell["n"] += 1
        if rec.get("no_answer"):
            cell["no_answer"] += 1
            continue
        if rec.get("answer_unknown"):
            # thinking arrived as an unsplitable content prefix: the number may
            # well be in there, but we cannot tell trace from answer, so the
            # row is not scorable rather than scored on raw content.
            cell["unsplitable"] += 1
            continue
        cell["answered"] += 1
        if _content_matches(rec.get("content_raw") or "", key):
            cell["ok"] += 1
    if not groups:
        return []
    lines = [
        "",
        "Table 5: answer spot-check vs the prompts' `check` values. ok = answers "
        "containing the check's last number (heuristic contains-check, NOT a full "
        "verification; check values are hand-derived). ok is scored over "
        "`answered` rows only - rows that delivered no answer, and rows whose "
        "thinking arrived as an unsplitable content prefix, are excluded.",
        "",
        "| model | effort | thinking | budget | prompt | n | answered | check ok (of answered) | key |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (model, effort, thinking, budget, prompt), cell in sorted(
        groups.items(),
        key=lambda kv: (
            kv[0][0],
            _effort_sort_key(kv[0][1]),
            kv[0][2],
            kv[0][3] is not None,
            kv[0][3] or 0,
            kv[0][4],
        ),
    ):
        budget_s = "-" if budget is None else str(budget)
        reason = "no answer" if cell["no_answer"] else (
            "unsplitable thinking" if cell["unsplitable"] else ""
        )
        ok_cell = (
            f"{cell['ok']}/{cell['answered']}"
            if cell["answered"]
            else f"- ({reason or 'no answer'})"
        )
        lines.append(
            f"| {model} | {effort} | {thinking} | {budget_s} | {prompt} | {cell['n']} | "
            f"{cell['answered']}/{cell['n']} | {ok_cell} | {cell['key']} |"
        )
    return lines


# Manifest fields added after the first generation. Older manifests legitimately
# lack them, so they are neither required (check_manifest's missing-key gate) nor
# compared for equality (_manifest_mismatches) - otherwise every additive schema
# change would make --resume structurally impossible for pre-existing runs.
# Required-ness of the fields that have always existed is enforced by
# _manifest_schema_errors().
MANIFEST_OPTIONAL_KEYS = ("prompt_checks",)


def build_manifest(
    args: argparse.Namespace,
    prompts_text: str,
    prompts: dict | None = None,
) -> dict[str, Any]:
    """Build the experiment configuration stored beside the result rows.

    `prompt_checks` records the resolved spot-check answers so that
    --resummarize of a run made from a custom --prompts file reproduces the
    same Table 5 keys instead of silently re-reading data/prompts.json. It is
    derived from the resolved `prompts` object, not from prompts_text, because
    that text is only a hash input (and for merged files is not even one JSON
    document)."""
    checks: dict[str, Any] = {}
    for p in (prompts or {}).get("prompts", []):
        if isinstance(p, dict) and p.get("id") and p.get("check") is not None:
            checks[p["id"]] = p["check"]
    return {
        "script_sha256": hashlib.sha256(
            (HERE / "run_bench.py").read_bytes()
        ).hexdigest()[:12],
        "prompts_sha256": hashlib.sha256(prompts_text.encode("utf-8")).hexdigest()[:12],
        "prompt_checks": checks,
        "base_url": BASE_URL,
        "models": args.models,
        "thinking": args.thinking,
        "efforts": args.efforts,
        "runs": args.runs,
        "max_tokens": args.max_tokens,
        "thinking_budgets": args.thinking_budgets,
        "seed": args.seed,
        "reload": args.reload,
        "no_unload": args.no_unload,
        "delay": args.delay,
        "timeout": args.timeout,
        "load_timeout": args.load_timeout,
    }


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    prompts_text: str,
    prompts: dict | None = None,
) -> None:
    """Write the current experiment configuration as formatted JSON."""
    manifest = {"created": datetime.now().isoformat(timespec="seconds")}
    manifest.update(build_manifest(args, prompts_text, prompts))
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def dump_model_settings(
    model: str, settings_path: Path, output_dir: Path
) -> str | None:
    """Dump oMLX model_settings.json settings for one model to
    ``<output_dir>/<model>.setting.json`` as formatted JSON text.

    Returns an error string (for stderr) or None on success. Failures (missing
    file, model not found, parse error) never raise: the benchmark must continue
    even if settings cannot be captured.

    The ``settings_path`` defaults to ``~/.omlx/model_settings.json`` and can be
    overridden via the ``$OMLX_SETTINGS_FILE`` environment variable (mirroring
    the ``$OMLX_BASE_URL`` pattern).
    """
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"model_settings.json not found at {settings_path}"
    except (json.JSONDecodeError, OSError) as e:
        return f"cannot read {settings_path}: {e}"
    if not isinstance(settings, dict):
        return f"{settings_path} is not a JSON object"
    models = settings.get("models")
    if not isinstance(models, dict):
        return f"{settings_path} has no 'models' object"
    if model not in models:
        return f"model {model!r} not found in {settings_path.name}"
    text = json.dumps(models[model], ensure_ascii=False, indent=2)
    (output_dir / f"{model}.setting.json").write_text(text + "\n", encoding="utf-8")
    return None


def dump_all_model_settings(models: list[str], output_dir: Path) -> None:
    """Dump settings for every unique model in ``models`` once each.

    Called once at run start so the oMLX configuration that produced each
    measurement is preserved alongside the result rows. Failures are reported
    on stderr but never abort the benchmark.
    """
    settings_path = Path(
        os.environ.get("OMLX_SETTINGS_FILE") or str(DEFAULT_SETTINGS_PATH)
    )
    seen: set[str] = set()
    for m in models:
        if m in seen:
            continue
        seen.add(m)
        err = dump_model_settings(m, settings_path, output_dir)
        if err is not None:
            print(f"# settings: {err}", file=sys.stderr, flush=True)
        else:
            print(f"# settings -> {output_dir / f'{m}.setting.json'}", flush=True)


def _manifest_schema_errors(saved: dict) -> list[str]:
    """Return schema violations for a loaded manifest (empty = valid)."""
    schema_errors: list[str] = []
    for key in ("script_sha256", "prompts_sha256", "base_url"):
        if not isinstance(saved.get(key), str) or not saved[key]:
            schema_errors.append(f"{key} must be a non-empty string")
    for key in ("models", "thinking", "efforts"):
        value = saved.get(key)
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item for item in value)
        ):
            schema_errors.append(f"{key} must be a non-empty array of strings")
    if (
        not isinstance(saved.get("runs"), int)
        or isinstance(saved["runs"], bool)
        or saved["runs"] < 1
    ):
        schema_errors.append("runs must be an integer >= 1")
    if (
        not isinstance(saved.get("max_tokens"), int)
        or isinstance(saved["max_tokens"], bool)
        or saved["max_tokens"] <= 0
    ):
        schema_errors.append("max_tokens must be an integer > 0")
    if saved.get("seed") is not None and (
        not isinstance(saved["seed"], int) or isinstance(saved["seed"], bool)
    ):
        schema_errors.append("seed must be null or an integer")
    budgets = saved.get("thinking_budgets")
    if budgets is not None and (
        not isinstance(budgets, list)
        or any(
            b is not None and (
                not isinstance(b, int) or isinstance(b, bool) or b < 0
            )
            for b in budgets
        )
    ):
        schema_errors.append(
            "thinking_budgets must be an array of integers >= 0 or null"
        )
    for key in ("reload", "no_unload"):
        if not isinstance(saved.get(key), bool):
            schema_errors.append(f"{key} must be a boolean")
    for key, minimum, inclusive in (
        ("delay", 0.0, True),
        ("timeout", 0.0, False),
        ("load_timeout", 0.0, False),
    ):
        value = saved.get(key)
        valid_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        valid_number = valid_number and is_finite_number(value)
        valid_number = valid_number and (
            value >= minimum if inclusive else value > minimum
        )
        if not valid_number:
            comparison = ">= 0" if inclusive else "> 0"
            schema_errors.append(f"{key} must be a finite number {comparison}")
    return schema_errors


def _manifest_mismatches(saved: dict, cur: dict[str, Any]) -> list[str]:
    """Return ``key: saved/current`` lines where the current args diverge
    from the saved experiment config.

    Resuming the same experiment on a SUBSET of its models is legitimate
    (e.g. finishing an interrupted full run by model); expanding beyond the
    saved model list is not."""
    bad: list[str] = []
    for k, v in cur.items():
        if k in MANIFEST_OPTIONAL_KEYS and k not in saved:
            # Additive field this manifest predates (see MANIFEST_OPTIONAL_KEYS).
            # Required fields are already gated by check_manifest's missing-key
            # check, so anything else absent from saved stays a mismatch.
            continue
        if k == "models":
            if not set(v) <= set(saved.get(k) or []):
                bad.append(
                    f"  {k}: saved={saved.get(k)!r}  current={v!r} (current must be a subset)"
                )
            continue
        if saved.get(k) != v:
            bad.append(f"  {k}: saved={saved.get(k)!r}  current={v!r}")
    return bad


def check_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    prompts_text: str,
    prompts: dict | None = None,
) -> int:
    """Verify the saved experiment config matches the current args.
    Returns 0 on ok, 2 on mismatch OR a corrupted/schemabroken manifest
    (valid JSON that is not an object, missing required keys, or invalid value
    types). A missing manifest (legacy run) warns and returns 0 (key-only
    resume)."""
    if not manifest_path.exists():
        print(
            f"# resume: no manifest for {manifest_path.name}; verifying keys only (legacy run)",
            file=sys.stderr,
            flush=True,
        )
        return 0
    try:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(
            f"# resume: unreadable manifest {manifest_path}: {e}; aborting",
            file=sys.stderr,
        )
        return 2
    if not isinstance(saved, dict):
        print(
            f"# resume: manifest {manifest_path} is not a JSON object "
            f"(got {type(saved).__name__}); aborting",
            file=sys.stderr,
            flush=True,
        )
        return 2
    cur = build_manifest(args, prompts_text, prompts)
    # Manifests predating the budget feature have neither key; manifests from
    # the single-value era carry thinking_budget. Normalize both into
    # thinking_budgets (empty list = no budget) so schema and mismatch checks
    # compare one canonical key across all manifest generations.
    if "thinking_budgets" not in saved:
        legacy = saved.get("thinking_budget")
        saved["thinking_budgets"] = [] if legacy is None else [legacy]
    missing = [
        k for k in cur if k not in saved and k not in MANIFEST_OPTIONAL_KEYS
    ]
    if missing:
        print(
            f"# resume: manifest {manifest_path} missing keys {missing}; aborting",
            file=sys.stderr,
            flush=True,
        )
        return 2
    schema_errors = _manifest_schema_errors(saved)
    if schema_errors:
        print(
            f"# resume: manifest {manifest_path} has invalid schema; aborting",
            file=sys.stderr,
        )
        for error in schema_errors:
            print(f"  {error}", file=sys.stderr)
        return 2
    bad = _manifest_mismatches(saved, cur)
    if bad:
        print(
            "# resume: experiment config mismatch with selected file:", file=sys.stderr
        )
        for line in bad:
            print(line, file=sys.stderr)
        print(
            "# resume: ABORTED (no rows written). Re-run without --resume to start a new run, "
            "or select a matching file to continue it.",
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0


def write_csv(results_path: Path, csv_path: Path, max_tokens: object = None) -> None:
    """Write scalar row fields to CSV, omitting prompts and raw payloads.

    The four answer-delivery columns are derived by annotate_row (they need
    the raw payloads and max_tokens), so CSV stays consistent with the summary
    tables for old JSONL as well as new."""
    rows = [
        annotate_row(rec, max_tokens) for _, rec in iter_row_records(results_path)
    ]
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fields = [
        "timestamp",
        "model",
        "thinking",
        "effort",
        "thinking_budget",
        "prompt_id",
        "run",
        "status",
        "thinking_mode",
        "trace_flag",
        "leak_rc",
        "leak_content",
        "load_s",
        "first_any_token_s",
        "first_reasoning_token_s",
        "ttft_s",
        "total_s",
        "reasoning_chars",
        "content_chars",
        "answer_chars",
        "thinking_chars",
        "content_tokens",
        "thinking_echo",
        "answer_unknown",
        "truncated",
        "no_answer",
        "finish_reason",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def resummarize_results(rows_arg: str | None) -> int:
    """Regenerate summary_<stamp>.md / rows_<stamp>.csv from an existing rows
    JSONL without any HTTP call or measurement (--resummarize).

    With no path, the newest data/rows_*.jsonl is used. The adjacent manifest
    supplies the experiment config for the header (missing manifest -> "?" in
    the header). Existing summary/csv artifacts in the same directory are
    overwritten with the regenerated ones."""
    if rows_arg:
        results_path = Path(rows_arg)
        if not results_path.exists():
            print(f"error: rows file not found: {results_path}", file=sys.stderr)
            return 2
    else:
        candidates = result_files(DATA_DIR)
        if not candidates:
            print("error: no rows_*.jsonl found under data/", file=sys.stderr)
            return 2
        results_path = candidates[-1]
    stamp = results_path.stem.removeprefix("rows_")
    out_dir = results_path.parent
    manifest_path = out_dir / f"manifest_{stamp}.json"
    rows = [rec for _, rec in iter_row_records(results_path)]
    if not rows:
        print(f"error: no usable rows in {results_path}", file=sys.stderr)
        return 2
    config = _summary_config(manifest_path, None)
    if not config:
        print(
            f"# resummarize: no readable manifest beside {results_path.name}; "
            "header values will show '?'",
            file=sys.stderr,
            flush=True,
        )
    synth = SimpleNamespace(
        models=config.get("models", ["?"]),
        thinking=config.get("thinking", ["?"]),
        efforts=config.get("efforts", ["?"]),
        runs=config.get("runs", "?"),
        max_tokens=config.get("max_tokens", "?"),
        seed=config.get("seed", "?"),
        thinking_budgets=config.get("thinking_budgets"),
        no_unload=config.get("no_unload", False),
    )
    prompt_ids = list(dict.fromkeys(r["prompt_id"] for r in rows))
    # Table 5 check values: the manifest's resolved prompt_checks win, so a run
    # made from a custom --prompts file regenerates with ITS OWN answers rather
    # than whatever data/prompts.json holds now (the same id can carry a
    # different check). Legacy manifests without the key fall back to the
    # default prompt file and say so.
    checks: dict = {}
    saved_checks = config.get("prompt_checks")
    if isinstance(saved_checks, dict) and saved_checks:
        checks = saved_checks
    else:
        try:
            pjson = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
            if isinstance(pjson, dict):
                checks = {
                    p.get("id"): p.get("check")
                    for p in pjson.get("prompts", [])
                    if isinstance(p, dict)
                }
        except (OSError, json.JSONDecodeError):
            pass
        if prompt_ids and not set(prompt_ids) <= set(checks):
            print(
                "# resummarize: no prompt_checks in the manifest; falling back to "
                f"{PROMPTS_FILE.name} (Table 5 may use different check values than "
                "the original run)",
                file=sys.stderr,
                flush=True,
            )
    prompts = {
        "prompts": [{"id": pid, "check": checks.get(pid)} for pid in prompt_ids]
    }
    summary_path = out_dir / f"summary_{stamp}.md"
    csv_path = out_dir / f"rows_{stamp}.csv"
    write_summary(results_path, summary_path, synth, manifest_path, prompts)
    write_csv(results_path, csv_path, config.get("max_tokens"))
    print(f"# resummarize: {len(rows)} rows from {results_path}")
    print(f"# summary -> {summary_path}")
    print(f"# csv     -> {csv_path}")
    return 0


def fmt(x: object, spec: str = ".3f") -> str:
    """Format a metric value, using ``-`` for missing or NaN values."""
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    if isinstance(x, (int, float)):
        return format(x, spec)
    return str(x)


# Table 1 metric columns as (row field, header label, format spec); the
# header and every condition row are driven from this one spec so the
# layout cannot drift. answer_chars (not content_chars) is the answer column:
# content_chars counts an echoed thinking block verbatim (see annotate_row).
# The content_tokens field is server usage.completion_tokens, which includes
# thinking tokens, so it is labelled accordingly.
TABLE1_METRICS = [
    ("ttft_s", "ttft_s", ".3f"),
    ("total_s", "total_s", ".3f"),
    ("thinking_chars", "thinking_chars", ".0f"),
    ("answer_chars", "answer_chars", ".0f"),
    ("content_tokens", "completion_tokens", ".0f"),
]


def _mean_median_str(values: list, spec: str = ".3f") -> str:
    """Return ``mean/median`` formatted per spec, or ``-/-`` when empty."""
    if not values:
        return "-/-"
    return (
        f"{format(statistics.mean(values), spec)}/{format(statistics.median(values), spec)}"
    )


def _metric_values(field: str, rows: list[dict]) -> list:
    """Collect the non-null values of one numeric field across rows."""
    return [v for v in (r.get(field) for r in rows) if v is not None]


def _summary_config(manifest_path: Path, args: argparse.Namespace) -> dict:
    """Load the saved experiment config for the summary header, falling back
    to the current args per key when unreadable or not a JSON object."""
    try:
        config = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return config if isinstance(config, dict) else {}


def _report_max_tokens(config: dict, args: argparse.Namespace | None) -> object:
    """max_tokens for the derived answer/truncation fields.

    The manifest value wins: it records what the rows were actually generated
    with, which is what --resummarize of an older run must reproduce."""
    return config.get(
        "max_tokens", getattr(args, "max_tokens", None) if args else None
    )


def _summary_header_lines(
    args: argparse.Namespace,
    manifest_path: Path,
    prompts: dict | None,
    rows: list[dict],
    ok_rows: list[dict],
) -> list[str]:
    """Build the title/config/count block above the summary tables."""
    summary_config = _summary_config(manifest_path, args)
    summary_models = summary_config.get("models", args.models)
    summary_thinking = summary_config.get("thinking", args.thinking)
    summary_efforts = summary_config.get("efforts", args.efforts)
    summary_runs = summary_config.get("runs", args.runs)
    summary_max_tokens = summary_config.get("max_tokens", getattr(args, "max_tokens", "?"))
    summary_budgets = summary_config.get("thinking_budgets")
    if summary_budgets is None:
        legacy = summary_config.get("thinking_budget")
        summary_budgets = [] if legacy is None else [legacy]
    summary_seed = summary_config.get("seed", args.seed)
    summary_prompt_ids = [p["id"] for p in prompts["prompts"]] if prompts else "?"
    no_answer = sum(1 for r in ok_rows if r.get("no_answer"))
    unknown = sum(1 for r in ok_rows if r.get("answer_unknown"))
    truncated = sum(1 for r in ok_rows if r.get("truncated"))
    return [
        "# LLM Benchmark Summary",
        "",
        f"- date: {datetime.now().isoformat(timespec='seconds')}",
        f"- manifest: {manifest_path.name}",
        f"- models: {', '.join(summary_models)}",
        f"- thinking: {', '.join(summary_thinking)}  efforts: {', '.join(summary_efforts)}  runs: {summary_runs}",
        f"- max_tokens: {summary_max_tokens}  seed: {summary_seed}"
        + (
            f"  thinking_budgets: {', '.join('none' if b is None else str(b) for b in summary_budgets)}"
            if summary_budgets
            else ""
        ),
        f"- prompts: {json.dumps(summary_prompt_ids, ensure_ascii=False)}",
        f"- rows: {len(rows)} total, {len(ok_rows)} ok, {len(rows) - len(ok_rows)} error/incomplete",
        f"- answered: {len(ok_rows) - no_answer - unknown}/{len(ok_rows)} ok rows  "
        f"(no_answer: {no_answer}  answer_unknown: {unknown}  truncated at max_tokens: {truncated})",
    ]


def _condition_rows_by_key(rows: list[dict]) -> dict[tuple, list[dict]]:
    """Group rows by (model, thinking, effort, budget); every attempted
    condition is present (ok rows appended, others yield an empty ok list)."""
    by_cond: dict[tuple, list[dict]] = {}
    for r in rows:
        by_cond.setdefault(
            (r["model"], r["thinking"], r["effort"], r.get("thinking_budget")), []
        )
    for r in rows:
        if is_row_done(r):
            by_cond[
                (r["model"], r["thinking"], r["effort"], r.get("thinking_budget"))
            ].append(r)
    return by_cond


def _cond_sort_key(key: tuple) -> tuple:
    """Order conditions by (model, thinking, effort, budget) with efforts in
    intensity order and the budget-free slot first (None before integers)."""
    model, thinking, effort, budget = key
    return (
        model,
        thinking,
        _effort_sort_key(effort),
        budget is not None,
        budget or 0,
    )


def _finish_counts(vals: list[dict]) -> str:
    """Return the stop/length/other finish_reason tally cell for ok rows."""
    frs = [r.get("finish_reason") or "?" for r in vals]
    stop = frs.count("stop")
    length = frs.count("length")
    other = len(frs) - stop - length
    return f"{stop}/{length}/{other}"


def _delivery_counts(vals: list[dict]) -> str:
    """Return the no_answer/answer_unknown/truncated tally cell.

    Derived, not server-reported: a row can be status=ok, finish_reason=stop
    and still have produced no answer, and a row whose thinking arrived as an
    unsplitable content prefix has an answer length we cannot know (see
    annotate_row)."""
    no_ans = sum(1 for r in vals if r.get("no_answer"))
    unknown = sum(1 for r in vals if r.get("answer_unknown"))
    trunc = sum(1 for r in vals if r.get("truncated"))
    return f"{no_ans}/{unknown}/{trunc}"


def _table1_lines(rows: list[dict], ok_rows: list[dict]) -> list[str]:
    """Build Table 1: mean/median over **ok rows only** per condition."""
    columns = " | ".join(f"{label} mean/med" for _, label, _ in TABLE1_METRICS)
    lines = [
        f"| model | thinking | effort | budget | n_ok | n_total | {columns} | finish (stop/length/other) | no_ans/unk/trunc |",
        "|---" * (6 + len(TABLE1_METRICS) + 2) + "|",
    ]
    by_cond = _condition_rows_by_key(rows)
    for key in sorted(by_cond, key=_cond_sort_key):
        model, thinking, effort, budget = key
        vals = by_cond[key]
        cells = [
            model,
            thinking,
            effort,
            "-" if budget is None else str(budget),
            str(len(vals)),
            str(
                sum(
                    1
                    for r in rows
                    if (r["model"], r["thinking"], r["effort"], r.get("thinking_budget"))
                    == key
                )
            ),
        ]
        cells += [
            _mean_median_str(_metric_values(field, vals), spec)
            for field, _, spec in TABLE1_METRICS
        ]
        cells.append(_finish_counts(vals))
        cells.append(_delivery_counts(vals))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _table2_lines(
    args: argparse.Namespace,
    results_path: Path,
    cond_load_events: dict[tuple, dict[str, float]] | None,
) -> list[str]:
    """Build Table 2: successful load events per condition.

    - in-process event map (this run's actual load cycles), merged with
      events already present in the JSONL (past cycles from a resumed run)
    - legacy rows use a load_s-based compatibility event id
    - under --no-unload no load cycle ever happens: print N/A honestly."""
    lines = [
        "",
        "Table 2: successful model load events per condition. `load_s` = /v1/models/<m>/load response time only (reload HTTP time and reload_wait polling are separate load_log rows).",
        "",
    ]
    if getattr(args, "no_unload", False):
        lines.append("_no load cycles performed (--no-unload)")
        lines.append("")
        return lines
    lines += [
        "| model | thinking | effort | budget | load events | load_s mean/med |",
        "|---|---|---|---|---|---|",
    ]
    load_by_cond: dict[tuple, dict[str, float]] = {}
    for key, events in load_events_from_jsonl(results_path).items():
        load_by_cond.setdefault(key, {}).update(events)
    for key, events in (cond_load_events or {}).items():
        load_by_cond.setdefault(key, {}).update(events)
    for key in sorted(load_by_cond, key=_cond_sort_key):
        model, thinking, effort, budget = key
        vals = list(load_by_cond[key].values())
        if not vals:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    thinking,
                    effort,
                    "-" if budget is None else str(budget),
                    str(len(vals)),
                    _mean_median_str(vals),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def write_summary(
    results_path: Path,
    md_path: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    prompts: dict | None = None,
    cond_load_events: dict[tuple, dict[str, float]] | None = None,
) -> None:
    """Write condition statistics and load-event statistics as Markdown.

    Uses the SAME done/ok predicate as --resume (is_row_done): complete,
    error-free rows only. ``prompts`` supplies the prompt id list for the
    header (``None`` renders ``?``)."""
    rows = [rec for _, rec in iter_row_records(results_path)]
    if not rows:
        md_path.write_text("# No data\n", encoding="utf-8")
        return
    config = _summary_config(manifest_path, args)
    annotate_rows(rows, _report_max_tokens(config, args))
    ok_rows = [r for r in rows if is_row_done(r)]
    lines = _summary_header_lines(args, manifest_path, prompts, rows, ok_rows)
    lines += [
        "",
        "Table 1: mean/median over **ok rows only** (prompt x run). A row is ok only if the SSE stream was complete ([DONE] + finish_reason, no bad chunks) - ok does NOT mean an answer was produced: `answer_chars` excludes echoed thinking blocks and is null (so omitted from the mean) when the thinking arrived as an unsplitable content prefix, and `no_ans/unk/trunc` counts rows with no answer / rows whose answer length is unknown / rows cut off at max_tokens.",
        "",
    ]
    lines += _table1_lines(rows, ok_rows)
    lines += _table2_lines(args, results_path, cond_load_events)
    lines += _natural_quantile_lines(rows)
    lines += _binding_table_lines(rows)
    lines += _check_table_lines(rows, prompts)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def validate_prompts(prompts: object) -> list[str]:
    """Validate prompts.json schema; returns a list of error messages."""
    errors: list[str] = []
    if not isinstance(prompts, dict):
        return ["top level must be a JSON object"]
    mt = prompts.get("max_tokens")
    if not isinstance(mt, int) or isinstance(mt, bool) or mt <= 0:
        errors.append('"max_tokens" must be an integer > 0')
    plist = prompts.get("prompts")
    if not isinstance(plist, list) or not plist:
        errors.append('"prompts" must be a non-empty array')
        return errors
    seen: set = set()
    for i, p in enumerate(plist):
        where = f"prompts[{i}]"
        if not isinstance(p, dict):
            errors.append(f"{where} must be an object")
            continue
        pid = p.get("id")
        if not isinstance(pid, str) or not pid:
            errors.append(f'{where} must have a non-empty string "id"')
        elif pid in seen:
            errors.append(f"{where} duplicate prompt id: {pid!r}")
        else:
            seen.add(pid)
        content = p.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f'{where} ({pid}) must have a non-empty string "content"')
        # "check" is optional, but a truthy non-string (e.g. the bare number 38)
        # would otherwise survive input validation and raise TypeError inside
        # the summary builder - i.e. AFTER the whole measurement finished.
        if "check" in p and p.get("check") is not None and not isinstance(
            p["check"], str
        ):
            errors.append(
                f'{where} ({pid}) "check" must be a string or null '
                f"(got {type(p['check']).__name__})"
            )
    return errors


class BenchConfigError(Exception):
    """Fatal configuration error (prompts files); aborts the run with exit 2."""


def _resolve_prompt_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve the --prompts paths: absolute paths as-is, relative paths
    against cwd with a data/<name> fallback."""
    prompts_paths = (
        [PROMPTS_FILE] if args.prompts is None else [Path(p) for p in args.prompts]
    )
    resolved: list[Path] = []
    for p in prompts_paths:
        cand = p if p.is_absolute() else (Path.cwd() / p)
        if not cand.exists():
            alt = DATA_DIR / p.name if not p.is_absolute() else p
            if alt.exists():
                cand = alt
        resolved.append(cand)
    return resolved


def _read_prompt_file(cand: Path) -> tuple[str, dict]:
    """Read, parse, and schema-check one prompts file; raises
    BenchConfigError on any failure."""
    try:
        txt = cand.read_text(encoding="utf-8")
    except OSError as e:
        raise BenchConfigError(f"cannot read prompts file {cand}: {e}")
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError as e:
        raise BenchConfigError(f"{cand} is not valid JSON: {e}")
    perr = validate_prompts(obj)
    if perr:
        raise BenchConfigError("\n".join(f"{cand}: {msg}" for msg in perr))
    return txt, obj


def _merge_prompt_files(resolved: list[Path], parsed_list: list[dict]) -> dict:
    """Merge multiple prompts files into one prompts object; duplicate prompt
    ids across files are fatal."""
    all_prompts: list[dict] = []
    seen_ids: set[str] = set()
    for obj in parsed_list:
        for pr in obj["prompts"]:
            if pr["id"] in seen_ids:
                raise BenchConfigError(
                    f"duplicate prompt id {pr['id']!r} across --prompts files"
                )
            seen_ids.add(pr["id"])
            all_prompts.append(pr)
    return {
        "description": f"merged {len(resolved)} files",
        "max_tokens": max(obj["max_tokens"] for obj in parsed_list),
        "prompts": all_prompts,
    }


def load_prompts(args: argparse.Namespace) -> tuple[dict, str]:
    """Load, validate, and (when several files are given) merge the prompts;
    returns (prompts, prompts_text). prompts_text feeds the manifest hash
    (single file: the file text; merged: the files' texts joined by \\n)."""
    resolved = _resolve_prompt_paths(args)
    prompts_texts: list[str] = []
    parsed_list: list[dict] = []
    for cand in resolved:
        txt, obj = _read_prompt_file(cand)
        prompts_texts.append(txt)
        parsed_list.append(obj)
    if len(parsed_list) == 1:
        prompts = parsed_list[0]
        prompts_text = prompts_texts[0]
    else:
        prompts = _merge_prompt_files(resolved, parsed_list)
        prompts_text = "\n".join(prompts_texts)
    prompt_errors = validate_prompts(prompts)
    if prompt_errors:
        raise BenchConfigError("\n".join(f"prompts: {msg}" for msg in prompt_errors))
    return prompts, prompts_text


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (its description is the module docstring)."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=MODELS,
        help=f"model ids (default: {len(MODELS)} models)",
    )
    ap.add_argument("--thinking", nargs="*", choices=THINKING, default=THINKING)
    ap.add_argument(
        "--efforts",
        nargs="*",
        choices=EFFORTS,
        default=DEFAULT_EFFORTS,
        help="effort values for thinking=on rows (default: low medium high; "
        "'high' is a real Qwen3.8 value. 'off' probes the template alias, "
        "'xhigh' is a Qwen template value; add either explicitly)",
    )
    ap.add_argument(
        "--thinking-budgets",
        nargs="*",
        type=_budget_token,
        default=None,
        help="reasoning budget tokens, sent as the top-level thinking_budget field "
        "(integer >= 0) on thinking=on rows; multiple values cross with the selected "
        "efforts (each effort x budget pair becomes its own condition). 'none' adds a "
        "no-budget (natural thinking length) condition. oMLX caps thinking with an "
        "engine-side <think> block monitor, so it is template-independent "
        "(default: not sent)",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=3,
        help="measurement runs per condition (default 3, >= 1)",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=None, help="override prompts.json max_tokens"
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="seconds to sleep between unload and load (default 5)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="per-request timeout seconds (default 600)",
    )
    ap.add_argument(
        "--reload",
        action="store_true",
        help="POST /admin/api/reload between unload and load (server may 500 briefly after)",
    )
    ap.add_argument(
        "--no-unload",
        action="store_true",
        help="for OpenAI-compatible SSE endpoints without load/unload management: "
        "skip the unload/reload/load/warmup cycle and never call admin endpoints "
        "(the server must already serve the selected models)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="fixed sampling seed for reproducibility (default: server default, varies)",
    )
    ap.add_argument(
        "--load-timeout",
        type=float,
        default=900.0,
        help="client timeout (s) for the model load call; must exceed the server-side load (default 900)",
    )
    ap.add_argument(
        "--base-url",
        default=None,
        help="oMLX base URL (default: $OMLX_BASE_URL or http://127.0.0.1:8000)",
    )
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="prompts JSON file(s) (default: data/prompts.json). Multiple files are merged; "
        "relative paths are resolved against data/ and cwd. Example: --prompts prompts.arith.deep.json prompts.words.deep.json",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the condition matrix and exit"
    )
    ap.add_argument(
        "--check-env",
        action="store_true",
        help="read-only preflight: probe GET /v1/models and GET /v1/models/status, "
        "print an environment report and exit (0 base API usable, 1 unreachable, "
        "3 unusable; chat benchmark capability is NOT verified)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="append to the newest data/<timestamp>/rows_*.jsonl after verifying "
        "manifest; also supports legacy data/rows_*.jsonl; skip done "
        "(status=ok) condition+prompt+run keys",
    )
    ap.add_argument(
        "--resummarize",
        nargs="?",
        const="",
        default=None,
        metavar="ROWS_JSONL",
        help="regenerate summary_<stamp>.md and rows_<stamp>.csv from an existing "
        "rows JSONL without any measurement (no value: use the newest "
        "data/rows_*.jsonl)",
    )
    return ap


def validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject empty selections and non-positive timeouts/runs; ap.error exits
    with code 2."""
    if not args.models:
        ap.error("--models must list at least one model id")
    if not args.thinking:
        ap.error("--thinking must list at least one of off on")
    if not args.efforts:
        ap.error("--efforts must list at least one value")
    if args.runs < 1:
        ap.error("--runs must be >= 1")
    if any(isinstance(b, int) and b < 0 for b in args.thinking_budgets or []):
        ap.error("--thinking-budgets must be integers >= 0 or 'none'")
    if not (
        is_finite_number(args.timeout)
        and is_finite_number(args.load_timeout)
        and args.timeout > 0
        and args.load_timeout > 0
    ):
        ap.error("--timeout/--load-timeout must be finite numbers > 0")
    if not is_finite_number(args.delay) or args.delay < 0:
        ap.error("--delay must be a finite number >= 0")


def _models_list_schema_error(body: object) -> str | None:
    """Describe a /v1/models schema violation, or None when valid.

    Valid = JSON object whose 'data' is a list of objects each carrying a
    non-empty string 'id'. Non-JSON payloads arrive wrapped as {'raw': ...}
    (see http_json) and are reported as such."""
    if not isinstance(body, dict):
        return "response is not a JSON object"
    if "raw" in body and "data" not in body:
        return "response is not JSON"
    data = body.get("data")
    if not isinstance(data, list):
        return f"'data' is not a list ({type(data).__name__})"
    for entry in data:
        if not isinstance(entry, dict):
            return f"'data' entry is not an object: {entry!r}"
        mid = entry.get("id")
        if not isinstance(mid, str) or not mid:
            return f"'data' entry has no non-empty string 'id': {entry!r}"
    return None


def _probe_status_endpoint() -> None:
    """Report-only probe of the admin /v1/models/status endpoint.

    Never influences the --check-env exit code. Same schema rule as
    loaded_models(): 'models' must be a list of {id: str}; empty ids are
    tolerated (str check only)."""
    status, body = http_json("/v1/models/status", timeout=10.0)
    if status == 0:
        print(
            f"# GET /v1/models/status -> unreachable ({body.get('error', body)}); "
            "full mode undecided, --no-unload is the safe choice"
        )
        return
    if status in (404, 501):
        print(
            f"# GET /v1/models/status -> HTTP {status}: no admin status endpoint; "
            "use --no-unload"
        )
        return
    if status == 405:
        print(
            "# GET /v1/models/status -> HTTP 405 (unexpected method mismatch); "
            "prefer --no-unload"
        )
        return
    if status == 429:
        print(
            "# GET /v1/models/status -> HTTP 429 (rate limited): full mode undecided; "
            "retry later. --no-unload is safe meanwhile"
        )
        return
    if status in (401, 403):
        print(
            f"# GET /v1/models/status -> HTTP {status}: admin endpoints need auth; "
            "full mode unavailable"
        )
        print(
            "#   generic (--no-unload) behavior with this server is NOT verified by "
            "this probe; do a minimal run"
        )
        return
    if status != 200 or not isinstance(body, dict):
        print(
            f"# GET /v1/models/status -> HTTP {status}: status endpoint error; "
            "full mode undecided; prefer --no-unload"
        )
        return
    models = body.get("models")
    if not isinstance(models, list) or not all(
        isinstance(m, dict) and isinstance(m.get("id"), str) for m in models
    ):
        print("# GET /v1/models/status -> HTTP 200 but unexpected shape; prefer --no-unload")
        return
    loaded = [
        m["id"]
        for m in models
        if m.get("source_model_id") is None
        and (m.get("loaded") is True or m.get("is_loading") is True)
    ]
    print(
        f"# GET /v1/models/status -> HTTP 200, {len(loaded)} loaded/loading; "
        "full mode looks available"
    )


def check_environment() -> int:
    """--check-env preflight: probe GET /v1/models and GET /v1/models/status.

    The exit code is decided by /v1/models only (0 reachable + valid schema,
    1 unreachable, 3 reachable but unusable); the admin status endpoint is
    report-only. Passing proves list-API reachability ONLY: chat completions,
    SSE streaming, usage accounting and acceptance of extended fields
    (thinking_budget, chat_template_kwargs, ...) remain unverified until a
    minimal benchmark run."""
    print(f"# check-env: base_url {BASE_URL}")
    status, body = http_json("/v1/models", timeout=10.0)
    if status == 0:
        print(f"# GET /v1/models -> unreachable ({body.get('error', body)})")
        print(
            "# exit 1: server unreachable (start oMLX, or check --base-url / "
            "$OMLX_BASE_URL)"
        )
        return 1
    if status != 200:
        if status in (401, 403):
            print(
                f"# GET /v1/models -> HTTP {status}: auth required; "
                "this bench sends no API key"
            )
        else:
            print(
                f"# GET /v1/models -> HTTP {status}: not a usable OpenAI-compatible "
                "list API"
            )
        print("# exit 3: base API unusable")
        _probe_status_endpoint()
        return 3
    err = _models_list_schema_error(body)
    if err is not None:
        print(f"# GET /v1/models -> HTTP 200 but invalid schema: {err}")
        print("# exit 3: base API unusable")
        _probe_status_endpoint()
        return 3
    entries = body["data"]
    if not entries:
        print(
            "# GET /v1/models -> HTTP 200, schema valid, but 0 models; "
            "download/load models first (see README)"
        )
        print(
            "# exit 0: base discovery probe passed / chat benchmark capability "
            "remains unverified"
        )
        _probe_status_endpoint()
        return 0
    ids = ", ".join(f"{entry['id']!r}" for entry in entries)
    print(f"# GET /v1/models -> HTTP 200, {len(entries)} model(s): {ids}")
    print(
        "# base discovery probe passed / chat benchmark capability remains unverified"
    )
    _probe_status_endpoint()
    print(
        "# exit 0: base discovery probe passed (list API reachability only; "
        "chat capability unverified)"
    )
    return 0


def main() -> int:
    """Parse options, run all selected conditions, and write output artifacts."""
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.resummarize is not None:
        return resummarize_results(args.resummarize or None)
    global BASE_URL
    if args.check_env:
        BASE_URL = args.base_url or os.environ.get("OMLX_BASE_URL") or DEFAULT_BASE_URL
        BASE_URL = BASE_URL.rstrip("/")
        return check_environment()
    args.models = unique_values(args.models)
    args.thinking = unique_values(args.thinking)
    args.efforts = unique_values(args.efforts)
    args.thinking_budgets = unique_values(args.thinking_budgets or [])
    validate_args(ap, args)
    if args.thinking_budgets and "on" not in args.thinking:
        print(
            "# note: --thinking-budgets applies to thinking=on rows only; "
            "no thinking=on condition is selected",
            file=sys.stderr,
            flush=True,
        )
    if args.reload and args.no_unload:
        print(
            "# note: --reload is ignored with --no-unload (no load/unload management)",
            file=sys.stderr,
            flush=True,
        )

    try:
        prompts, prompts_text = load_prompts(args)
    except BenchConfigError as e:
        for line in str(e).splitlines():
            print(f"error: {line}", file=sys.stderr)
        return 2
    if args.max_tokens is None:
        args.max_tokens = prompts["max_tokens"]
    if not (args.max_tokens > 0):
        ap.error("--max-tokens must be a number > 0")
    BASE_URL = args.base_url or os.environ.get("OMLX_BASE_URL") or DEFAULT_BASE_URL
    BASE_URL = BASE_URL.rstrip("/")

    paths = resolve_output_paths(args)
    conds = condition_matrix(args)
    if args.dry_run:
        print_dry_run(args, conds, prompts, paths)
        return 0

    done = load_completed(paths.results_path) if args.resume else set()
    if args.resume:
        rc = check_manifest(paths.manifest_path, args, prompts_text, prompts)
        if rc != 0:
            return rc
    if not paths.manifest_path.exists():
        write_manifest(paths.manifest_path, args, prompts_text, prompts)

    dump_all_model_settings(args.models, paths.output_dir)

    logger = LoadLogger(paths.loadlog_path)
    print(f"# oMLX bench start {datetime.now().isoformat(timespec='seconds')}")
    print(f"# base_url: {BASE_URL}  reload: {args.reload}  seed: {args.seed}")
    print(
        f"# {len(conds)} conditions, {args.runs} runs, {len(prompts['prompts'])} prompts, max_tokens={args.max_tokens}"
        + (
            f", thinking_budgets={args.thinking_budgets}"
            if args.thinking_budgets
            else ""
        )
    )
    print(f"# results -> {paths.results_path}")

    stats = RunStats()
    cond_load_events: dict[tuple, dict[str, float]] = {}
    t_start = time.perf_counter()
    try:
        for c in conds:
            cond_keys = [
                row_key(c.model, c.thinking, c.effort, c.budget, p["id"], run)
                for run in range(1, args.runs + 1)
                for p in prompts["prompts"]
            ]
            if done and all(k in done for k in cond_keys):
                print(
                    f"[{c.model} | {c.thinking} | {c.effort} | {c.budget}] all rows done; skip",
                    flush=True,
                )
                continue

            load_s: float | None = None
            load_event_id: str | None = None
            load_err: str | None = None
            if not args.no_unload:
                load_s, load_event_id, load_err = run_condition_setup(
                    c, args, logger, prompts
                )
                if load_event_id is not None:
                    cond_load_events.setdefault(
                        (c.model, c.thinking, c.effort, c.budget), {}
                    )[load_event_id] = load_s
            if load_err is not None:
                print(
                    f"  !! condition setup failed: {load_err}",
                    file=sys.stderr,
                    flush=True,
                )

            for run in range(1, args.runs + 1):
                for p in prompts["prompts"]:
                    if (
                        row_key(
                            c.model, c.thinking, c.effort, c.budget, p["id"], run
                        )
                        in done
                    ):
                        continue
                    stats.add(
                        run_measurement_request(
                            args,
                            c,
                            p,
                            run,
                            load_s,
                            load_event_id,
                            load_err,
                            paths.results_path,
                        )
                    )
    finally:
        # Best-effort final cleanup, skipped under --no-unload (user-managed);
        # see cleanup_loaded_models.
        if not args.no_unload:
            cleanup_loaded_models(logger)

    elapsed = time.perf_counter() - t_start
    print(
        f"\n# done: {stats.ok} ok, {stats.incomplete} incomplete, {stats.fatal} error(s), elapsed {elapsed:.1f}s"
    )
    write_summary(
        paths.results_path,
        paths.summary_path,
        args,
        paths.manifest_path,
        prompts,
        cond_load_events=cond_load_events,
    )
    write_csv(
        paths.results_path,
        paths.csv_path,
        _report_max_tokens(_summary_config(paths.manifest_path, args), args),
    )
    print(f"# summary -> {paths.summary_path}")
    print(f"# csv     -> {paths.csv_path}")
    return 0 if (stats.incomplete + stats.fatal) == 0 else 1


@dataclass
class OutputPaths:
    """Output artifact paths for one run (all share the same stamp)."""

    output_dir: Path
    results_path: Path
    csv_path: Path
    loadlog_path: Path
    summary_path: Path
    manifest_path: Path


@dataclass
class RunStats:
    """Row-outcome counters for one benchmark run."""

    ok: int = 0
    incomplete: int = 0
    fatal: int = 0

    def add(self, status: str) -> None:
        """Count one row outcome (ok / incomplete / error)."""
        if status == "ok":
            self.ok += 1
        elif status == "incomplete":
            self.incomplete += 1
        else:
            self.fatal += 1


def row_key(
    model: str,
    thinking: str,
    effort: str,
    budget: int | None,
    prompt_id: str,
    run: int,
) -> str:
    """Unique per-attempt key: model|thinking|effort|budget|prompt_id|run."""
    return f"{model}|{thinking}|{effort}|{budget}|{prompt_id}|{run}"


def resolve_output_paths(args: argparse.Namespace) -> OutputPaths:
    """Create/pick the output directory and artifact paths. With --resume the
    newest existing rows_*.jsonl (data/<timestamp>/ first, then legacy
    data/) is selected and its stamp/directory reused."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DATA_DIR / stamp
    results_path = output_dir / f"rows_{stamp}.jsonl"
    if args.resume:
        prev = result_files(DATA_DIR)
        if prev:
            results_path = prev[-1]
            output_dir = results_path.parent
            stamp = results_path.stem.removeprefix("rows_")
            print(f"# resume: appending to {results_path}", file=sys.stderr, flush=True)
        else:
            print("# resume: no previous rows_*.jsonl; starting fresh", file=sys.stderr)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        output_dir=output_dir,
        results_path=results_path,
        csv_path=output_dir / f"rows_{stamp}.csv",
        loadlog_path=output_dir / f"load_log_{stamp}.tsv",
        summary_path=output_dir / f"summary_{stamp}.md",
        manifest_path=output_dir / f"manifest_{stamp}.json",
    )


def print_dry_run(
    args: argparse.Namespace,
    conds: list[Condition],
    prompts: dict,
    paths: OutputPaths,
) -> None:
    """Print the condition matrix and output destinations (--dry-run)."""
    print(
        f"# {len(conds)} conditions x {args.runs} runs x {len(prompts['prompts'])} prompts = {len(conds) * args.runs * len(prompts['prompts'])} requests"
        + (
            f", thinking_budgets={args.thinking_budgets}"
            if args.thinking_budgets
            else ""
        )
    )
    for c in conds:
        print(
            f"  {c.model:32s} thinking={c.thinking:4s} effort={c.effort:8s} "
            f"budget={c.budget if c.budget is not None else '-'}"
        )
    print(
        f"# base_url: {BASE_URL}\n# results: {paths.results_path}\n# csv: {paths.csv_path}\n# loads: {paths.loadlog_path}\n# summary: {paths.summary_path}\n# manifest: {paths.manifest_path}"
    )


def _unload_all_models(
    args: argparse.Namespace, logger: LoadLogger, loaded_now: list[str]
) -> str | None:
    """Unload every selected + currently-loaded physical model (deduped,
    selected order first). Returns an error string or None."""
    try:
        for m in dict.fromkeys(list(args.models) + list(loaded_now)):
            unload_model(m, logger)
    except RuntimeError as e:
        return str(e)
    return None


def _reload_and_wait(logger: LoadLogger) -> str | None:
    """POST /admin/api/reload, then poll until /v1/models answers again (the
    admin endpoint 500s briefly during re-discovery). Returns an error string
    or None."""
    t_reload = time.perf_counter()
    status, body = http_json("/admin/api/reload", {}, timeout=30.0)
    logger.log("*", "reload", status, body, time.perf_counter() - t_reload)
    if status != 200:
        return f"reload failed: status={status} body={body}"
    t_wait = time.perf_counter()
    available = wait_until_available(timeout_total=120.0)
    logger.log(
        "*",
        "reload_wait",
        200 if available else 504,
        {"available": available},
        time.perf_counter() - t_wait,
    )
    if not available:
        return "server unavailable after reload (120s)"
    return None


def run_condition_setup(
    c: Condition, args: argparse.Namespace, logger: LoadLogger, prompts: dict
) -> tuple[float | None, str | None, str | None]:
    """Run one condition's full isolation load cycle: model status ->
    unload all -> [reload + wait] -> delay -> load -> warmup.

    Returns (load_s, load_event_id, load_err). On any failure load_err is
    set (the caller records error rows and sends no measurement request)
    while load_s / load_event_id still carry the load outcome when the load
    itself succeeded (e.g. warmup failure)."""
    try:
        loaded_now = loaded_models()
    except Exception as e:  # noqa: BLE001
        return None, None, f"model status: {e}"
    err = _unload_all_models(args, logger, loaded_now)
    if err is not None:
        return None, None, err
    if args.reload:
        err = _reload_and_wait(logger)
        if err is not None:
            return None, None, err
    time.sleep(args.delay)
    try:
        load_s = load_model(c.model, logger, timeout=args.load_timeout)
    except RuntimeError as e:
        return None, None, f"load: {e}"
    load_event_id = uuid.uuid4().hex
    print(f"  warming up {c.model}...", file=sys.stderr, flush=True)
    try:
        warmup(
            c.model,
            prompts["prompts"][0]["content"],
            c.thinking,
            c.effort,
            timeout=min(args.timeout, 300.0),
            seed=args.seed,
            thinking_budget=c.budget,
        )
    except Exception as e:  # noqa: BLE001
        return load_s, load_event_id, f"warmup: {e}"
    return load_s, load_event_id, None


def _append_row(results_path: Path, rec: dict) -> None:
    """Append one row record to the results JSONL."""
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def new_row(
    c: Condition,
    prompt: dict,
    run: int,
    load_s: float | None,
    load_event_id: str | None,
) -> dict:
    """Build the initial JSONL record for one attempted request."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": c.model,
        "thinking": c.thinking,
        "effort": c.effort,
        "thinking_budget": c.budget,
        "prompt_id": prompt["id"],
        "run": run,
        "status": "pending",
        "error": None,
        "load_event_id": load_event_id,
        "thinking_mode": "not_observed",  # overwritten per-response by summarize_thinking
        "load_s": load_s,
        "prompt": prompt["content"],
        "usage": None,
    }


def _apply_stream_result(rec: dict, res: dict, c: Condition) -> str:
    """Fold one measured stream result into the row record and print the
    one-line outcome. Returns the row status (ok / incomplete)."""
    fallback_mode, header = THINKING_MODES.get(c.model, DEFAULT_THINKING_MODE)
    (
        reasoning_chars,
        content_chars,
        thinking_chars,
        trace_flag,
        leak_rc,
        leak_content,
        mode,
    ) = summarize_thinking(res, fallback_mode, header, c.thinking == "off")
    complete = res["protocol_error"] is None
    rec["thinking_mode"] = mode
    rec.update(
        status="ok" if complete else "incomplete",
        first_any_token_s=round(res["first_any_token_s"], 6)
        if res["first_any_token_s"] is not None
        else None,
        first_reasoning_token_s=round(res["first_reasoning_token_s"], 6)
        if res["first_reasoning_token_s"] is not None
        else None,
        ttft_s=round(res["ttft_s"], 6) if res["ttft_s"] is not None else None,
        total_s=round(res["total_s"], 6),
        reasoning_chars=reasoning_chars,
        content_chars=content_chars,
        thinking_chars=thinking_chars,
        trace_flag=trace_flag,
        leak_rc=leak_rc,
        leak_content=leak_content,
        content_raw=res["content"],
        reasoning_raw=res["reasoning_content"],
        usage=res["usage"],
        content_tokens=res["content_tokens"],
        finish_reason=res["finish_reason"],
        error=res["protocol_error"],
    )
    leak = " LEAK!" if (c.thinking == "off" and (leak_rc or leak_content)) else ""
    print(
        f"  [{rec['status']}] ttft={fmt(rec['ttft_s'], '.2f')}s total={fmt(res['total_s'], '.2f')}s "
        f"think={thinking_chars}ch content={content_chars}ch tok={rec['content_tokens']} "
        f"finish={rec['finish_reason']}{leak}"
        + (f" err={rec['error']}" if rec["error"] else ""),
        flush=True,
    )
    return rec["status"]


def run_measurement_request(
    args: argparse.Namespace,
    c: Condition,
    prompt: dict,
    run: int,
    load_s: float | None,
    load_event_id: str | None,
    load_err: str | None,
    results_path: Path,
) -> str:
    """Run and record one (prompt x run) measurement request for a condition;
    appends exactly one JSONL row. Returns the row status ok / incomplete /
    error (error = HTTP/timeout failure, or a failed condition setup which
    records the error row without sending a request)."""
    rec = new_row(c, prompt, run, load_s, load_event_id)
    if load_err is not None:
        rec["status"] = "error"
        rec["error"] = f"condition setup failed: {load_err}"
        status = "error"
        print(f"  !! {rec['error']}", file=sys.stderr, flush=True)
    else:
        print(
            f"[{c.model} | {c.thinking} | {c.effort} | {c.budget} | {prompt['id']} #{run}] request...",
            flush=True,
        )
        try:
            res = run_stream_request(
                c.model,
                prompt["content"],
                c.thinking,
                c.effort,
                args.max_tokens,
                timeout=args.timeout,
                seed=args.seed,
                thinking_budget=c.budget,
            )
            status = _apply_stream_result(rec, res, c)
        except Exception as e:  # noqa: BLE001
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"
            status = "error"
            print(f"  !! error: {rec['error']}", file=sys.stderr, flush=True)
    _append_row(results_path, rec)
    return status


def cleanup_loaded_models(logger: LoadLogger) -> None:
    """Leave the server empty like the bench started it: RE-QUERY the loaded
    models and unload whatever is still loaded (best-effort).

    Re-querying also covers setup states a single tracked variable would
    miss (load succeeded but warmup failed, unload loop aborted midway,
    Ctrl-C mid-condition)."""
    try:
        still_loaded = loaded_models(timeout=30.0)
    except Exception as e:  # noqa: BLE001
        print(
            f"  ! cleanup: cannot list models: {e}", file=sys.stderr, flush=True
        )
        return
    for m in still_loaded:
        try:
            print(f"# cleanup: unloading {m}", file=sys.stderr, flush=True)
            unload_model(m, logger)
        except Exception as e:  # noqa: BLE001
            print(f"  ! final unload {m}: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
