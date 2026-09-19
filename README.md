# llm-bench

A benchmark runner that measures **thinking** × **effort** × **thinking_budget**
behavior of reasoning LLMs served through an oMLX-compatible API: request
latency, token throughput, thinking/output split, budget-binding effectiveness,
and self-check answer accuracy.

**Tested with oMLX 0.7.0.dev2 (development build).** Stable releases may not
expose the admin endpoints; run `--check-env` first and fall back to
`--no-unload` if the status endpoint is missing.

## Requirements

- Apple Silicon Mac with macOS 15+
- [oMLX](https://github.com/jundot/omlx) installed and running
- Python 3.9+ (stdlib only; no third-party packages needed for the benchmark)
- Optional: `uv` + matplotlib for summary charts
- Run one benchmark instance per server; runs are strictly sequential
  (a single loop of requests, no concurrency)

## Obtaining models

Defaults are pinned Hugging Face repos with verified oMLX-format builds.
Quantization build differences make latencies/tokens non-comparable — use the
exact repo + revision below for reproducible numbers.

| Model id in this repo | Default HF repo | Pinned revision |
|---|---|---|
| `Ornith-1.5-35B-A3B-oQ4e-mtp` | `scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp` | `5465dc4cfc70aefda40177ffacd5e5cde27c2a0d` |
| `Qwen3.8-27B-oQ4e-mtp` | `Jundot/Qwen3.8-27B-oQ4e-mtp` | `04dc5509edd8670fc78cc8c2f74bf9b77b1f2acc` |

Alternative repos exist (e.g. `mlx-works/Ornith-1.5-35B-A3B-oQ4e-mtp`,
`LookUpMark/Ornith-1.5-35B-A3B-oQ4e-mtp`) but are separate builds; do not mix
builds within one measurement series.

**Local-quantization id collision**: oMLX displays model ids without the org
prefix, so a locally quantized copy of the base model can shadow a pinned HF
repo under the same id (e.g. `Qwen/Qwen3.8-27B-oQ4e-mtp` and
`Jundot/Qwen3.8-27B-oQ4e-mtp` both surface as `Qwen3.8-27B-oQ4e-mtp`). A
local quantization is a **different weight set** from the pinned repo —
make sure the server serves the pinned repo when reproducing published
numbers.

**Effort contract for the pinned Qwen template**: the template accepts
`xhigh` / `medium` / `low` (its built-in default when unspecified is
`xhigh`). The CLI default matrix instead runs `low medium high`
(`DEFAULT_EFFORTS`); `high` depends on oMLX's alias fallback (one alias
attempt on exception, then dropping `reasoning_effort` to the template
default), so the `high` row may degrade or error depending on the engine
build. To measure deterministically against the pinned repo, prefer
`--efforts low medium xhigh`.

`Ornith-1.0-35B-oQ4e` has no oMLX-format HF repo; the base model is
`ornith-ai/Ornith-1.0-35B` and `lmstudio-community/Ornith-1.0-35B-MLX-4bit`
is an alternative MLX build (not pinned, not comparable).

## Modes

**Primary: `--no-unload`** — for OpenAI-compatible SSE endpoints without
load/unload management. The runner skips the unload/reload/load/warmup cycle
and never calls admin endpoints; the server must already serve the selected
models. It relies on these server features: `chat_template_kwargs`,
`reasoning_effort`, `thinking_budget` (top-level field), `stream_options.include_usage`,
SSE `[DONE]` + `finish_reason`, and `usage.completion_tokens`. Strict servers
may reject unknown fields. Tables 3/4 (budget analysis) depend on
`usage.generation_tokens_per_second`, which is oMLX-specific; other servers
will leave those columns empty. This mode is *designed* for endpoints with
the features above; verify behavior on your server with a minimal run —
`--check-env` only proves list-API reachability. `--reload` is ignored with
`--no-unload` (a note is printed).

**Secondary: full mode** (default) — oMLX-managed load cycle. Per condition:
unload → reload (optional, `--reload`) → load → warmup, keeping the measured
requests isolated from the in-server load cost.

## Quickstart

| Command | Requests |
|---|---|
| `python3 scripts/run_bench.py --check-env` | 0 (probe only) |
| `python3 scripts/run_bench.py --dry-run` | 0 (prints the condition matrix) |
| Minimal run: `python3 scripts/run_bench.py --no-unload --models Qwen3.8-27B-oQ4e-mtp --thinking on --efforts medium --runs 1` | **1 condition × 3 prompts = 3 requests** (no warmup under `--no-unload`) |
| Default matrix (1 model, no budgets, runs=3) | 4 conditions × 3 prompts × 3 runs = **36 measurement requests**; full mode adds one warmup per condition (+4) and a load cycle. On the pinned Qwen repo the `high` row may error/degrade depending on the engine build (see Effort contract above) |
| Charts: `uv run --with matplotlib python3 scripts/plot_summary_charts.py result/<ts>/summary_<ts>.md result/<ts>/charts` (add `--per-model` for per-model panels) | — |

Note: `--thinking off` alone auto-appends an ON × effort=`off` alias-probe
condition, so it runs **2 conditions** (6 requests per run). The minimal run
above uses `--thinking on --efforts medium` to pin exactly one condition.

## Options

| Option | Meaning |
|---|---|
| `--models ...` | model ids (default: the 2 pinned defaults) |
| `--thinking off on ...` | thinking toggles (default: `off on`) |
| `--efforts ...` | effort values for thinking=on rows (default: `low medium high`; `off` probes the template alias; `xhigh` is a Qwen template value) |
| `--thinking-budgets ...` | reasoning budget tokens sent as the top-level `thinking_budget` field (integer ≥ 0, or `none` for a natural-length condition); each effort × budget pair becomes its own condition |
| `--runs N` | measurement runs per condition (default 3) |
| `--max-tokens N` | override `max_tokens` from prompts.json |
| `--prompts FILE...` | prompts JSON file(s) relative to `data/` (merged; default `data/prompts.json`) |
| `--reload` | POST `/admin/api/reload` between unload and load (full mode; ignored with `--no-unload`) |
| `--no-unload` | run without load/unload management (see Modes) |
| `--seed N` | fixed sampling seed (default: server default) |
| `--timeout S` | per-request timeout (default 600) |
| `--load-timeout S` | client timeout for model load (default 900) |
| `--delay S` | sleep between unload and load (default 5) |
| `--base-url URL` | server base URL (default: `$OMLX_BASE_URL` or `http://127.0.0.1:8000`) |
| `--dry-run` | print the condition matrix and exit |
| `--check-env` | read-only preflight probe; prints an environment report and exits (0 base API usable, 1 unreachable, 3 unusable; **chat capability is NOT verified**) |
| `--resume` | append to the newest `result/<timestamp>/rows_*.jsonl` after verifying the manifest; skips done rows |
| `--resummarize [ROWS_JSONL]` | regenerate summary/csv from an existing rows file without measuring |

## Output format

One run writes everything under `result/<YYYYMMDD_HHMMSS>/` (created
automatically; git-ignored). Prompt files stay under `data/` (git-tracked) —
the two directories are independent:

- `rows_<stamp>.jsonl` — one JSON object per request: model, condition
  (thinking/effort/budget), prompt id, run, timings, token usage, and the
  raw response fields
- `rows_<stamp>.csv` — flattened view with derived columns
- `summary_<stamp>.md` — human-readable tables (aggregate metrics; Table 3/4
  analyze how budgets bind measured behavior)
- `load_log_<stamp>.tsv` — unload/load/reload events (full mode)
- `manifest_<stamp>.json` — run configuration plus `script_sha256` /
  `prompts_sha256` for reproducibility
- `<model>.setting.json` — dump of each model's server-side settings

The per-row `status` column records request completion, **not** answer
correctness: `ok` ≠ answered. Check-value scoring in the summary is a
hand-computed spot check (heuristic), not a strict grader.

## Methodology

- Warmup requests (full mode) absorb the hidden in-server load cost before
  measurement; `--no-unload` skips them entirely.
- SSE streams are read to completion; a response counts as complete only
  with a terminal event.
- Thinking routing is auto-detected per response (`reasoning_content` vs
  content trace prefix); the per-row `thinking_mode` reports what that
  response actually showed, and `not_observed` rows never inherit a
  configured fallback label.
- `--resume` verifies `script_sha256` and `prompts_sha256` in the manifest
  against the current script/prompts; a script-version mismatch is rejected
  (exit 2), so old runs resume only with the matching script version.
- Table 3/4 use oMLX's `usage.generation_tokens_per_second` to analyze
  budget binding; see Modes for the portability caveat.

## Prompts

**4 prompt files / 12 prompt entries** (3 entries each): `prompts.json`,
`prompts.arith.deep.json`, `prompts.math.deep.json`, `prompts.words.deep.json`.
Multiple files merge:

```
python3 scripts/run_bench.py --prompts prompts.arith.deep.json prompts.words.deep.json
```

Check values are hand-computed spot checks (heuristic), not strict graders.

## Limitations

- Behavior depends on the oMLX version; admin endpoints and
  `generation_tokens_per_second` may differ on other builds.
- `thinking_budget` can be ineffective on engine paths that never bind it;
  such groups carry a "budget INEFFECTIVE" note in the summary.
- Effort values are template-dependent (see the Effort contract above);
  template behavior may reject or alias values.

## License

Apache-2.0 (see [LICENSE](LICENSE)).
