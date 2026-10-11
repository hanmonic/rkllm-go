# rkllm-go v2 design

OpenAI-compatible server for the RK3588 NPU that keeps the KV cache across
the turns of agent conversations, parks/restores several sessions per model,
shares a cached system+tools prefix between sessions, and swaps several
models in one process. Runtime behaviour it relies on is measured in
`docs/kv-cache-probe-findings.md` (ground truth); templates and tool-call
parsing are in `docs/templates-and-toolparse.md`.

Status (2026-10-10): runs on a NanoPi M6 (in an LXC container) as a dev server on :8081
(`/opt/rkllm-go-v2/run.sh`). Production v1 (`/opt/rkllm-go`,
`rkllm-server.service`, :8080) is untouched.

## 1. Architecture

```
HTTP (internal/api)          POST /v1/chat/completions, GET /v1/models, /healthz, /debug/state
  render prompt (internal/prompt, server-side template, raw mode in the runtime)
  parse tool calls / reasoning (internal/toolparse)
        |
        v  core.Job {Prompt, StaticPrefix, SessionID, NoCache, MaxNewTokens, Sampling}
Pool (internal/core/pool.go)  one Slot per loaded model; memory budget + IOMMU domains; LRU swap
  Slot = engine.Handle + bounded Queue + one worker goroutine
  worker: global NPU mutex -> strategy (prefix | append | snapshot | off) -> engine.Run
  SessionStore (per model)  session id -> last prompt, cache file
  FileStore                 prompt-cache files in /dev/shm under a byte budget (LRU)
  RawMap (per model)        exact generated text per returned assistant message
        |
        v
engine.Engine / Handle (internal/engine)  runtime-neutral interface
  rk   (cgo, build tag rk131 = default, or rk123)  -> librkllmrt.so
  fake (pure Go model of the probe findings; all core tests run on it)
```

- Prompts are always rendered by the server (`rkllm_set_chat_template(h,"","","")` at load = raw mode), so the
  bytes sent to the runtime are known exactly and are append-stable (`TestAppendStability`).
- `RawMap`: when the client sends back an assistant message the server produced, history is rendered from the
  exact generated text, so the re-sent prompt equals what was prefilled/generated before.
- One NPU job at a time across all handles (`Pool.npu` mutex); handles of different models stay resident.
- The `fake` engine reproduces the probe's semantics (LCP with the previous kh=0 prompt only, kh=1 append, one-shot
  prompt-cache seeds, stale-KV corruption on mode mixing) so strategies are unit-tested on macOS.
- `internal/rkllm` is the v1 binding (unused by v2, kept for reference).

## 2. Cache strategies per family

| Family | Models | Default | Mechanism |
|---|---|---|---|
| `qwen3` | qwen3-0.6b, -1.7b, -4b-16k, -8b-allnpu | `prefix` | keep_history=0, full prompt each request; the runtime reuses the longest common token prefix with the previous run's prompt |
| `gemma3` | gemma3n-e4b | `prefix` | same (LCP works on Gemma; BOS is added by the runtime) |
| `qwen3.5` | q35-2b, q35-4b, q35-9b | `snapshot` | the runtime does no LCP reuse on hybrid linear attention; every run saves a prompt-cache file, the next request of the session loads it |
| any | | `append` (opt-in) | keep_history=1, send only the delta when the new prompt extends the cached text |
| any | | `off` (per request: `"cache": false`, global: `--cache off`) | full clear + cold prefill |

Strategy resolution: `models[name].strategy` > `strategies[family]` > built-in default
(`config.DefaultStrategies`). Never mix keep_history modes on one handle without a full clear (probe: stale KV,
2-4x slower prefill); the slot tracks the mode and clears on every switch.

### prefix
Per request: if the session is live on the handle, run the full prompt (event `hit`, or `partial` if history
diverged; the runtime truncates at the first differing token itself). Otherwise park the live session, then
restore this session's file if it covers more than the current prefix record (`restored`), else load / create
the shared prefix cache (`cold`). Per-turn prefill = previous reply + new message + template glue.

### append
Live session and the new prompt extends `cached` (= previous prompt + generated text) -> run only the delta with
keep_history=1 (`hit`). Anything else -> full clear + full prompt (`miss` / `cold`). keep_history=1 state cannot be
saved in a restorable form (probe Q4b), so a session switch always costs a cold prefill. Useful for a single
long-lived Qwen3.5 session (lowest per-turn TTFT), bad for interleaved sessions.

### snapshot (Qwen3.5 default)
Every run is `run(kh=0, prompt, save_prompt_cache=<session file>)`. The file covers the prompt only (generated
tokens are not saved). Next request of the session: `rkllm_load_prompt_cache(file)` then the full prompt; the
runtime prefills only the suffix. The load happens on every request (also when the session is still live,
because Qwen3.5 has no LCP reuse), ~20-40 ms on q35-2b, ~90-100 ms on q35-4b.

Detail: the generation prompt ends in `</think>\n\n`. The runtime tokenizes a trailing `\n\n` as one token (271) but
`\n\n` followed by the reply text as `\n`,`\n` (198,198), so a snapshot of the full prompt would never be a token
prefix of the next prompt. Snapshot therefore sends the prompt without its trailing newlines (the file ends at
`</think>`), lets the model generate the newlines, and strips up to that many leading `\n` from the output.

#### Snapshot verification (2026-10-10, q35-2b, temperature 0, `scripts/snapshot_ab.py`)
The earlier report (snapshot session B "refused the weather for Mirabel" and was "unable to retrieve the weather
for Tokyo") was traced as follows:
1. Live interleaved A/B runs, append vs snapshot: 8/16 outputs identical. The first difference is request 3 (B1
   reply wording); everything later runs on different histories. The Mirabel refusal happens in **both** modes
   (in both, B1 first calls a non-existent tool `weather(city=Mirabel)`; the test's tool runner answers
   `unknown tool`). In the snapshot run, B2 then called `weather` instead of `get_weather`, got `unknown tool`, and
   correctly said it could not retrieve Tokyo's weather. Not a cache error.
2. Replay of the append transcript (identical message histories, same interleaving, fresh server per mode):
   append replay = append live (16/16, deterministic); snapshot vs append 12/16; cold (`cache:false`) vs append
   11/16; snapshot vs cold 11/16. With identical inputs all three paths disagree on 4-5 borderline greedy
   choices ("Turquoise" vs "Turquoise." vs "Your favourite colour is turquoise.", `weather` vs `get_weather`).
   Snapshot is not further from the cold baseline than append is.
3. `RKLLM_LOG_LEVEL=2` token check (`snapshot_ab.py logcheck`, `tokcmp`): for all 16 requests the snapshot prompt
   tokens equal the cold prompt tokens minus the final `\n\n` (271), the first difference is at the last position;
   all 16 restores loaded exactly the session's previous prompt (`Loaded prompt cache with N tokens`, `Current kv
   cache size is N`, first N prompt tokens == the previous prompt of that session). No cross-session leakage.

Verdict: snapshot restores are exact; output differences come from different numeric paths (file-restored state +
suffix prefill vs one prefill vs keep_history decode, W8A8 NPU kernels) on a 2B model with many near-ties, and in
the live run from histories that diverged after the first such tie. Append differs from cold for the same reason
and additionally because its KV holds the generated tokens (e.g. 271 after `</think>`) while re-rendered history
is re-tokenized (198,198). Snapshot is the Qwen3.5 default; `append` stays selectable.

## 3. Sessions, park / restore

- Session id: `X-Session-Id` request header, else the OpenAI `user` field, else the session whose last prompt is the
  longest prefix of this prompt (continuation), else a new random id. The id is always returned in the
  `X-Session-Id` response header and in `usage.rkllm.session_id`. 1000 sessions per model (LRU).
- Live session = the one whose state is in the handle's KV cache.
- **Park** (prefix strategy, lazy): when another session takes the handle, the live session's last prompt is saved
  with a 1-token kh=0 run, `save_prompt_cache=1` to `/dev/shm/rkllm-sessions/<model>/<session>.cache`. The hot path
  (consecutive turns of one session) never writes files. Skipped if the session's file already covers that prompt.
- **Restore**: load the file, then the full prompt (one-shot seed, probe Q4b).
- Snapshot: the file is written by every run, so park is free.
- Append: nothing restorable; a returning session is re-prefilled.
- Files live in `--session-dir` (wiped at start) under `--session-budget` bytes, evicted LRU (`FileStore`). File size:
  qwen3-1.7b 615 KB + 115 KB/token; q35-2b ~20 MB + 12 KB/token.
- A `"cache": false` request discards the live state (parked files are kept); the affected session's next request
  reports `miss` or `restored`.

## 4. Shared prefix caches

`StaticPrefix` = the append-stable system + tools block of the rendered prompt (`prompt.Render` stable part of
the first turn). For a new session whose prompt starts with it (>= `--min-prefix-bytes`, default 1024), the slot
loads `/dev/shm/rkllm-sessions/<model>/prefix-<sha>.cache`, creating it once with a 1-token save run if missing
(`usage.rkllm.prefix_cache` = `created` | `loaded`). Not needed when the handle's prefix record already starts
with the block (prefix strategy, same tools). Measured: a new qwen3-1.7b session with a 550-token system+tools
block starts at TTFT 370-440 ms (prefill 21-25) instead of ~2.9 s.

## 5. Model pool: memory and IOMMU domains

Step 0 (`cmd/rkllm-memtest`, `/opt/rkllm-go-v2/span.log`): the RKNPU driver maps each handle's NPU buffers into an
IOMMU domain (`RKLLMExtendParam.base_domain_id`, ids 0..15). One domain has ~4 GiB of IOVA space shared by
everything mapped into it; a large model fills one or more domains and a further `rkllm_init` into a full domain
fails with `failed to malloc npu memory`. With distinct domains several models coexist in one process.

The pool gives each model `ceil(mem_mb / 3500)` consecutive free domains (conservative: memtest shows e.g.
qwen3-8b-allnpu occupying domains 0-1, the pool reserves 3) and keeps the sum of `mem_mb` of resident models under
`--mem-budget-mb` (default 18000). Loading a model that does not fit evicts idle models least-recently-used first
(their live session is parked); a model larger than the budget is allowed alone. If init fails on the chosen
domains it retries once with one extra domain.

Measured resident memory (VmRSS after `rkllm_init`, ctx 16384, runtime 1.3.1) and load time:

| model | family | RSS MB | `mem_mb` | domains | init (warm) |
|---|---|---|---|---|---|
| qwen3-0.6b | qwen3 | 2,655 | 2700 | 1 | 1.5-2.2 s |
| qwen3-1.7b | qwen3 | 3,756 | 3800 | 2 | 3.4-3.7 s |
| qwen3-4b-16k | qwen3 | 6,758 | 6800 | 2 | 33-36 s |
| qwen3-8b-allnpu | qwen3 | 10,228 | 10300 | 3 | 10-24 s |
| q35-2b | qwen3.5 | 2,446 | 2500 | 1 | 3.9-6.9 s |
| q35-4b | qwen3.5 | 5,358 | 5400 | 2 | 32-36 s |
| q35-9b | qwen3.5 | 9,506 | 9600 | 3 | 62-64 s |
| gemma3n-e4b | gemma3 | 5,819 | 5900 | 2 | 42-46 s |

Fits under the 18,000 MB default budget (observed in the 8-model smoke test, `/opt/rkllm-go-v2/smoke8.txt`):
{q35-2b, q35-4b, q35-9b} 17,500; {qwen3-4b-16k, qwen3-8b-allnpu} 17,100; {gemma3n-e4b, q35-9b} 15,500;
{qwen3-0.6b, qwen3-1.7b, qwen3-4b-16k} 13,300; {qwen3-1.7b, q35-4b, qwen3-0.6b} 11,900. Does not fit:
qwen3-8b-allnpu + q35-9b (19,900), any 8B/9B + a 4B + the 1.7B. Actual process RSS with gemma3n-e4b + q35-9b
resident was 15.9 GB (estimate 15.5 GB). Keep `mem-budget-mb` + `session-budget` below the container's 28 GiB.

## 6. Queue

Per model: a bounded FIFO (`--queue-depth`, default 4 waiting jobs; the running job does not count). Full -> HTTP
503 `server_busy`. Jobs whose client disconnected are dropped before they run; a disconnect during a run calls
`rkllm_abort` (the handle is then marked dirty and fully cleared before the next job). One worker per slot, and
the global NPU mutex serializes runs across models. `usage.rkllm.queue_ms` is the wait; `model_load_ms` is set when
the request waited for a load.

## 7. Flags (`rkllm-server-131` / `rkllm-server-123`)

| flag | default | meaning |
|---|---|---|
| `--config` | `models.json` next to the binary | model list |
| `--addr` | `:8081` | listen address |
| `--max-context` | 4096 | max_context_len for models without one |
| `--max-tokens` | 2048 | init max_new_tokens (per-request `max_tokens` on 1.3.1 only) |
| `--temperature`, `--top-k`, `--top-p`, `--repeat-penalty` | 0, 1, 1.0, 1.0 | default sampling (greedy) |
| `--cache` | `on` | `off`: full clear + cold prefill on every request |
| `--queue-depth` | 4 | waiting requests per model before 503 |
| `--session-dir` | `/dev/shm/rkllm-sessions` | cache files (wiped at start) |
| `--session-budget` | `8GiB` | byte budget for cache files; 0 disables parking and prefix caches |
| `--mem-budget-mb` | 18000 | resident model budget (LRU swap above it) |
| `--prefix-cache` | true | shared system+tools prefix caches |
| `--min-prefix-bytes` | 1024 | minimum static block size for a prefix cache |
| `--preload` | "" | extra models to preload (adds to `models.json` `preload`) |

Env: `RKLLM_LOG_LEVEL=2` makes the runtime log every run's prompt text, token list and reused kv size.

## 8. models.json

```json
{
  "default": "qwen3-1.7b",
  "preload": ["qwen3-1.7b"],
  "strategies": {"qwen3": "prefix", "qwen3.5": "snapshot", "gemma3": "prefix"},
  "models": {
    "q35-2b": {"path": "/opt/rkllm/models/q35-2b/model.rkllm", "family": "qwen3.5", "strategy": "snapshot",
               "max_context_len": 16384, "mem_mb": 2500, "preload": false},
    "llama3.2-1b": "/opt/rkllm/models/llama3.2-1b/model.rkllm"
  },
  "families": {"gemma3-1b": "gemma3"}
}
```

- `default` (required): model for requests without `model`.
- `preload`: loaded at start, before listening.
- `strategies`: per-family default (`prefix|append|snapshot|off`).
- `models.<name>`: a path string (v1 form) or an object: `path`, `family` (`qwen3|qwen3.5|gemma3|gemma4`; inferred
  from the name if absent, unknown families fall back to the qwen3 template), `strategy`, `max_context_len`
  (all 8 benchmark models: 16384, the conversion limit), `mem_mb` (resident estimate; if absent: file size +
  0.12 MB/token of context + 300), `preload`.
- `families`: v1 name -> family map, still honoured.

## 9. API contract

`POST /v1/chat/completions`: OpenAI request (`messages`, `tools`, `stream`, `stream_options.include_usage`,
`max_tokens` / `max_completion_tokens`, `temperature`, `top_p`) plus `top_k`, `enable_thinking` (or
`chat_template_kwargs.enable_thinking`, default off) and `cache` (`false` = cold prefill for this request).

- Request header `X-Session-Id` (or body `user`) pins the session; response header `X-Session-Id` always carries it.
- `message.reasoning_content`: the thinking text (`<think>` / Gemma thought channel) when thinking is on; streamed
  as `delta.reasoning_content`. `message.content` never contains it.
- `message.tool_calls` with `finish_reason: "tool_calls"`; streaming sends tool calls as one delta with `index`,
  after holding back any partial call markup.
- `usage`: `prompt_tokens`, `completion_tokens`, `total_tokens`, `prompt_tokens_details.cached_tokens`, and
  `usage.rkllm`:

| field | meaning |
|---|---|
| `prompt_tokens` | prompt length in tokens (from the kv identity) |
| `prefill_tokens` | tokens actually prefilled (runtime perf) |
| `cached_tokens` | prompt_tokens - prefill_tokens |
| `prefill_ms`, `generate_tokens`, `generate_ms` | runtime perf stats |
| `ttft_ms` | HTTP arrival to first token (includes queue, load, park, restore) |
| `queue_ms`, `model_load_ms` | queue wait; load time if the request waited for a load |
| `cache_strategy` | `prefix` / `append` / `snapshot` / `off` |
| `cache_event` | `hit` (live session, cache covered all previous input), `partial` (live, history diverged; reused up to the difference), `miss` (live state lost or not extendable), `restored` (from a parked / snapshot file), `cold` (new session, prefix cache at most, or cache off) |
| `session_id`, `kv_cache_size` | session; KV size after the run |
| extras | `completion_tokens`, `park_ms`, `restore_ms`, `save_ms`, `prefix_cache` (`created` / `loaded`) |

- `GET /v1/models`: id, `loaded`, `family`, `context_length`, `cache_strategy`.
- `GET /debug/state`: runtime, budgets, `resident_mb_estimate`, per handle (strategy, domains, live session,
  `kv_mode`, `dirty`, kv size, queue), `cache_files`, sessions per model, `iommu_domain_owners`.
- `GET /healthz`; `POST /v1/embeddings` -> 501.
- Errors (OpenAI envelope): 404 `model_not_found`, 400 `context_length_exceeded`, 503 `server_busy` / `unavailable`,
  500 `inference_error`.

## 10. Integration numbers (`scripts/integration_v2.py`, temperature 0)

Logs on the board: `/opt/rkllm-go-v2/it-131-final.txt` (1.3.1), `it-123.txt` (1.2.3), `it-q35-4b.txt`,
`gemma3-check.txt`, `ab/` (snapshot A/B).

| model / runtime | case | TTFT | prefill |
|---|---|---|---|
| qwen3-1.7b 1.3.1 prefix | cold full prompt (cache off, 1,049 tok) | 4.75 s | 1,049 |
| | turn of a live session (10-request tool conversation) | 200-400 ms | 25-62 |
| | new session, shared prefix loaded | 370-440 ms | 21-25 |
| | session switch: park 160-225 ms + restore 140-165 ms | 500-710 ms | 23-48 |
| qwen3-1.7b 1.2.3 prefix | cold full prompt (967 tok) | 4.4 s | 967 |
| | live turn | 200-360 ms | 25-62 |
| | switch: park 245-320 ms + restore 104-118 ms | 630-810 ms | 24-48 |
| q35-2b 1.3.1 snapshot | cold full prompt (1,389 tok) | 9.5 s | 1,389 |
| | live turn (restore 20-40 ms) | 440-935 ms | 34-78 |
| | session switch (restore 20-110 ms) | 420-530 ms | 23-63 |
| q35-2b 1.3.1 append | live turn | 340-400 ms | 24-41 |
| | session switch (cold) | 5.1-7.9 s | full |
| q35-4b 1.3.1 snapshot | session switch (restore 92-104 ms) | 1.0-1.2 s | 23-48 |
| | live turn | ~2.1 s | 66-70 |
| gemma3n-e4b 1.3.1 prefix | 3-turn tool conversation, recall OK | 0.5-2.7 s | 22-162 |

All checks pass (solo recall, A/B recall without cross-talk, cache off, streaming tool calls) on qwen3-1.7b (both
runtimes), q35-2b and q35-4b. Snapshot live turns cost more than append live turns because the previous reply is
re-prefilled and the file is loaded and re-saved each turn; switches are 10-15x cheaper.

## 11. Differences between rk131 and rk123 builds

| | 1.3.1 (`-tags rk131`, default) | 1.2.3 (`-tags rk123`) |
|---|---|---|
| header | `third_party/rkllm/1.3.1` | `third_party/rkllm/1.2.3` (= `/opt/rkllm/include/rkllm.h`) |
| library | `/opt/rkllm-131/server/lib` | `/opt/rkllm/runtime` |
| `rkllm_init` callback | `RKLLMCallback*` struct | function pointer |
| per-request `max_tokens`, sampling | `RKLLMInferParam.max_new_tokens`, `sampling_params` | not available: init values (`--max-tokens`, `--temperature`, ...) apply to every request |
| 1-token prompt-cache save runs | `max_new_tokens=1` | stopped with `rkllm_abort` after the first token (`RunRequest.StopAfter`), then full clear |
| models | all 8 | Qwen3 and gemma3n load; Qwen3.5 (toolkit 1.3.0) fails `rkllm_init` |
| LLMCallState | NORMAL 0, WAITING 1, FINISH 2, ERROR 3 | same |

**1.2.3 crash (fixed).** `rkllm-server-123` segfaulted at address 0 inside `rkllm_run` on the first run after a
run that saved a prompt cache. Root cause: once any run on a handle has saved a prompt cache, 1.2.3 dereferences
`RKLLMInferParam.prompt_cache_params` on every later run; the shim passed NULL when not saving. Reproduced with the
probe (save run, then any run with NULL params -> SIGSEGV; with a zeroed struct -> fine). Fix: `rk_run` always
passes a zeroed `RKLLMPromptCacheParam` (`save_prompt_cache=0`), on both runtimes. Struct layouts were checked
against both headers and are not involved.

**1.2.3 pause bug (worked around).** The old 1.2.3 save path paused the run (callback returns 1). On 1.2.3 that
paused state survives `rkllm_clear_kv_cache`: the next `rkllm_load_prompt_cache` is silently ignored (full prefill)
and later prefills get 2-4x slower with stale-KV answers (the first post-fix integration run showed restored
sessions with `cached=0` and 17 s prefills). Ending the save run with `rkllm_abort` instead leaves a clean handle
(load + full prompt -> prefill 19 of 737 tokens). The engine no longer pauses at all.

## 12. Known limitations

- Qwen3.5 has no runtime prefix reuse: snapshot reloads a 20-30 MB+ file each request and re-prefills the previous
  reply; append is cheaper for one live session but loses everything on a switch.
- Qwen3 / Gemma prefix strategy re-prefills the previous assistant reply every turn (LCP matches prompts only).
- One prefix record per handle: with interleaved sessions every switch costs a park (lazy save) + restore.
- A history edit inside a Qwen3.5 conversation (not an extension of the snapshot's prompt) is a cold prefill.
- Greedy outputs depend on the cache path (cold vs restored vs append) for borderline tokens; benchmark
  comparisons between strategies at temperature 0 will show such differences on small models.
- Generated tokens are re-tokenized when history is re-rendered (`</think>\n\n` -> 198,198 instead of 271); the
  prefix/snapshot KV therefore differs slightly from what the model generated.
- 1.2.3: no per-request `max_tokens` / temperature; Qwen3.5 models do not load.
- Gemma 4 (gemma4-e2b) is unreliable with its own chat format (probe T2) and is not in the benchmark set.
- `mem_mb` is a static estimate; prompt-cache loads add ~file size transiently to RSS.
- Cache files are in /dev/shm (RAM); the session budget competes with model memory.
- The session store and files are in memory/tmpfs only; a restart loses all sessions.

## 13. Running on the board

Build and run only under `/opt/rkllm-go-v2` (sources in `src/`, a copy of this repo). Only one process may use the
NPU; `run.sh start` refuses if a probe, memtest, flask or other rkllm server is running.

```sh
/opt/rkllm-go-v2/run.sh status
/opt/rkllm-go-v2/run.sh build                 # rkllm-server-131 + rkllm-server-123 from src/
/opt/rkllm-go-v2/run.sh start [flags]         # :8081, models.json in /opt/rkllm-go-v2, log server.log
RUNTIME=123 /opt/rkllm-go-v2/run.sh restart   # 1.2.3 build
LOG_LEVEL=2 /opt/rkllm-go-v2/run.sh restart   # runtime prompt/token logging
/opt/rkllm-go-v2/run.sh stop                  # SIGTERM: parks sessions, destroys handles
python3 /opt/rkllm-go-v2/src/scripts/integration_v2.py --model qwen3-1.7b
```
The script is versioned as `scripts/run.sh`.
