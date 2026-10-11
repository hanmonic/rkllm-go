# rkllm-go

An OpenAI-compatible LLM server in Go for the Rockchip RK3588/RK3588S NPU (RKLLM runtime), built for agent workloads. It keeps the KV cache between the turns of a conversation, so a multi-turn tool-calling agent re-reads only the new tokens instead of the whole prompt on every request.

Benchmarks of 15 models on this stack, with numbers: see the [write-up and raw results](https://github.com/hanmonic/rk3588-npu-llm-benchmarks) and `docs/benchmarks-2026-10-10.md`.

## What it does

- `POST /v1/chat/completions` (streaming and non-streaming), `GET /v1/models`, `/healthz`, `/debug/state`.
- Server-side chat templates and tool-call parsing for Qwen3 (JSON in tags), Qwen3.5 (XML), Gemma 4 and the generic JSON formats. Tool calls come back as OpenAI `tool_calls`.
- KV-cache reuse, chosen per model family:
  - `prefix` (Qwen3, Gemma3): send the full prompt each time, the runtime reuses the longest common token prefix.
  - `snapshot` (Qwen3.5, hybrid linear attention, where the runtime has no automatic prefix reuse): save a prompt-cache file after each turn and reload it.
  - `append` (opt-in) and `off`.
  - Cache hits cut prefill by 82-99% in the benchmark.
- Several sessions per model: sessions are parked to files under `/dev/shm` and restored, with a shared system+tools prefix cache.
- Several models in one process with a memory budget, IOMMU-domain planning and LRU swapping.
- Works with RKLLM runtime 1.3.1 and 1.2.3 (build tags `rk131` / `rk123`).
- An experimental `--live-append` mode for Qwen3.5. It cuts time-to-first-token 3-4x on long coding sessions but lowered accuracy on short tasks in testing, so it is off by default. Details in `docs/benchmarks-2026-10-10.md`.

## Build and run

You need an arm64 RK3588 board with the RKNPU driver, the RKLLM runtime library (`librkllmrt.so`, from [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm)) and converted `.rkllm` models. Conversion needs x86_64 Linux and the RKLLM toolkit; context is capped at 16,384 tokens by the toolkit.

```sh
# on the board
make          # fetches the RKLLM headers, then builds rkllm-server-131 and rkllm-server-123
./rkllm-server-131 --config models.json --addr :8081
```

The cgo flags in `internal/engine/rk/flags_131.go` and `flags_123.go` point at `/opt/rkllm-131/server/lib` and `/opt/rkllm/runtime`; change them to where you put `librkllmrt.so`.

Edit `models.json` for your model paths, families, context lengths and memory estimates. On macOS or any machine without the runtime, `make test` runs the pure-Go tests against a fake engine that models the runtime behaviour we measured.

```sh
curl localhost:8081/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"hi"}]}'
```

Send an `X-Session-Id` header (or the session field the server returns) to keep a conversation on the cache.

## Layout

| Path | What |
|---|---|
| `cmd/server` | the server |
| `cmd/rkllm-bench` | agent benchmark harness (house batches, multi-step chains, coding tasks, interleaved sessions) |
| `cmd/rkllm-probe` | scripted experiments against the raw runtime (the source of `docs/kv-cache-probe-findings.md`) |
| `cmd/rkllm-memtest` | NPU allocation limits |
| `internal/core` | pool, slots, cache strategies, sessions |
| `internal/prompt`, `internal/toolparse` | chat templates and tool-call parsers |
| `internal/engine` | runtime-neutral interface: cgo binding and a fake engine for tests |
| `docs/` | design, runtime probe findings, template notes, benchmarks |

## Status

Hobby project, tested on NanoPi M6 and R6S boards. Expect rough edges. Issues and pull requests are welcome.

## License

MIT for this repository (see `LICENSE`). Rockchip's RKLLM headers are not included. `make` downloads them unmodified from [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm) (`scripts/fetch-headers.sh`); see `NOTICE`.
