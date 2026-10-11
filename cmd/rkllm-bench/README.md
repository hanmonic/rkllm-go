# rkllm-bench

Multi-turn agent benchmark for rkllm-go. Every task is a real agent loop: the harness sends the
conversation, executes the model's tool calls against a deterministic simulator, appends the
assistant turn (with its `tool_calls`) and the `tool` results to history, and repeats. Each
(model, scenario) runs with KV-cache reuse on (default) and off (`"cache": false` on every
request), so the report answers: how long until the agent produces a useful result, and does
caching change correctness? Stdlib only; the server is treated as a black box over HTTP.

## Scenarios

| scenario | tasks (smoke task first) | max turns | context growth |
|---|---|---|---|
| `house-batch` | S1_five, S2a_goodnight_open, S2b_goodnight_closed, S3_ten (port of toolbench3.py) | 12 | ~1.0k tokens, +100-300 per turn, 2-4 turns |
| `coding-agent` | bulk-discount, csv-stock, two-bugs: a 11-file Python repo (`scenarios/testdata/invoicer`) with injected bugs | 20 | ~0.5k at start; file reads 300-1100 tok, `run_tests` 450-1500 tok; 2-5k by the end (two-bugs exceeds 4096) |
| `hard-multistep` | answer_from_result, conditional_restart_yes, conditional_restart_no, injection (toolbench2.py MULTI) | 6 | ~0.9k, 2-3 turns |
| `interleaved` | 3 coding-agent sessions (2 in smoke), one turn each in round-robin, separate `X-Session-Id`s | 20 | as coding-agent; every switch forces a session park/restore |

Scoring:

- house-batch: unchanged from toolbench3.py (end-state checks, bath order, conditional status
  check, forbidden call, zero precondition/unknown-tool errors; python truthiness kept).
- hard-multistep: unchanged stage-1/stage-2 matching from toolbench2.py. A passing "call"
  stage 2 is executed and the loop continues until the model stops calling tools.
- coding-agent: `run_tests` is a deterministic semantic check of the edited source (the bulk
  condition is parsed and evaluated, the stock column index is resolved) and prints
  pytest-shaped output. Pass = all 20 simulated tests pass on the final files. Tests are
  read-only. "Solved" (time to useful result) = the turn where `run_tests` first passed.

Tool calls are taken only from the response's `tool_calls`. Raw tool-call text left in
`content` (`<tool_call>`, `<function=`, `call:x{`) and arguments that aren't a JSON object count
as `malformed`.

## Run

```sh
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o rkllm-bench ./cmd/rkllm-bench
scp rkllm-bench <board>:/opt/rkllm-bench/

./rkllm-bench run --server http://127.0.0.1:8081 --out /opt/rkllm-bench/res-$(date +%m%d) \
    --models qwen3-4b-16k,q35-4b --scenarios house-batch,coding-agent,hard-multistep,interleaved --modes on,off
./rkllm-bench run ... --smoke --out /tmp/smoke      # one short case per scenario
./rkllm-bench report DIR                             # rebuild summary.json + report.md
./rkllm-bench list                                   # scenarios and task ids
```

Flags: `--max-turns N` (overrides the per-scenario cap), `--timeout 30m` (per request),
`--ctx model=N,...` or `--ctx N` (context window; default is a context field from `/v1/models`
if present, else 16384 for `*16k*`, `q35*`, `*8b-allnpu`, otherwise 4096), `--seed 42`,
`--no-stream`, `--no-warmup`, `--parallel-interleave` (concurrent instead of round-robin).

Context budget: before each request the harness estimates the prompt (last `prompt_tokens` +
`completion_tokens` + new messages at 3 chars/token). Tool outputs are truncated to fit
`ctx - max_tokens`; when even that is impossible the task stops with `ctx_exhausted`.

Resume: results are written per combo; a combo is complete once `done/<key>.json` exists.
Rerunning with the same `--out` skips completed combos and reruns any combo that was interrupted.
Ctrl-C leaves no done marker for the combo in flight.

Each model gets one untimed warmup request (cache off) before its first pending combo, so
model load time isn't counted in the first task.

## Output (DIR)

- `turns/<model>__<scenario>__<mode>.jsonl`: one line per request. Contains client wall, TTFB,
  TTFT (first content/tool delta when streaming), the `usage` fields including the full
  `usage.rkllm`, hit ratio, decode tok/s, tool calls, malformed/unknown counts, truncation
  flag, estimated context, and whether the task was solved after this turn.
- `tasks/<key>.jsonl`: one line per task with pass/score/detail, stop reason, wall,
  time-to-success, token sums, and max context.
- `done/<key>.json`: resume markers. `state/<key>.json`: `/debug/state` snapshot after the combo.
- `all_turns.jsonl`, `summary.json`, and `report.md`. The report has a cache on/off speedup
  table, a time to useful result table, per-model tables, and TTFT by `cache_event`
  (`restored` rows = park/restore cost).

## Fake server

`rkllm-bench fakeserver --addr 127.0.0.1:8081 --time-scale 0.01` implements the contract with
synthetic RK3588-ish timing: prefix matching per session, one resident KV session per model
(`restored` on switch), model load on switch, and the queue. Its "model" plays each task's
scripted reference solution, so every task should pass against it (two-bugs on 4096-context
models ends `ctx_exhausted` by design). `--malformed-every N` injects broken tool calls.
The unit tests run the whole harness against it through `httptest`:

```sh
go test ./cmd/rkllm-bench/...
```
