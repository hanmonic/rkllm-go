# rkllm-go v2 benchmark — 2026-10-10

Production-style agent benchmark of rkllm-go v2 (KV-cache reuse) on the RK3588 NPU, runtime 1.3.1, a NanoPi M6, dev server on :8081. Full generated reports: `bench-2026-10-10/full-report.md` and `bench-2026-10-10/coding-off-report.md`.

## Setup

- Harness: `cmd/rkllm-bench`. Each task is a real multi-turn agent loop (tool calls executed against a simulated environment, results fed back) and is scored on final state, not on the format of individual calls.
- Scenarios:
  - `house-batch` (4 tasks): 5-10 home commands in one prompt, with tool preconditions.
  - `hard-multistep` (4): conditional chains, including a prompt-injection case.
  - `coding-agent` (3): read, edit and test a small repo, up to 20 turns and ~15k context.
  - `interleaved` (3): coding tasks run concurrently with separate sessions, so sessions are parked and restored.
- Models: qwen3-0.6b, qwen3-1.7b, qwen3-4b-16k, qwen3-8b-allnpu, q35-2b, q35-4b, q35-9b, gemma3n-e4b. gemma4-e2b was dropped because its conversion is broken.
- Cache strategy:
  - Qwen3 and Gemma use `prefix`.
  - Qwen3.5 uses `snapshot`.
  - Cache off means `"cache": false`, which is a cold prefill of the whole prompt every turn.
- Phases:
  1. All 4 scenarios with the cache on.
  2. `house-batch` and `hard-multistep` with the cache off.
  3. `coding-agent` with the cache on and off: smoke set, 10-turn cap, on 1.7b, q35-2b and q35-4b. A full coding run with the cache off was dropped because it costs ~225 s per turn at 9k tokens.
- Single deterministic run per task. n is small, so treat pass rates as indicative.

## Headline

1. **The cache saves 82-99% of prefill on every model and scenario.** Turns after the first prefill only the delta: about 40-500 tokens instead of 1.2k-4k.
2. **TTFT improves 3-7x on short agent tasks, and up to 24x on coding with the cache off as the baseline.** For qwen3-1.7b coding turns, the cached TTFT p50 is 0.6 s against 11 s cold, which is a 24x TTFT speedup, and wall time is 5.6x faster.
3. **End-to-end task time improves less (1.2-3x), because decode dominates.** Decode speed doesn't change: 1.5-16 tok/s depending on model size. Speedup grows with turns and context, so coding gains more than batch.
4. **Time to a correct result with the cache on versus off**, for pairs where both runs passed:
   - q35-2b hard-multistep: 11 s vs 30 s.
   - q35-9b hard-multistep: 58 s vs 124 s.
   - qwen3-8b hard-multistep: 30 s vs 64 s.
   - q35-4b coding: 168 s vs 433 s.

## Accuracy (cache on, tasks passed)

| model | house-batch /4 | hard-multistep /4 | coding /3 | interleaved /3 | total /14 |
|---|---|---|---|---|---|
| q35-9b | 3 | 4 | 3 | 3 | **13** |
| q35-4b | 3 | 3 | 1 | 1 | 8 |
| qwen3-8b-allnpu | 1 | 4 | 0 | 1 | 6 |
| qwen3-4b-16k | 3 | 1 | 1 | 0 | 5 |
| q35-2b | 0 | 4 | 0 | 0 | 4 |
| qwen3-0.6b | 0 | 3 | 0 | 0 | 3 |
| qwen3-1.7b | 0 | 1 | 0 | 0 | 1 |
| gemma3n-e4b | 0 | 0 | 0 | 0 | 0 |

q35-9b is the only model that completes the coding tasks. It is slow, at 2.1 tok/s decode and ~10 min per coding task, but it is reliable.

## Speed (cached TTFT p50, decode)

| model | cached TTFT p50 (hit turns) | cold TTFT p50 (~1.2-1.6k prompt) | decode tok/s |
|---|---|---|---|
| qwen3-0.6b | 0.4-1.6 s | 3.5-6.3 s | 13-16 |
| qwen3-1.7b | 0.6-1.3 s | 5.7-9.3 s | 6-8 |
| q35-2b | 1.0-1.9 s | 11-22 s | 8-10 |
| qwen3-4b-16k | 1.1-6.0 s | 13.5-13.8 s | 3.3-4.5 |
| q35-4b | 2.5-13 s | 30-31 s | 3-4.4 |
| qwen3-8b-allnpu | 1.8-7.6 s | 20-24 s | 1.5-2.7 |
| q35-9b | 3.2-14 s | 37-45 s | 2.1-2.5 |
| gemma3n-e4b | 4.8 s | 30-31 s | 3.7-4 |

Session park/restore, from the `restored` event in `interleaved`, costs little on Qwen3: 1.9 s p50 restored vs 1.0 s hit on 1.7b, with ~130 tokens re-prefilled.

## Issues found

1. **Qwen3.5 snapshot mode re-prefills the previous generation on live sessions.** Cached TTFT on q35-4b and q35-9b is 9-13 s, against 1.8-5 s for Qwen3 models of similar size. The proposed fix is to use `append` while the session stays live and to snapshot only when it is parked. Not implemented yet.
2. **Cache on vs off changes outcomes for some models**: qwen3-1.7b hard-multistep passed 1/4 with the cache on and 3/4 with it off; qwen3-0.6b was the reverse at 3/4 vs 1/4. Decoding is greedy, so the difference comes from numerics: chunked/incremental prefill does not produce bit-identical logits. On small models this flips borderline decisions. The cache does not bias results consistently in one direction.
3. **gemma3n-e4b house-batch with the cache on**: the client TTFT p50 is 306 s while the server's is 2.5 s. One task produced a very long generation before the first parsed output. Not investigated. Gemma3n fails every scenario either way, so this has no practical effect.
4. **q35-2b in coding** hits unknown tools 9 times and exhausts its context once. Small Qwen3.5 models degenerate into loops on long tasks.

## Recommendation

| use | model | why |
|---|---|---|
| Fast home control (short, mostly single action) | qwen3-1.7b or q35-2b | ~1 s cached TTFT; good on simple and conditional calls |
| Batch house commands | qwen3-4b-16k | 3/4 like q35-4b and q35-9b, but ~2x faster per task |
| Hard or coding agent work | q35-9b | 13/14; slow but the only model that reliably finishes coding |

After the Qwen3.5 live-append fix, re-run q35-4b and q35-9b. Their cached TTFT should fall towards the Qwen3 numbers, which would make q35-4b the best all-round choice.

## Update 2026-10-11: Qwen3.5 live-append fix

`snapshotAppend` in `internal/core/slot.go` (disable with `--no-live-append`). The first turn of a session is a keep_history=0 run that saves a snapshot, as before. On the second turn the snapshot is loaded and the rest of the prompt is sent as a keep_history=1 delta (probe on q35-2b: the loaded state is attended by a kh=1 run). From the third turn on, deltas are appended to the live cache, so the previous reply is no longer re-prefilled. An unmatched history falls back to the snapshot restore. Parked sessions still restore from the older snapshot and re-prefill the tail.

Bug found on the way: `clear()` drops the live-session marker, which made the first version never append (all turns `restored`). Fixed with `setLive` after the clear.

Re-run: q35-4b and q35-9b, cache on, all four scenarios (`bench-2026-10-11-append/report.md`).

| metric | before | after |
|---|---|---|
| q35-4b coding cached TTFT p50 (hit turns) | 13.1 s | 3.3 s |
| q35-9b coding cached TTFT p50 (hit turns) | 9.1 s | 2.6 s |
| q35-4b coding prefill per hit turn (mean) | 561 tok | 290 tok |
| q35-4b coding pass | 1/3 | 1/3 |
| q35-9b coding pass | 3/3 | 2/3 (bulk-discount hit the 20-turn cap at score 0.90) |
| q35-9b interleaved pass | 3/3 | 3/3 |
| q35-4b / q35-9b hard-multistep | 3/4 / 4/4 | 3/4 / 3/4 |
| q35-4b / q35-9b house-batch | 3/4 / 3/4 | 3/4 / 2/4 |

- Coding turns with many steps gain the most (3-4x lower TTFT on `hit` turns, hits are 40 of 46 turns for q35-4b). Two-turn scenarios (house-batch, hard-multistep) gain nothing, because the second turn is the snapshot-to-append conversion and still prefills the reply.
- Interleaved is dominated by park/restore (31 of 34 q35-9b turns `restored`, p50 9.2 s), which the fix does not change. Parked sessions have no fresh snapshot; saving one at park time (load old snapshot, kh=0 run over the cached text, save) is the next step.
- Accuracy deltas are within single-run noise (greedy decoding, n=3-4) but three tasks moved down by one (q35-9b hard-multistep, house-batch, coding). Treat as a possible small cost of the load-then-kh=1 conversion and re-run before relying on q35-9b for batch work.
- Not committed.

### Accuracy control (2026-10-11, q35-9b, house-batch + hard-multistep, cache on)

| run | hard-multistep | house-batch |
|---|---|---|
| live-append, run 1 | 3/4 | 2/4 |
| live-append, run 2 (identical) | 3/4 | 2/4 |
| `--no-live-append` (control) | 4/4 | 3/4 |

Decoding is greedy, so the two append runs reproduce exactly and the drop is real, not noise. The failing tasks stop one turn early (2 turns instead of 3), so the damage is on the conversion turn (snapshot loaded, then a kh=1 suffix). Suspect: after `rkllm_load_prompt_cache` the runtime reports kv_size without the loaded tokens (probe Q7: kv 30 for a 24-token suffix on a 26-token snapshot), so positions of the delta may be off. Pure kh=1-to-kh=1 appends (3rd turn on) were not separately tested for accuracy.

Decision: live-append is now opt-in (`--live-append`, default off). Raw data: the two noise runs (append and control) from the benchmark harness. Next: (1) test whether turn-3+ appends are clean by starting a session in kh=1 with no snapshot load; (2) if so, start long-running Qwen3.5 sessions in kh=1 and keep snapshots for short ones.
