# rkllm-bench report

Generated 2026-10-10T11:45:20Z from `coding-off` (runs: 1010T1120-3432dd).

Times are wall-clock as seen by the client unless marked srv (server-reported `usage.rkllm`). Task wall = sum of the task's request times. Time to useful result = task wall up to the turn where the task's success condition first held (passed tasks only). Prefill/turn = tokens actually prefilled (not served from KV cache).

## Cache on vs off

| model | scenario | success on/off | task wall on / off (s) | wall speedup | TTFT on / off (ms) | TTFT speedup | prefill/turn on / off | prefill saved | paired time-to-success speedup (n) |
|---|---|---|---|---|---|---|---|---|---|
| q35-2b | coding-agent (smoke) | 0% / 0% | 52.3 / 168.4 | 3.22x | 2324 / 13901 | 5.98x | 212 / 1941 | 89% | - (0) |
| q35-4b | coding-agent (smoke) | 100% / 100% | 191.0 / 543.7 | 2.85x | 13588 / 63749 | 4.69x | 533 / 2918 | 82% | 2.59x (1) |
| qwen3-1.7b | coding-agent (smoke) | 0% / 0% | 64.5 / 364.0 | 5.64x | 812 / 19598 | 24.13x | 80 / 2513 | 97% | - (0) |

## Time to useful result

Mean seconds from the first request to the turn where the task was solved, over passed tasks (pass count in brackets).

| model | scenario | cache on | cache off |
|---|---|---|---|
| q35-2b | coding-agent (smoke) | never (0/1) | never (0/1) |
| q35-4b | coding-agent (smoke) | 167.5s (1/1) | 433.4s (1/1) |
| qwen3-1.7b | coding-agent (smoke) | never (0/1) | never (0/1) |

## q35-2b

| scenario | cache | pass | score | turns/task | task wall s (p50) | turn wall s | TTFT srv p50/p90 ms | TTFT client p50 ms | prefill/turn | cached/turn | hit | prefill t/s | decode t/s | max ctx | malformed | unknown | errors | cache events | queue s | load s | stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coding-agent (smoke) | on | 0/1 | 0.90 | 10.0 | 52.3 (52.3) | 5.2 | 963/4468 | 4086 | 212 | 1715 | 89% | 129.0 | 9.80 | 2898 | 0 | 0 | 0 | cold:1 hit:9 | 0.0 | - | max_turns:1 |
| coding-agent (smoke) | off | 0/1 | 0.90 | 10.0 | 168.4 (168.4) | 16.8 | 12076/20591 | 15963 | 1941 | 0 | 0% | 139.9 | 9.81 | 2915 | 0 | 0 | 0 | cold:10 | 0.0 | - | max_turns:1 |

## q35-4b

| scenario | cache | pass | score | turns/task | task wall s (p50) | turn wall s | TTFT srv p50/p90 ms | TTFT client p50 ms | prefill/turn | cached/turn | hit | prefill t/s | decode t/s | max ctx | malformed | unknown | errors | cache events | queue s | load s | stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coding-agent (smoke) | on | 1/1 | 1.00 | 7.0 | 191.0 (191.0) | 27.3 | 11009/34474 | 11224 | 533 | 2328 | 81% | 47.7 | 4.11 | 4501 | 0 | 0 | 0 | cold:1 hit:6 | 0.0 | - | done:1 |
| coding-agent (smoke) | off | 1/1 | 1.00 | 7.0 | 543.7 (543.7) | 77.7 | 67663/114159 | 78550 | 2918 | 0 | 0% | 45.8 | 3.91 | 4605 | 0 | 0 | 0 | cold:7 | 0.0 | - | done:1 |

## qwen3-1.7b

| scenario | cache | pass | score | turns/task | task wall s (p50) | turn wall s | TTFT srv p50/p90 ms | TTFT client p50 ms | prefill/turn | cached/turn | hit | prefill t/s | decode t/s | max ctx | malformed | unknown | errors | cache events | queue s | load s | stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coding-agent (smoke) | on | 0/1 | 0.90 | 10.0 | 64.5 (64.5) | 6.5 | 595/879 | 596 | 80 | 919 | 92% | 147.9 | 8.96 | 1337 | 0 | 0 | 0 | cold:1 hit:9 | 0.0 | - | max_turns:1 |
| coding-agent (smoke) | off | 0/1 | 0.90 | 10.0 | 364.0 (364.0) | 36.4 | 11000/40672 | 11000 | 2513 | 0 | 0% | 129.0 | 4.39 | 5084 | 0 | 0 | 0 | cold:10 | 0.0 | - | max_turns:1 |

## TTFT by cache event

`restored` rows in the interleaved scenario are the session park/restore cost.

| model | scenario | cache | event | n | TTFT srv mean ms | p50 | p90 | prefill mean | cached mean |
|---|---|---|---|---|---|---|---|---|---|
| q35-2b | coding-agent (smoke) | off | cold | 10 | 13901 | 12076 | 20591 | 1941 | 0 |
| q35-2b | coding-agent (smoke) | on | cold | 1 | 6183 | 6183 | 6183 | 34 | 773 |
| q35-2b | coding-agent (smoke) | on | hit | 9 | 1896 | 963 | 4468 | 232 | 1819 |
| q35-4b | coding-agent (smoke) | off | cold | 7 | 63749 | 67663 | 114159 | 2918 | 0 |
| q35-4b | coding-agent (smoke) | on | cold | 1 | 15761 | 15761 | 15761 | 34 | 773 |
| q35-4b | coding-agent (smoke) | on | hit | 6 | 13226 | 5789 | 34474 | 616 | 2587 |
| qwen3-1.7b | coding-agent (smoke) | off | cold | 10 | 19598 | 11000 | 40672 | 2513 | 0 |
| qwen3-1.7b | coding-agent (smoke) | on | cold | 1 | 2552 | 2552 | 2552 | 34 | 534 |
| qwen3-1.7b | coding-agent (smoke) | on | hit | 9 | 619 | 595 | 879 | 85 | 961 |

