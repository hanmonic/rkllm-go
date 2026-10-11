# rkllm-bench report

Generated 2026-10-11T03:10:58Z from `full-append` (runs: 1011T0020-7281f5).

Times are wall-clock as seen by the client unless marked srv (server-reported `usage.rkllm`). Task wall = sum of the task's request times. Time to useful result = task wall up to the turn where the task's success condition first held (passed tasks only). Prefill/turn = tokens actually prefilled (not served from KV cache).

## Time to useful result

Mean seconds from the first request to the turn where the task was solved, over passed tasks (pass count in brackets).

| model | scenario | cache on | cache off |
|---|---|---|---|
| q35-4b | coding-agent | 128.8s (1/3) | n/a |
| q35-4b | hard-multistep | 40.8s (3/4) | n/a |
| q35-4b | house-batch | 97.3s (3/4) | n/a |
| q35-4b | interleaved | 156.6s (1/3) | n/a |
| q35-9b | coding-agent | 1052.6s (2/3) | n/a |
| q35-9b | hard-multistep | 75.5s (3/4) | n/a |
| q35-9b | house-batch | 157.8s (2/4) | n/a |
| q35-9b | interleaved | 619.8s (3/3) | n/a |

## q35-4b

| scenario | cache | pass | score | turns/task | task wall s (p50) | turn wall s | TTFT srv p50/p90 ms | TTFT client p50 ms | prefill/turn | cached/turn | hit | prefill t/s | decode t/s | max ctx | malformed | unknown | errors | cache events | queue s | load s | stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coding-agent | on | 1/3 | 0.87 | 15.3 | 556.8 (636.5) | 36.3 | 3278/19112 | 3555 | 319 | 4805 | 94% | 39.6 | 3.27 | 9133 | 0 | 0 | 0 | cold:3 hit:40 restored:3 | 0.0 | - | done:1 max_turns:2 |
| hard-multistep | on | 3/4 | 0.75 | 2.0 | 37.7 (21.0) | 18.9 | 2274/60864 | 4186 | 60 | 769 | 93% | 25.3 | 3.89 | 1579 | 0 | 0 | 0 | cold:4 restored:4 | 0.0 | - | task_stop:4 |
| house-batch | on | 3/4 | 0.97 | 2.0 | 125.6 (115.8) | 62.8 | 7292/29009 | 13240 | 295 | 752 | 72% | 48.6 | 4.52 | 1606 | 0 | 0 | 0 | cold:4 restored:4 | 0.0 | - | done:4 |
| interleaved | on | 1/3 | 0.87 | 10.0 | 533.2 (330.5) | 53.3 | 15540/50268 | 24017 | 644 | 3377 | 84% | 25.7 | 3.17 | 7656 | 0 | 0 | 0 | cold:3 hit:12 restored:15 | 0.0 | - | done:2 max_turns:1 |

## q35-9b

| scenario | cache | pass | score | turns/task | task wall s (p50) | turn wall s | TTFT srv p50/p90 ms | TTFT client p50 ms | prefill/turn | cached/turn | hit | prefill t/s | decode t/s | max ctx | malformed | unknown | errors | cache events | queue s | load s | stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coding-agent | on | 2/3 | 0.97 | 15.0 | 906.1 (474.1) | 60.4 | 2669/47085 | 19905 | 330 | 2709 | 89% | 22.5 | 1.99 | 11909 | 0 | 0 | 0 | cold:3 hit:37 restored:5 | 0.0 | - | done:2 max_turns:1 |
| hard-multistep | on | 3/4 | 0.75 | 2.0 | 82.3 (61.9) | 41.1 | 3160/38165 | 3214 | 60 | 646 | 92% | 26.7 | 2.38 | 1333 | 0 | 0 | 0 | cold:4 restored:4 | 0.0 | - | task_stop:4 |
| house-batch | on | 2/4 | 0.92 | 2.0 | 267.8 (244.5) | 133.9 | 14478/38323 | 18564 | 270 | 639 | 70% | 29.9 | 2.26 | 1378 | 0 | 0 | 0 | cold:4 restored:4 | 0.0 | - | done:4 |
| interleaved | on | 3/3 | 1.00 | 11.3 | 690.5 (612.8) | 60.9 | 7654/43263 | 14424 | 460 | 3566 | 89% | 29.2 | 2.03 | 8070 | 0 | 0 | 0 | cold:3 restored:31 | 0.0 | - | done:3 |

## TTFT by cache event

`restored` rows in the interleaved scenario are the session park/restore cost.

| model | scenario | cache | event | n | TTFT srv mean ms | p50 | p90 | prefill mean | cached mean |
|---|---|---|---|---|---|---|---|---|---|
| q35-4b | coding-agent | on | cold | 3 | 1140 | 1133 | 1163 | 38 | 773 |
| q35-4b | coding-agent | on | hit | 40 | 7818 | 3278 | 15868 | 290 | 5468 |
| q35-4b | coding-agent | on | restored | 3 | 18685 | 15388 | 30022 | 989 | 0 |
| q35-4b | hard-multistep | on | cold | 4 | 16591 | 2213 | 60864 | 30 | 1538 |
| q35-4b | hard-multistep | on | restored | 4 | 3301 | 2579 | 4186 | 90 | 0 |
| q35-4b | house-batch | on | cold | 4 | 8984 | 2332 | 29009 | 78 | 1505 |
| q35-4b | house-batch | on | restored | 4 | 10510 | 10731 | 13239 | 513 | 0 |
| q35-4b | interleaved | on | cold | 3 | 1132 | 1141 | 1174 | 38 | 773 |
| q35-4b | interleaved | on | hit | 12 | 37282 | 8773 | 117712 | 504 | 4873 |
| q35-4b | interleaved | on | restored | 15 | 20463 | 20519 | 34716 | 876 | 2700 |
| q35-9b | coding-agent | on | cold | 3 | 7959 | 1747 | 20531 | 36 | 673 |
| q35-9b | coding-agent | on | hit | 37 | 16425 | 2556 | 48021 | 353 | 3221 |
| q35-9b | coding-agent | on | restored | 5 | 10011 | 5277 | 26905 | 334 | 142 |
| q35-9b | hard-multistep | on | cold | 4 | 10799 | 1852 | 38165 | 30 | 1292 |
| q35-9b | hard-multistep | on | restored | 4 | 3244 | 3214 | 3384 | 89 | 0 |
| q35-9b | house-batch | on | cold | 4 | 12059 | 3177 | 38323 | 77 | 1278 |
| q35-9b | house-batch | on | restored | 4 | 15594 | 14491 | 18564 | 464 | 0 |
| q35-9b | interleaved | on | cold | 3 | 1595 | 1611 | 1617 | 36 | 673 |
| q35-9b | interleaved | on | restored | 31 | 17593 | 9171 | 43263 | 501 | 3846 |

