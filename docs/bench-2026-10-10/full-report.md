# rkllm-bench report

Generated 2026-10-10T11:44:54Z from `full` (runs: 1010T0133-3fed39, 1010T0812-1d9761).

Times are wall-clock as seen by the client unless marked srv (server-reported `usage.rkllm`). Task wall = sum of the task's request times. Time to useful result = task wall up to the turn where the task's success condition first held (passed tasks only). Prefill/turn = tokens actually prefilled (not served from KV cache).

## Cache on vs off

| model | scenario | success on/off | task wall on / off (s) | wall speedup | TTFT on / off (ms) | TTFT speedup | prefill/turn on / off | prefill saved | paired time-to-success speedup (n) |
|---|---|---|---|---|---|---|---|---|---|
| gemma3n-e4b | hard-multistep | 0% / 50% | 19.5 / 184.8 | 9.49x | 8821 / 33026 | 3.74x | 18 / 1998 | 99% | - (0) |
| gemma3n-e4b | house-batch | 0% / 0% | 422.4 / 189.8 | 0.45x | 12731 / 35533 | 2.79x | 330 / 1878 | 82% | - (0) |
| q35-2b | hard-multistep | 100% / 100% | 11.6 / 33.2 | 2.87x | 1974 / 11285 | 5.72x | 61 / 1630 | 96% | 2.71x (4) |
| q35-2b | house-batch | 0% / 0% | 464.5 / 982.8 | 2.12x | 10057 / 39643 | 3.94x | 566 / 4187 | 86% | - (0) |
| q35-4b | hard-multistep | 75% / 75% | 39.9 / 123.5 | 3.10x | 8723 / 35276 | 4.04x | 65 / 1631 | 96% | 2.11x (3) |
| q35-4b | house-batch | 75% / 75% | 126.1 / 179.2 | 1.42x | 10366 / 38174 | 3.68x | 295 / 1839 | 84% | 1.16x (3) |
| q35-9b | hard-multistep | 100% / 100% | 66.2 / 142.8 | 2.16x | 6483 / 37630 | 5.80x | 66 / 1388 | 95% | 2.13x (4) |
| q35-9b | house-batch | 75% / 75% | 217.4 / 304.3 | 1.40x | 12604 / 48983 | 3.89x | 246 / 1624 | 85% | 1.25x (3) |
| qwen3-0.6b | hard-multistep | 75% / 25% | 6.8 / 8.9 | 1.30x | 866 / 3513 | 4.06x | 52 / 1175 | 96% | 2.77x (1) |
| qwen3-0.6b | house-batch | 0% / 0% | 21.6 / 42.3 | 1.96x | 1507 / 6492 | 4.31x | 192 / 1351 | 86% | - (0) |
| qwen3-1.7b | hard-multistep | 25% / 75% | 9.9 / 21.5 | 2.17x | 2041 / 5712 | 2.80x | 40 / 1194 | 97% | 1.00x (1) |
| qwen3-1.7b | house-batch | 0% / 0% | 111.7 / 109.2 | 0.98x | 1373 / 9590 | 6.99x | 133 / 1657 | 92% | - (0) |
| qwen3-4b-16k | hard-multistep | 25% / 50% | 16.6 / 45.3 | 2.72x | 3112 / 14346 | 4.61x | 39 / 1190 | 97% | - (0) |
| qwen3-4b-16k | house-batch | 75% / 75% | 106.9 / 123.7 | 1.16x | 5818 / 17680 | 3.04x | 220 / 1366 | 84% | 1.02x (2) |
| qwen3-8b-allnpu | hard-multistep | 100% / 75% | 32.7 / 70.3 | 2.15x | 3614 / 21041 | 5.82x | 49 / 1191 | 96% | 2.12x (3) |
| qwen3-8b-allnpu | house-batch | 25% / 50% | 135.5 / 174.7 | 1.29x | 7604 / 25174 | 3.31x | 207 / 1370 | 85% | 1.06x (1) |

## Time to useful result

Mean seconds from the first request to the turn where the task was solved, over passed tasks (pass count in brackets).

| model | scenario | cache on | cache off |
|---|---|---|---|
| gemma3n-e4b | coding-agent | never (0/3) | n/a |
| gemma3n-e4b | hard-multistep | never (0/4) | 200.0s (2/4) |
| gemma3n-e4b | house-batch | never (0/4) | never (0/4) |
| gemma3n-e4b | interleaved | never (0/3) | n/a |
| q35-2b | coding-agent | never (0/3) | n/a |
| q35-2b | hard-multistep | 11.0s (4/4) | 29.8s (4/4) |
| q35-2b | house-batch | never (0/4) | never (0/4) |
| q35-2b | interleaved | never (0/3) | n/a |
| q35-4b | coding-agent | 168.3s (1/3) | n/a |
| q35-4b | hard-multistep | 43.4s (3/4) | 91.4s (3/4) |
| q35-4b | house-batch | 97.8s (3/4) | 113.4s (3/4) |
| q35-4b | interleaved | 158.1s (1/3) | n/a |
| q35-9b | coding-agent | 621.7s (3/3) | n/a |
| q35-9b | hard-multistep | 58.4s (4/4) | 124.1s (4/4) |
| q35-9b | house-batch | 159.8s (3/4) | 200.4s (3/4) |
| q35-9b | interleaved | 616.6s (3/3) | n/a |
| qwen3-0.6b | coding-agent | never (0/3) | n/a |
| qwen3-0.6b | hard-multistep | 6.0s (3/4) | 9.5s (1/4) |
| qwen3-0.6b | house-batch | never (0/4) | never (0/4) |
| qwen3-0.6b | interleaved | never (0/3) | n/a |
| qwen3-1.7b | coding-agent | never (0/3) | n/a |
| qwen3-1.7b | hard-multistep | 18.3s (1/4) | 18.3s (3/4) |
| qwen3-1.7b | house-batch | never (0/4) | never (0/4) |
| qwen3-1.7b | interleaved | never (0/3) | n/a |
| qwen3-4b-16k | coding-agent | 118.4s (1/3) | n/a |
| qwen3-4b-16k | hard-multistep | 25.0s (1/4) | 40.7s (2/4) |
| qwen3-4b-16k | house-batch | 75.7s (3/4) | 78.5s (3/4) |
| qwen3-4b-16k | interleaved | never (0/3) | n/a |
| qwen3-8b-allnpu | coding-agent | never (0/3) | n/a |
| qwen3-8b-allnpu | hard-multistep | 30.3s (4/4) | 64.1s (3/4) |
| qwen3-8b-allnpu | house-batch | 85.1s (1/4) | 105.1s (2/4) |
| qwen3-8b-allnpu | interleaved | 395.4s (1/3) | n/a |

## gemma3n-e4b

| scenario | cache | pass | score | turns/task | task wall s (p50) | turn wall s | TTFT srv p50/p90 ms | TTFT client p50 ms | prefill/turn | cached/turn | hit | prefill t/s | decode t/s | max ctx | malformed | unknown | errors | cache events | queue s | load s | stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coding-agent | on | 0/3 | 0.80 | 4.0 | 267.8 (317.1) | 67.0 | 4833/13464 | 4881 | 337 | 990 | 75% | 58.5 | 3.80 | 2117 | 0 | 0 | 0 | cold:3 hit:7 partial:2 | 0.0 | - | done:3 |
| hard-multistep | on | 0/4 | 0.00 | 1.0 | 19.5 (12.9) | 19.5 | 909/32342 | 2902 | 18 | 1928 | 99% | 36.6 | 4.02 | 1956 | 4 | 0 | 0 | cold:4 | 0.0 | - | task_stop:4 |
| hard-multistep | off | 2/4 | 0.50 | 1.5 | 184.8 (186.5) | 123.2 | 31337/36522 | 34330 | 1998 | 0 | 0% | 60.7 | 3.89 | 2114 | 0 | 0 | 0 | cold:6 | 0.0 | - | task_stop:4 |
| house-batch | on | 0/4 | 0.13 | 1.2 | 422.4 (307.4) | 337.9 | 2507/30432 | 306093 | 330 | 1828 | 85% | 51.2 | 3.69 | 3275 | 2 | 0 | 0 | cold:4 hit:1 | 0.0 | - | done:4 |
| house-batch | off | 0/4 | 0.03 | 1.0 | 189.8 (33.9) | 189.8 | 29978/52620 | 29989 | 1878 | 0 | 0% | 53.0 | 3.94 | 1902 | 1 | 0 | 0 | cold:4 | 0.0 | - | done:4 |
| interleaved | on | 0/3 | 0.80 | 3.7 | 348.7 (351.0) | 95.1 | 5182/13064 | 5801 | 348 | 934 | 73% | 53.6 | 3.75 | 2108 | 0 | 0 | 0 | cold:3 restored:8 | 0.0 | - | done:3 |

## q35-2b

| scenario | cache | pass | score | turns/task | task wall s (p50) | turn wall s | TTFT srv p50/p90 ms | TTFT client p50 ms | prefill/turn | cached/turn | hit | prefill t/s | decode t/s | max ctx | malformed | unknown | errors | cache events | queue s | load s | stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coding-agent | on | 0/3 | 0.80 | 19.7 | 223.8 (269.8) | 11.4 | 1424/33654 | 4382 | 486 | 4283 | 90% | 62.2 | 9.28 | 15416 | 0 | 9 | 0 | cold:3 hit:56 | 0.0 | - | ctx_exhausted:1 max_turns:2 |
| hard-multistep | on | 4/4 | 1.00 | 2.2 | 11.6 (8.4) | 5.1 | 968/11493 | 3171 | 61 | 1565 | 96% | 92.7 | 10.11 | 1742 | 0 | 0 | 0 | cold:4 hit:5 | 0.0 | - | done:1 task_stop:3 |
| hard-multistep | off | 4/4 | 1.00 | 2.2 | 33.2 (28.6) | 14.7 | 11378/12300 | 12300 | 1630 | 0 | 0% | 144.7 | 9.81 | 1769 | 0 | 0 | 0 | cold:9 | 0.0 | - | done:1 task_stop:3 |
| house-batch | on | 0/4 | 0.53 | 6.8 | 464.5 (57.4) | 68.8 | 1881/35396 | 8929 | 566 | 3479 | 86% | 62.7 | 8.00 | 14223 | 0 | 6 | 0 | cold:4 hit:23 | 0.0 | - | ctx_exhausted:1 done:2 max_turns:1 |
| house-batch | off | 0/4 | 0.51 | 11.5 | 982.8 (378.0) | 85.5 | 21701/74811 | 44298 | 4187 | 0 | 0% | 105.9 | 8.28 | 14201 | 0 | 0 | 0 | cold:46 | 0.0 | - | ctx_exhausted:1 max_turns:3 |
| interleaved | on | 0/3 | 0.80 | 19.7 | 221.2 (259.8) | 11.2 | 1311/31034 | 4314 | 486 | 4283 | 90% | 62.1 | 9.34 | 15416 | 0 | 9 | 0 | cold:3 restored:56 | 0.0 | - | ctx_exhausted:1 max_turns:2 |

## q35-4b

| scenario | cache | pass | score | turns/task | task wall s (p50) | turn wall s | TTFT srv p50/p90 ms | TTFT client p50 ms | prefill/turn | cached/turn | hit | prefill t/s | decode t/s | max ctx | malformed | unknown | errors | cache events | queue s | load s | stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coding-agent | on | 1/3 | 0.87 | 10.0 | 518.9 (302.5) | 51.9 | 13041/34980 | 13687 | 509 | 5388 | 91% | 31.6 | 3.02 | 10256 | 0 | 0 | 0 | cold:3 hit:27 | 0.0 | - | done:2 max_turns:1 |
| hard-multistep | on | 3/4 | 0.75 | 2.2 | 39.9 (21.9) | 17.7 | 2228/61131 | 8939 | 65 | 1565 | 96% | 34.1 | 4.31 | 1770 | 0 | 0 | 0 | cold:4 hit:5 | 0.0 | - | done:1 task_stop:3 |
| hard-multistep | off | 3/4 | 0.75 | 2.2 | 123.5 (118.0) | 54.9 | 30944/60644 | 36120 | 1631 | 0 | 0% | 46.3 | 4.52 | 1772 | 0 | 0 | 0 | cold:9 | 0.0 | - | done:1 task_stop:3 |
| house-batch | on | 3/4 | 0.97 | 2.0 | 126.1 (117.4) | 63.1 | 7264/29712 | 17389 | 295 | 1544 | 84% | 44.8 | 4.42 | 2274 | 0 | 0 | 0 | cold:4 hit:4 | 0.0 | - | done:4 |
| house-batch | off | 3/4 | 0.97 | 2.0 | 179.2 (169.2) | 89.6 | 29751/73366 | 73367 | 1839 | 0 | 0% | 48.2 | 4.44 | 2275 | 0 | 0 | 0 | cold:8 | 0.0 | - | done:4 |
| interleaved | on | 1/3 | 0.87 | 10.0 | 528.3 (303.9) | 52.8 | 11061/30658 | 13562 | 509 | 5388 | 91% | 31.9 | 3.33 | 10256 | 0 | 0 | 0 | cold:3 hit:13 restored:14 | 0.0 | - | done:2 max_turns:1 |

## q35-9b

| scenario | cache | pass | score | turns/task | task wall s (p50) | turn wall s | TTFT srv p50/p90 ms | TTFT client p50 ms | prefill/turn | cached/turn | hit | prefill t/s | decode t/s | max ctx | malformed | unknown | errors | cache events | queue s | load s | stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coding-agent | on | 3/3 | 1.00 | 11.3 | 702.5 (608.7) | 62.0 | 8802/43875 | 15421 | 460 | 3803 | 89% | 29.3 | 2.08 | 8239 | 0 | 0 | 0 | cold:3 hit:31 | 0.0 | - | done:3 |
| hard-multistep | on | 4/4 | 1.00 | 2.2 | 66.2 (46.9) | 29.4 | 3207/36858 | 3612 | 66 | 1320 | 95% | 28.6 | 2.49 | 1548 | 0 | 0 | 0 | cold:4 hit:5 | 0.0 | - | done:1 task_stop:3 |
| hard-multistep | off | 4/4 | 1.00 | 2.2 | 142.8 (119.3) | 63.5 | 37528/42500 | 37699 | 1388 | 0 | 0% | 36.9 | 2.47 | 1553 | 0 | 0 | 0 | cold:9 | 0.0 | - | done:1 task_stop:3 |
| house-batch | on | 3/4 | 0.94 | 2.2 | 217.4 (205.9) | 96.6 | 10122/36128 | 26482 | 246 | 1377 | 85% | 28.9 | 2.41 | 1913 | 0 | 0 | 0 | cold:4 hit:5 | 0.0 | - | done:4 |
| house-batch | off | 3/4 | 0.94 | 2.2 | 304.3 (302.5) | 135.3 | 44881/82666 | 82667 | 1624 | 0 | 0% | 33.2 | 2.41 | 1914 | 0 | 0 | 0 | cold:9 | 0.0 | - | done:4 |
| interleaved | on | 3/3 | 1.00 | 11.3 | 697.4 (603.2) | 61.5 | 7230/43643 | 15264 | 460 | 3803 | 89% | 27.6 | 2.12 | 8239 | 0 | 0 | 0 | cold:3 hit:1 restored:30 | 0.0 | - | done:3 |

## qwen3-0.6b

| scenario | cache | pass | score | turns/task | task wall s (p50) | turn wall s | TTFT srv p50/p90 ms | TTFT client p50 ms | prefill/turn | cached/turn | hit | prefill t/s | decode t/s | max ctx | malformed | unknown | errors | cache events | queue s | load s | stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coding-agent | on | 0/3 | 0.80 | 1.7 | 3.7 (2.9) | 2.2 | 370/1793 | 1205 | 67 | 582 | 90% | 261.7 | 16.19 | 810 | 0 | 0 | 0 | cold:3 hit:2 | 0.0 | - | done:3 |
| hard-multistep | on | 3/4 | 0.75 | 2.0 | 6.8 (6.2) | 3.4 | 425/3945 | 568 | 52 | 1144 | 96% | 179.6 | 13.61 | 1337 | 0 | 0 | 0 | cold:4 hit:4 | 0.0 | - | done:1 task_stop:3 |
| hard-multistep | off | 1/4 | 0.25 | 1.8 | 8.9 (9.5) | 5.1 | 3481/3727 | 4705 | 1175 | 0 | 0% | 344.5 | 13.80 | 1222 | 0 | 0 | 0 | cold:7 | 0.0 | - | task_stop:4 |
| house-batch | on | 0/4 | 0.53 | 2.0 | 21.6 (18.9) | 10.8 | 1379/3806 | 1801 | 192 | 1153 | 86% | 206.4 | 13.02 | 1569 | 0 | 0 | 0 | cold:4 hit:4 | 0.0 | - | done:4 |
| house-batch | off | 0/4 | 0.49 | 2.0 | 42.3 (32.4) | 21.2 | 6269/9271 | 9272 | 1351 | 0 | 0% | 211.4 | 8.98 | 1583 | 0 | 0 | 0 | cold:8 | 0.0 | - | done:4 |
| interleaved | on | 0/3 | 0.80 | 1.7 | 5.3 (2.5) | 3.2 | 315/769 | 1169 | 67 | 581 | 90% | 228.4 | 12.61 | 809 | 1 | 0 | 0 | cold:3 hit:1 restored:1 | 0.0 | - | done:3 |

## qwen3-1.7b

| scenario | cache | pass | score | turns/task | task wall s (p50) | turn wall s | TTFT srv p50/p90 ms | TTFT client p50 ms | prefill/turn | cached/turn | hit | prefill t/s | decode t/s | max ctx | malformed | unknown | errors | cache events | queue s | load s | stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coding-agent | on | 0/3 | 0.80 | 20.0 | 216.6 (217.8) | 10.8 | 1043/2000 | 1276 | 146 | 1894 | 93% | 108.2 | 6.28 | 4525 | 0 | 0 | 0 | cold:3 hit:57 | 0.0 | - | max_turns:3 |
| hard-multistep | on | 1/4 | 0.25 | 1.5 | 9.9 (7.1) | 6.6 | 582/9541 | 4009 | 40 | 1126 | 97% | 116.8 | 8.27 | 1213 | 0 | 0 | 0 | cold:4 hit:2 | 0.0 | - | task_stop:4 |
| hard-multistep | off | 3/4 | 0.75 | 2.2 | 21.5 (18.4) | 9.5 | 5681/6280 | 5916 | 1194 | 0 | 0% | 212.7 | 8.18 | 1290 | 0 | 0 | 0 | cold:9 | 0.0 | - | done:1 task_stop:3 |
| house-batch | on | 0/4 | 0.88 | 7.2 | 111.7 (63.3) | 15.4 | 1270/3128 | 15486 | 133 | 1816 | 93% | 109.3 | 6.26 | 3124 | 0 | 14 | 0 | cold:4 hit:25 | 0.0 | - | done:2 max_turns:2 |
| house-batch | off | 0/4 | 0.73 | 5.5 | 109.2 (66.5) | 19.9 | 9315/12394 | 15085 | 1657 | 0 | 0% | 175.1 | 6.55 | 2140 | 0 | 0 | 0 | cold:22 | 0.0 | - | done:3 max_turns:1 |
| interleaved | on | 0/3 | 0.80 | 20.0 | 285.9 (281.9) | 14.3 | 1841/2567 | 1868 | 130 | 1865 | 93% | 117.1 | 6.08 | 3348 | 0 | 0 | 0 | cold:3 restored:57 | 0.0 | - | max_turns:3 |

## qwen3-4b-16k

| scenario | cache | pass | score | turns/task | task wall s (p50) | turn wall s | TTFT srv p50/p90 ms | TTFT client p50 ms | prefill/turn | cached/turn | hit | prefill t/s | decode t/s | max ctx | malformed | unknown | errors | cache events | queue s | load s | stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coding-agent | on | 1/3 | 0.87 | 3.3 | 53.8 (14.2) | 16.1 | 3468/9209 | 6439 | 278 | 1285 | 82% | 54.7 | 3.28 | 3255 | 0 | 0 | 0 | cold:3 hit:7 | 0.0 | - | done:3 |
| hard-multistep | on | 1/4 | 0.25 | 1.5 | 16.6 (13.8) | 11.1 | 1103/13585 | 7766 | 39 | 1126 | 97% | 60.1 | 4.48 | 1218 | 0 | 0 | 0 | cold:4 hit:2 | 0.0 | - | task_stop:4 |
| hard-multistep | off | 2/4 | 0.50 | 2.0 | 45.3 (38.4) | 22.6 | 13843/19900 | 19182 | 1190 | 0 | 0% | 83.6 | 4.43 | 1290 | 0 | 0 | 0 | cold:8 | 0.0 | - | done:1 task_stop:3 |
| house-batch | on | 3/4 | 1.00 | 2.0 | 106.9 (94.8) | 53.5 | 3581/13330 | 10610 | 220 | 1153 | 84% | 55.9 | 3.67 | 1684 | 0 | 0 | 0 | cold:4 hit:4 | 0.0 | - | done:4 |
| house-batch | off | 3/4 | 0.97 | 2.0 | 123.7 (111.8) | 61.8 | 13541/34868 | 34869 | 1366 | 0 | 0% | 77.8 | 3.82 | 1684 | 0 | 0 | 0 | cold:8 | 0.0 | - | done:4 |
| interleaved | on | 0/3 | 0.80 | 7.3 | 156.0 (8.6) | 21.3 | 2727/5717 | 2728 | 149 | 2285 | 94% | 45.7 | 2.43 | 3759 | 0 | 0 | 0 | cold:3 hit:18 restored:1 | 0.0 | - | done:2 max_turns:1 |

## qwen3-8b-allnpu

| scenario | cache | pass | score | turns/task | task wall s (p50) | turn wall s | TTFT srv p50/p90 ms | TTFT client p50 ms | prefill/turn | cached/turn | hit | prefill t/s | decode t/s | max ctx | malformed | unknown | errors | cache events | queue s | load s | stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coding-agent | on | 0/3 | 0.87 | 20.0 | 635.1 (638.0) | 31.8 | 4930/31921 | 18148 | 274 | 2436 | 90% | 24.9 | 1.67 | 7441 | 0 | 0 | 0 | cold:3 hit:57 | 0.0 | - | max_turns:3 |
| hard-multistep | on | 4/4 | 1.00 | 2.2 | 32.7 (30.7) | 14.5 | 1757/19822 | 10559 | 49 | 1143 | 96% | 40.2 | 2.72 | 1290 | 0 | 0 | 0 | cold:4 hit:5 | 0.0 | - | done:1 task_stop:3 |
| hard-multistep | off | 3/4 | 0.75 | 2.2 | 70.3 (62.6) | 31.3 | 20320/30001 | 29282 | 1191 | 0 | 0% | 57.0 | 2.66 | 1290 | 0 | 0 | 0 | cold:9 | 0.0 | - | done:1 task_stop:3 |
| house-batch | on | 1/4 | 0.82 | 2.0 | 135.5 (120.2) | 67.7 | 4923/19674 | 13638 | 207 | 1153 | 85% | 42.1 | 2.45 | 1588 | 0 | 0 | 0 | cold:4 hit:4 | 0.0 | - | done:4 |
| house-batch | off | 2/4 | 0.92 | 2.0 | 174.7 (150.2) | 87.3 | 23592/40216 | 40217 | 1370 | 0 | 0% | 54.7 | 2.54 | 1712 | 0 | 0 | 0 | cold:8 | 0.0 | - | done:4 |
| interleaved | on | 1/3 | 0.87 | 15.3 | 618.2 (419.1) | 40.3 | 6100/19019 | 16347 | 264 | 2411 | 90% | 30.0 | 1.48 | 6485 | 0 | 0 | 0 | cold:3 hit:3 restored:40 | 0.0 | - | done:2 max_turns:1 |

## TTFT by cache event

`restored` rows in the interleaved scenario are the session park/restore cost.

| model | scenario | cache | event | n | TTFT srv mean ms | p50 | p90 | prefill mean | cached mean |
|---|---|---|---|---|---|---|---|---|---|
| gemma3n-e4b | coding-agent | on | cold | 3 | 5182 | 1138 | 13464 | 31 | 803 |
| gemma3n-e4b | coding-agent | on | hit | 7 | 6586 | 4833 | 17152 | 379 | 1109 |
| gemma3n-e4b | coding-agent | on | partial | 2 | 10775 | 8189 | 13362 | 651 | 853 |
| gemma3n-e4b | hard-multistep | off | cold | 6 | 33026 | 31337 | 36522 | 1998 | 0 |
| gemma3n-e4b | hard-multistep | on | cold | 4 | 8821 | 909 | 32342 | 18 | 1928 |
| gemma3n-e4b | house-batch | off | cold | 4 | 35533 | 29978 | 52620 | 1878 | 0 |
| gemma3n-e4b | house-batch | on | cold | 4 | 8985 | 2259 | 30432 | 55 | 1824 |
| gemma3n-e4b | house-batch | on | hit | 1 | 27717 | 27717 | 27717 | 1429 | 1846 |
| gemma3n-e4b | interleaved | on | cold | 3 | 908 | 912 | 957 | 30 | 804 |
| gemma3n-e4b | interleaved | on | restored | 8 | 9147 | 5422 | 24159 | 468 | 982 |
| q35-2b | coding-agent | on | cold | 3 | 2451 | 775 | 6093 | 38 | 773 |
| q35-2b | coding-agent | on | hit | 56 | 8414 | 1424 | 33654 | 510 | 4471 |
| q35-2b | hard-multistep | off | cold | 9 | 11285 | 11378 | 12300 | 1630 | 0 |
| q35-2b | hard-multistep | on | cold | 4 | 3222 | 486 | 11493 | 30 | 1538 |
| q35-2b | hard-multistep | on | hit | 5 | 975 | 977 | 1007 | 86 | 1586 |
| q35-2b | house-batch | off | cold | 46 | 39643 | 21701 | 74811 | 4187 | 0 |
| q35-2b | house-batch | on | cold | 4 | 6468 | 962 | 23005 | 78 | 1505 |
| q35-2b | house-batch | on | hit | 23 | 10681 | 1899 | 35396 | 650 | 3822 |
| q35-2b | interleaved | on | cold | 3 | 435 | 432 | 445 | 38 | 773 |
| q35-2b | interleaved | on | restored | 56 | 8406 | 1405 | 31034 | 510 | 4471 |
| q35-4b | coding-agent | on | cold | 3 | 6089 | 1157 | 16006 | 38 | 773 |
| q35-4b | coding-agent | on | hit | 27 | 18321 | 13060 | 40319 | 561 | 5901 |
| q35-4b | hard-multistep | off | cold | 9 | 35276 | 30944 | 60644 | 1631 | 0 |
| q35-4b | hard-multistep | on | cold | 4 | 16151 | 1235 | 61131 | 30 | 1538 |
| q35-4b | hard-multistep | on | hit | 5 | 2781 | 2524 | 4207 | 94 | 1586 |
| q35-4b | house-batch | off | cold | 8 | 38174 | 29751 | 73366 | 1839 | 0 |
| q35-4b | house-batch | on | cold | 4 | 9165 | 2277 | 29712 | 78 | 1505 |
| q35-4b | house-batch | on | hit | 4 | 11567 | 10930 | 17059 | 512 | 1582 |
| q35-4b | interleaved | on | cold | 3 | 1185 | 1161 | 1293 | 38 | 773 |
| q35-4b | interleaved | on | hit | 13 | 13160 | 11061 | 18319 | 215 | 9140 |
| q35-4b | interleaved | on | restored | 14 | 22755 | 19596 | 40065 | 883 | 2893 |
| q35-9b | coding-agent | on | cold | 3 | 7745 | 1579 | 20108 | 36 | 673 |
| q35-9b | coding-agent | on | hit | 31 | 17536 | 9099 | 43875 | 501 | 4106 |
| q35-9b | hard-multistep | off | cold | 9 | 37630 | 37528 | 42500 | 1388 | 0 |
| q35-9b | hard-multistep | on | cold | 4 | 10426 | 1689 | 36858 | 30 | 1292 |
| q35-9b | hard-multistep | on | hit | 5 | 3329 | 3223 | 3825 | 95 | 1342 |
| q35-9b | house-batch | off | cold | 9 | 48983 | 44881 | 82666 | 1624 | 0 |
| q35-9b | house-batch | on | cold | 4 | 11627 | 3080 | 36128 | 77 | 1278 |
| q35-9b | house-batch | on | hit | 5 | 13386 | 14289 | 25941 | 380 | 1456 |
| q35-9b | interleaved | on | cold | 3 | 1557 | 1543 | 1611 | 36 | 673 |
| q35-9b | interleaved | on | hit | 1 | 11215 | 11215 | 11215 | 169 | 8070 |
| q35-9b | interleaved | on | restored | 30 | 18852 | 8823 | 43643 | 512 | 3974 |
| qwen3-0.6b | coding-agent | on | cold | 3 | 791 | 298 | 1793 | 33 | 538 |
| qwen3-0.6b | coding-agent | on | hit | 2 | 452 | 370 | 535 | 118 | 650 |
| qwen3-0.6b | hard-multistep | off | cold | 7 | 3513 | 3481 | 3727 | 1175 | 0 |
| qwen3-0.6b | hard-multistep | on | cold | 4 | 1263 | 369 | 3945 | 20 | 1122 |
| qwen3-0.6b | hard-multistep | on | hit | 4 | 469 | 446 | 567 | 84 | 1164 |
| qwen3-0.6b | house-batch | off | cold | 8 | 6492 | 6269 | 9271 | 1351 | 0 |
| qwen3-0.6b | house-batch | on | cold | 4 | 1395 | 684 | 3806 | 57 | 1125 |
| qwen3-0.6b | house-batch | on | hit | 4 | 1619 | 1647 | 1800 | 326 | 1182 |
| qwen3-0.6b | interleaved | on | cold | 3 | 297 | 292 | 315 | 32 | 539 |
| qwen3-0.6b | interleaved | on | hit | 1 | 625 | 625 | 625 | 90 | 719 |
| qwen3-0.6b | interleaved | on | restored | 1 | 769 | 769 | 769 | 151 | 568 |
| qwen3-1.7b | coding-agent | on | cold | 3 | 834 | 819 | 1043 | 33 | 538 |
| qwen3-1.7b | coding-agent | on | hit | 57 | 1483 | 1048 | 2364 | 152 | 1965 |
| qwen3-1.7b | hard-multistep | off | cold | 9 | 5712 | 5681 | 6280 | 1194 | 0 |
| qwen3-1.7b | hard-multistep | on | cold | 4 | 2739 | 460 | 9541 | 20 | 1122 |
| qwen3-1.7b | hard-multistep | on | hit | 2 | 645 | 604 | 686 | 81 | 1132 |
| qwen3-1.7b | house-batch | off | cold | 22 | 9590 | 9315 | 12394 | 1657 | 0 |
| qwen3-1.7b | house-batch | on | cold | 4 | 1068 | 634 | 2124 | 57 | 1125 |
| qwen3-1.7b | house-batch | on | hit | 25 | 1422 | 1281 | 3128 | 146 | 1926 |
| qwen3-1.7b | interleaved | on | cold | 3 | 674 | 542 | 997 | 32 | 539 |
| qwen3-1.7b | interleaved | on | restored | 57 | 2004 | 1870 | 3132 | 135 | 1935 |
| qwen3-4b-16k | coding-agent | on | cold | 3 | 2831 | 1301 | 6434 | 33 | 538 |
| qwen3-4b-16k | coding-agent | on | hit | 7 | 7118 | 3859 | 26510 | 383 | 1606 |
| qwen3-4b-16k | hard-multistep | off | cold | 8 | 14346 | 13843 | 19900 | 1190 | 0 |
| qwen3-4b-16k | hard-multistep | on | cold | 4 | 4043 | 789 | 13585 | 20 | 1122 |
| qwen3-4b-16k | hard-multistep | on | hit | 2 | 1249 | 1107 | 1392 | 78 | 1132 |
| qwen3-4b-16k | house-batch | off | cold | 8 | 17680 | 13541 | 34868 | 1366 | 0 |
| qwen3-4b-16k | house-batch | on | cold | 4 | 4797 | 1764 | 13330 | 57 | 1125 |
| qwen3-4b-16k | house-batch | on | hit | 4 | 6839 | 6047 | 10609 | 384 | 1182 |
| qwen3-4b-16k | interleaved | on | cold | 3 | 743 | 750 | 751 | 32 | 539 |
| qwen3-4b-16k | interleaved | on | hit | 18 | 3879 | 3018 | 7450 | 166 | 2671 |
| qwen3-4b-16k | interleaved | on | restored | 1 | 2727 | 2727 | 2727 | 190 | 574 |
| qwen3-8b-allnpu | coding-agent | on | cold | 3 | 5044 | 3126 | 9813 | 33 | 538 |
| qwen3-8b-allnpu | coding-agent | on | hit | 57 | 11611 | 5281 | 33198 | 286 | 2536 |
| qwen3-8b-allnpu | hard-multistep | off | cold | 9 | 21041 | 20320 | 30001 | 1191 | 0 |
| qwen3-8b-allnpu | hard-multistep | on | cold | 4 | 5918 | 1070 | 19822 | 20 | 1122 |
| qwen3-8b-allnpu | hard-multistep | on | hit | 5 | 1770 | 1757 | 1962 | 72 | 1159 |
| qwen3-8b-allnpu | house-batch | off | cold | 8 | 25174 | 23592 | 40216 | 1370 | 0 |
| qwen3-8b-allnpu | house-batch | on | cold | 4 | 6608 | 2395 | 19674 | 57 | 1125 |
| qwen3-8b-allnpu | house-batch | on | hit | 4 | 8599 | 7575 | 13637 | 356 | 1182 |
| qwen3-8b-allnpu | interleaved | on | cold | 3 | 1576 | 1355 | 2353 | 32 | 539 |
| qwen3-8b-allnpu | interleaved | on | hit | 3 | 7496 | 7385 | 7727 | 120 | 6245 |
| qwen3-8b-allnpu | interleaved | on | restored | 40 | 10938 | 6100 | 19019 | 292 | 2264 |

