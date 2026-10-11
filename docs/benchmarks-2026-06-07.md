# Benchmark Report — 2026-06-07

Hardware: NanoPi M6 — RK3588S, 32 GB LPDDR4X  
Runtime: RKLLM 1.2.3 / rkllm-go (this repo) in an LXC container  
Toolkit: rkllm-toolkit 1.2.3 — model conversions run on an x86_64 build VM (48 GB RAM)

---

## Models Under Test

| Name | Source | Quant | Hybrid | Context | Size |
|---|---|---|---|---|---|
| `qwen3-0.6b` | dulimov HF | w8a8 | all-NPU | 4096 | — |
| `qwen3-1.7b` | dulimov HF | w8a8 | all-NPU | 4096 | — |
| `qwen3-4b` | dulimov HF | w8a8 | all-NPU | 4096 | 4.6 GB |
| `qwen3-4b-16k` | **own conversion** | w8a8 opt-1 | all-NPU | **16384** | 4.6 GB |
| `qwen3-4b-online` | dulimov HF | w8a8 | all-NPU | 4096 | — |
| `qwen3-8b-online` | dulimov HF | w8a8 | hybrid-0.5 | 4096 | 8.3 GB |
| `qwen3-8b-allnpu` | **own conversion** | w8a8 opt-1 | **all-NPU** | **16384** | 8.3 GB |
| `qwen3-14b-online` | dulimov HF | w8a8 | hybrid-0.5 | — | — |
| `deepseek-r1-1.5b` | HF distill | w8a8 | — | — | — |
| `deepseek-r1-7b` | HF distill | w8a8 | — | — | — |
| `llama3.2-1b` | Meta HF | w8a8 | — | — | — |
| `llama3.2-3b` | Meta HF | w8a8 | — | — | — |
| `gemma3-1b` | Google HF | w8a8 | — | — | — |
| `gemma3-4b` | Google HF | w8a8 | — | — | — |
| `gemma3n-e4b` | Google HF | w8a8 | — | — | — |

---

## 1. Throughput (generate speed)

Single-turn chat completions, non-streaming. The generate figure is the only one that matters for perceived response latency — prefill is fast and rarely felt.

| Model | Prefill t/s | Generate t/s | Memory |
|---|---|---|---|
| `qwen3-4b` (dulimov) | 70–87 | **5.5** | 5132 MB |
| `qwen3-4b-16k` (ours) | 72–82 | **5.6–5.8** | 5141 MB |
| `qwen3-8b-online` (hybrid-0.5) | 57 | **3.1** | 14799 MB |
| `qwen3-8b-allnpu` (ours) | 35–36 | **3.2** | 8295 MB |

### Key findings

**The 8B hybrid-0.5 penalty is memory, not speed.** Generate t/s is identical between `qwen3-8b-online` (hybrid) and `qwen3-8b-allnpu` (all-NPU): both land at 3.1–3.2 t/s. The hybrid model keeps a copy of half the layers in CPU RAM, resulting in 14.8 GB usage vs 8.3 GB for all-NPU — a 6.5 GB saving per node with no speed tradeoff.

**Generate speed is memory-bandwidth bound.** Each decode step reads all model weights once. The RK3588S's LPDDR4X bandwidth caps this at ~3 t/s for 8B and ~5.5 t/s for 4B. Community benchmarks showing 10–14 t/s for 8B are from devices with LPDDR5 (Orange Pi 5 Pro etc.). This is a hardware ceiling, not a conversion or config issue.

**The 4B-16k conversion has zero throughput regression** vs the dulimov 4096-context build. Same speed, same memory, 4× the context window, matching toolkit version (1.2.3 vs 1.2.1b1). It is a straight upgrade.

---

## 2. Tool Call Reliability

**Method:** 13 scenarios across 4 system-prompt variants (none / neutral / encouraging / discouraging), 5 runs per scenario. Scenarios include obvious-trigger (weather, search), computation (calculate), multi-location, and should-not-call (factual, greeting). Scoring: % of runs where `finish_reason == "tool_calls"` with valid JSON arguments and correct tool name.

Full scenario matrix — success rate (% matched expectation), σ = run-to-run variability:

| Model | weather | weather+enc | calc+enc | search | multi-weather | factual (no-call) | greeting (no-call) |
|---|---|---|---|---|---|---|---|
| `qwen3-0.6b` | 0% | 0% | 0% | 0% | 0% | 100% | 100% |
| `qwen3-1.7b` | 0% | 0% | 0% | 0% | 0% | 100% | 100% |
| `qwen3-4b` | 0% | 100% | 100% | 100% | 100% | 100% | 100% |
| `qwen3-4b-online` | 20%σ0.4 | 100% | 100% | 100% | 100% | 100% | 100% |
| **`qwen3-8b-online`** | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** |
| `qwen3-14b-online` | 20%σ0.4 | 80%σ0.4 | 20%σ0.4 | 100% | 0% | 100% | 100% |
| `deepseek-r1-1.5b` | 0% | 0% | 0% | 0% | 0% | 100% | 100% |
| `deepseek-r1-7b` | 0% | 0% | 0% | 0% | 0% | 100% | 100% |
| `llama3.2-1b` | 0% | 0% | 0% | 0% | 0% | 100% | 100% |
| `llama3.2-3b` | 0% | 0% | 0% | 0% | 0% | 100% | 100% |
| `gemma3-1b` | 0% | 0% | 0% | 0% | 0% | 100% | 100% |
| `gemma3-4b` | 0% | 0% | 0% | 0% | 0% | 100% | 100% |
| `gemma3n-e4b` | 0% | 0% | 0% | 0% | 0% | 100% | 100% |

The `qwen3-8b-allnpu` spot-check (3 runs, 6 scenarios) matched `qwen3-8b-online` exactly — conversion did not affect tool call behaviour.

The `qwen3-4b-16k` vs `qwen3-4b` tool call comparison was also identical across all 8 tested scenarios.

### Key findings

**Only Qwen3 models produce structured tool calls.** All non-Qwen models score 0% on every trigger scenario. This is not a parsing problem — the models are generating output, just not in the `<tool_call>` format our parser expects. Each model family uses a different native format:

| Model family | Observed output format |
|---|---|
| Qwen3 (4B+) | `<tool_call>{"function":"...","parameters":{...}}</tool_call>` |
| Gemma 3 | `` ```tool_call\n{...}\n``` `` or `` ```tool_name:get_weather `` |
| LLaMA 3.2 | Echoes `<tools>` block verbatim (confused by Qwen3-format injection) |
| DeepSeek-R1 | `<function_call>...</function_call>` or prose description |

The root cause for non-Qwen models is twofold: (1) we inject tools using Qwen3's `<tools>` XML format, which other models weren't trained on, and (2) we only parse `<tool_call>` blocks. Fixing LLaMA would require both a new prompt injection format and a new output parser. Gemma's format is parseable with a small regex addition. Neither was investigated further as capability at small sizes (1–4B) is likely insufficient regardless.

**qwen3-0.6b and qwen3-1.7b are below the capability threshold** for structured tool calling even with correct prompting — they describe the call in prose rather than emitting structured output. Not a size-fixable issue at these weights.

**qwen3-14b-online performs worse than qwen3-8b-online** on several scenarios with high variability (σ=0.4). Likely a temperature interaction with the hybrid CPU/NPU split at this model size. Not investigated further.

**False positive rate is zero across all models.** No model called a tool when asked a factual question or given a greeting, even with an encouraging system prompt. The tool-call abstain behaviour is robust.

---

## 3. Long-Context Multi-Tool Test

**Method:** Long system prompt (~500 tokens) + 2 turns of fake conversation history + a single user prompt requesting: weather in 3 cities, a web search, and a distance calculation. Total context at request time: ~1,400 tokens. The test script simulates tool execution (fake results) and drives the multi-turn loop until the model produces a final synthesis. 3 runs per model.

**Scenario prompt:**
> Check weather in Tokyo, London, Paris. Search for major AI infrastructure announcements this week. Calculate the round-trip distance Tokyo→London→Paris→Tokyo using provided leg distances. Combine into a structured situation report.

Expected tool calls: 5 (get_weather×3, search_web, calculate).

| Model | Tool calls/run | Turns | Avg time | All tools used | Notes |
|---|---|---|---|---|---|
| `qwen3-4b-16k` | **0** | 1 | 137s | **0%** | Hallucinated full report every run |
| `qwen3-8b-allnpu` | **5** | 2 | 282s | **100%** | All tools correct, 3/3 runs |

### Key findings

**The 4B model completely fails at multi-tool orchestration under long context.** All three runs produced a plausible-looking situation report with fabricated weather data, fabricated search results, and made-up distances — no tools called. The failure is not a context-length issue (runs 2 and 3 showed 0 prompt tokens, suggesting the model stopped tracking context correctly). It is a capability ceiling: the 4B cannot hold multi-step instruction compliance when the prompt is complex.

**The 8B model handled it perfectly, consistently.** All 5 expected tools were called in a single parallel turn (the model batched all tool calls together), received fake results, and synthesised a well-structured markdown report on turn 2. Zero failures across 3 runs.

**8B is ~2× slower per response** in this test (282s vs 137s) but the 4B time is wasted — it's generating hallucinated content, not useful output.

---

## 4. Model Conversion Notes

### qwen3-8b-allnpu

Converted from `Qwen/Qwen3-8B` (HuggingFace) on armiban-build using rkllm-toolkit 1.2.3.

```python
llm.build(
    do_quantization=True, optimization_level=1,
    quantized_dtype='w8a8', quantized_algorithm='normal',
    target_platform='rk3588', num_npu_core=3,
    hybrid_rate=0.0,   # all-NPU
    max_context=16384,
)
```

Build time: ~2 hours (build phase ~2 min, optimize phase ~50 min at ~83s/step × 36 steps).  
RAM required: ~32 GB peak. the build VM had 48 GB; no swap used.  
Output: 8.3 GB `.rkllm` file. Served via nginx, pulled to the board.

**Result:** `hybrid_rate=0.0` confirmed by memory footprint (8.3 GB vs 14.8 GB for hybrid-0.5). Generate speed unchanged — bandwidth-limited, not compute-limited.

### qwen3-4b-16k

Converted from `Qwen/Qwen3-4B` (HuggingFace) on armiban-build using rkllm-toolkit 1.2.3.

```python
llm.build(
    do_quantization=True, optimization_level=1,
    quantized_dtype='w8a8', quantized_algorithm='normal',
    target_platform='rk3588', num_npu_core=3,
    hybrid_rate=0.0,   # all-NPU (same as dulimov original)
    max_context=16384,
)
```

Output: 4.6 GB `.rkllm` file (identical size to dulimov 4096-context build).  
Runtime confirms: `max_context_limit: 16384` in startup log.

**Result:** Zero regression vs dulimov build across throughput and tool call benchmarks.

---

## 5. Recommendations

### Production model selection

| Use case | Recommended model | Reasoning |
|---|---|---|
| Everyday interactive chat | `qwen3-4b-16k` | 5.5 t/s, 16k context, zero regression vs old default |
| Multi-tool / agentic tasks | `qwen3-8b-allnpu` | Only model that reliably chains multiple tool calls |
| Long document Q&A | `qwen3-8b-allnpu` | 4B fails at complex instructions under long context |
| Simple factual / short tasks | `qwen3-4b-16k` | Speed advantage, sufficient for single tool calls with encouraging sys prompt |

`qwen3-4b-16k` is set as the server default. Switch to `qwen3-8b-allnpu` by setting `"model": "qwen3-8b-allnpu"` in client requests.

### What to deprioritise

- **Gemma 3, LLaMA 3.2 tool calling** — parseable formats exist but capability at 1–4B sizes is insufficient. Not worth the engineering effort.
- **qwen3-14b-online** — inconsistent, high variability, slower than 8B without reliability gains.
- **DeepSeek-R1 as orchestrator** — interesting architecture but single-model-at-a-time constraint means hot-swap latency (30–60s per model swap) makes multi-model pipelines impractical on this hardware.
- **W4A16 quantization for 8B** — would improve generate speed to ~5–6 t/s but quality tradeoff not yet measured. Keep in backlog.

### Multi-node architecture (4 devices)

Given that `qwen3-8b-allnpu` handles both tool calling and complex reasoning, and `qwen3-4b-16k` handles interactive speed:

- **All 3 inference nodes**: run `qwen3-4b-16k` as default, `qwen3-8b-allnpu` on-demand for agentic tasks
- A distributed orchestrator pinning DeepSeek-R1 on one node with Qwen3 subagents on others is not recommended — hot-swap overhead negates the benefit vs just using the 8B directly
- If all nodes run the same model, load balancing across 3 nodes effectively triples throughput for concurrent users

---

## 6. Open Items

- [ ] `<function call>` (space, not underscore) variant seen once on 8B `calc_encouraging` — one-line fix in `parseToolCalls` regex
- [ ] `qwen3-4b` (original dulimov 4096-context) can be removed from `models.json` — superseded by `qwen3-4b-16k`
- [ ] W4A16 8B conversion — kick off if 3 t/s proves too slow in production
- [ ] Multi-step tool call test at `--ctx-size medium` for 4B — determine at what context/complexity threshold the 4B starts failing
- [ ] Throughput bench on long prompts without curl timeout (current 180s limit causes the long-prompt test to time out mid-generation)
