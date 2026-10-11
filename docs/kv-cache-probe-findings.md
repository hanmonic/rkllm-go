# RKLLM KV-cache probe findings (runtime 1.3.1, cross-checked on 1.2.3)

Date: 2026-10-09. Hardware: NanoPi M6 (RK3588, NPU driver 0.9.8, 4 big cores `0xF0`).
Probe: `cmd/rkllm-probe/` (stdlib Go + cgo, one binary per runtime). All runs use `top_k=1, temperature=0`.
Main model: `qwen3-1.7b` (toolkit 1.2.3, max_context_limit 16384). Key tests repeated on `q35-2b`
(Qwen3.5, hybrid linear attention) and `gemma4-e2b`. Raw logs: `/opt/rkllm-probe/out/*.log` on the board.

## How to reproduce

```sh
# on the board, sources in /opt/rkllm-probe/src, 1.3.1 header copied to /opt/rkllm-probe/inc131
CGO_CFLAGS="-I/opt/rkllm-probe/inc131 -DRKLLM_V131" CGO_LDFLAGS="-L/opt/rkllm-131/server/lib" go build -o probe131 .
CGO_CFLAGS="-I/opt/rkllm/include"                   CGO_LDFLAGS="-L/opt/rkllm/runtime"        go build -o probe123 .
LD_LIBRARY_PATH=/opt/rkllm-131/server/lib RKLLM_LOG_LEVEL=1 ./probe131 run q2_history [-model PATH] [-ctx N] [-fmt qwen|gemma4] [-v]
./probe131 list          # experiment scripts (scripts/*.json, embedded)
```

Each script is a JSON list of ops (`init`, `template`, `functools`, `run`, `kv`, `clear`, `load`, `release`, ...).
Every `run` prints `kv_before`/`kv_after` (`rkllm_get_kv_cache_size`), `perf` (RKLLMPerfStat), callback counts,
generated token ids, wall time and time to first callback. Prompt macros: `{{SYS:..}} {{U:..}} {{A}} {{END}}`
(rendered for `-fmt`), `{{FILLER:n[:seed]}}` (n deterministic sentences, ~20.5 Qwen tokens each).
`cmd/rkllm-probe/fmtdump` (pure Go, imports `internal/prompt`) prints `prompt.Format()` output for the template diff.

**Debug log that answers most questions:** `RKLLM_LOG_LEVEL=2` makes 1.3.1 print, per run,
`Keep history status`, `Prompt text: "..."` (the rendered prompt), `Prompt tokens: [ 'piece':id, ... ]`
(the exact tokenisation, including any BOS) and `Current kv cache size is N` (the reused prefix length, see Q2d).
At init it prints BOS/EOS ids and `add_bos_token`. This is a free tokenizer oracle.

KV accounting identity used throughout (verified on every run, all three models):
`kv_after = reused_prefix + prompt_tokens + n_emitted_tokens - 1`.
The last emitted token (EOS, or the token that hit max_new_tokens) is **not** fed back into the cache.

---

## Q1. Raw mode (`rkllm_set_chat_template(h, "", "", "")`)

Calls: `init` -> `run(role="user", "...")` (built-in template) -> `set_chat_template("","","")` -> `run("<|im_start|>user\n...")`.

| model | input | prompt tokens (log / prefill) | output |
|---|---|---|---|
| qwen3-1.7b | role mode, `What is the capital of France? Answer in one word.` | 24 | `Paris` |
| qwen3-1.7b | raw, same question hand-formatted in ChatML + empty think block | 24 (identical token list) | `Paris` |
| qwen3-1.7b | raw `a` | `['a':64]`, 1 token | - |
| q35-2b | raw `a` | `['a':64]`, 1 token | - |
| gemma4-e2b | raw `a` | `['<bos>':2, ' a':496]`, 2 tokens | - |
| gemma4-e2b | raw `<bos>a` | `['<bos>','<bos>',' a']`, double BOS | - |

- Qwen3 / Qwen3.5: raw mode adds nothing (`add_bos_token: 0`). Role mode = ChatML + `<think>\n\n</think>\n\n` when `enable_thinking=false`; no default system prompt.
- Gemma 4: `add_bos_token: 1`. The runtime prepends `<bos>` to **every text-prompt run**, raw mode included, and also to every `keep_history=1` delta (a BOS lands mid-sequence). `RKLLM_INPUT_TOKEN` runs get no BOS.
- `set_chat_template` logs "disables the internal automatic chat template parsing, including enable_thinking"; it also clears the KV cache (kv 25 -> 0).
- `role="tool"` in raw mode: the runtime insists the prompt is a JSON array, `rkllm_run` returns -1 and **no callback at all fires** (no ERROR callback).

Conclusion: raw mode is byte-exact for Qwen; for Gemma 4 the server must not add `<bos>` itself (`Options.AddBOS=false`) and cannot avoid the runtime's per-run BOS except by using token input.

## Q2. keep_history semantics

### 2a/2b. keep_history=1 (script `q2_history`, raw mode)

| run | input | prefill | kv before -> after | answer |
|---|---|---|---|---|
| T1 kh=1 | sys + user("My name is Zorblax, number 7342, reply OK") + gen prompt (46 tok) | 46 | 0 -> 48 | `OK.` (ids `OK . <|im_end|>`) |
| T2 kh=1 | delta `<|im_end|>\n<|im_start|>user\nWhat is my name and my favorite number?...` | 23 (= delta only) | 48 -> 89 | `My name is Zorblax and my favorite number is 7342.` |
| T3 kh=1 | delta without `<|im_end|>` | 27 | 89 -> 118 | answered from history |
| T4 kh=1 | T1 prompt again (full) | 46 | 118 -> 166 | appended a duplicate, no prefix matching |

- Prefill is exactly the delta; the model remembers. Same on q35-2b (T2 prefill 23, correct recall) and 1.2.3.
- Generated tokens are in the cache **except the last one**: 48 = 46 + 3 - 1. The `<|im_end|>` / `<turn|>` stop token is not in the cache and nothing is appended automatically. The next delta must begin with the stop token **and** the template's `\n`: `<|im_end|>\n<|im_start|>user\n...` (Gemma 4: `<turn|>\n<|turn>user\n...`). Omitting it (T3) does not error, it just leaves a malformed turn in history.
- Role mode with kh=1 (script `q2_role`, and log level 2 on all three models): turn 2 is rendered as `<|im_start|>user\nWhat is my name?<|im_end|>\n<|im_start|>assistant\n...` only (17 tokens). The runtime does **not** close the previous assistant turn, so role-mode history is `...OK.<|im_start|>user...`. Recall still worked for Qwen.
- kh=1 is pure append: no prefix matching, no truncation (T4).

### 2c. keep_history=0: is history discarded?

Yes. After T1-T4 (kh=1), `run(kh=0, "What is my name?")`: kv 166 -> 69 (fresh), answer "I'm not sure who you are". Role mode (`q2_role` r3/r4) and gemma4 identical. **The official flask server (kh=0, newest message only) gives the model no conversation history**; any benchmark run that way measured single-turn behaviour.

### 2d. Automatic prefix reuse with keep_history=0 — yes, and it is the most useful mechanism

Script `q2d_prefix` (qwen3-1.7b, ctx 8192, raw mode, kh=0 on every run):

| run | prompt | prefill tokens | TTFT |
|---|---|---|---|
| P1 | cold, 2020 tok | 2020 | 10.7 s |
| P1 again | identical | **0** | 0.14 s |
| P2 | same first ~900 tok, then different | 1118 | 7.2 s |
| P3 | P2 + "There are 100 records." + new user turn | **31** | 0.34 s |
| P4 | unrelated | 2215 | 12.0 s |
| P5 | P1 again | 2019 (only 1 token reused) | 10.7 s |

Rules (all verified by the kv identity and by the `Current kv cache size is N` log line, which prints N = reused tokens):
- Each kh=0 run reuses the **longest common token prefix with the previous kh=0 run's prompt**, truncates the cache there and prefills the rest. A one-token mismatch at position k simply truncates at k (reuse = k); nothing errors. Identical prompt: reuse n-1, the last prompt token is re-fed (prefill 0, one extra generate step).
- Only the previous **prompt** is matched, never the generated tokens: `q2_history` t8 resent t7's prompt + t7's answer verbatim (same token ids) and reuse was 67 (= t7 prompt), not 77. Cost per conversational turn = previous answer + new message + template glue.
- One slot only: the record is replaced on every run (P5).
- Same on 1.2.3 (identical numbers) and gemma4-e2b (log shows `Current kv cache size is 63` = previous prompt length).
- **Not on q35-2b** (hybrid linear attention): every kh=0 run prefilled the full prompt, including an identical resend (2021 tok, 14.1 s each time). Qwen3.5 only gets reuse via kh=1 append or prompt-cache files (Q4).

### Mode transitions (surprising, both runtimes)

- A kh=1 run right after a kh=0 run starts from an empty cache (kv 2027 -> 46, prefill = delta only), and a kh=0 run after kh=1 runs reuses nothing.
- Those transition runs **see stale KV**: e.g. kh=1 "What is my name?" after a kh=0 run whose prompt mentioned Zorblax answered "Hello! I'm Zorblax"; another produced `</think></think></think>` garbage; on 1.2.3 a kh=0 run after kh=1 runs answered "I'm Zorblax". The cache length is reset but old slots are still attended.
- Transition runs are 2-4x slower: 2020-token prefill took 20.6 s (kh=1 after kh=0) and 20.7-41.7 s (kh=0 after kh=1) vs 10.7 s normally.
- After `rkllm_clear_kv_cache(h, 0, NULL, NULL)` both effects are gone (cold 6006-token kh=1 prefill 71.3 s = same as kh=0 cold; correct answers).

Conclusion: pick one mode per handle, or full-clear on every switch.

## Q3. Truncation

`rkllm_clear_kv_cache(h, 0, &start, &end)`:

| state | result |
|---|---|
| after a finished kh=1 run | ret -1, log: "Start_pos and end_pos are only valid when keep_history == 0 and the generation has been paused by returning 1 in the callback!" |
| kh=1 run paused by callback returning 1 | ret -1 (same message) |
| finished kh=0 run | ret -1 |
| kh=0 run paused by callback returning 1 | **ret 0**, kv 41 -> 20 (q35-2b: same, ret 0) |

Working sequence (script `q3_truncate` C2, `q3b_stale` E2):
1. `rkllm_run(kh=0, prompt)`, callback returns 1 on the Nth NORMAL token. `rkllm_run` returns 0, no FINISH callback, no perf stats, `rkllm_is_running`=0. kv = prompt + N - 1 (the paused token is pending, not cached).
2. `rkllm_clear_kv_cache(h, 0, &start, &end)` with `end = current kv` -> ret 0, kv = start.
3. `rkllm_run(kh=0, delta)` -> **appends** the delta at `start` (prefill = delta only, kv 20 -> 72). A run after a pause is always "resume/append", not LCP.
   - Resume without truncation: `rkllm_run(kh=0, "")` continues generation; the pending token is emitted **again** (duplicate callback for the token already delivered before the pause).

Other findings:
- Pausing on the EOS token does not work: returning 1 from the callback that carries `<|im_end|>` is ignored; the run FINISHes.
- So a finished conversation cannot be range-truncated. The only truncation primitives are (a) kh=0 automatic LCP (cuts at the first differing prompt token), (b) the pause trick above (mid-generation only), (c) full clear.
- Full clear `rkllm_clear_kv_cache(h,0,NULL,NULL)`: ret 0, kv -> 0, ~18 ms (qwen3-1.7b, ~37 ms at ctx 8192; q35 ~3 ms). It also resets the LCP record (next identical prompt is fully prefilled).
- `keep_system_prompt=1` with a system prompt set via `set_chat_template`: kv -> 14 (the system prompt), 32 us; the next run prefilled only the user turn and kept the persona. In raw mode (empty system prompt) it is equivalent to a full clear.
- On q35-2b the paused range clear returns 0, but whether the recurrent (linear-attention) state is actually rewound was not verified.

## Q4. Prompt cache files

### 4a. What is saved
`run(kh=0, prompt, prompt_cache_params={1, path})`:
- qwen3-1.7b file size = **615,192 + 114,704 x prompt_tokens** bytes, exact for 39, 531, 533, 2026 tokens (6006 tok: 689.5 MB). 114,704 B/token ~ fp16 K+V for 28 layers x 8 kv heads x 128 (114,688) + 16. Generated tokens are **not** saved (531 tok + 1 gen and 533 tok + 31 gen fit the same line).
- q35-2b: 26 tok 21.5 MB, 2013 tok 46.0 MB -> ~20.2 MiB fixed (recurrent state) + ~12.3 KB/token.
- Write time to /dev/shm (included before the first token callback): ~55 ms @531 tok, ~180 ms @2026, ~465 ms @6006 (~1.5 GB/s).
- Saving on a kh=1 delta run writes the whole cache (11.4 MB ~ 94 tokens = history + delta), but see 4b: such files are not reusable.
- Saving on an empty-prompt run writes nothing.

### 4b. Loading (`q4b_load`, `q4c_load_full`, fresh process)
- `rkllm_load_prompt_cache` returns 0 but `kv_size` stays 0. It does **not** make the content part of the cache for whatever you send next. It seeds the kh=0 LCP record + KV: the next kh=0 run must send the **full prompt** (cached prefix + suffix).
  - suffix only after load: prefill = suffix, model does not know the cached content ("1234" instead of PINEAPPLE-42).
  - full prompt after load: prefill = suffix only, correct answer. 2013-token prefix: TTFT **0.31 s** vs 10.8 s cold (qwen3-1.7b); **0.36 s vs 13.9 s on q35-2b** (this is the "linear-attention cache reuse").
- Load time from /dev/shm: 3.6 MB 29 ms, 61 MB 83 ms, 231 MB 251-357 ms, 690 MB 664 ms (q35 46 MB 48 ms). Peak RSS rises by about the file size during load (HWM 2.3 -> 2.55 GB for 231 MB; 2.8 -> 3.5 GB for 690 MB).
- The loaded content is a one-shot seed: after one other run (or `release`), it is gone (P5-style full prefill). Load it again before each request that needs it.
- Files saved from kh=1 runs (`mid_delta.cache`, `conv.cache`) gave **zero** reuse when the full conversation text was sent; with kh=1 after load, stale slots leaked into the output ("Hello! I'm Zorbl...</think>"). Only save from kh=0 full-prompt runs.
- Loading into a handle with existing cache is fine (it replaces the LCP record). Nonexistent file: ret -1, log error.

### 4c. Mid-conversation save
Works if the conversation is driven in kh=0 mode: send the full conversation as the prompt with `save_prompt_cache=1` (and `max_new_tokens=1` if you only want the snapshot). The file covers the whole prompt. Not possible from kh=1 state (4b), and an empty-prompt run does not save.

### 4d. `rkllm_release_prompt_cache`
Returns 0, does not change `kv_size` or RSS. It discards a loaded-but-unconsumed cache (load -> release -> full prompt = full prefill) and resets the LCP record (after release the next run reused 0 tokens of the previous prompt).

## Q5. Multiple handles in one process (`q5_multi`, `q5b_multi_models`)

- Two handles of qwen3-1.7b: init 2.6 s + 2.8 s; VmRSS 8 MB -> 2,326 MB -> 4,639 MB (+2.3 GB each, no weight sharing). qwen3-1.7b + q35-2b: 2,326 -> 4,555 MB. `perf.memory_usage_mb` is process VmHWM, not per handle.
- Third handle fails in both cases: `E RKNN: failed to malloc npu memory, size: 1880096768` (q35-2b) / `595984384` (qwen3-0.6b), `errno 14 Bad address`, `rkllm_init` returns -1. NPU-side allocation ceiling around 4.5 GB in this process (whether per process or device-wide was not tested).
- Caches are independent: interleaving A(2k prompt) / B(2k prompt) / A(same prefix, new question) gave A prefill 15 and B prefill 15 (each handle keeps its own LCP record); kh=1 recall per handle correct (Zorblax / Quux).
- Alternating handles works; runs are serialized by the caller (one NPU job at a time).

## Q6. Throughput (qwen3-1.7b, ctx 8192, `q6_bench`)

| prompt | prefill time | prefill tok/s | decode tok/s (64 tok) |
|---|---|---|---|
| 502 | 1.97 s | 254 | 10.8 |
| 1,939 | 10.1 s | 192 | 7.2 |
| 6,006 | 70.5 s | 85 | 1.7-2.9 |

Short prompts (24-46 tok) prefill at 165-230 tok/s; decode ~12 tok/s at short context.

200-token delta on a 6,000-token context:

| path | prefill tokens | TTFT |
|---|---|---|
| cold 6,254-token prompt | 6,254 | **92.0 s** |
| kh=0 LCP (previous prompt = the 6,006 prefix) | 248 | 8.7 s |
| kh=1 append | 246 | 5.5 s |
| load 690 MB cache file (0.66 s) + kh=0 full prompt | 248 | 5.6 s (+0.66 s load) |

The deep-context delta prefills at only 30-45 tok/s (attention over 6k positions), and decode at 6k context is 1.5-2.9 tok/s, so long contexts stay slow even when cached. Still, caching is a 10-17x TTFT win at 6k. q35-2b: cold 2,021 tokens 13.9 s (146 tok/s); minimum prefill latency on q35 is ~0.3-0.7 s even for 17 tokens.
qwen3-4b-16k: **not measured** (time).

## Q7. Other 1.3.1 behaviour

- Perf stats (`RKLLMPerfStat`) are non-zero only in the FINISH callback. A paused run has no FINISH and no perf. `generate_tokens` = emitted tokens - 1 (the first token comes from prefill); a 1-token prompt reports prefill 0 and is processed as a decode step.
- `token_id` is present on every NORMAL and WAITING callback. WAITING callbacks (incomplete UTF-8, e.g. emoji) carry a token id and no text; the bytes arrive with a later NORMAL. Count tokens from callbacks, not from text.
- The stop token is delivered as a NORMAL callback with its id (151645 / 248046 / 106 / 1) and empty text (skip_special_token=true).
- No ERROR callback was observed; failures surface as a -1 return from `rkllm_run`/`rkllm_init` with no callback at all.
- `rkllm_set_chat_template` resets the KV cache.
- Init time: 2.5-3.5 s warm (14.7 s first cold load from disk).
- 1.2.3: same LLMCallState values as 1.3.1 (NORMAL=0, WAITING=1, FINISH=2, ERROR=3; the repo CLAUDE.md lists old values). All Q2 results reproduce on 1.2.3 (`probe123`), except per-run `max_new_tokens`/sampling, which 1.2.3 lacks.

---

## Questions from the template work

### T1. Which template is embedded, and does it match `internal/prompt`?
Enable with `RKLLM_LOG_LEVEL=2` (prints `Prompt text: "..."` plus the token list; levels 3 and 5 print the same). Script `t_tmpl` rendered the runtime's own template (role mode, `rkllm_set_function_tools` for tools), and `fmtdump` rendered `prompt.Format()` for the same messages:

| model | 1 user msg, thinking off | thinking on | system + 1 tool + user |
|---|---|---|---|
| qwen3-1.7b | identical | identical | identical |
| q35-2b | identical | identical (`...assistant\n<think>\n`) | identical |
| gemma4-e2b | identical (`<|turn>user\n...<turn|>\n<|turn>model\n`) | identical (`<|turn>system\n<|think|>\n<turn|>\n...`) | identical (`<|tool>declaration:get_weather{...}<tool|>`) |

All nine are byte-identical. Multi-turn cannot be compared: role mode never re-renders history, it renders only the new user turn (see 2b).

### T2. BOS in raw mode
Qwen3 / Qwen3.5: none (`add_bos_token: 0`). Gemma 4: the runtime always prepends `<bos>` (id 2) to text input, raw mode included, and again on every kh=1 delta (mid-sequence). Writing `<bos>` yourself gives a double BOS. Token input (`RKLLM_INPUT_TOKEN`) gets no BOS.

**Gemma 4 tokenizer quirk:** the runtime tokenizes each text chunk between special tokens with a SentencePiece dummy-prefix space: `<|turn>user\n` -> `'<|turn>' ' user'(2430) '\n'` (in-chunk `user` is 2364), `<turn|>\n` -> `'<turn|>' ' '(236743) '\n'`, `<|tool_call>call:` -> `' call'`, `<|"|>Paris` -> `' Paris'`. This applies to the runtime's own template too. Feeding the "clean" token ids via `RKLLM_INPUT_TOKEN` (`t_g4fixed`) did **not** improve output, though: gemma4-e2b answers "Please provide the text..." to "What is the capital of France? Answer in one word." either way, and fails 2-turn recall. With Gemma-3 tags (`<start_of_turn>`) or plain completion it answers "Paris". The converted gemma4-e2b looks unreliable with its own chat format. This needs separate investigation before relying on it.

### T3. Special tokens in callback text
Yes, skip_special_token=true drops them from `text`, but `token_id` is always present. Forced Gemma 4 tool-call continuation (`t_g4special*`):
- skip=true: text `Paris}`; ids `52(<|"|>) 50429 52 236783 49(<tool_call|>) 1(<eos>)`, the specials have empty text.
- skip=false: text `<|"|>Paris<|"|>}<tool_call|><eos>`; the stop token's text is also delivered (`<turn|>` in other runs).
Qwen's `<tool_call>`/`</tool_call>` (151657/151658) are added tokens, not special: their text arrives even with skip=true, and only `<|im_end|>` is blanked. Recommendation: run with skip_special_token=false (or map ids yourself) and strip the stop token in the server.

### T4. The `\n` after the stop token; tolerance of mismatches
- After a kh=1 turn the cache does **not** end with the stop token. The stop token is emitted but never fed back (identity above; gemma4 kv 22 + 3 - 1 = 24). So the next delta must start with **both** the stop token and the `\n`: Qwen `<|im_end|>\n<|im_start|>user\n...`, Gemma 4 `<turn|>\n<|turn>user\n...`. For Gemma 4 in text mode that delta becomes `<bos> <turn|> ' ' \n <|turn> ' user' ...` (log, `q2_history -fmt gemma4`), which is wrong twice over. Send Gemma kh=1 deltas as token ids.
- kh=1 does no matching at all; whatever you send is appended verbatim. A missing `\n` is not "tolerated", it becomes history.
- kh=0 LCP compares tokens exactly and truncates at the first mismatch (`q3c` t3: reuse 7; `q2d` P2: reuse ~900 of 2020). A one-token difference at position k costs re-prefilling from k, it never fails. Because only the previous prompt is matched (not generated tokens), the re-rendered assistant reply plus its `<|im_end|>\n` is always re-prefilled in kh=0 mode, so its exact rendering only has to be stable between consecutive requests.

---

## Design implications

**(a) Reuse across turns of one conversation (Qwen3 / Gemma, full-attention models).** Use **kh=0 and send the full rendered conversation every request**, in raw mode with the server's own template. The runtime reuses the LCP with the previous request's prompt automatically. Per-turn prefill = previous assistant reply + new message + glue (P3: 31 tokens, 0.34 s TTFT on a 2.1k context). No tokenizer, no state tracking, and edits or regenerations are handled automatically. kh=1 deltas save re-prefilling the last reply (5.5 s vs 8.7 s on a 6k context in Q6) but give up all truncation, so they are only worth it as an optimisation for long replies.
For **q35-2b (Qwen3.5)**, kh=0 does no reuse. Use kh=1 deltas within a conversation, or save a prompt-cache file after each turn and load it before the next full-prompt kh=0 request (load ~50 ms, Q4b).

**(b) History diverging from the cache.** With kh=0 + full prompt this is automatic: the cache is cut at the first differing token. With kh=1 there is no way back: range clear is refused (-1), so full clear and re-prefill (or load a snapshot file). Never mix kh=0/kh=1 on a handle without a full clear in between (stale-KV leakage, 2-4x slower prefill). Gemma 4: send deltas as token ids or not at all (BOS + dummy-space corruption).

**(c) Park/restore multiple sessions.** Possible via prompt-cache files: on each kh=0 turn of a session, set `save_prompt_cache=1` to `/dev/shm/<session>.cache` (cost ~90 ms per 1k tokens for qwen3-1.7b, 115 KB/token; q35 ~20 MB + 12 KB/token). To resume: `rkllm_load_prompt_cache` (~110 ms/1k tokens) then a kh=0 run with the full prompt. A loaded file is a one-shot seed, so reload before every request of a non-current session. kh=1-produced files are not restorable. A second warm handle (+2.3 GB RSS, max ~2 small models per process) keeps a second session fully resident without files.

**(d) Shared static system + tools prefix.** Save the prefix once (`run(kh=0, prefix, save=1, max_new=1)`) and load it before the first request of any session whose prompt starts with it (TTFT 0.31 s vs 10.8 s for a 2k prefix). Combined with (a), a session's own follow-ups then reuse via LCP, so the file is only needed when switching sessions on a handle.

**Impossible on this runtime (1.3.1 and 1.2.3):** range truncation of a finished cache (only mid-generation, kh=0, after a pause); more than one resident prefix per handle (the LCP record is a single slot, replaced every run); reusing generated tokens via LCP; automatic prefix reuse on Qwen3.5 hybrid models; restoring files saved from kh=1 runs; sharing weights between handles; BOS-free text input on Gemma 4.
