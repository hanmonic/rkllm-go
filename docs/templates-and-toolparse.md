# Chat templates and tool-call parsing

Packages: `internal/prompt` (templates, registry), `internal/toolparse` (call
extraction + schema coercion), `internal/pyjson` (order-preserving JSON that
serialises like Python `json.dumps`, needed to reproduce Jinja `tojson`).
None of them import cgo or `internal/rkllm`.

## Where the templates come from

The `.rkllm` files are obfuscated: `grep -a` over the whole of
`qwen3-0.6b`, `q35-2b` and `gemma4-e2b` finds no `im_start`, `start_of_turn`
or `<|turn` at all, so the embedded template cannot be read from the model
files. `librkllmrt.so` 1.3.1 contains minja (a Jinja engine) and the string
`tokenizer.chat_template`, i.e. the runtime renders the HF Jinja template
stored in the model. The runtime's hardcoded fallbacks (llama.cpp's list) only
cover plain ChatML / `<start_of_turn>` without tools or thinking.

So the ground truth is the upstream HF template of each source checkpoint,
downloaded on 2026-10-09 and kept in `internal/prompt/testdata/hf/`:

| Family | Models | Template file (source) |
|---|---|---|
| `qwen3` | qwen3-0.6b, -1.7b, -4b, -4b-16k, -8b-allnpu | `qwen3.jinja` = `chat_template` in `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-4B` tokenizer_config.json (identical) |
| `qwen3.5` | q35-2b, q35-4b, q35-9b | `qwen3.5.jinja` = `Qwen/Qwen3.5-4B/chat_template.jinja` (9B identical; 2B differs only in defaulting thinking off) |
| `gemma3` | gemma3n-e4b, gemma3-1b, gemma3-4b | `gemma3n.jinja` / `gemma3.jinja` from `unsloth/gemma-3n-E4B-it`, `unsloth/gemma-3-4b-it` (google repos are gated; differ only in multimodal tokens) |
| `gemma4` | gemma4-e2b | `gemma4.jinja` = `google/gemma-4-E2B-it/chat_template.jinja` ("Canonical", published 2026-07-09); EOS ids `[1,106,50]` = `<eos>`, `<turn|>`, `<|tool_response>` from generation_config.json |

`testdata/render_hf.py` renders `testdata/cases.json` through these templates
with jinja2 exactly as transformers does; `TestGoldenMatchesUpstreamTemplates`
asserts byte equality (23 case/family pairs) wherever our policy coincides with
upstream. Regenerate with
`python render_hf.py internal/prompt/testdata` (needs jinja2).

Family resolution: `models.json` gets an optional `"families"` map
(model name -> `qwen3` | `qwen3.5` | `gemma3` | `gemma4`); unlisted models are
inferred from the name (`q35`/`qwen3.5` before `qwen3`, `gemma4`, `gemma3`).
Llama 3.2 / DeepSeek-R1 have no template here; `chat.go` falls back to qwen3
with a log line, and `toolparse` handles their JSON calls via the generic parser.

## API

```go
tpl, _ := prompt.Resolve(cfg.Families[name], name)
stable, tail := tpl.Render(msgs, tools, prompt.Options{EnableThinking: on})
full := tpl.Format(msgs, tools, opts)            // == stable + tail
n := prompt.StablePrefixLen(tpl, msgs, tools, opts)
tpl.StopStrings()

content, calls, err := toolparse.Parse(string(tpl.Family()), output, toolparse.SchemasFromTools(tools))
reasoning, answer := toolparse.SplitReasoning(output)
safe, inCall := toolparse.Holdback(family, soFar) // SSE: stream soFar[:safe], hold the rest
```

`prompt.Message` has `Raw`: the exact text the model generated for an
assistant turn (without the stop token). If set it is rendered verbatim after
the generation prompt that preceded it, so the history is byte-identical to
what is in the KV cache. Without `Raw`, the turn is re-rendered canonically
from content / reasoning / tool_calls, which matches the generation only when
the model produced canonical output.

## Append stability

`Format(msgs) = stable + tail`, and `stable` is a prefix of
`Format(append(msgs, anything...))` (tested for every family, case and split
point, thinking on and off: `TestAppendStability`). `tail` = generation prompt
plus closings that depend on the next message. Additionally
`TestGenerationContinuity` checks the KV-cache property itself: the full
prompt (incl. generation prompt) that produced assistant turn *i* is a prefix
of every later prompt.

Holds only while `tools` and `Options` stay fixed for the conversation (both
change the top block). With zero messages nothing is stable (the first
message decides how the system/tools block is laid out).

## Per family

### qwen3
- Template: `<|im_start|>{role}\n{content}<|im_end|>\n`; generation prompt
  `<|im_start|>assistant\n` + `<think>\n\n</think>\n\n` when thinking is off.
- Tools: system turn = `{system}\n\n` + `# Tools … <tools>\n{tool|tojson}\n…</tools>\n\nFor each function call, return a json object … </tool_call>`; with no system message the block stands alone (no default "You are a helpful assistant" — the old formatter added one, upstream does not).
- Call: `<tool_call>\n{"name": "NAME", "arguments": ARGS}\n</tool_call>`, several separated by `\n`; ARGS is the OpenAI arguments string verbatim.
- Tool result: consecutive `tool` messages share one `<|im_start|>user` turn, each `\n<tool_response>\n{content}\n</tool_response>`; `<|im_end|>\n` after the last (that closing is in the tail).
- Thinking: soft switch `/no_think` is NOT used (appending it to "the last user message" moves on the next turn and breaks the prefix). Off = empty think block prefill, which is what upstream `enable_thinking=false` does. Prior assistant turns get the same prefill; with thinking on, `<think>\n{reasoning}\n</think>\n\n` is re-rendered (from `Reasoning` or a `<think>` block in content).
- Deviation from upstream: upstream drops the think block of assistant turns before the last user query and, with thinking off, renders intermediate tool-call turns without the empty block. Both would make the history differ from what was prefilled; we keep them.

### qwen3.5
- Same ChatML, but content of every message is trimmed; generation prompt is `<|im_start|>assistant\n<think>\n` (thinking on) or `…<think>\n\n</think>\n\n` (off).
- Tools: system turn = `# Tools\n\nYou have access to the following functions:\n\n<tools>…</tools>` + the XML-format instructions + `<IMPORTANT>` reminder block, then `\n\n{system}`.
- Call: `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n…</function>\n</tool_call>`; first call preceded by `\n\n` if there is content. V is Jinja `|string` for scalars (`True`/`False`/`None`, which is why models emit `True`) and `tojson` for arrays/objects.
- Tool result: as qwen3 (trimmed).
- Thinking/deviation: as qwen3 (upstream renders `<think>…</think>` only after the last user query).

### gemma3 (Gemma 3, Gemma 3n)
- Template: `<bos>` (only with `Options.AddBOS`), `<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n`, `<start_of_turn>model\n{content}<end_of_turn>\n`, generation prompt `<start_of_turn>model\n`. No system role; contents trimmed.
- Tools: no native support. We use the polyfill the runtime's minja applies to tool-less templates — `librkllmrt.so` contains `You can call any of the following tools to satisfy the user's requests: ` and `Example tool call syntax:`, and the old gemma3/3n outputs echo its example id `"call_1___"`. The text (tools JSON indent 2 + example `{"tool_calls": [{"name": "tool_name", "arguments": {"arg1": "some_value"}, "id": "call_1___"}]}`) is appended to the system text, which is folded into the first user turn.
- Call: ```` ```json\n{"tool_calls": [{"name", "arguments", "id"}]}\n``` ```` (indent 2), matching what gemma3n produced (100% right_args in bench1).
- Tool result: one user turn per run of `tool` messages, each `{"tool_response": {"tool", "tool_call_id", "content"}}` (indent 2, content embedded as JSON when it parses as an object/array).
- Thinking: none.

### gemma4
- Template: `<|turn>system\n[<|think|>\n]{system}<|tool>declaration:…<tool|>…<turn|>\n`, `<|turn>user\n{user}<turn|>\n`, generation prompt `<|turn>model\n`. Tool declarations are ported macro-for-macro (`format_function_declaration`, `format_parameters`, `format_argument`, `dictsort`) and match upstream byte-for-byte.
- Call: `<|tool_call>call:NAME{k:v,…}<tool_call|>`, keys dictsorted, strings `<|"|>…<|"|>`. The model then emits `<|tool_response>` (an EOS id); the result is spliced in as `response:NAME{value:<|"|>{content}<|"|>}<tool_response|>` and the model continues **in the same turn** (generation prompt after tool results is empty, or `<|channel>thought\n` with thinking on).
- Thinking: `<|think|>` in the system turn, reasoning as `<|channel>thought\n…\n<channel|>`; re-rendered always (upstream only after the last user message unless `preserve_thinking`). Toggling thinking changes the very first turn, i.e. invalidates the whole cache.
- Deviation: an assistant message with both content and tool calls renders content after the calls and before the tool responses (generation order). Upstream puts it after the responses, which rewrites earlier bytes once results arrive.

## Tool-call parsing (`internal/toolparse`)

Ported from `/opt/rkllm-131/toolbench.py` (`parse_text_calls`,
`parse_gemma_calls`, `parse_json_calls`, `_coerce`). All syntaxes are tried in
a family-specific order, stopping at the first that matches:

- tagged `<tool_call>` blocks, JSON body (also `{"function": "name", "parameters": …}`) or Qwen3.5 XML body; unterminated blocks (max_tokens) are accepted.
- Gemma 4 `call:NAME{…}`, with tokens (`<|tool_call>`, `<|"|>`) or bare as the 1.3.1 runtime returns it (`call:set_light{room:living room,on:false}` in result-m63-gemma4-e2b.json). Bare form only at a word boundary and for a declared tool; commas inside unquoted strings are disambiguated by checking whether `key:` follows with a schema property name.
- generic JSON anywhere in the text: `{"name","arguments"|"parameters"}`, `{"function":{…}}`, `{"tool_calls":[…]}`, top-level arrays; surrounding ```` ```json ```` fences are removed. Names must be declared tools (when tools are given) — prose JSON and hallucinated names like `"tool_name"` stay content.

Coercion follows the tool's JSON schema (property `type`, incl. type lists,
`anyOf`, nested `properties`/`items`): `"21.5"`→21.5, `"15"`/`45.0`→integer,
`True`/`false`/`"yes"`/`1`→bool, `'["a","b"]'` or `"a, b"`→array, JSON-string
objects→object, numbers→string when the schema says string. Without a schema
type, XML/bare values get `_coerce` (bools in any case, null/None, JSON). JSON
bodies are parsed leniently (Python `True/False/None`, single quotes, trailing
commas). `arguments` is re-serialised Python-style (`{"a": 1, "b": "x"}`) — for
Qwen3 that equals what the model emits, which keeps re-rendered history close
to the cache. Malformed markup returns `ErrMalformed` (wrapped), keeps the
parsed calls, and leaves the bad block in content. Reasoning spans
(`<think>`, including the opener-less form Qwen3.5 produces because `<think>\n`
is prefilled, and Gemma's thought channel) are skipped and left in content;
`SplitReasoning` separates them.

`Holdback(family, text)` returns how many bytes can be streamed and whether a
call has started: it looks for `<tool_call>`, `<|tool_call>`, bare
`call:NAME{` (gemma4) or `{`/```` ``` ````/`[` followed by a
`"name"|"function"|"tool_calls"|"type"` key (generic), and holds back any
trailing partial marker.

Fixtures: `internal/toolparse/testdata/outputs.json` (25 real outputs from
res1-*.json / result-m63-*.json), `tools.json` (bench schemas).

## Open questions

1. **Embedded template revision unverifiable from the files.** The runtime
   logs `Prompt text: "%s"` (string in librkllmrt.so; probably needs
   `RKLLM_LOG_LEVEL=1`). Running each model once with the runtime's own
   template + `rkllm_set_function_tools` and diffing that log against
   `Format()` would confirm the template revision, especially Gemma 4 (the
   google and unsloth revisions differ; the Oct 7 conversion's source is unknown).
2. **Special tokens in callback text.** The Gemma 4 bench output has no
   `<|tool_call>` / `<|"|>`, so the 1.3.1 callback seems to drop special-token
   text. Then `Raw` for Gemma 4 (and the `<think>` handling if those are special
   for a model) cannot be reconstructed from text; it would need `token_id`s
   from the callback. Qwen's `<tool_call>`/`<think>` are ordinary added tokens
   and appear in text.
3. **BOS.** Gemma templates start with `<bos>`. Whether the runtime tokenizer
   adds BOS itself with an empty chat template is unknown; `Options.AddBOS`
   defaults to false.
4. **End-of-turn in the cache.** After generation the cache holds the stop
   token (`<|im_end|>`, `<end_of_turn>`, `<turn|>`) but not the `\n` the
   template adds after it, so the next prompt diverges one token after the
   generated text at most — whether the runtime's "full prefix match" reuse
   tolerates that needs the probe's answer.
5. **Gemma 4 `dictsort`** is ported as Jinja's case-insensitive sort; minja's
   may be case-sensitive. Only matters for mixed-case argument/property names.
6. Thinking with Qwen3.5-2B: upstream defaults it off for 2B and on for 4B/9B;
   here it is whatever the request says (`enable_thinking`).
