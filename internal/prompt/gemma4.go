package prompt

import (
	"encoding/json"
	"sort"
	"strings"

	"rkllm-go/internal/pyjson"
)

// Gemma 4 template. Source: chat_template.jinja of google/gemma-4-E2B-it
// ("Gemma 4 Canonical Chat Template", published 2026-07-09); special tokens
// and EOS ids from its tokenizer_config.json / generation_config.json
// (eos_token_id [1,106,50] = <eos>, <turn|>, <|tool_response>).
//
// Shape:
//
//	<|turn>system\n[<|think|>\n]{system}<|tool>declaration:NAME{…}<tool|>…<turn|>\n
//	<|turn>user\n{user}<turn|>\n
//	<|turn>model\n[<|channel>thought\n{reasoning}\n<channel|>]<|tool_call>call:NAME{k:v,…}<tool_call|><|tool_response>
//	response:NAME{value:<|"|>{tool content}<|"|>}<tool_response|>{final answer}<turn|>\n
//
// The model stops on <|tool_response> after emitting calls; the tool result is
// spliced in and the model continues in the same model turn (no new header).
//
// Deviation from upstream (for append stability): an assistant message's
// text content is rendered after its tool calls and before the tool
// responses (generation order). Upstream renders it after the responses,
// which would rewrite earlier bytes once results arrive.

const (
	g4Q        = `<|"|>`
	g4TurnEnd  = "<turn|>\n"
	g4ModelHdr = "<|turn>model\n"
	g4Thought  = "<|channel>thought\n"
	g4ThoughtE = "\n<channel|>"
	g4RespOpen = "<|tool_response>"
)

type gemma4 struct{}

func (gemma4) Family() Family { return FamilyGemma4 }
func (gemma4) StopStrings() []string {
	return []string{"<turn|>", "<|tool_response>", "<eos>"}
}
func (t gemma4) Format(m []Message, tools []json.RawMessage, o Options) string {
	s, tail := t.Render(m, tools, o)
	return s + tail
}

type g4State int

const (
	g4None       g4State = iota // last thing written closed its turn
	g4AfterText                 // assistant text turn open, closing deferred
	g4AfterCalls                // assistant emitted calls + <|tool_response> opener
	g4AfterResp                 // one or more tool responses written
)

func (t gemma4) Render(msgs []Message, tools []json.RawMessage, o Options) (string, string) {
	var b strings.Builder
	if o.AddBOS {
		b.WriteString("<bos>")
	}
	i := 0
	hasSys := len(msgs) > 0 && (msgs[0].Role == "system" || msgs[0].Role == "developer")
	if o.EnableThinking || len(tools) > 0 || hasSys {
		b.WriteString("<|turn>system\n")
		if o.EnableThinking {
			b.WriteString("<|think|>\n")
		}
		if hasSys {
			b.WriteString(strings.TrimSpace(msgs[0].Content))
			i = 1
		}
		for _, tl := range tools {
			if v, ok := parseTool(tl); ok {
				b.WriteString("<|tool>" + strings.TrimSpace(g4Declaration(v)) + "<tool|>")
			}
		}
		b.WriteString(g4TurnEnd)
	}

	state := g4None
	openerPending := false // <|tool_response> already written for the first response
	var lastCalls []ToolCall
	for ; i < len(msgs); i++ {
		m := msgs[i]
		if m.Role == "tool" {
			if !openerPending {
				b.WriteString(g4RespOpen)
			}
			openerPending = false
			name := nameForTool(m, lastCalls)
			if name == "" {
				name = "unknown"
			}
			b.WriteString("response:" + name + "{value:" + g4Q + m.Content + g4Q + "}<tool_response|>")
			state = g4AfterResp
			continue
		}
		isModel := m.Role == "assistant"
		// deferred closing of the previous turn
		if !isModel && (state == g4AfterText || state == g4AfterResp) {
			b.WriteString(g4TurnEnd)
		}
		openerPending = false
		if !isModel {
			b.WriteString("<|turn>" + m.Role + "\n" + strings.TrimSpace(m.Content) + g4TurnEnd)
			state = g4None
			continue
		}
		prefill := g4GenPrompt(state, o)
		b.WriteString(prefill)
		b.WriteString(g4Body(m, strings.HasSuffix(prefill, g4Thought)))
		lastCalls = m.ToolCalls
		if len(m.ToolCalls) > 0 {
			b.WriteString(g4RespOpen)
			openerPending = true
			state = g4AfterCalls
		} else {
			state = g4AfterText
		}
	}
	var tail string
	switch state {
	case g4AfterText:
		tail = g4TurnEnd + g4ModelHdr
	default:
		tail = g4GenPrompt(state, o)
	}
	return split(msgs, b.String(), tail)
}

// g4GenPrompt is what precedes model output in a given state.
func g4GenPrompt(state g4State, o Options) string {
	switch state {
	case g4AfterResp:
		if o.EnableThinking {
			return g4Thought
		}
		return ""
	case g4AfterCalls, g4AfterText:
		return "" // continuation of the same model turn
	}
	return g4ModelHdr
}

func g4Body(m Message, thoughtPrefilled bool) string {
	if m.Raw != "" {
		return m.Raw
	}
	var b strings.Builder
	switch {
	case thoughtPrefilled:
		b.WriteString(m.Reasoning + g4ThoughtE)
	case m.Reasoning != "":
		b.WriteString(g4Thought + m.Reasoning + g4ThoughtE)
	}
	for _, tc := range m.ToolCalls {
		b.WriteString("<|tool_call>call:" + tc.Name + "{")
		if a, ok := argsValue(tc.Arguments); ok && a.Kind == pyjson.Object {
			for j, kv := range dictsort(a) {
				if j > 0 {
					b.WriteString(",")
				}
				b.WriteString(kv.k + ":" + g4Arg(kv.v, false))
			}
		} else if !ok {
			// unparseable: render non-fatally like the unsloth variant
			s := strings.TrimSpace(tc.Arguments)
			if strings.HasPrefix(s, "{") && strings.HasSuffix(s, "}") {
				s = s[1 : len(s)-1]
			}
			b.WriteString(s)
		}
		b.WriteString("}<tool_call|>")
	}
	b.WriteString(g4StripThinking(m.Content))
	return b.String()
}

// g4StripThinking mirrors the template's strip_thinking macro.
func g4StripThinking(text string) string {
	var r strings.Builder
	for _, part := range strings.Split(text, "<channel|>") {
		if k := strings.Index(part, "<|channel>"); k >= 0 {
			r.WriteString(part[:k])
		} else {
			r.WriteString(part)
		}
	}
	return strings.TrimSpace(r.String())
}

type kv struct {
	k string
	v pyjson.Value
}

// dictsort mirrors Jinja's dictsort (case-insensitive by key, stable).
func dictsort(v pyjson.Value) []kv {
	out := make([]kv, len(v.K))
	for i := range v.K {
		out[i] = kv{v.K[i], v.V[i]}
	}
	sort.SliceStable(out, func(a, b int) bool { return strings.ToLower(out[a].k) < strings.ToLower(out[b].k) })
	return out
}

// g4Arg mirrors the template's format_argument macro.
func g4Arg(v pyjson.Value, escapeKeys bool) string {
	switch v.Kind {
	case pyjson.Null:
		return "null"
	case pyjson.String:
		return g4Q + v.S + g4Q
	case pyjson.Bool:
		if v.B {
			return "true"
		}
		return "false"
	case pyjson.Number:
		return v.S
	case pyjson.Object:
		var b strings.Builder
		b.WriteString("{")
		for j, e := range dictsort(v) {
			if j > 0 {
				b.WriteString(",")
			}
			if escapeKeys {
				b.WriteString(g4Q + e.k + g4Q)
			} else {
				b.WriteString(e.k)
			}
			b.WriteString(":" + g4Arg(e.v, escapeKeys))
		}
		b.WriteString("}")
		return b.String()
	case pyjson.Array:
		var b strings.Builder
		b.WriteString("[")
		for j, e := range v.A {
			if j > 0 {
				b.WriteString(",")
			}
			b.WriteString(g4Arg(e, escapeKeys))
		}
		b.WriteString("]")
		return b.String()
	}
	return ""
}

func truthy(v pyjson.Value, ok bool) bool {
	if !ok {
		return false
	}
	switch v.Kind {
	case pyjson.Null:
		return false
	case pyjson.Bool:
		return v.B
	case pyjson.String:
		return v.S != ""
	case pyjson.Number:
		return v.S != "0" && v.S != "0.0"
	case pyjson.Array:
		return len(v.A) > 0
	case pyjson.Object:
		return len(v.K) > 0
	}
	return false
}

// upperType renders `x | upper` for a schema "type" (missing -> "").
func upperType(v pyjson.Value, ok bool) string {
	if !ok {
		return ""
	}
	if v.Kind == pyjson.String {
		return strings.ToUpper(v.S)
	}
	if v.Kind == pyjson.Array { // Python str(list).upper(), e.g. "['STRING', 'NULL']"
		parts := make([]string, len(v.A))
		for i, e := range v.A {
			parts[i] = "'" + e.PyStr() + "'"
		}
		return strings.ToUpper("[" + strings.Join(parts, ", ") + "]")
	}
	return strings.ToUpper(v.PyStr())
}

func g4Required(v pyjson.Value) string {
	var b strings.Builder
	b.WriteString("required:[")
	for j, e := range v.A {
		if j > 0 {
			b.WriteString(",")
		}
		b.WriteString(g4Q + e.PyStr() + g4Q)
	}
	b.WriteString("]")
	return b.String()
}

var g4StandardKeys = map[string]bool{"description": true, "type": true, "properties": true, "required": true, "nullable": true}

// g4Params mirrors the template's format_parameters macro.
func g4Params(b *strings.Builder, props pyjson.Value, filterKeys bool) {
	first := true
	for _, e := range dictsort(props) {
		if filterKeys && g4StandardKeys[e.k] {
			continue
		}
		if !first {
			b.WriteString(",")
		}
		first = false
		val := e.v
		b.WriteString(e.k + ":{")
		addComma := false
		comma := func() {
			if addComma {
				b.WriteString(",")
			} else {
				addComma = true
			}
		}
		if d, ok := val.Get("description"); truthy(d, ok) {
			b.WriteString("description:" + g4Q + d.PyStr() + g4Q)
			addComma = true
		}
		tv, tok := val.Get("type")
		typ := upperType(tv, tok)
		switch typ {
		case "STRING":
			if en, ok := val.Get("enum"); truthy(en, ok) {
				comma()
				b.WriteString("enum:" + g4Arg(en, true))
			}
		case "ARRAY":
			if items, ok := val.Get("items"); ok && items.Kind == pyjson.Object && len(items.K) > 0 {
				comma()
				b.WriteString("items:{")
				firstItem := true
				for _, ie := range dictsort(items) {
					if ie.v.Kind == pyjson.Null {
						continue
					}
					if !firstItem {
						b.WriteString(",")
					}
					firstItem = false
					switch ie.k {
					case "properties":
						b.WriteString("properties:{")
						if ie.v.Kind == pyjson.Object {
							g4Params(b, ie.v, false)
						}
						b.WriteString("}")
					case "required":
						b.WriteString(g4Required(ie.v))
					case "type":
						if ie.v.Kind == pyjson.String {
							b.WriteString("type:" + g4Arg(pyjson.Str(strings.ToUpper(ie.v.S)), true))
						} else {
							up := pyjson.NewArray()
							for _, x := range ie.v.A {
								up.A = append(up.A, pyjson.Str(strings.ToUpper(x.PyStr())))
							}
							b.WriteString("type:" + g4Arg(up, true))
						}
					default:
						b.WriteString(ie.k + ":" + g4Arg(ie.v, true))
					}
				}
				b.WriteString("}")
			}
		}
		if n, ok := val.Get("nullable"); truthy(n, ok) {
			comma()
			b.WriteString("nullable:true")
		}
		if typ == "OBJECT" {
			if p, ok := val.Get("properties"); ok && p.Kind == pyjson.Object {
				comma()
				b.WriteString("properties:{")
				g4Params(b, p, false)
				b.WriteString("}")
			} else if val.Kind == pyjson.Object {
				comma()
				b.WriteString("properties:{")
				g4Params(b, val, true)
				b.WriteString("}")
			}
			if r, ok := val.Get("required"); truthy(r, ok) {
				comma()
				b.WriteString(g4Required(r))
			}
		}
		comma()
		b.WriteString("type:" + g4Q + typ + g4Q + "}")
	}
}

// g4Declaration mirrors format_function_declaration.
func g4Declaration(tool pyjson.Value) string {
	fn, _ := tool.Get("function")
	var b strings.Builder
	b.WriteString("declaration:" + fn.GetString("name") + "{description:" + g4Q + fn.GetString("description") + g4Q)
	if params, ok := fn.Get("parameters"); truthy(params, ok) {
		b.WriteString(",parameters:{")
		if p, ok := params.Get("properties"); truthy(p, ok) {
			b.WriteString("properties:{")
			g4Params(&b, p, false)
			b.WriteString("},")
		}
		if r, ok := params.Get("required"); truthy(r, ok) {
			b.WriteString(g4Required(r) + ",")
		}
		if ty, ok := params.Get("type"); truthy(ty, ok) {
			b.WriteString("type:" + g4Q + upperType(ty, ok) + g4Q + "}")
		}
	}
	if resp, ok := fn.Get("response"); ok {
		b.WriteString(",response:{")
		if d, ok := resp.Get("description"); truthy(d, ok) {
			b.WriteString("description:" + g4Q + d.PyStr() + g4Q + ",")
		}
		if ty, ok := resp.Get("type"); upperType(ty, ok) == "OBJECT" {
			b.WriteString("type:" + g4Q + "OBJECT" + g4Q + "}")
		}
	}
	b.WriteString("}")
	return b.String()
}
