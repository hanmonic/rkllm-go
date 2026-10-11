package prompt

import (
	"encoding/json"
	"strings"

	"rkllm-go/internal/pyjson"
)

// Gemma 3 / Gemma 3n template. Source: chat_template.jinja of
// unsloth/gemma-3-4b-it and unsloth/gemma-3n-E4B-it (mirrors of the gated
// google/* repos; the two differ only in multimodal placeholders).
//
// No system role: the system message is prepended to the first turn's
// content followed by "\n\n". No native tool format: tools are described with
// the same polyfill the RKLLM runtime (minja) uses for templates without tool
// support — strings "You can call any of the following tools to satisfy the
// user's requests: " and "Example tool call syntax:" are in librkllmrt.so
// 1.3.1, and the old benchmarks (gemma3n-e4b 100% right_args) ran with it.
// Calls are emitted as JSON {"tool_calls": [{"name", "arguments", "id"}]}.
// Tool results go back as a user turn holding {"tool_response": {...}} JSON.

const (
	g3Start = "<start_of_turn>"
	g3End   = "<end_of_turn>\n"

	g3ToolsIntro   = "You can call any of the following tools to satisfy the user's requests: "
	g3ToolsExample = "\n\nExample tool call syntax:\n\n"
)

type gemma3 struct{}

func (gemma3) Family() Family        { return FamilyGemma3 }
func (gemma3) StopStrings() []string { return []string{"<end_of_turn>", "<eos>"} }
func (t gemma3) Format(m []Message, tools []json.RawMessage, o Options) string {
	s, tail := t.Render(m, tools, o)
	return s + tail
}

func gemma3ToolsPrompt(tools []json.RawMessage) string {
	arr := pyjson.NewArray()
	for _, tl := range tools {
		if v, ok := parseTool(tl); ok {
			arr.A = append(arr.A, v)
		}
	}
	ex := pyjson.NewObject()
	call := pyjson.NewObject()
	call.Set("name", pyjson.Str("tool_name"))
	a := pyjson.NewObject()
	a.Set("arg1", pyjson.Str("some_value"))
	call.Set("arguments", a)
	call.Set("id", pyjson.Str("call_1___"))
	ex.Set("tool_calls", pyjson.NewArray(call))
	return g3ToolsIntro + arr.DumpIndent("  ") + g3ToolsExample + ex.DumpIndent("  ")
}

func (t gemma3) Render(msgs []Message, tools []json.RawMessage, o Options) (string, string) {
	all := msgs
	var b strings.Builder
	if o.AddBOS {
		b.WriteString("<bos>")
	}
	sys := ""
	hasSys := false
	if len(msgs) > 0 && msgs[0].Role == "system" {
		sys, hasSys = msgs[0].Content, true
		msgs = msgs[1:]
	}
	if len(tools) > 0 {
		tp := gemma3ToolsPrompt(tools)
		if hasSys && sys != "" {
			sys = sys + "\n\n" + tp
		} else {
			sys = tp
		}
		hasSys = true
	}
	prefix := ""
	if hasSys {
		prefix = sys + "\n\n"
	}
	first := true
	takePrefix := func() string {
		if first {
			first = false
			return prefix
		}
		return ""
	}

	pending := ""
	var lastCalls []ToolCall
	for _, m := range msgs {
		if m.Role == "tool" {
			if pending == "" {
				b.WriteString(g3Start + "user\n" + takePrefix())
			} else {
				b.WriteString("\n")
			}
			b.WriteString(gemma3ToolResponse(m, lastCalls))
			pending = g3End
			continue
		}
		b.WriteString(pending)
		pending = ""
		if m.Role == "assistant" {
			b.WriteString(g3Start + "model\n" + takePrefix())
			b.WriteString(gemma3Body(m))
			b.WriteString(g3End)
			lastCalls = m.ToolCalls
			continue
		}
		// user, or a non-leading system message (no system role in Gemma)
		b.WriteString(g3Start + "user\n" + takePrefix() + strings.TrimSpace(m.Content) + g3End)
	}
	return split(all, b.String(), pending+g3Start+"model\n")
}

func gemma3Body(m Message) string {
	if m.Raw != "" {
		return m.Raw
	}
	content := strings.TrimSpace(m.Content)
	if len(m.ToolCalls) == 0 {
		return content
	}
	arr := pyjson.NewArray()
	for _, tc := range m.ToolCalls {
		c := pyjson.NewObject()
		c.Set("name", pyjson.Str(tc.Name))
		if a, ok := argsValue(tc.Arguments); ok {
			c.Set("arguments", a)
		} else {
			c.Set("arguments", pyjson.Str(tc.Arguments))
		}
		if tc.ID != "" {
			c.Set("id", pyjson.Str(tc.ID))
		}
		arr.A = append(arr.A, c)
	}
	obj := pyjson.NewObject()
	obj.Set("tool_calls", arr)
	js := "```json\n" + obj.DumpIndent("  ") + "\n```"
	if content != "" {
		return content + "\n\n" + js
	}
	return js
}

func gemma3ToolResponse(m Message, calls []ToolCall) string {
	inner := pyjson.NewObject()
	if name := nameForTool(m, calls); name != "" {
		inner.Set("tool", pyjson.Str(name))
	}
	if m.ToolCallID != "" {
		inner.Set("tool_call_id", pyjson.Str(m.ToolCallID))
	}
	if v, err := pyjson.ParseString(strings.TrimSpace(m.Content)); err == nil && (v.Kind == pyjson.Object || v.Kind == pyjson.Array) {
		inner.Set("content", v)
	} else {
		inner.Set("content", pyjson.Str(m.Content))
	}
	obj := pyjson.NewObject()
	obj.Set("tool_response", inner)
	return obj.DumpIndent("  ")
}
