package prompt

import (
	"encoding/json"
	"strings"

	"rkllm-go/internal/pyjson"
)

// Qwen3 and Qwen3.5 templates. Source: HF tokenizer_config.json chat_template
// of Qwen/Qwen3-{0.6B,4B} and chat_template.jinja of Qwen/Qwen3.5-{2B,4B,9B}
// (identical except Qwen3.5-2B defaults enable_thinking to false; here
// thinking is always explicit via Options).

const (
	imStart = "<|im_start|>"
	imEnd   = "<|im_end|>\n"

	qwen3ToolsHead = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
	qwen3ToolsTail = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"

	qwen35ToolsHead = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
	qwen35ToolsTail = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"
)

var qwenStops = []string{"<|im_end|>", "<|endoftext|>"}

// ---------------- Qwen3 ----------------

type qwen3 struct{}

func (qwen3) Family() Family        { return FamilyQwen3 }
func (qwen3) StopStrings() []string { return append([]string(nil), qwenStops...) }
func (t qwen3) Format(m []Message, tools []json.RawMessage, o Options) string {
	s, tail := t.Render(m, tools, o)
	return s + tail
}

func (qwen3) genPrompt(o Options) string {
	if o.EnableThinking {
		return imStart + "assistant\n"
	}
	return imStart + "assistant\n<think>\n\n</think>\n\n"
}

func (t qwen3) Render(msgs []Message, tools []json.RawMessage, o Options) (string, string) {
	var b strings.Builder
	i := 0
	if len(tools) > 0 {
		b.WriteString(imStart + "system\n")
		if len(msgs) > 0 && msgs[0].Role == "system" {
			b.WriteString(msgs[0].Content + "\n\n")
			i = 1
		}
		b.WriteString(qwen3ToolsHead)
		for _, tl := range tools {
			b.WriteString("\n")
			b.WriteString(toolJSON(tl))
		}
		b.WriteString(qwen3ToolsTail + imEnd)
	} else if len(msgs) > 0 && msgs[0].Role == "system" {
		b.WriteString(imStart + "system\n" + msgs[0].Content + imEnd)
		i = 1
	}
	pending := ""
	for ; i < len(msgs); i++ {
		m := msgs[i]
		if m.Role == "tool" {
			if pending == "" {
				b.WriteString(imStart + "user")
			}
			b.WriteString("\n<tool_response>\n" + m.Content + "\n</tool_response>")
			pending = imEnd
			continue
		}
		b.WriteString(pending)
		pending = ""
		if m.Role == "assistant" {
			b.WriteString(t.genPrompt(o))
			b.WriteString(t.body(m, o))
			b.WriteString(imEnd)
			continue
		}
		b.WriteString(imStart + m.Role + "\n" + m.Content + imEnd)
	}
	return split(msgs, b.String(), pending+t.genPrompt(o))
}

// body is everything after the generation prompt, up to <|im_end|>.
func (qwen3) body(m Message, o Options) string {
	if m.Raw != "" {
		return m.Raw
	}
	content, reasoning := splitThink(m)
	var b strings.Builder
	if o.EnableThinking {
		b.WriteString("<think>\n" + strings.Trim(reasoning, "\n") + "\n</think>\n\n" + strings.TrimLeft(content, "\n"))
	} else {
		b.WriteString(content)
	}
	for j, tc := range m.ToolCalls {
		if j > 0 || content != "" {
			b.WriteString("\n")
		}
		args := strings.TrimSpace(tc.Arguments)
		if args == "" {
			args = "{}"
		}
		b.WriteString("<tool_call>\n{\"name\": \"" + tc.Name + "\", \"arguments\": " + args + "}\n</tool_call>")
	}
	return b.String()
}

// ---------------- Qwen3.5 ----------------

type qwen35 struct{}

func (qwen35) Family() Family        { return FamilyQwen35 }
func (qwen35) StopStrings() []string { return append([]string(nil), qwenStops...) }
func (t qwen35) Format(m []Message, tools []json.RawMessage, o Options) string {
	s, tail := t.Render(m, tools, o)
	return s + tail
}

func (qwen35) genPrompt(o Options) string {
	if o.EnableThinking {
		return imStart + "assistant\n<think>\n"
	}
	return imStart + "assistant\n<think>\n\n</think>\n\n"
}

func (t qwen35) Render(msgs []Message, tools []json.RawMessage, o Options) (string, string) {
	var b strings.Builder
	i := 0
	hasSys := len(msgs) > 0 && msgs[0].Role == "system"
	if len(tools) > 0 {
		b.WriteString(imStart + "system\n" + qwen35ToolsHead)
		for _, tl := range tools {
			b.WriteString("\n")
			b.WriteString(toolJSON(tl))
		}
		b.WriteString(qwen35ToolsTail)
		if hasSys {
			if c := strings.TrimSpace(msgs[0].Content); c != "" {
				b.WriteString("\n\n" + c)
			}
		}
		b.WriteString(imEnd)
	} else if hasSys {
		b.WriteString(imStart + "system\n" + strings.TrimSpace(msgs[0].Content) + imEnd)
	}
	if hasSys {
		i = 1
	}
	pending := ""
	for ; i < len(msgs); i++ {
		m := msgs[i]
		if m.Role == "tool" {
			if pending == "" {
				b.WriteString(imStart + "user")
			}
			b.WriteString("\n<tool_response>\n" + strings.TrimSpace(m.Content) + "\n</tool_response>")
			pending = imEnd
			continue
		}
		b.WriteString(pending)
		pending = ""
		if m.Role == "assistant" {
			b.WriteString(t.genPrompt(o))
			b.WriteString(t.body(m, o))
			b.WriteString(imEnd)
			continue
		}
		// user, and (non-HF, which raises) a system message not in first position.
		b.WriteString(imStart + m.Role + "\n" + strings.TrimSpace(m.Content) + imEnd)
	}
	return split(msgs, b.String(), pending+t.genPrompt(o))
}

func (qwen35) body(m Message, o Options) string {
	if m.Raw != "" {
		return m.Raw
	}
	mm := m
	mm.Content = strings.TrimSpace(m.Content)
	content, reasoning := splitThink(mm)
	var b strings.Builder
	if o.EnableThinking {
		// the generation prompt already holds "<think>\n"
		b.WriteString(strings.TrimSpace(reasoning) + "\n</think>\n\n")
	}
	b.WriteString(content)
	for j, tc := range m.ToolCalls {
		switch {
		case j > 0:
			b.WriteString("\n")
		case strings.TrimSpace(content) != "":
			b.WriteString("\n\n")
		}
		b.WriteString("<tool_call>\n<function=" + tc.Name + ">\n")
		if args, ok := argsValue(tc.Arguments); ok {
			for k, key := range args.K {
				v := args.V[k]
				b.WriteString("<parameter=" + key + ">\n")
				if v.Kind == pyjson.Array || v.Kind == pyjson.Object { // tojson
					b.WriteString(v.Dump())
				} else {
					b.WriteString(v.PyStr()) // Jinja `| string`: True/False/None
				}
				b.WriteString("\n</parameter>\n")
			}
		}
		b.WriteString("</function>\n</tool_call>")
	}
	return b.String()
}
