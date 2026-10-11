package api

import (
	"encoding/json"
	"strings"
	"testing"

	"rkllm-go/internal/core"
	"rkllm-go/internal/prompt"
)

func TestContentForms(t *testing.T) {
	var m []Message
	err := json.Unmarshal([]byte(`[{"role":"user","content":"a"},{"role":"assistant","content":null},{"role":"user","content":[{"type":"text","text":"x"},{"type":"text","text":"y"}]}]`), &m)
	if err != nil || m[0].Content != "a" || m[1].Content != "" || m[2].Content != "xy" {
		t.Fatalf("%v %+v", err, m)
	}
}

func TestStaticPrefix(t *testing.T) {
	tools := []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"f","parameters":{}}}`)}
	for _, f := range []prompt.Family{prompt.FamilyQwen3, prompt.FamilyQwen35, prompt.FamilyGemma3} {
		tpl, _ := prompt.Get(f)
		msgs := []prompt.Message{{Role: "system", Content: "be nice"}, {Role: "user", Content: "hello"}}
		sp := staticPrefix(tpl, msgs, tools, prompt.Options{})
		full := tpl.Format(msgs, tools, prompt.Options{})
		if sp == "" || !strings.HasPrefix(full, sp) || !strings.Contains(sp, "be nice") {
			t.Fatalf("%s: %q", f, sp)
		}
		// also a prefix of later turns
		later := tpl.Format(append(msgs, prompt.Message{Role: "assistant", Content: "hi"}, prompt.Message{Role: "user", Content: "more"}), tools, prompt.Options{})
		if !strings.HasPrefix(later, sp) {
			t.Fatalf("%s: not stable", f)
		}
	}
}

func TestRawSubstitution(t *testing.T) {
	raw := core.NewRawMap(10)
	gen := "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>"
	o := parseOutput(prompt.FamilyQwen3, gen, []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"a":{"type":"integer"}}}}}`)}, false)
	if len(o.calls) != 1 {
		t.Fatalf("calls %+v", o)
	}
	(&Handler{}).remember(raw, o, gen)
	in := []Message{{Role: "assistant", Content: Content(o.content), ToolCalls: []ToolCall{{ID: "call_1_0", Type: "function", Function: o.calls[0].Function}}}}
	pm := convertMessages(in, raw)
	if pm[0].Raw != gen {
		t.Fatalf("raw not substituted: %+v", pm[0])
	}
}

func TestSplitLive(t *testing.T) {
	r, c := splitLive("<think>\nabc</thi", true)
	if r != "abc" || c != "" {
		t.Fatalf("%q %q", r, c)
	}
	r, c = splitLive("<think>\nabc\n</think>\n\nanswer", true)
	if r != "abc" || c != "answer" {
		t.Fatalf("%q %q", r, c)
	}
	if r, c := splitLive("plain", false); r != "" || c != "plain" {
		t.Fatal("no thinking")
	}
}
