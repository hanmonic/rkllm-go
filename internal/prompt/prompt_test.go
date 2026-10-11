package prompt

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// testdata/cases.json holds OpenAI-shaped conversations; testdata/golden/<family>/<case>.txt
// is the upstream HF template rendered by testdata/render_hf.py (jinja2, as
// transformers renders it). Cases list in "hf" the families for which our
// rendering policy coincides with upstream (see package doc for where and why
// it deviates).

type caseFile struct {
	Tools []json.RawMessage `json:"tools"`
	Cases []struct {
		Name     string       `json:"name"`
		HF       []string     `json:"hf"`
		Tools    bool         `json:"tools"`
		Thinking bool         `json:"thinking"`
		Messages []apiMessage `json:"messages"`
	} `json:"cases"`
}

type apiMessage struct {
	Role       string `json:"role"`
	Content    string `json:"content"`
	Reasoning  string `json:"reasoning_content"`
	ToolCallID string `json:"tool_call_id"`
	Name       string `json:"name"`
	ToolCalls  []struct {
		ID       string `json:"id"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
}

func loadCases(t *testing.T) caseFile {
	t.Helper()
	b, err := os.ReadFile("testdata/cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cf caseFile
	if err := json.Unmarshal(b, &cf); err != nil {
		t.Fatal(err)
	}
	return cf
}

func toMessages(in []apiMessage) []Message {
	out := make([]Message, len(in))
	for i, m := range in {
		out[i] = Message{Role: m.Role, Content: m.Content, Reasoning: m.Reasoning, ToolCallID: m.ToolCallID, Name: m.Name}
		for _, tc := range m.ToolCalls {
			out[i].ToolCalls = append(out[i].ToolCalls, ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
		}
	}
	return out
}

func optsFor(f Family, thinking bool) Options {
	return Options{EnableThinking: thinking, AddBOS: f == FamilyGemma3 || f == FamilyGemma4}
}

func firstDiff(a, b string) string {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	lo := i - 60
	if lo < 0 {
		lo = 0
	}
	clip := func(s string) string {
		hi := i + 60
		if hi > len(s) {
			hi = len(s)
		}
		if lo > len(s) {
			return ""
		}
		return s[lo:hi]
	}
	return "at byte " + itoa(i) + "\n got: " + strings.ReplaceAll(clip(a), "\n", `\n`) + "\nwant: " + strings.ReplaceAll(clip(b), "\n", `\n`)
}

func itoa(i int) string { b, _ := json.Marshal(i); return string(b) }

func TestGoldenMatchesUpstreamTemplates(t *testing.T) {
	cf := loadCases(t)
	n := 0
	for _, c := range cf.Cases {
		for _, fam := range c.HF {
			f := Family(fam)
			tpl, err := Get(f)
			if err != nil {
				t.Fatal(err)
			}
			want, err := os.ReadFile(filepath.Join("testdata", "golden", fam, c.Name+".txt"))
			if err != nil {
				t.Fatal(err)
			}
			var tools []json.RawMessage
			if c.Tools {
				tools = cf.Tools
			}
			got := tpl.Format(toMessages(c.Messages), tools, optsFor(f, c.Thinking))
			if got != string(want) {
				t.Errorf("%s/%s differs from upstream template %s", fam, c.Name, firstDiff(got, string(want)))
			}
			n++
		}
	}
	if n < 20 {
		t.Fatalf("only %d golden comparisons ran", n)
	}
}

// TestAppendStability: for every family, case and split point, the stable
// part of Format(msgs[:n]) is a prefix of Format(msgs[:m]) for all m > n, and
// Format == stable + tail.
func TestAppendStability(t *testing.T) {
	cf := loadCases(t)
	for _, f := range Families() {
		tpl, _ := Get(f)
		for _, c := range cf.Cases {
			var tools []json.RawMessage
			if c.Tools {
				tools = cf.Tools
			}
			for _, thinking := range []bool{false, true} {
				o := optsFor(f, thinking)
				msgs := toMessages(c.Messages)
				for n := 0; n <= len(msgs); n++ {
					stable, tail := tpl.Render(msgs[:n], tools, o)
					if full := tpl.Format(msgs[:n], tools, o); full != stable+tail {
						t.Fatalf("%s/%s n=%d: Format != stable+tail", f, c.Name, n)
					}
					if StablePrefixLen(tpl, msgs[:n], tools, o) != len(stable) {
						t.Fatalf("%s/%s n=%d: StablePrefixLen mismatch", f, c.Name, n)
					}
					for m := n + 1; m <= len(msgs); m++ {
						longer := tpl.Format(msgs[:m], tools, o)
						if !strings.HasPrefix(longer, stable) {
							t.Errorf("%s/%s thinking=%v: stable(msgs[:%d]) not a prefix of Format(msgs[:%d]) %s",
								f, c.Name, thinking, n, m, firstDiff(longer, stable))
						}
					}
				}
			}
		}
	}
}

// TestGenerationContinuity is the KV-cache property proper: the full prompt
// that generated assistant message i (including the generation prompt) is a
// prefix of the prompt for every later turn, as long as the assistant turn is
// re-rendered exactly as generated. Raw is used to model "exactly as
// generated" with the canonical body.
func TestGenerationContinuity(t *testing.T) {
	cf := loadCases(t)
	for _, f := range Families() {
		tpl, _ := Get(f)
		for _, c := range cf.Cases {
			var tools []json.RawMessage
			if c.Tools {
				tools = cf.Tools
			}
			for _, thinking := range []bool{false, true} {
				o := optsFor(f, thinking)
				msgs := toMessages(c.Messages)
				for i, m := range msgs {
					if m.Role != "assistant" {
						continue
					}
					if f == FamilyGemma4 && i > 0 && msgs[i-1].Role == "assistant" {
						continue // upstream continuation quirk: assistant directly after assistant
					}
					prompt := tpl.Format(msgs[:i], tools, o)
					later := tpl.Format(msgs, tools, o)
					if !strings.HasPrefix(later, prompt) {
						t.Errorf("%s/%s thinking=%v: prompt for assistant #%d not a prefix of later prompt %s",
							f, c.Name, thinking, i, firstDiff(later, prompt))
					}
				}
			}
		}
	}
}

func TestRawIsRenderedVerbatim(t *testing.T) {
	gen := "<tool_call>\n<function=get_weather>\n<parameter=city>\nSydney\n</parameter>\n</function>\n</tool_call>"
	msgs := []Message{
		{Role: "user", Content: "weather in sydney"},
		{Role: "assistant", Raw: gen, ToolCalls: []ToolCall{{ID: "x", Name: "get_weather", Arguments: `{"city":"Sydney"}`}}},
		{Role: "tool", ToolCallID: "x", Content: "sunny"},
	}
	for _, f := range Families() {
		tpl, _ := Get(f)
		o := Options{}
		before := tpl.Format(msgs[:1], nil, o)
		after := tpl.Format(msgs, nil, o)
		if !strings.HasPrefix(after, before+gen) {
			t.Errorf("%s: Raw not rendered right after the generation prompt:\n%q", f, after)
		}
	}
}

func TestQwen3NoThinkPrefill(t *testing.T) {
	tpl, _ := Get(FamilyQwen3)
	got := tpl.Format([]Message{{Role: "user", Content: "hi"}}, nil, Options{})
	want := "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
	if got != want {
		t.Fatalf("got %q", got)
	}
	// prior assistant content with a think block is split, reasoning kept when thinking is on
	msgs := []Message{{Role: "user", Content: "q"}, {Role: "assistant", Content: "<think>\nhmm\n</think>\n\nA"}, {Role: "user", Content: "q2"}}
	got = tpl.Format(msgs, nil, Options{EnableThinking: true})
	if !strings.Contains(got, "<|im_start|>assistant\n<think>\nhmm\n</think>\n\nA<|im_end|>\n") {
		t.Fatalf("reasoning not re-rendered: %q", got)
	}
}

func TestToolsInSystemQwen3(t *testing.T) {
	tpl, _ := Get(FamilyQwen3)
	tools := []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"f","description":"d <x> & y","parameters":{"type":"object","properties":{"a":{"type":"string"}}}}}`)}
	got := tpl.Format([]Message{{Role: "user", Content: "hi"}}, tools, Options{})
	want := `{"type": "function", "function": {"name": "f", "description": "d <x> & y", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}}`
	if !strings.Contains(got, "<tools>\n"+want+"\n</tools>") {
		t.Fatalf("tool not rendered python-style: %q", got)
	}
}

func TestGemma3ToolPolyfill(t *testing.T) {
	cf := loadCases(t)
	tpl, _ := Get(FamilyGemma3)
	msgs := []Message{
		{Role: "system", Content: "You are a home automation agent."},
		{Role: "user", Content: "Turn off the living room light."},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c1", Name: "set_light", Arguments: `{"room": "living room", "on": false}`}}},
		{Role: "tool", ToolCallID: "c1", Content: `{"ok": true}`},
		{Role: "tool", ToolCallID: "c1", Content: `done`},
	}
	got := tpl.Format(msgs, cf.Tools, Options{})
	for _, want := range []string{
		"<start_of_turn>user\nYou are a home automation agent.\n\nYou can call any of the following tools to satisfy the user's requests: [\n  {\n    \"type\": \"function\",",
		"Example tool call syntax:\n\n{\n  \"tool_calls\": [\n    {\n      \"name\": \"tool_name\",",
		"\n\nTurn off the living room light.<end_of_turn>\n<start_of_turn>model\n```json\n{\n  \"tool_calls\": [\n    {\n      \"name\": \"set_light\",\n      \"arguments\": {\n        \"room\": \"living room\",\n        \"on\": false\n      },\n      \"id\": \"c1\"",
		"<start_of_turn>user\n{\n  \"tool_response\": {\n    \"tool\": \"set_light\",\n    \"tool_call_id\": \"c1\",\n    \"content\": {\n      \"ok\": true\n    }\n  }\n}\n{\n  \"tool_response\": {",
		"\"content\": \"done\"\n  }\n}<end_of_turn>\n<start_of_turn>model\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in\n%s", want, got)
		}
	}
	if strings.Contains(got, "<start_of_turn>system") {
		t.Error("gemma3 must not emit a system role")
	}
}

func TestGemma4ContentBeforeResponses(t *testing.T) {
	tpl, _ := Get(FamilyGemma4)
	msgs := []Message{
		{Role: "user", Content: "Weather in Paris?"},
		{Role: "assistant", Content: "Checking.", ToolCalls: []ToolCall{{ID: "c1", Name: "get_weather", Arguments: `{"city": "Paris"}`}}},
		{Role: "tool", ToolCallID: "c1", Content: "sunny"},
	}
	got := tpl.Format(msgs, nil, Options{})
	want := "<|turn>user\nWeather in Paris?<turn|>\n<|turn>model\n<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>Checking.<|tool_response>response:get_weather{value:<|\"|>sunny<|\"|>}<tool_response|>"
	if got != want {
		t.Fatalf("got  %q\nwant %q", got, want)
	}
}

func TestRegistry(t *testing.T) {
	cases := map[string]Family{
		"qwen3-0.6b": FamilyQwen3, "qwen3-1.7b": FamilyQwen3, "qwen3-4b-16k": FamilyQwen3, "qwen3-8b-allnpu": FamilyQwen3,
		"q35-2b": FamilyQwen35, "q35-4b": FamilyQwen35, "q35-9b": FamilyQwen35, "Qwen3.5-4B": FamilyQwen35,
		"gemma3n-e4b": FamilyGemma3, "gemma3-1b": FamilyGemma3, "gemma3-4b": FamilyGemma3,
		"gemma4-e2b": FamilyGemma4, "/opt/rkllm/models/gemma4-e2b/model.rkllm": FamilyGemma4,
		"llama3.2-1b": "", "deepseek-r1-7b": "",
	}
	for name, want := range cases {
		if got := FamilyForModel(name); got != want {
			t.Errorf("FamilyForModel(%q) = %q, want %q", name, got, want)
		}
	}
	if tpl, err := Resolve("gemma3n", "whatever"); err != nil || tpl.Family() != FamilyGemma3 {
		t.Errorf("Resolve configured: %v %v", tpl, err)
	}
	if _, err := Resolve("", "llama3.2-1b"); err == nil {
		t.Error("expected error for unknown family")
	}
	if _, err := Resolve("bogus", "q35-4b"); err == nil {
		t.Error("expected error for bogus configured family")
	}
	for _, f := range Families() {
		tpl, _ := Get(f)
		if len(tpl.StopStrings()) == 0 || tpl.Family() != f {
			t.Errorf("%s: bad stop strings/family", f)
		}
	}
}
