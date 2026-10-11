package toolparse

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
)

// testdata/outputs.json: real model outputs copied from the benchmark result
// saved benchmark outputs ("raw" fields).
// testdata/tools.json: the benchmark's tool schemas (toolbench.py + toolbench2.py).

func loadTools(t *testing.T) []ToolSchema {
	t.Helper()
	b, err := os.ReadFile("testdata/tools.json")
	if err != nil {
		t.Fatal(err)
	}
	var raw []json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatal(err)
	}
	ts := SchemasFromTools(raw)
	if len(ts) != 12 {
		t.Fatalf("got %d tools", len(ts))
	}
	return ts
}

func loadOutputs(t *testing.T) map[string]struct{ Family, Text string } {
	t.Helper()
	b, err := os.ReadFile("testdata/outputs.json")
	if err != nil {
		t.Fatal(err)
	}
	var rows []struct{ ID, Family, Text string }
	if err := json.Unmarshal(b, &rows); err != nil {
		t.Fatal(err)
	}
	m := map[string]struct{ Family, Text string }{}
	for _, r := range rows {
		m[r.ID] = struct{ Family, Text string }{r.Family, r.Text}
	}
	return m
}

type wantCall struct{ name, args string }

func check(t *testing.T, id, content string, calls []ToolCall, wantContent string, want []wantCall) {
	t.Helper()
	if content != wantContent {
		t.Errorf("%s: content = %q, want %q", id, content, wantContent)
	}
	if len(calls) != len(want) {
		t.Fatalf("%s: got %d calls %+v, want %d", id, len(calls), calls, len(want))
	}
	for i, w := range want {
		c := calls[i]
		if c.Type != "function" || !strings.HasPrefix(c.ID, "call_") {
			t.Errorf("%s: bad id/type %+v", id, c)
		}
		if c.Function.Name != w.name || c.Function.Arguments != w.args {
			t.Errorf("%s call %d: got %s %s, want %s %s", id, i, c.Function.Name, c.Function.Arguments, w.name, w.args)
		}
		if !json.Valid([]byte(c.Function.Arguments)) {
			t.Errorf("%s: arguments not valid JSON: %s", id, c.Function.Arguments)
		}
	}
}

func TestRealOutputs(t *testing.T) {
	tools := loadTools(t)
	outs := loadOutputs(t)
	cases := []struct {
		id      string
		content string // "=" means unchanged (trimmed) input
		calls   []wantCall
	}{
		{"qwen3_4b_weather", "", []wantCall{{"get_weather", `{"city": "Sydney"}`}}},
		{"qwen3_4b_bedroom", "", []wantCall{{"set_light", `{"room": "bedroom", "on": true, "brightness": 30}`}}},
		{"qwen3_0.6b_reminder", "", []wantCall{{"create_reminder", `{"text": "Check the oven", "minutes_from_now": 15}`}}},
		{"qwen3_4b_chat", "=", nil},
		// Qwen3.5 XML: lowercase and Python-style bools, integer strings
		{"q35_2b_living_false", "", []wantCall{{"set_light", `{"room": "living room", "on": false}`}}},
		{"q35_4b_bedroom_True", "", []wantCall{{"set_light", `{"room": "bedroom", "on": true, "brightness": 30}`}}},
		{"q35_4b_reminder", "", []wantCall{{"create_reminder", `{"text": "Check the oven", "minutes_from_now": 15}`}}},
		{"q35_9b_math", "=", nil},
		// Gemma 4 with special tokens stripped by the runtime
		{"gemma4_living_off", "", []wantCall{{"set_light", `{"room": "living room", "on": false}`}}},
		{"gemma4_bedroom", "", []wantCall{{"set_light", `{"brightness": 30, "room": "bedroom"}`}}},
		{"gemma4_kitchen", "", []wantCall{{"set_light", `{"brightness": 100, "on": true, "room": "kitchen"}`}}},
		{"gemma4_restart", "", []wantCall{{"restart_service", `{"host": "web01", "service": "nginx"}`}}},
		{"gemma4_chat", "=", nil},
		// Gemma 3/3n polyfill JSON in fences, with preambles
		{"gemma3n_preamble", "Okay, I understand.", []wantCall{{"get_weather", `{"city": "Sydney"}`}}},
		{"gemma3n_grafana", "Okay, I understand. Here's the tool call to restart the grafana service on lgtm:", []wantCall{{"restart_service", `{"host": "lgtm", "service": "grafana"}`}}},
		{"gemma3n_bedroom", "", []wantCall{{"set_light", `{"room": "bedroom", "on": true, "brightness": 30}`}}},
		{"gemma3_4b_tool_name", "=", nil}, // hallucinated tool name: left as content
		{"gemma3_1b_broken", "=", nil},    // invalid JSON: left as content
		// Llama 3.2: "parameters" key, stringly-typed values coerced by schema
		{"llama32_3b_reminder_str", "", []wantCall{{"create_reminder", `{"text": "Check the oven", "minutes_from_now": 15}`}}},
		{"llama32_3b_on_false_str", "", []wantCall{{"set_light", `{"room": "living room", "on": false}`}}},
		{"llama32_1b_list", "", []wantCall{{"get_weather", `{"city": "Sydney", "unit": "c"}`}}},
		{"llama32_1b_fn_string", "", []wantCall{{"set_light", `{"room": "bedroom", "on": true, "brightness": 30}`}}},
		{"llama32_1b_two", ";", []wantCall{{"get_weather", `{"city": "London", "unit": "c"}`}, {"create_reminder", `{"text": "Get a coffee", "minutes_from_now": 30}`}}},
		{"llama32_1b_nested_fn", "", []wantCall{{"set_light", `{"room": "kitchen", "on": true, "brightness": 100}`}}},
		{"deepseek_7b_prose", "=", nil},
	}
	if len(cases) != len(outs) {
		t.Fatalf("%d cases for %d fixtures", len(cases), len(outs))
	}
	for _, c := range cases {
		o, ok := outs[c.id]
		if !ok {
			t.Fatalf("no fixture %s", c.id)
		}
		content, calls, err := Parse(o.Family, o.Text, tools)
		if err != nil {
			t.Errorf("%s: unexpected error %v", c.id, err)
		}
		want := c.content
		if want == "=" {
			want = strings.TrimSpace(o.Text)
		}
		check(t, c.id, content, calls, want, c.calls)
	}
}

func TestSynthetic(t *testing.T) {
	tools := loadTools(t)
	cases := []struct {
		name, family, text, content string
		calls                       []wantCall
	}{
		{"q35 parallel + float + array + bool", FamilyQwen35,
			"I'll do both.\n\n<tool_call>\n<function=set_thermostat>\n<parameter=temp>\n21.5\n</parameter>\n<parameter=mode>\ncool\n</parameter>\n<parameter=zone>\nupstairs\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=calendar_add>\n<parameter=title>\nPlanning\n</parameter>\n<parameter=start>\n2026-10-10T09:00:00\n</parameter>\n<parameter=duration_minutes>\n45.0\n</parameter>\n<parameter=attendees>\n[\"alice\", \"bob\"]\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=run_backup>\n<parameter=target>\nphotos\n</parameter>\n<parameter=dry_run>\nTrue\n</parameter>\n</function>\n</tool_call>",
			"I'll do both.",
			[]wantCall{
				{"set_thermostat", `{"temp": 21.5, "mode": "cool", "zone": "upstairs"}`},
				{"calendar_add", `{"title": "Planning", "start": "2026-10-10T09:00:00", "duration_minutes": 45, "attendees": ["alice", "bob"]}`},
				{"run_backup", `{"target": "photos", "dry_run": true}`},
			}},
		{"q35 multi-line string value kept, comma-list array", FamilyQwen35,
			"<tool_call>\n<function=send_message>\n<parameter=channel>\nops\n</parameter>\n<parameter=text>\nline one\nline two\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=calendar_add>\n<parameter=title>\nX\n</parameter>\n<parameter=attendees>\nalice, bob\n</parameter>\n</function>\n</tool_call>",
			"",
			[]wantCall{
				{"send_message", `{"channel": "ops", "text": "line one\nline two"}`},
				{"calendar_add", `{"title": "X", "attendees": ["alice", "bob"]}`},
			}},
		{"q35 thinking prefilled: call inside reasoning ignored", FamilyQwen35,
			"I could emit <tool_call>\n<function=get_weather>\n</function>\n</tool_call> but no.\n</think>\n\nIt is sunny.",
			"I could emit <tool_call>\n<function=get_weather>\n</function>\n</tool_call> but no.\n</think>\n\nIt is sunny.",
			nil},
		{"qwen3 think then call", FamilyQwen3,
			"<think>\nNeed weather.\n</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\", \"unit\": \"f\"}}\n</tool_call>",
			"<think>\nNeed weather.\n</think>",
			[]wantCall{{"get_weather", `{"city": "Tokyo", "unit": "f"}`}}},
		{"qwen3 python literals and number-as-string", FamilyQwen3,
			"<tool_call>\n{\"name\": \"set_thermostat\", \"arguments\": {\"temp\": \"21.5\", \"mode\": \"heat\", \"zone\": \"hall\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"run_backup\", \"arguments\": {\"target\": \"db\", \"dry_run\": True}}\n</tool_call>",
			"",
			[]wantCall{{"set_thermostat", `{"temp": 21.5, "mode": "heat", "zone": "hall"}`}, {"run_backup", `{"target": "db", "dry_run": true}`}}},
		{"qwen3 unterminated call at max_tokens", FamilyQwen3,
			"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}",
			"",
			[]wantCall{{"get_weather", `{"city": "Oslo"}`}}},
		{"qwen3 rkllm variant keys", FamilyQwen3,
			"<tool_call>\n{\"function\": \"http_check\", \"parameters\": {\"url\": \"http://x\"}}\n</tool_call>",
			"",
			[]wantCall{{"http_check", `{"url": "http://x"}`}}},
		{"gemma4 tokens visible, nested and arrays", FamilyGemma4,
			"<|tool_call>call:calendar_add{attendees:[<|\"|>alice<|\"|>,<|\"|>bob, jr<|\"|>],duration_minutes:30,start:<|\"|>2026-10-10T09:00:00<|\"|>,title:<|\"|>Sync: Q4, plans<|\"|>}<tool_call|><|tool_response>",
			"",
			[]wantCall{{"calendar_add", `{"attendees": ["alice", "bob, jr"], "duration_minutes": 30, "start": "2026-10-10T09:00:00", "title": "Sync: Q4, plans"}`}}},
		{"gemma4 bare value with comma and colon", FamilyGemma4,
			"call:send_message{channel:ops,text:Deploy done: all green, no errors}",
			"",
			[]wantCall{{"send_message", `{"channel": "ops", "text": "Deploy done: all green, no errors"}`}}},
		{"gemma4 bare float + two calls", FamilyGemma4,
			"call:set_thermostat{mode:heat,temp:21.5,zone:living}\ncall:set_light{on:true,room:hall}",
			"",
			[]wantCall{{"set_thermostat", `{"mode": "heat", "temp": 21.5, "zone": "living"}`}, {"set_light", `{"on": true, "room": "hall"}`}}},
		{"gemma4 bare 'call:' in prose for unknown tool is content", FamilyGemma4,
			"To do that, call:someone{now} or recall:x{y}.",
			"To do that, call:someone{now} or recall:x{y}.",
			nil},
		{"gemma4 thought channel", FamilyGemma4,
			"<|channel>thought\nuse the light tool<channel|><|tool_call>call:set_light{on:false,room:<|\"|>den<|\"|>}<tool_call|>",
			"<|channel>thought\nuse the light tool<channel|>",
			[]wantCall{{"set_light", `{"on": false, "room": "den"}`}}},
		{"generic: json inside prose not a call", FamilyGemma3,
			"Here is an example config: {\"name\": \"x\", \"size\": 3}.",
			"Here is an example config: {\"name\": \"x\", \"size\": 3}.",
			nil},
		{"generic: integer given as float", FamilyGemma3,
			"```json\n{\"tool_calls\": [{\"name\": \"set_light\", \"arguments\": {\"room\": \"den\", \"on\": 1, \"brightness\": 40.0}}]}\n```",
			"",
			[]wantCall{{"set_light", `{"room": "den", "on": true, "brightness": 40}`}}},
		{"no tools declared: generic accepts any name, no coercion", FamilyGeneric,
			"{\"name\": \"anything\", \"arguments\": {\"n\": \"5\"}}",
			"",
			[]wantCall{{"anything", `{"n": "5"}`}}},
	}
	for _, c := range cases {
		ts := tools
		if c.family == FamilyGeneric {
			ts = nil
		}
		content, calls, err := Parse(c.family, c.text, ts)
		if err != nil {
			t.Errorf("%s: unexpected error %v", c.name, err)
		}
		check(t, c.name, content, calls, c.content, c.calls)
	}
}

func TestMalformed(t *testing.T) {
	text := "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \n</tool_call>\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}\n</tool_call>"
	content, calls, err := Parse(FamilyQwen3, text, nil)
	if !errors.Is(err, ErrMalformed) {
		t.Fatalf("want ErrMalformed, got %v", err)
	}
	if len(calls) != 1 || calls[0].Function.Arguments != `{"city": "Rome"}` {
		t.Fatalf("calls %+v", calls)
	}
	if !strings.Contains(content, `"city": `) {
		t.Fatalf("malformed markup should stay in content: %q", content)
	}
}

func TestSplitReasoning(t *testing.T) {
	cases := []struct{ in, r, c string }{
		{"<think>\nabc\n</think>\n\nanswer", "abc", "answer"},
		{"prefilled reasoning\n</think>\n\nanswer", "prefilled reasoning", "answer"},
		{"<|channel>thought\nhmm<channel|>Done.", "hmm", "Done."},
		{"plain", "", "plain"},
		{"<think>\nunterminated", "unterminated", ""},
	}
	for _, c := range cases {
		r, content := SplitReasoning(c.in)
		if r != c.r || content != c.c {
			t.Errorf("SplitReasoning(%q) = %q, %q; want %q, %q", c.in, r, content, c.r, c.c)
		}
	}
}

func TestHoldback(t *testing.T) {
	cases := []struct {
		family, text string
		safe         int
		inCall       bool
	}{
		{FamilyQwen3, "Hello world", 11, false},
		{FamilyQwen3, "Sure <tool", 5, false},
		{FamilyQwen3, "Sure <tool_call>\n{\"na", 5, true},
		{FamilyQwen3, "a < b", 5, false},
		{FamilyQwen35, "x\n\n<tool_call>\n<function=", 3, true},
		{FamilyGemma4, "ok call:set_l", 3, false},
		{FamilyGemma4, "ok call:set_light{room", 3, true},
		{FamilyGemma4, "recall: the", 11, false},
		{FamilyGemma4, "<|tool_c", 0, false},
		{FamilyGemma3, "Okay.\n\n```js", 7, false},
		{FamilyGemma3, "Okay.\n\n```json\n{\n  \"tool_calls\"", 7, true},
		{FamilyGemma3, "Use {braces} freely", 19, false},
		{FamilyGemma3, "set {", 4, false},
		{FamilyGeneric, "[{\"name\": \"x\"", 0, true},
	}
	for _, c := range cases {
		safe, in := Holdback(c.family, c.text)
		if safe != c.safe || in != c.inCall {
			t.Errorf("Holdback(%s, %q) = %d,%v; want %d,%v", c.family, c.text, safe, in, c.safe, c.inCall)
		}
	}
}

func TestStreamingNeverLeaksMarkup(t *testing.T) {
	// Feed real outputs byte by byte: whatever Holdback lets through must be a
	// prefix of the final parsed content.
	tools := loadTools(t)
	for id, o := range loadOutputs(t) {
		content, calls, _ := Parse(o.Family, o.Text, tools)
		streamed := 0
		for i := 1; i <= len(o.Text); i++ {
			safe, _ := Holdback(o.Family, o.Text[:i])
			if safe > streamed {
				streamed = safe
			}
		}
		emitted := strings.TrimSpace(o.Text[:streamed])
		if len(calls) > 0 && !strings.HasPrefix(content, emitted) {
			t.Errorf("%s: streamed %q is not a prefix of content %q", id, emitted, content)
		}
	}
}
