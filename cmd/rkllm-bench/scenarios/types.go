// Package scenarios holds the multi-turn agent tasks run by rkllm-bench:
// simulated environments (a house, a homelab, a small Python repo), the tool
// definitions the model sees, deterministic tool executors, scoring, and a
// scripted reference solution per task (used by the fake server and tests).
package scenarios

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Message is an OpenAI chat message as sent to the server.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// ToolCall is the OpenAI tool_call shape.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall carries the function name and JSON-encoded arguments.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Call is a parsed tool call handed to a Task.
type Call struct {
	ID   string
	Name string
	Raw  string         // arguments exactly as returned
	Args map[string]any // decoded arguments (empty when Bad)
	Bad  bool           // arguments were not a JSON object
}

// ParseCall decodes one OpenAI tool call. A string-encoded JSON object is the
// normal case; an already-decoded object is tolerated.
func ParseCall(tc ToolCall) Call {
	c := Call{ID: tc.ID, Name: tc.Function.Name, Raw: tc.Function.Arguments, Args: map[string]any{}}
	raw := strings.TrimSpace(tc.Function.Arguments)
	if raw == "" {
		return c
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(raw), &m); err != nil || m == nil {
		c.Bad = true
		return c
	}
	c.Args = m
	return c
}

// Reply is one scripted assistant turn (reference solution).
type Reply struct {
	Content string
	Calls   []RefCall
}

// RefCall is a scripted tool call.
type RefCall struct {
	Name string
	Args map[string]any
}

// Score is the final outcome of a task.
type Score struct {
	Pass   bool           `json:"pass"`
	Score  float64        `json:"score"` // 0..1 partial credit
	Detail map[string]any `json:"detail,omitempty"`
}

// Task is one stateful multi-turn agent task. A fresh Task is created per run.
type Task interface {
	ID() string
	Scenario() string
	Initial() []Message
	Tools() []any
	MaxTokens() int
	// OnAssistant sees every assistant turn before its tool calls are executed.
	// Returning stop=true ends the task (e.g. a scored stage failed).
	OnAssistant(turn int, content string, calls []Call) (stop bool)
	// Exec executes one tool call and returns the tool message content.
	Exec(c Call) string
	// Solved reports whether the useful result has been reached right now.
	Solved() bool
	// Score is the final score after the loop ends.
	Score() Score
	// Reference is a scripted solution: Reference()[i] is the reply for assistant turn i.
	Reference() []Reply
}

// Spec names a task and constructs fresh instances of it.
type Spec struct {
	ID       string
	Scenario string
	New      func() Task
}

// ---- tool schema helpers ----

func fn(name, desc string, props map[string]any, req []string) any {
	if req == nil {
		req = []string{}
	}
	return map[string]any{"type": "function", "function": map[string]any{
		"name": name, "description": desc,
		"parameters": map[string]any{"type": "object", "properties": props, "required": req},
	}}
}

var (
	tS = map[string]any{"type": "string"}
	tI = map[string]any{"type": "integer"}
	tB = map[string]any{"type": "boolean"}
	tN = map[string]any{"type": "number"}
)

func enum(vals ...string) map[string]any {
	return map[string]any{"type": "string", "enum": vals}
}

func toolNames(tools []any) map[string]bool {
	out := map[string]bool{}
	for _, t := range tools {
		f := t.(map[string]any)["function"].(map[string]any)
		out[f["name"].(string)] = true
	}
	return out
}

func jsonStr(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// ---- python-ish value helpers (keep toolbench scoring semantics) ----

// norm lower-cases and trims strings, leaves other values alone.
func norm(v any) any {
	if s, ok := v.(string); ok {
		return strings.ToLower(strings.TrimSpace(s))
	}
	if f, ok := v.(float64); ok {
		return f
	}
	if i, ok := v.(int); ok {
		return float64(i)
	}
	return v
}

// pyStr approximates python str(v).
func pyStr(v any) string {
	switch x := v.(type) {
	case nil:
		return "None"
	case string:
		return x
	case bool:
		if x {
			return "True"
		}
		return "False"
	case float64:
		if x == float64(int64(x)) {
			return fmt.Sprintf("%d", int64(x))
		}
		return fmt.Sprint(x)
	default:
		return jsonStr(x)
	}
}

// truthy is python bool(v).
func truthy(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case float64:
		return x != 0
	case string:
		return x != ""
	case []any:
		return len(x) > 0
	case map[string]any:
		return len(x) > 0
	}
	return true
}

// pyFloat is python float(v); ok=false where python would raise.
func pyFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case bool:
		if x {
			return 1, true
		}
		return 0, true
	case string:
		var f float64
		if _, err := fmt.Sscanf(strings.TrimSpace(x), "%g", &f); err == nil {
			return f, true
		}
	}
	return 0, false
}

// numEq is python ==, for a JSON value against a number.
func numEq(v any, x float64) bool {
	switch y := v.(type) {
	case float64:
		return y == x
	case bool:
		return (y && x == 1) || (!y && x == 0)
	}
	return false
}

func sortedKeys[V any](m map[string]V) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}
