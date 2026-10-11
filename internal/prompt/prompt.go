// Package prompt renders OpenAI-style chat histories into the exact prompt
// text each served model family was trained on. The RKLLM runtime's own chat
// template is disabled (set to empty strings), so this package owns the
// template, and it is written to be append-stable so the server can reuse the
// KV cache across turns of an agent conversation.
//
// # Append stability
//
// Every Template splits its output into a stable part and a tail:
//
//	Format(msgs) == stable + tail
//
// The guarantee is: for any messages ext,
//
//	strings.HasPrefix(Format(append(msgs, ext...)), stable)
//
// The tail is the generation prompt for the next assistant turn plus any
// closing marker whose form depends on what message comes next (e.g. Qwen's
// `<|im_end|>` after a group of tool results is only written once the group
// ends). StablePrefixLen returns len(stable).
//
// The guarantee holds only while tools and Options are unchanged across the
// conversation; changing them changes the system block at the top. Prior
// assistant turns are re-rendered from Message fields, so the prompt matches
// what the model actually generated (and what sits in the KV cache) only if
// the re-render is byte-identical to the generation. Set Message.Raw to the
// generated text to guarantee that; otherwise the canonical rendering of
// content / reasoning / tool calls is used.
//
// # Thinking policy
//
// What the model saw or generated is what is re-rendered. Options.EnableThinking
// is treated as a conversation-level setting: the generation prompt's thinking
// prefill (e.g. Qwen's empty `<think>\n\n</think>\n\n` when thinking is off)
// is repeated for every prior assistant turn, and prior reasoning is kept
// rather than stripped. This deviates from the upstream HF templates, which
// drop reasoning of turns before the last user query; that costs context
// tokens but keeps the KV cache valid. See docs/templates-and-toolparse.md.
package prompt

import (
	"encoding/json"
	"fmt"
	"strings"

	"rkllm-go/internal/pyjson"
)

// Family identifies a chat template / tool-call format.
type Family string

const (
	FamilyQwen3  Family = "qwen3"
	FamilyQwen35 Family = "qwen3.5"
	FamilyGemma3 Family = "gemma3" // Gemma 3 and Gemma 3n share the template
	FamilyGemma4 Family = "gemma4"
)

// Message is one chat message. It mirrors api.Message (no import cycle).
type Message struct {
	Role       string // system | user | assistant | tool (developer is treated as system by gemma4)
	Content    string
	Reasoning  string // assistant reasoning_content; also extracted from <think>…</think> in Content (Qwen)
	ToolCalls  []ToolCall
	ToolCallID string // role "tool": id of the call this result answers
	Name       string // role "tool": function name, if the client sent it

	// Raw, if non-empty on an assistant message, is the exact text the model
	// generated for that turn (decoded output, excluding the stop token). It is
	// rendered verbatim in place of Reasoning/Content/ToolCalls, immediately
	// after the generation prompt that preceded it, followed by the family's
	// closing marker. This makes the re-rendered history byte-identical to what
	// was prefilled + generated, i.e. to the KV cache contents.
	Raw string
}

// ToolCall is an assistant tool call in OpenAI shape (arguments is a JSON string).
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// Options are per-conversation rendering options.
type Options struct {
	// EnableThinking turns on chain-of-thought where the family supports it.
	EnableThinking bool
	// AddBOS prepends the family's BOS token text (Gemma: "<bos>"). Leave
	// false if the runtime tokenizer adds BOS itself.
	AddBOS bool
}

// Template renders chat histories for one model family.
type Template interface {
	Family() Family
	// Render returns the append-stable part and the tail (generation prompt
	// plus any closing that depends on the next message).
	Render(msgs []Message, tools []json.RawMessage, opts Options) (stable, tail string)
	// Format returns stable + tail: the full prompt ending with the
	// generation prompt for the assistant turn.
	Format(msgs []Message, tools []json.RawMessage, opts Options) string
	// StopStrings are end-of-turn markers; generation should stop on any.
	StopStrings() []string
}

// StablePrefixLen returns the length of the prefix of Format(msgs, …) that is
// guaranteed to be a prefix of Format(append(msgs, more...), …).
func StablePrefixLen(t Template, msgs []Message, tools []json.RawMessage, opts Options) int {
	s, _ := t.Render(msgs, tools, opts)
	return len(s)
}

// ---- registry ----

var registry = map[Family]Template{
	FamilyQwen3:  qwen3{},
	FamilyQwen35: qwen35{},
	FamilyGemma3: gemma3{},
	FamilyGemma4: gemma4{},
}

// Families lists the supported families.
func Families() []Family {
	return []Family{FamilyQwen3, FamilyQwen35, FamilyGemma3, FamilyGemma4}
}

// ParseFamily normalises a configured family name ("qwen3.5", "q35", "gemma3n", …).
func ParseFamily(s string) (Family, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "qwen3", "qwen":
		return FamilyQwen3, true
	case "qwen3.5", "qwen35", "qwen3_5", "q35":
		return FamilyQwen35, true
	case "gemma3", "gemma3n", "gemma-3", "gemma-3n", "gemma":
		return FamilyGemma3, true
	case "gemma4", "gemma-4":
		return FamilyGemma4, true
	}
	return "", false
}

// FamilyForModel infers the family from a model name or path such as
// "q35-4b", "qwen3-8b-allnpu", "gemma3n-e4b", "/opt/rkllm/models/gemma4-e2b/model.rkllm".
// It returns "" if nothing matches.
func FamilyForModel(name string) Family {
	n := strings.ToLower(name)
	switch {
	case strings.Contains(n, "q35"), strings.Contains(n, "qwen3.5"), strings.Contains(n, "qwen3_5"), strings.Contains(n, "qwen35"):
		return FamilyQwen35
	case strings.Contains(n, "qwen3"):
		return FamilyQwen3
	case strings.Contains(n, "gemma4"), strings.Contains(n, "gemma-4"):
		return FamilyGemma4
	case strings.Contains(n, "gemma3"), strings.Contains(n, "gemma-3"):
		return FamilyGemma3
	}
	return ""
}

// Get returns the template for a family.
func Get(f Family) (Template, error) {
	t, ok := registry[f]
	if !ok {
		return nil, fmt.Errorf("prompt: unknown family %q", f)
	}
	return t, nil
}

// Resolve picks a template: an explicitly configured family (models.json
// "families") wins, otherwise the family is inferred from the model name.
func Resolve(configured, modelName string) (Template, error) {
	if configured != "" {
		f, ok := ParseFamily(configured)
		if !ok {
			return nil, fmt.Errorf("prompt: unknown family %q configured for model %q", configured, modelName)
		}
		return Get(f)
	}
	if f := FamilyForModel(modelName); f != "" {
		return Get(f)
	}
	return nil, fmt.Errorf("prompt: cannot infer template family for model %q; set it in models.json \"families\"", modelName)
}

// ---- shared helpers ----

// split finalises a render. With no messages nothing is stable: whether the
// first message is a system message decides how the leading block (which
// also carries the tools) is rendered.
func split(msgs []Message, stable, tail string) (string, string) {
	if len(msgs) == 0 {
		return "", stable + tail
	}
	return stable, tail
}

// splitThink mirrors the HF Qwen templates: if content contains </think>,
// reasoning is the text between <think> and </think>, content the rest.
// An explicit Message.Reasoning wins.
func splitThink(m Message) (content, reasoning string) {
	content = m.Content
	reasoning = m.Reasoning
	if i := strings.Index(content, "</think>"); i >= 0 {
		before := content[:i]
		after := content[strings.LastIndex(content, "</think>")+len("</think>"):]
		if reasoning == "" {
			r := strings.TrimRight(before, "\n")
			if j := strings.LastIndex(r, "<think>"); j >= 0 {
				r = r[j+len("<think>"):]
			}
			reasoning = strings.TrimLeft(r, "\n")
		}
		content = strings.TrimLeft(after, "\n")
	}
	return content, reasoning
}

// parseTool parses one tool definition, normalising a bare function object
// ({"name":…, "parameters":…}) into {"type":"function","function":{…}}.
func parseTool(raw json.RawMessage) (pyjson.Value, bool) {
	v, err := pyjson.Parse(raw)
	if err != nil || v.Kind != pyjson.Object {
		return pyjson.Value{}, false
	}
	if _, ok := v.Get("function"); !ok {
		if _, hasName := v.Get("name"); hasName {
			w := pyjson.NewObject()
			w.Set("type", pyjson.Str("function"))
			w.Set("function", v)
			return w, true
		}
	}
	return v, true
}

// toolJSON renders a tool like Jinja's `tool | tojson` (Python separators,
// client key order). Unparseable input is written as given.
func toolJSON(raw json.RawMessage) string {
	v, ok := parseTool(raw)
	if !ok {
		return strings.TrimSpace(string(raw))
	}
	return v.Dump()
}

// argsValue parses a tool call's arguments JSON string (lenient).
func argsValue(args string) (pyjson.Value, bool) {
	if strings.TrimSpace(args) == "" {
		return pyjson.NewObject(), true
	}
	v, err := pyjson.ParseLenient(args)
	if err != nil {
		return pyjson.Value{}, false
	}
	return v, true
}

func nameForTool(m Message, calls []ToolCall) string {
	for _, c := range calls {
		if c.ID != "" && c.ID == m.ToolCallID {
			return c.Name
		}
	}
	return m.Name
}
