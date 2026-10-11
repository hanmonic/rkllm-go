// Package toolparse extracts OpenAI-shaped tool calls from raw model output
// for every served model family, and coerces argument values against the
// tool's JSON schema.
//
// Accepted call syntaxes (all families try all of them, in a family-specific
// order, stopping at the first syntax that matches anything):
//
//   - tagged:  <tool_call>\n{"name": …, "arguments": {…}}\n</tool_call>   (Qwen3)
//     <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call> (Qwen3.5)
//   - gemma:   <|tool_call>call:NAME{k:<|"|>v<|"|>,n:3}<tool_call|>   (Gemma 4), and the
//     form seen when the runtime strips special tokens: call:NAME{k:v,n:3}
//   - generic: a JSON object/array anywhere in the text holding
//     {"name", "arguments"|"parameters"}, {"function": {…}} or {"tool_calls": [...]}
//     (Gemma 3/3n polyfill, Llama 3.2, DeepSeek). Only calls naming a known
//     tool are accepted when tools are given.
//
// Ported from the benchmark parsers in /opt/rkllm-131/toolbench.py (incl. the
// True/float/array coercion fixes), plus schema-driven coercion.
package toolparse

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"rkllm-go/internal/pyjson"
)

// Family names accepted by Parse/Holdback (same strings as prompt.Family).
const (
	FamilyQwen3   = "qwen3"
	FamilyQwen35  = "qwen3.5"
	FamilyGemma3  = "gemma3"
	FamilyGemma4  = "gemma4"
	FamilyGeneric = "generic"
)

// ToolSchema is the part of an OpenAI tool definition the parser needs.
type ToolSchema struct {
	Name       string
	Parameters json.RawMessage // JSON schema of the arguments object; may be nil
}

// ToolCall is an OpenAI tool call.
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

// ErrMalformed is returned (wrapped) when call markup was found but could not
// be parsed. Successfully parsed calls are still returned, and the malformed
// markup is left in content.
var ErrMalformed = errors.New("toolparse: malformed tool call")

// SchemasFromTools converts OpenAI tool objects ({"type":"function","function":{…}}
// or bare function objects) into ToolSchemas. Unparseable entries are skipped.
func SchemasFromTools(tools []json.RawMessage) []ToolSchema {
	var out []ToolSchema
	for _, raw := range tools {
		var t struct {
			Function *struct {
				Name       string          `json:"name"`
				Parameters json.RawMessage `json:"parameters"`
			} `json:"function"`
			Name       string          `json:"name"`
			Parameters json.RawMessage `json:"parameters"`
		}
		if json.Unmarshal(raw, &t) != nil {
			continue
		}
		if t.Function != nil {
			out = append(out, ToolSchema{Name: t.Function.Name, Parameters: t.Function.Parameters})
		} else if t.Name != "" {
			out = append(out, ToolSchema{Name: t.Name, Parameters: t.Parameters})
		}
	}
	return out
}

// found is one parsed call (or a malformed one) and its byte span in text.
type found struct {
	start, end int
	name       string
	args       pyjson.Value
	loose      bool // values came from untyped text (XML / bare gemma)
	err        error
}

type parser func(text string, skip []span, tools *toolset) []found

// Parse extracts tool calls from model output text. content is text with call
// markup removed and trimmed; reasoning (<think>…</think>, Gemma thought
// channel) is left in content untouched and calls inside it are ignored — use
// SplitReasoning to separate it.
func Parse(family, text string, tools []ToolSchema) (content string, calls []ToolCall, err error) {
	ts := newToolset(tools)
	text = strings.ReplaceAll(text, "<|python_tag|>", "")
	skip := reasoningSpans(text)

	var order []parser
	switch family {
	case FamilyQwen3, FamilyQwen35:
		order = []parser{parseTagged, parseGemma, parseGeneric}
	case FamilyGemma4:
		order = []parser{parseGemma, parseTagged, parseGeneric}
	default: // gemma3, generic, unknown
		order = []parser{parseGeneric, parseTagged, parseGemma}
	}
	var fs []found
	for _, p := range order {
		if fs = p(text, skip, ts); len(fs) > 0 {
			break
		}
	}
	if len(fs) == 0 {
		return strings.TrimSpace(text), nil, nil
	}

	var b strings.Builder
	last := 0
	var errs []string
	for _, f := range fs {
		if f.err != nil {
			errs = append(errs, f.err.Error())
			continue // leave malformed markup in content
		}
		b.WriteString(text[last:f.start])
		last = f.end
		args := f.args
		if s := ts.schema(f.name); s != nil {
			args = coerce(args, *s, f.loose)
		} else if f.loose {
			args = coerce(args, pyjson.Value{}, true)
		}
		calls = append(calls, ToolCall{
			ID:       newID(),
			Type:     "function",
			Function: FunctionCall{Name: f.name, Arguments: argsString(args)},
		})
	}
	b.WriteString(text[last:])
	content = cleanContent(b.String())
	if len(errs) > 0 {
		err = fmt.Errorf("%w: %s", ErrMalformed, strings.Join(errs, "; "))
	}
	return content, calls, err
}

func argsString(v pyjson.Value) string {
	switch v.Kind {
	case pyjson.Object:
		return v.Dump()
	case pyjson.Null:
		return "{}"
	case pyjson.String:
		if inner, err := pyjson.ParseLenient(v.S); err == nil && inner.Kind == pyjson.Object {
			return inner.Dump()
		}
	}
	return v.Dump()
}

var emptyFenceRE = regexp.MustCompile("(?s)```[a-zA-Z]*\\s*```")

func cleanContent(s string) string {
	s = emptyFenceRE.ReplaceAllString(s, "")
	return strings.TrimSpace(s)
}

func newID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return "call_" + hex.EncodeToString(b[:])
}

// ---- reasoning ----

type span struct{ start, end int }

func inSpans(i int, ss []span) bool {
	for _, s := range ss {
		if i >= s.start && i < s.end {
			return true
		}
	}
	return false
}

// reasoningSpans finds <think>…</think> and <|channel>…<channel|> regions.
// A closing tag without an opener (Qwen3.5 prefills "<think>\n"; Gemma 4
// prefills "<|channel>thought\n" after tool results) opens at offset 0. An
// opener without a closer runs to the end of text.
func reasoningSpans(text string) []span {
	var out []span
	for _, p := range [][2]string{{"<think>", "</think>"}, {"<|channel>", "<channel|>"}} {
		open, close := p[0], p[1]
		pos := 0
		for pos <= len(text) {
			o := strings.Index(text[pos:], open)
			c := strings.Index(text[pos:], close)
			if o < 0 && c < 0 {
				break
			}
			if c >= 0 && (o < 0 || c < o) { // orphan closer
				out = append(out, span{pos, pos + c + len(close)})
				pos += c + len(close)
				continue
			}
			start := pos + o
			c2 := strings.Index(text[start:], close)
			if c2 < 0 {
				out = append(out, span{start, len(text)})
				break
			}
			out = append(out, span{start, start + c2 + len(close)})
			pos = start + c2 + len(close)
		}
	}
	return out
}

// SplitReasoning separates chain-of-thought from the answer. It handles
// Qwen <think>…</think> (with or without the opening tag, which Qwen3.5
// prefills) and the Gemma 4 thought channel. reasoning is trimmed.
func SplitReasoning(text string) (reasoning, content string) {
	spans := reasoningSpans(text)
	if len(spans) == 0 {
		return "", text
	}
	var r, c strings.Builder
	last := 0
	for _, s := range spans {
		c.WriteString(text[last:s.start])
		seg := text[s.start:s.end]
		for _, tag := range []string{"<think>", "</think>", "<|channel>", "<channel|>"} {
			seg = strings.ReplaceAll(seg, tag, "")
		}
		seg = strings.TrimPrefix(strings.TrimLeft(seg, "\n"), "thought\n")
		if r.Len() > 0 {
			r.WriteString("\n")
		}
		r.WriteString(strings.TrimSpace(seg))
		last = s.end
	}
	c.WriteString(text[last:])
	return r.String(), strings.TrimLeft(c.String(), "\n")
}

// ---- tagged <tool_call> ----

func parseTagged(text string, skip []span, ts *toolset) []found {
	const open, close = "<tool_call>", "</tool_call>"
	var out []found
	pos := 0
	for {
		o := strings.Index(text[pos:], open)
		if o < 0 {
			break
		}
		start := pos + o
		bodyStart := start + len(open)
		if inSpans(start, skip) {
			pos = bodyStart
			continue
		}
		end, bodyEnd := len(text), len(text)
		if c := strings.Index(text[bodyStart:], close); c >= 0 {
			bodyEnd = bodyStart + c
			end = bodyEnd + len(close)
		}
		if n := strings.Index(text[bodyStart:bodyEnd], open); n >= 0 { // missing closer before next block
			bodyEnd = bodyStart + n
			end = bodyEnd
		}
		body := strings.TrimSpace(text[bodyStart:bodyEnd])
		f := found{start: start, end: end}
		if strings.Contains(body, "<function=") {
			f.name, f.args, f.err = parseXMLCall(body)
			f.loose = true
		} else {
			f.name, f.args, f.err = parseJSONCall(stripFence(body))
		}
		out = append(out, f)
		pos = end
	}
	return out
}

func stripFence(s string) string {
	if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```")
		s = strings.TrimPrefix(s, "json")
		s = strings.TrimSuffix(strings.TrimSpace(s), "```")
	}
	return strings.TrimSpace(s)
}

var (
	funcRE  = regexp.MustCompile(`<function=([^>\s]+)>`)
	paramRE = regexp.MustCompile(`<parameter=([^>]+)>`)
)

// parseXMLCall parses the Qwen3.5 body <function=N><parameter=K>\nV\n</parameter></function>.
// A value ends at </parameter>, or leniently at the next <parameter=, </function> or end.
func parseXMLCall(body string) (string, pyjson.Value, error) {
	m := funcRE.FindStringSubmatchIndex(body)
	if m == nil {
		return "", pyjson.Value{}, errors.New("no <function=…> in <tool_call>")
	}
	name := body[m[2]:m[3]]
	rest := body[m[1]:]
	if e := strings.Index(rest, "</function>"); e >= 0 {
		rest = rest[:e]
	}
	args := pyjson.NewObject()
	locs := paramRE.FindAllStringSubmatchIndex(rest, -1)
	for i, l := range locs {
		key := strings.TrimSpace(rest[l[2]:l[3]])
		vEnd := len(rest)
		if i+1 < len(locs) {
			vEnd = locs[i+1][0]
		}
		val := rest[l[1]:vEnd]
		if c := strings.Index(val, "</parameter>"); c >= 0 {
			val = val[:c]
		}
		val = strings.TrimPrefix(val, "\n")
		val = strings.TrimSuffix(val, "\n")
		args.Set(key, pyjson.Str(val))
	}
	return name, args, nil
}

// parseJSONCall parses {"name":…, "arguments":…} and its variants.
func parseJSONCall(body string) (string, pyjson.Value, error) {
	v, err := pyjson.ParseLenient(body)
	if err != nil {
		return "", pyjson.Value{}, fmt.Errorf("invalid JSON in <tool_call>: %v", err)
	}
	if v.Kind == pyjson.Array && len(v.A) == 1 {
		v = v.A[0]
	}
	name, args, ok := callFromObject(v)
	if !ok {
		return "", pyjson.Value{}, errors.New("<tool_call> JSON has no function name")
	}
	return name, args, nil
}

// callFromObject recognises {"name","arguments"|"parameters"},
// {"function": {"name","arguments"}} and {"function": "name", "parameters": …}.
func callFromObject(v pyjson.Value) (string, pyjson.Value, bool) {
	if v.Kind != pyjson.Object {
		return "", pyjson.Value{}, false
	}
	if fn, ok := v.Get("function"); ok {
		if fn.Kind == pyjson.Object {
			return callFromObject(fn)
		}
		if fn.Kind == pyjson.String {
			args, _ := v.Get("parameters")
			if a, ok := v.Get("arguments"); ok {
				args = a
			}
			return fn.S, normArgs(args), true
		}
	}
	name, ok := v.Get("name")
	if !ok || name.Kind != pyjson.String || name.S == "" {
		return "", pyjson.Value{}, false
	}
	args, has := v.Get("arguments")
	if !has {
		args, has = v.Get("parameters")
	}
	if !has {
		return "", pyjson.Value{}, false
	}
	return name.S, normArgs(args), true
}

func normArgs(a pyjson.Value) pyjson.Value {
	switch a.Kind {
	case pyjson.Object:
		return a
	case pyjson.String: // arguments given as a JSON string
		if v, err := pyjson.ParseLenient(a.S); err == nil && v.Kind == pyjson.Object {
			return v
		}
	case pyjson.Null:
		return pyjson.NewObject()
	}
	return a
}

// ---- generic JSON ----

func parseGeneric(text string, skip []span, ts *toolset) []found {
	var out []found
	for i := 0; i < len(text); i++ {
		c := text[i]
		if c != '{' && c != '[' {
			continue
		}
		if inSpans(i, skip) {
			continue
		}
		end := balancedEnd(text, i)
		if end < 0 {
			if c == '{' {
				break // unbalanced object: the rest is truncated
			}
			continue
		}
		v, err := pyjson.ParseLenient(text[i:end])
		if err != nil {
			continue
		}
		var cands []pyjson.Value
		switch {
		case v.Kind == pyjson.Array:
			cands = v.A
		case v.Kind == pyjson.Object:
			if tc, ok := v.Get("tool_calls"); ok && tc.Kind == pyjson.Array {
				cands = tc.A
			} else {
				cands = []pyjson.Value{v}
			}
		}
		var fs []found
		for _, cand := range cands {
			name, args, ok := callFromObject(cand)
			if !ok || !ts.allowed(name) {
				fs = nil
				break
			}
			fs = append(fs, found{name: name, args: args})
		}
		if len(fs) == 0 {
			continue
		}
		start, stop := i, end
		// swallow a surrounding ```json fence
		if s, e, ok := fenceAround(text, start, stop); ok {
			start, stop = s, e
		}
		// all calls of one JSON value share its span: give it to the last one
		for k := range fs {
			fs[k].start, fs[k].end = start, start
		}
		fs[len(fs)-1].end = stop
		out = append(out, fs...)
		i = stop - 1
	}
	return out
}

// fenceAround reports whether text[s:e] is the sole content of a ``` fence.
func fenceAround(text string, s, e int) (int, int, bool) {
	before := strings.TrimRight(text[:s], " \t\n")
	if strings.HasSuffix(before, "```json") {
		before = strings.TrimSuffix(before, "```json")
	} else if strings.HasSuffix(before, "```") {
		before = strings.TrimSuffix(before, "```")
	} else {
		return 0, 0, false
	}
	after := text[e:]
	trimmed := strings.TrimLeft(after, " \t\n")
	if !strings.HasPrefix(trimmed, "```") {
		return 0, 0, false
	}
	return len(before), e + (len(after) - len(trimmed)) + 3, true
}

// balancedEnd returns the index just past the bracket matching text[i], or -1.
func balancedEnd(text string, i int) int {
	depth := 0
	inStr, esc := false, false
	for j := i; j < len(text); j++ {
		c := text[j]
		if inStr {
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{', '[':
			depth++
		case '}', ']':
			depth--
			if depth == 0 {
				return j + 1
			}
		}
	}
	return -1
}

// ---- toolset ----

type toolset struct {
	names   map[string]bool
	schemas map[string]*pyjson.Value
}

func newToolset(tools []ToolSchema) *toolset {
	ts := &toolset{names: map[string]bool{}, schemas: map[string]*pyjson.Value{}}
	for _, t := range tools {
		ts.names[t.Name] = true
		if len(t.Parameters) > 0 {
			if v, err := pyjson.Parse(t.Parameters); err == nil {
				vv := v
				ts.schemas[t.Name] = &vv
			}
		}
	}
	return ts
}

// allowed is true when no tools were declared or name is one of them.
func (ts *toolset) allowed(name string) bool {
	return len(ts.names) == 0 || ts.names[name]
}

func (ts *toolset) schema(name string) *pyjson.Value { return ts.schemas[name] }

// properties returns the property names of a tool's schema.
func (ts *toolset) properties(name string) map[string]bool {
	s := ts.schemas[name]
	if s == nil {
		return nil
	}
	p, ok := s.Get("properties")
	if !ok || p.Kind != pyjson.Object {
		return nil
	}
	m := map[string]bool{}
	for _, k := range p.K {
		m[k] = true
	}
	return m
}
