package toolparse

import (
	"encoding/json"
	"errors"
	"regexp"
	"strings"

	"rkllm-go/internal/pyjson"
)

const gemmaQ = `<|"|>`

var gemmaCallRE = regexp.MustCompile(`(<\|tool_call>)?call:([A-Za-z_][A-Za-z0-9_.\-]*)\{`)

// parseGemma finds Gemma 4 calls. With special tokens visible they look like
// <|tool_call>call:NAME{key:<|"|>str<|"|>,n:3}<tool_call|>; the RKLLM runtime
// strips special tokens from its text output, which leaves
// call:NAME{key:str,n:3} (seen in result-m63-gemma4-e2b.json). The bare form
// is only accepted at a word boundary and for a declared tool name.
func parseGemma(text string, skip []span, ts *toolset) []found {
	var out []found
	pos := 0
	for _, l := range gemmaCallRE.FindAllStringSubmatchIndex(text, -1) {
		start := l[0]
		if start < pos || inSpans(start, skip) {
			continue
		}
		tagged := l[2] >= 0
		name := text[l[4]:l[5]]
		if !tagged {
			if start > 0 && isWordByte(text[start-1]) {
				continue
			}
			if !ts.allowed(name) {
				continue
			}
		}
		p := &gparser{s: text, i: l[1]}
		args, err := p.object(ts.properties(name))
		stop := p.i
		for _, t := range []string{"<tool_call|>", "<|tool_response>"} {
			if strings.HasPrefix(text[stop:], t) {
				stop += len(t)
			}
		}
		out = append(out, found{start: start, end: stop, name: name, args: args, err: err})
		pos = stop
	}
	return out
}

func isWordByte(c byte) bool {
	return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
}

// gparser parses Gemma's argument DSL: {key:value,…} with bare or <|"|>-quoted
// keys, values that are <|"|>strings<|"|>, "json strings", nested {…}, […]
// or bare scalars. Truncated input yields what was parsed so far.
type gparser struct {
	s string
	i int
}

func (p *gparser) eof() bool { return p.i >= len(p.s) }
func (p *gparser) ws() {
	for !p.eof() && strings.IndexByte(" \t\r\n", p.s[p.i]) >= 0 {
		p.i++
	}
}

// object parses after an opening '{' through the matching '}'. known, if
// non-nil, is the set of property names used to disambiguate commas inside
// unquoted string values.
func (p *gparser) object(known map[string]bool) (pyjson.Value, error) {
	obj := pyjson.NewObject()
	for {
		p.ws()
		if p.eof() {
			return obj, nil
		}
		switch p.s[p.i] {
		case '}':
			p.i++
			return obj, nil
		case ',':
			p.i++
			continue
		}
		key, ok := p.key()
		if !ok {
			return obj, errors.New("gemma call: expected key:value")
		}
		obj.Set(key, p.value(true, known))
	}
}

func (p *gparser) key() (string, bool) {
	var k string
	switch {
	case strings.HasPrefix(p.s[p.i:], gemmaQ):
		e := strings.Index(p.s[p.i+len(gemmaQ):], gemmaQ)
		if e < 0 {
			return "", false
		}
		k = p.s[p.i+len(gemmaQ) : p.i+len(gemmaQ)+e]
		p.i += 2*len(gemmaQ) + e
	case p.s[p.i] == '"':
		e := strings.IndexByte(p.s[p.i+1:], '"')
		if e < 0 {
			return "", false
		}
		k = p.s[p.i+1 : p.i+1+e]
		p.i += e + 2
	default:
		e := strings.IndexAny(p.s[p.i:], ":,}")
		if e < 0 || p.s[p.i+e] != ':' {
			return "", false
		}
		k = strings.TrimSpace(p.s[p.i : p.i+e])
		p.i += e
	}
	p.ws()
	if p.eof() || p.s[p.i] != ':' {
		return "", false
	}
	p.i++
	return k, k != ""
}

func (p *gparser) value(inObject bool, known map[string]bool) pyjson.Value {
	p.ws()
	if p.eof() {
		return pyjson.Str("")
	}
	switch {
	case strings.HasPrefix(p.s[p.i:], gemmaQ):
		start := p.i + len(gemmaQ)
		e := strings.Index(p.s[start:], gemmaQ)
		if e < 0 {
			p.i = len(p.s)
			return pyjson.Str(p.s[start:])
		}
		p.i = start + e + len(gemmaQ)
		return pyjson.Str(p.s[start : start+e])
	case p.s[p.i] == '"':
		j, esc := p.i+1, false
		for ; j < len(p.s); j++ {
			if esc {
				esc = false
			} else if p.s[j] == '\\' {
				esc = true
			} else if p.s[j] == '"' {
				break
			}
		}
		if j >= len(p.s) {
			v := p.s[p.i+1:]
			p.i = len(p.s)
			return pyjson.Str(v)
		}
		lit := p.s[p.i : j+1]
		p.i = j + 1
		var s string
		if json.Unmarshal([]byte(lit), &s) == nil {
			return pyjson.Str(s)
		}
		return pyjson.Str(lit[1 : len(lit)-1])
	case p.s[p.i] == '{':
		p.i++
		v, _ := p.object(nil)
		return v
	case p.s[p.i] == '[':
		p.i++
		arr := pyjson.NewArray()
		for {
			p.ws()
			if p.eof() {
				return arr
			}
			switch p.s[p.i] {
			case ']':
				p.i++
				return arr
			case ',':
				p.i++
				continue
			}
			before := p.i
			arr.A = append(arr.A, p.value(false, nil))
			if p.i == before { // no progress (stray '}'), bail out
				p.i++
			}
		}
	}
	// bare scalar
	start := p.i
	for !p.eof() {
		c := p.s[p.i]
		if c == '}' || c == ']' {
			break
		}
		if c == ',' && (!inObject || p.looksLikeKey(p.i+1, known)) {
			break
		}
		p.i++
	}
	return bareScalar(strings.TrimSpace(p.s[start:p.i]))
}

// looksLikeKey reports whether s[j:] starts a new `key:` pair.
func (p *gparser) looksLikeKey(j int, known map[string]bool) bool {
	s := p.s
	for j < len(s) && strings.IndexByte(" \t\r\n", s[j]) >= 0 {
		j++
	}
	if strings.HasPrefix(s[j:], gemmaQ) {
		j += len(gemmaQ)
	} else if j < len(s) && s[j] == '"' {
		j++
	}
	k := j
	for k < len(s) && isWordByte(s[k]) {
		k++
	}
	if k == j {
		return false
	}
	name := s[j:k]
	if strings.HasPrefix(s[k:], gemmaQ) {
		k += len(gemmaQ)
	} else if k < len(s) && s[k] == '"' {
		k++
	}
	for k < len(s) && (s[k] == ' ' || s[k] == '\t') {
		k++
	}
	if k >= len(s) || s[k] != ':' {
		return false
	}
	return known == nil || known[name]
}

// bareScalar mirrors toolbench.py: strip matching quotes, else _coerce
// (true/false/null/none case-insensitively, then JSON, else string).
func bareScalar(s string) pyjson.Value {
	if len(s) >= 2 && s[0] == s[len(s)-1] && (s[0] == '"' || s[0] == '\'') {
		return pyjson.Str(s[1 : len(s)-1])
	}
	return pyCoerce(s)
}
