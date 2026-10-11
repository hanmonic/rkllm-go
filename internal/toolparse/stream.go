package toolparse

import (
	"strings"
)

// Holdback is a streaming aid. Given everything generated so far, it returns
// safe: the number of leading bytes that cannot be part of tool-call markup
// and may be streamed as content now; and inCall: whether a tool call has
// started (from text[safe:] on). Bytes past safe should be held back until
// more text arrives or generation ends (then call Parse on the full text).
//
// It is deliberately simple: it looks for the start markers of the family's
// call syntaxes and holds back any trailing partial marker. It does not look
// inside reasoning spans specially — a marker mentioned in <think> holds back
// the rest of the stream, which then gets parsed at the end anyway.
func Holdback(family, text string) (safe int, inCall bool) {
	var ms []marker
	switch family {
	case FamilyQwen3, FamilyQwen35:
		ms = []marker{lit("<tool_call>")}
	case FamilyGemma4:
		ms = []marker{lit("<|tool_call>"), gemmaBare{}, lit("<tool_call>")}
	default:
		ms = []marker{lit("<tool_call>"), jsonStart{}}
	}
	for i := 0; i < len(text); i++ {
		for _, m := range ms {
			switch m.at(text, i) {
			case full:
				return i, true
			case partial:
				return i, false
			}
		}
	}
	return len(text), false
}

type match int

const (
	none match = iota
	partial
	full
)

type marker interface {
	at(text string, i int) match
}

type lit string

func (l lit) at(text string, i int) match {
	s := string(l)
	rest := text[i:]
	if strings.HasPrefix(rest, s) {
		return full
	}
	if len(rest) < len(s) && strings.HasPrefix(s, rest) {
		return partial
	}
	return none
}

// gemmaBare matches `call:NAME{` at a word boundary.
type gemmaBare struct{}

func (gemmaBare) at(text string, i int) match {
	if i > 0 && isWordByte(text[i-1]) {
		return none
	}
	rest := text[i:]
	const p = "call:"
	if len(rest) < len(p) {
		if strings.HasPrefix(p, rest) {
			return partial
		}
		return none
	}
	if !strings.HasPrefix(rest, p) {
		return none
	}
	j := len(p)
	for j < len(rest) && (isWordByte(rest[j]) || rest[j] == '.' || rest[j] == '-') {
		j++
	}
	if j == len(rest) {
		return partial
	}
	if j > len(p) && rest[j] == '{' {
		return full
	}
	return none
}

// jsonStart matches the start of a generic JSON call: an optional ```json
// fence, optional '[', then '{' followed by a "name"/"function"/"tool_calls" key.
type jsonStart struct{}

func (jsonStart) at(text string, i int) match {
	rest := text[i:]
	c := rest[0]
	if c != '`' && c != '{' && c != '[' {
		return none
	}
	j := 0
	if c == '`' {
		for _, f := range []string{"```json", "```"} {
			if strings.HasPrefix(rest, f) {
				j = len(f)
				break
			}
			if strings.HasPrefix(f, rest) {
				return partial
			}
		}
		if j == 0 {
			return none
		}
	}
	skip := func() {
		for j < len(rest) && strings.IndexByte(" \t\r\n", rest[j]) >= 0 {
			j++
		}
	}
	skip()
	if j < len(rest) && rest[j] == '[' {
		j++
		skip()
	}
	if j == len(rest) {
		return partial
	}
	if rest[j] != '{' {
		return none
	}
	j++
	skip()
	tail := rest[j:]
	for _, k := range []string{`"name"`, `"function"`, `"tool_calls"`, `"type"`} {
		if strings.HasPrefix(tail, k) {
			return full
		}
		if strings.HasPrefix(k, tail) {
			return partial
		}
	}
	return none
}
