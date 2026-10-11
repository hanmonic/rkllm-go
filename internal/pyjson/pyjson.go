// Package pyjson is an order-preserving JSON value type that serialises the
// way Python's json.dumps does (", " / ": " separators, ensure_ascii=False).
//
// Chat templates are Jinja and run under Python (transformers) or minja (the
// RKLLM runtime); both render `x | tojson` with Python's separators and key
// order. Reproducing that byte-for-byte matters for KV-cache prefix reuse, and
// Go's encoding/json can do neither (it sorts map keys and emits compact
// output with HTML escaping).
package pyjson

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// Kind is the JSON type of a Value.
type Kind uint8

const (
	Null Kind = iota
	Bool
	Number
	String
	Array
	Object
)

// Value is a parsed JSON value. Objects keep key order (duplicates allowed).
type Value struct {
	Kind Kind
	B    bool     // Bool
	S    string   // String value, or the literal text of a Number
	A    []Value  // Array items
	K    []string // Object keys
	V    []Value  // Object values, parallel to K
}

// Str, Num, Boolean, NullValue construct scalar values.
func Str(s string) Value        { return Value{Kind: String, S: s} }
func Num(lit string) Value      { return Value{Kind: Number, S: lit} }
func Boolean(b bool) Value      { return Value{Kind: Bool, B: b} }
func NullValue() Value          { return Value{Kind: Null} }
func NewObject() Value          { return Value{Kind: Object} }
func NewArray(a ...Value) Value { return Value{Kind: Array, A: a} }

// Set appends (or replaces the first existing) key on an object value.
func (v *Value) Set(k string, val Value) {
	for i, kk := range v.K {
		if kk == k {
			v.V[i] = val
			return
		}
	}
	v.K = append(v.K, k)
	v.V = append(v.V, val)
}

// Get returns the first value stored under key k of an object.
func (v Value) Get(k string) (Value, bool) {
	if v.Kind != Object {
		return Value{}, false
	}
	for i, kk := range v.K {
		if kk == k {
			return v.V[i], true
		}
	}
	return Value{}, false
}

// GetString returns v[k] if it is a string.
func (v Value) GetString(k string) string {
	x, ok := v.Get(k)
	if !ok || x.Kind != String {
		return ""
	}
	return x.S
}

// Equal reports deep equality (object key order matters).
func (v Value) Equal(o Value) bool {
	if v.Kind != o.Kind {
		return false
	}
	switch v.Kind {
	case Null:
		return true
	case Bool:
		return v.B == o.B
	case Number, String:
		return v.S == o.S
	case Array:
		if len(v.A) != len(o.A) {
			return false
		}
		for i := range v.A {
			if !v.A[i].Equal(o.A[i]) {
				return false
			}
		}
		return true
	case Object:
		if len(v.K) != len(o.K) {
			return false
		}
		for i := range v.K {
			if v.K[i] != o.K[i] || !v.V[i].Equal(o.V[i]) {
				return false
			}
		}
		return true
	}
	return false
}

// Parse parses strict JSON, preserving object key order and number literals.
func Parse(data []byte) (Value, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	v, err := parseValue(dec)
	if err != nil {
		return Value{}, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return Value{}, errors.New("pyjson: trailing data after JSON value")
	}
	return v, nil
}

// ParseString is Parse for a string.
func ParseString(s string) (Value, error) { return Parse([]byte(s)) }

func parseValue(dec *json.Decoder) (Value, error) {
	tok, err := dec.Token()
	if err != nil {
		return Value{}, err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			obj := Value{Kind: Object}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return Value{}, err
				}
				key, ok := kt.(string)
				if !ok {
					return Value{}, fmt.Errorf("pyjson: object key is %T", kt)
				}
				val, err := parseValue(dec)
				if err != nil {
					return Value{}, err
				}
				obj.K = append(obj.K, key)
				obj.V = append(obj.V, val)
			}
			if _, err := dec.Token(); err != nil {
				return Value{}, err
			}
			return obj, nil
		case '[':
			arr := Value{Kind: Array}
			for dec.More() {
				val, err := parseValue(dec)
				if err != nil {
					return Value{}, err
				}
				arr.A = append(arr.A, val)
			}
			if _, err := dec.Token(); err != nil {
				return Value{}, err
			}
			return arr, nil
		}
		return Value{}, fmt.Errorf("pyjson: unexpected delimiter %v", t)
	case bool:
		return Boolean(t), nil
	case nil:
		return NullValue(), nil
	case json.Number:
		return Num(string(t)), nil
	case string:
		return Str(t), nil
	}
	return Value{}, fmt.Errorf("pyjson: unexpected token %T", tok)
}

// ParseLenient tries strict JSON first, then repairs the mistakes small
// models make: Python literals (True/False/None) outside strings, single-quoted
// strings, and trailing commas.
func ParseLenient(s string) (Value, error) {
	v, err := ParseString(s)
	if err == nil {
		return v, nil
	}
	fixed := repair(s)
	if fixed == s {
		return Value{}, err
	}
	if v2, err2 := ParseString(fixed); err2 == nil {
		return v2, nil
	}
	return Value{}, err
}

func repair(s string) string {
	var b strings.Builder
	inStr := byte(0)
	esc := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inStr != 0 {
			if esc {
				esc = false
				b.WriteByte(c)
				continue
			}
			switch {
			case c == '\\':
				esc = true
				b.WriteByte(c)
			case c == inStr:
				inStr = 0
				b.WriteByte('"')
			case c == '"' && inStr == '\'':
				b.WriteString(`\"`)
			default:
				b.WriteByte(c)
			}
			continue
		}
		switch c {
		case '"', '\'':
			inStr = c
			b.WriteByte('"')
			continue
		case ',':
			// drop trailing comma before } or ]
			j := i + 1
			for j < len(s) && isSpace(s[j]) {
				j++
			}
			if j < len(s) && (s[j] == '}' || s[j] == ']') {
				continue
			}
		}
		if isIdentStart(c) && (i == 0 || !isIdentChar(s[i-1])) {
			j := i
			for j < len(s) && isIdentChar(s[j]) {
				j++
			}
			switch s[i:j] {
			case "True":
				b.WriteString("true")
				i = j - 1
				continue
			case "False":
				b.WriteString("false")
				i = j - 1
				continue
			case "None":
				b.WriteString("null")
				i = j - 1
				continue
			}
		}
		b.WriteByte(c)
	}
	return b.String()
}

func isSpace(c byte) bool      { return c == ' ' || c == '\t' || c == '\n' || c == '\r' }
func isIdentStart(c byte) bool { return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') }
func isIdentChar(c byte) bool  { return isIdentStart(c) || (c >= '0' && c <= '9') }

// Quote returns s as a JSON string literal escaped like Python's
// json.dumps(ensure_ascii=False): only ", \ and control characters.
func Quote(s string) string {
	var b strings.Builder
	quoteTo(&b, s)
	return b.String()
}

func quoteTo(b *strings.Builder, s string) {
	const hex = "0123456789abcdef"
	b.WriteByte('"')
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if c < 0x20 {
				b.WriteString(`\u00`)
				b.WriteByte(hex[c>>4])
				b.WriteByte(hex[c&0xf])
			} else {
				b.WriteByte(c)
			}
		}
	}
	b.WriteByte('"')
}

// Dump serialises like Python json.dumps(v, ensure_ascii=False).
func (v Value) Dump() string {
	var b strings.Builder
	v.dump(&b, "", "", 0)
	return b.String()
}

// DumpIndent serialises like Python json.dumps(v, indent=len(indent)).
func (v Value) DumpIndent(indent string) string {
	var b strings.Builder
	v.dump(&b, indent, "", 0)
	return b.String()
}

func (v Value) dump(b *strings.Builder, indent, cur string, _ int) {
	switch v.Kind {
	case Null:
		b.WriteString("null")
	case Bool:
		if v.B {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case Number:
		b.WriteString(v.S)
	case String:
		quoteTo(b, v.S)
	case Array:
		if len(v.A) == 0 {
			b.WriteString("[]")
			return
		}
		b.WriteByte('[')
		next := cur + indent
		for i, it := range v.A {
			if i > 0 {
				sep(b, indent)
			}
			if indent != "" {
				b.WriteByte('\n')
				b.WriteString(next)
			}
			it.dump(b, indent, next, 0)
		}
		if indent != "" {
			b.WriteByte('\n')
			b.WriteString(cur)
		}
		b.WriteByte(']')
	case Object:
		if len(v.K) == 0 {
			b.WriteString("{}")
			return
		}
		b.WriteByte('{')
		next := cur + indent
		for i, k := range v.K {
			if i > 0 {
				sep(b, indent)
			}
			if indent != "" {
				b.WriteByte('\n')
				b.WriteString(next)
			}
			quoteTo(b, k)
			b.WriteString(": ")
			v.V[i].dump(b, indent, next, 0)
		}
		if indent != "" {
			b.WriteByte('\n')
			b.WriteString(cur)
		}
		b.WriteByte('}')
	}
}

func sep(b *strings.Builder, indent string) {
	if indent == "" {
		b.WriteString(", ")
	} else {
		b.WriteByte(',')
	}
}

// PyStr renders a scalar the way Jinja's `value | string` does under Python:
// True/False/None for bool/null, the literal for numbers, the raw string for
// strings, and json-ish Python repr is NOT attempted for containers (callers
// use Dump for those, as the Qwen3.5 template does).
func (v Value) PyStr() string {
	switch v.Kind {
	case Null:
		return "None"
	case Bool:
		if v.B {
			return "True"
		}
		return "False"
	case Number, String:
		return v.S
	}
	return v.Dump()
}
