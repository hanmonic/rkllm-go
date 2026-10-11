package toolparse

import (
	"math"
	"strconv"
	"strings"

	"rkllm-go/internal/pyjson"
)

// pyCoerce is toolbench.py's _coerce: decode a textual parameter value when
// possible (bools in any case, null/None, JSON numbers/arrays/objects), else
// keep it as a string. Used only where no schema type is known.
func pyCoerce(s string) pyjson.Value {
	t := strings.TrimSpace(s)
	switch strings.ToLower(t) {
	case "true":
		return pyjson.Boolean(true)
	case "false":
		return pyjson.Boolean(false)
	case "null", "none":
		return pyjson.NullValue()
	}
	if v, err := pyjson.ParseString(t); err == nil {
		return v
	}
	return pyjson.Str(s)
}

// schemaTypes returns the JSON-schema type(s) of sch, inferring from
// properties/items/enum when "type" is absent.
func schemaTypes(sch pyjson.Value) []string {
	if sch.Kind != pyjson.Object {
		return nil
	}
	if t, ok := sch.Get("type"); ok {
		switch t.Kind {
		case pyjson.String:
			return []string{t.S}
		case pyjson.Array:
			var out []string
			for _, x := range t.A {
				if x.Kind == pyjson.String {
					out = append(out, x.S)
				}
			}
			return out
		}
	}
	if _, ok := sch.Get("properties"); ok {
		return []string{"object"}
	}
	if _, ok := sch.Get("items"); ok {
		return []string{"array"}
	}
	for _, k := range []string{"anyOf", "oneOf"} {
		if alts, ok := sch.Get(k); ok && alts.Kind == pyjson.Array {
			var out []string
			for _, a := range alts.A {
				out = append(out, schemaTypes(a)...)
			}
			return out
		}
	}
	return nil
}

func matches(v pyjson.Value, typ string) bool {
	switch typ {
	case "string":
		return v.Kind == pyjson.String
	case "number":
		return v.Kind == pyjson.Number
	case "integer":
		return v.Kind == pyjson.Number && isIntLiteral(v.S)
	case "boolean":
		return v.Kind == pyjson.Bool
	case "array":
		return v.Kind == pyjson.Array
	case "object":
		return v.Kind == pyjson.Object
	case "null":
		return v.Kind == pyjson.Null
	}
	return false
}

func isIntLiteral(s string) bool {
	_, err := strconv.ParseInt(s, 10, 64)
	return err == nil
}

// coerce converts v toward schema sch. loose marks values that came from
// untyped text (Qwen3.5 XML parameters): without a schema type they get
// pyCoerce. Containers are coerced recursively via properties / items.
func coerce(v pyjson.Value, sch pyjson.Value, loose bool) pyjson.Value {
	types := schemaTypes(sch)
	if len(types) == 0 {
		switch {
		case v.Kind == pyjson.String && loose:
			return pyCoerce(v.S)
		case v.Kind == pyjson.Object:
			return coerceObject(v, sch, loose)
		case v.Kind == pyjson.Array:
			return coerceArray(v, sch, loose)
		}
		return v
	}
	for _, t := range types {
		if matches(v, t) {
			return recurse(v, sch, loose)
		}
	}
	for _, t := range types {
		if out, ok := convert(v, t); ok {
			return recurse(out, sch, loose)
		}
	}
	return recurse(v, sch, loose)
}

func recurse(v, sch pyjson.Value, loose bool) pyjson.Value {
	switch v.Kind {
	case pyjson.Object:
		return coerceObject(v, sch, loose)
	case pyjson.Array:
		return coerceArray(v, sch, loose)
	}
	return v
}

func coerceObject(v, sch pyjson.Value, loose bool) pyjson.Value {
	props, _ := sch.Get("properties")
	out := pyjson.NewObject()
	for i, k := range v.K {
		ps, _ := props.Get(k)
		out.K = append(out.K, k)
		out.V = append(out.V, coerce(v.V[i], ps, loose))
	}
	return out
}

func coerceArray(v, sch pyjson.Value, loose bool) pyjson.Value {
	items, _ := sch.Get("items")
	out := pyjson.NewArray()
	for _, it := range v.A {
		out.A = append(out.A, coerce(it, items, loose))
	}
	return out
}

// convert tries to turn v into JSON type typ.
func convert(v pyjson.Value, typ string) (pyjson.Value, bool) {
	s := strings.TrimSpace(v.S)
	switch typ {
	case "string":
		switch v.Kind {
		case pyjson.Number:
			return pyjson.Str(v.S), true
		case pyjson.Bool:
			return pyjson.Str(strconv.FormatBool(v.B)), true
		case pyjson.Object, pyjson.Array:
			return pyjson.Str(v.Dump()), true
		}
	case "number":
		switch v.Kind {
		case pyjson.String:
			if f, err := strconv.ParseFloat(s, 64); err == nil && !math.IsInf(f, 0) && !math.IsNaN(f) {
				return pyjson.Num(numLiteral(s, f)), true
			}
		}
	case "integer":
		switch v.Kind {
		case pyjson.String, pyjson.Number:
			if n, err := strconv.ParseInt(s, 10, 64); err == nil {
				return pyjson.Num(strconv.FormatInt(n, 10)), true
			}
			if f, err := strconv.ParseFloat(s, 64); err == nil && f == math.Trunc(f) && math.Abs(f) < 1<<53 {
				return pyjson.Num(strconv.FormatInt(int64(f), 10)), true
			}
		}
	case "boolean":
		switch v.Kind {
		case pyjson.String:
			switch strings.ToLower(s) {
			case "true", "yes", "on", "1":
				return pyjson.Boolean(true), true
			case "false", "no", "off", "0":
				return pyjson.Boolean(false), true
			}
		case pyjson.Number:
			switch v.S {
			case "1":
				return pyjson.Boolean(true), true
			case "0":
				return pyjson.Boolean(false), true
			}
		}
	case "array":
		if v.Kind == pyjson.String {
			if a, err := pyjson.ParseLenient(s); err == nil && a.Kind == pyjson.Array {
				return a, true
			}
			if s == "" {
				return pyjson.NewArray(), true
			}
			arr := pyjson.NewArray()
			for _, part := range strings.Split(s, ",") {
				arr.A = append(arr.A, pyjson.Str(strings.TrimSpace(part)))
			}
			return arr, true
		}
		if v.Kind != pyjson.Null {
			return pyjson.NewArray(v), true
		}
	case "object":
		if v.Kind == pyjson.String {
			if o, err := pyjson.ParseLenient(s); err == nil && o.Kind == pyjson.Object {
				return o, true
			}
		}
	case "null":
		if v.Kind == pyjson.String {
			switch strings.ToLower(s) {
			case "null", "none", "":
				return pyjson.NullValue(), true
			}
		}
	}
	return v, false
}

// numLiteral keeps a model's number text when it is already valid JSON,
// otherwise reformats it ("+5" -> "5", ".5" -> "0.5").
func numLiteral(s string, f float64) string {
	if v, err := pyjson.ParseString(s); err == nil && v.Kind == pyjson.Number {
		return s
	}
	return strconv.FormatFloat(f, 'f', -1, 64)
}
