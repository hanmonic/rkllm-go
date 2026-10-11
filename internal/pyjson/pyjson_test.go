package pyjson

import "testing"

func TestDumpMatchesPython(t *testing.T) {
	// expected strings produced by Python: json.dumps(json.loads(in), ensure_ascii=False)
	cases := []struct{ in, want, indent2 string }{
		{`{"b":1,"a":[true,null,1.50,"x"]}`, `{"b": 1, "a": [true, null, 1.50, "x"]}`,
			"{\n  \"b\": 1,\n  \"a\": [\n    true,\n    null,\n    1.50,\n    \"x\"\n  ]\n}"},
		{`"<a> & °C   \"q\" \\ \n\t\u0001"`, "\"<a> & °C   \\\"q\\\" \\\\ \\n\\t\\u0001\"", ""},
		{`{"e":{},"l":[]}`, `{"e": {}, "l": []}`, "{\n  \"e\": {},\n  \"l\": []\n}"},
	}
	for _, c := range cases {
		v, err := ParseString(c.in)
		if err != nil {
			t.Fatal(err)
		}
		if got := v.Dump(); got != c.want {
			t.Errorf("Dump(%s) = %s, want %s", c.in, got, c.want)
		}
		if c.indent2 != "" {
			if got := v.DumpIndent("  "); got != c.indent2 {
				t.Errorf("DumpIndent(%s) = %q, want %q", c.in, got, c.indent2)
			}
		}
	}
}

func TestParseLenient(t *testing.T) {
	cases := map[string]string{
		`{"a": True, "b": None, "c": False,}`: `{"a": true, "b": null, "c": false}`,
		`{'a': 'it"s', "True": "True"}`:       `{"a": "it\"s", "True": "True"}`,
		`[1, 2, ]`:                            `[1, 2]`,
	}
	for in, want := range cases {
		v, err := ParseLenient(in)
		if err != nil {
			t.Fatalf("%s: %v", in, err)
		}
		if got := v.Dump(); got != want {
			t.Errorf("ParseLenient(%s) = %s, want %s", in, got, want)
		}
	}
	if _, err := ParseLenient(`{"a": 1} trailing`); err == nil {
		t.Error("expected error for trailing data")
	}
}

func TestPyStr(t *testing.T) {
	for in, want := range map[string]string{`true`: "True", `false`: "False", `null`: "None", `21.5`: "21.5", `"s"`: "s", `[1,"a"]`: `[1, "a"]`} {
		v, _ := ParseString(in)
		if got := v.PyStr(); got != want {
			t.Errorf("PyStr(%s) = %q, want %q", in, got, want)
		}
	}
}
