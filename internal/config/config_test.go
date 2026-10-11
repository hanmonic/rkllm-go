package config

import (
	"os"
	"testing"
)

func TestParseV1AndV2(t *testing.T) {
	v1 := `{"default":"a","models":{"a":"/m/a.rkllm","q35-2b":"/m/q"},"families":{"a":"gemma3"}}`
	c, err := Parse([]byte(v1), 4096)
	if err != nil {
		t.Fatal(err)
	}
	if c.Models["a"].Family != "gemma3" || c.Models["a"].Strategy != StrategyPrefix || c.Models["a"].MaxContextLen != 4096 {
		t.Fatalf("v1 a: %+v", c.Models["a"])
	}
	if c.Models["q35-2b"].Strategy != StrategySnapshot {
		t.Fatal("qwen3.5 default strategy")
	}
	v2 := `{"default":"x","preload":["x"],"strategies":{"qwen3.5":"append"},
	  "models":{"x":{"path":"/x","family":"qwen3","max_context_len":16384,"strategy":"append","mem_mb":3},
	            "y":{"path":"/y","family":"qwen3.5"}}}`
	c, err = Parse([]byte(v2), 4096)
	if err != nil {
		t.Fatal(err)
	}
	x, y := c.Models["x"], c.Models["y"]
	if !x.Preload || x.MaxContextLen != 16384 || x.Strategy != StrategyAppend || x.MemMB != 3 || y.Strategy != StrategyAppend {
		t.Fatalf("v2: %+v %+v", x, y)
	}
	if _, err := Parse([]byte(`{"default":"x","models":{"x":{"path":"/x","strategy":"bogus"}}}`), 1); err == nil {
		t.Fatal("bad strategy accepted")
	}
}

func TestRepoModelsJSON(t *testing.T) {
	b, err := os.ReadFile("../../models.json")
	if err != nil {
		t.Skip(err)
	}
	c, err := Parse(b, 4096)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"qwen3-0.6b", "qwen3-1.7b", "qwen3-4b-16k", "qwen3-8b-allnpu", "q35-2b", "q35-4b", "q35-9b", "gemma3n-e4b"} {
		if m := c.Models[n]; m == nil || m.MaxContextLen != 16384 {
			t.Fatalf("%s missing or wrong context", n)
		}
	}
}
