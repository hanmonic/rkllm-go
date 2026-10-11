// Package config loads models.json.
//
// Backward compatible with v1: "models" maps a name to either a path string
// or an object with per-model settings, and the old top-level "families" map
// still works. v2 fields (all optional):
//
//	{
//	  "default": "qwen3-1.7b",
//	  "preload": ["qwen3-1.7b"],
//	  "strategies": {"qwen3": "prefix", "qwen3.5": "snapshot", "gemma3": "prefix"},
//	  "models": {
//	    "qwen3-1.7b": "/opt/rkllm/models/qwen3-1.7b/model.rkllm",
//	    "q35-2b": {"path": "/opt/rkllm/models/q35-2b/model.rkllm", "family": "qwen3.5",
//	               "max_context_len": 16384, "strategy": "snapshot", "preload": true, "mem_mb": 2500}
//	  }
//	}
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"rkllm-go/internal/prompt"
)

// Strategy is a per-handle KV-cache strategy.
type Strategy string

const (
	// StrategyPrefix: keep_history=0, full prompt every request; the runtime
	// reuses the longest common token prefix with the previous run's prompt.
	StrategyPrefix Strategy = "prefix"
	// StrategyAppend: keep_history=1, send only the delta when the new prompt
	// extends what is in the cache; otherwise full clear + full prompt.
	StrategyAppend Strategy = "append"
	// StrategySnapshot: keep_history=0, save a prompt-cache file on every run
	// and load it before the next run of the session (for models without
	// runtime prefix reuse, i.e. Qwen3.5).
	StrategySnapshot Strategy = "snapshot"
	// StrategyOff: full clear before every request.
	StrategyOff Strategy = "off"
)

// ParseStrategy validates a strategy name.
func ParseStrategy(s string) (Strategy, error) {
	switch Strategy(strings.ToLower(strings.TrimSpace(s))) {
	case StrategyPrefix:
		return StrategyPrefix, nil
	case StrategyAppend:
		return StrategyAppend, nil
	case StrategySnapshot:
		return StrategySnapshot, nil
	case StrategyOff:
		return StrategyOff, nil
	}
	return "", fmt.Errorf("unknown cache strategy %q (prefix|append|snapshot|off)", s)
}

// DefaultStrategies per family (docs/kv-cache-probe-findings.md: prefix reuse
// works on Qwen3 and Gemma, not on Qwen3.5). Qwen3.5 uses snapshot: restores
// are token-exact (verified with RKLLM_LOG_LEVEL=2, docs/v2-design.md) and a
// session switch costs a ~20-60 ms file load instead of a cold prefill;
// append stays available per model or per family in models.json.
var DefaultStrategies = map[prompt.Family]Strategy{
	prompt.FamilyQwen3:  StrategyPrefix,
	prompt.FamilyQwen35: StrategySnapshot,
	prompt.FamilyGemma3: StrategyPrefix,
	prompt.FamilyGemma4: StrategyPrefix,
}

// Model is one resolved model entry.
type Model struct {
	Name          string
	Path          string
	Family        prompt.Family
	Template      prompt.Template
	MaxContextLen int
	Strategy      Strategy
	Preload       bool
	// MemMB is the expected resident memory of a loaded handle (0 = estimate
	// from the file size and context length).
	MemMB int
}

// File is the raw JSON shape.
type File struct {
	Default    string                `json:"default"`
	Preload    []string              `json:"preload,omitempty"`
	Models     map[string]ModelEntry `json:"models"`
	Families   map[string]string     `json:"families,omitempty"`
	Strategies map[string]string     `json:"strategies,omitempty"`
}

// ModelEntry accepts a path string or an object.
type ModelEntry struct {
	Path          string `json:"path"`
	Family        string `json:"family,omitempty"`
	MaxContextLen int    `json:"max_context_len,omitempty"`
	Strategy      string `json:"strategy,omitempty"`
	Preload       bool   `json:"preload,omitempty"`
	MemMB         int    `json:"mem_mb,omitempty"`
}

func (e *ModelEntry) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		*e = ModelEntry{Path: s}
		return nil
	}
	type plain ModelEntry
	var p plain
	if err := json.Unmarshal(b, &p); err != nil {
		return err
	}
	*e = ModelEntry(p)
	return nil
}

// Config is the resolved configuration.
type Config struct {
	Default string
	Models  map[string]*Model
}

// Names returns model names sorted.
func (c *Config) Names() []string {
	n := make([]string, 0, len(c.Models))
	for k := range c.Models {
		n = append(n, k)
	}
	sort.Strings(n)
	return n
}

// Load reads and resolves a models.json file.
func Load(path string, defaultCtx int) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(b, defaultCtx)
}

// Parse resolves a models.json document.
func Parse(b []byte, defaultCtx int) (*Config, error) {
	var f File
	if err := json.Unmarshal(b, &f); err != nil {
		return nil, fmt.Errorf("models.json: %w", err)
	}
	if len(f.Models) == 0 {
		return nil, fmt.Errorf("models.json: no models defined")
	}
	famStrat := map[prompt.Family]Strategy{}
	for k, v := range DefaultStrategies {
		famStrat[k] = v
	}
	for fam, s := range f.Strategies {
		pf, ok := prompt.ParseFamily(fam)
		if !ok {
			return nil, fmt.Errorf("models.json strategies: unknown family %q", fam)
		}
		st, err := ParseStrategy(s)
		if err != nil {
			return nil, err
		}
		famStrat[pf] = st
	}
	pre := map[string]bool{}
	for _, n := range f.Preload {
		pre[n] = true
	}
	c := &Config{Default: f.Default, Models: map[string]*Model{}}
	for name, e := range f.Models {
		fam := e.Family
		if fam == "" {
			fam = f.Families[name]
		}
		tpl, err := prompt.Resolve(fam, name)
		if err != nil {
			// v1 configs list models without a template (llama, deepseek):
			// fall back to qwen3 like v1 did.
			tpl, _ = prompt.Get(prompt.FamilyQwen3)
		}
		m := &Model{
			Name:          name,
			Path:          e.Path,
			Family:        tpl.Family(),
			Template:      tpl,
			MaxContextLen: e.MaxContextLen,
			Preload:       e.Preload || pre[name],
			MemMB:         e.MemMB,
		}
		if m.MaxContextLen <= 0 {
			m.MaxContextLen = defaultCtx
		}
		if e.Strategy != "" {
			st, err := ParseStrategy(e.Strategy)
			if err != nil {
				return nil, fmt.Errorf("model %s: %w", name, err)
			}
			m.Strategy = st
		} else {
			m.Strategy = famStrat[m.Family]
			if m.Strategy == "" {
				m.Strategy = StrategyPrefix
			}
		}
		c.Models[name] = m
	}
	for n := range pre {
		if _, ok := c.Models[n]; !ok {
			return nil, fmt.Errorf("models.json preload: unknown model %q", n)
		}
	}
	if c.Default == "" {
		return nil, fmt.Errorf("models.json: missing \"default\"")
	}
	if _, ok := c.Models[c.Default]; !ok {
		return nil, fmt.Errorf("models.json: default model %q not in models", c.Default)
	}
	return c, nil
}
