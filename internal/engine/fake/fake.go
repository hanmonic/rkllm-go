// Package fake is an in-memory engine.Engine that mimics the RKLLM runtime
// cache semantics measured in docs/kv-cache-probe-findings.md, with one
// "token" per byte:
//
//   - keep_history=0: reuse the longest common prefix with the previous
//     keep_history=0 prompt (or a loaded prompt cache), unless NoLCP.
//   - keep_history=1: append to the cache, no matching.
//   - the last emitted token (the stop token) is not cached.
//   - switching keep_history modes without a clear corrupts the output.
//   - prompt cache files hold the prompt only; a load seeds the next
//     keep_history=0 run once.
package fake

import (
	"context"
	"fmt"
	"os"
	"sync"

	"rkllm-go/internal/engine"
)

// Reply produces the model output for the full context text the model sees.
type Reply func(context string) string

type Engine struct {
	NoLCP    bool
	Reply    Reply
	PerRunMT bool // report per-run max tokens support (1.3.1)

	mu      sync.Mutex
	Handles []*Handle
}

func (e *Engine) Version() string       { return "fake" }
func (e *Engine) PerRunMaxTokens() bool { return e.PerRunMT }

func (e *Engine) Load(p engine.Params) (engine.Handle, error) {
	h := &Handle{e: e, mode: -1}
	e.mu.Lock()
	e.Handles = append(e.Handles, h)
	e.mu.Unlock()
	return h, nil
}

type Handle struct {
	e      *Engine
	mu     sync.Mutex
	kvText string // what the cache holds
	record string // previous keep_history=0 prompt
	seed   *string
	mode   int // -1 clean, 0, 1
	paused bool

	// observability for tests
	LastContext string
	LastPrefill int
	Runs        int
	Corrupted   bool
	Loads       int
	Saves       int
	Clears      int
}

func lcp(a, b string) int {
	i := 0
	for i < len(a) && i < len(b) && a[i] == b[i] {
		i++
	}
	return i
}

func (h *Handle) Run(ctx context.Context, req engine.RunRequest) (engine.RunResult, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.Runs++
	var res engine.RunResult
	want := 0
	if req.KeepHistory {
		want = 1
	}
	corrupt := h.mode != -1 && h.mode != want && !h.paused
	if corrupt {
		h.Corrupted = true
	}
	var context string
	prefill := 0
	if h.paused {
		// resume/append semantics after a pause
		context = h.kvText + req.Prompt
		prefill = len(req.Prompt)
		h.paused = false
	} else if req.KeepHistory {
		context = h.kvText + req.Prompt
		prefill = len(req.Prompt)
	} else {
		rec := h.record
		if h.seed != nil {
			rec = *h.seed
			h.seed = nil
		} else if h.e.NoLCP {
			rec = ""
		}
		reuse := lcp(req.Prompt, rec)
		if reuse == len(req.Prompt) && reuse > 0 {
			reuse-- // the last prompt token is re-fed
		}
		context = req.Prompt
		prefill = len(req.Prompt) - reuse
		h.record = req.Prompt
	}
	h.mode = want
	if req.SaveCachePath != "" {
		if err := os.WriteFile(req.SaveCachePath, []byte(context), 0o644); err != nil {
			return res, err
		}
		h.Saves++
	}
	h.LastContext = context
	h.LastPrefill = prefill
	out := "CORRUPT"
	if !corrupt && h.e.Reply != nil {
		out = h.e.Reply(context)
	}
	max := req.MaxNewTokens
	toks := []engine.Token{}
	for i := 0; i < len(out); i++ {
		toks = append(toks, engine.Token{ID: int32(out[i]), Text: out[i : i+1]})
	}
	toks = append(toks, engine.Token{ID: -1}) // stop token, empty text
	if max > 0 && len(toks) > max {
		toks = toks[:max]
	}
	for i, t := range toks {
		if ctx.Err() != nil {
			res.Aborted = true
			h.kvText = context
			return res, nil
		}
		res.Tokens = append(res.Tokens, t)
		if req.OnToken != nil {
			req.OnToken(t)
		}
		if req.StopAfter > 0 && i+1 >= req.StopAfter {
			res.Stopped = true
			h.paused = true
			h.kvText = context
			return res, nil
		}
	}
	gen := ""
	for i := 0; i+1 < len(res.Tokens); i++ {
		gen += res.Tokens[i].Text
	}
	h.kvText = context + gen
	res.Finished = true
	res.Perf = &engine.Perf{PrefillTokens: prefill, GenerateTokens: len(res.Tokens) - 1}
	return res, nil
}

func (h *Handle) ClearKV() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.kvText, h.record, h.seed, h.mode, h.paused = "", "", nil, -1, false
	h.Clears++
	return nil
}

func (h *Handle) KVSize() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.kvText)
}

func (h *Handle) LoadPromptCache(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("fake load: %w", err)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	s := string(b)
	h.seed = &s
	h.Loads++
	return nil
}

func (h *Handle) ReleasePromptCache() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.seed, h.record = nil, ""
	return nil
}

func (h *Handle) SetChatTemplate(a, b, c string) error { return h.ClearKV() }
func (h *Handle) Destroy()                             {}
