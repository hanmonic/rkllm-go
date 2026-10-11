// Package fakeserver is a stand-in for rkllm-go that honours the KV-cache
// server contract (X-Session-Id, "cache": false, usage.rkllm, streaming with
// include_usage) with synthetic timing. Its "model" plays each scenario's
// scripted reference solution, so a harness run against it should pass every
// task. It models one NPU: one request at a time, one resident model, one
// resident KV session per model; switching sessions costs a restore.
package fakeserver

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"rkllm-go/cmd/rkllm-bench/scenarios"
)

// ModelSpec describes one fake model.
type ModelSpec struct {
	ID         string
	Family     string
	Context    int
	PrefillTPS float64 // tokens/s
	DecodeTPS  float64 // tokens/s
	LoadMs     float64
}

// Options configures the fake.
type Options struct {
	Models         []ModelSpec
	TimeScale      float64 // multiply all synthetic sleeps (0 = no sleeping)
	RestorePerTok  float64 // ms per KV token to restore a parked session
	MalformedEvery int     // every Nth completion leaks raw tool-call text instead of tool_calls (0 = never)
	ExposeContext  bool    // include "max_context" in /v1/models
}

// DefaultModels mirrors the models to be benchmarked, with rough RK3588 speeds.
func DefaultModels() []ModelSpec {
	return []ModelSpec{
		{"qwen3-0.6b", "qwen3", 4096, 400, 18, 2000},
		{"qwen3-1.7b", "qwen3", 4096, 200, 10, 4000},
		{"qwen3-4b-16k", "qwen3", 16384, 80, 5.5, 8000},
		{"qwen3-8b-allnpu", "qwen3", 16384, 36, 3.2, 15000},
		{"q35-2b", "qwen3.5", 16384, 150, 9.9, 5000},
		{"q35-4b", "qwen3.5", 16384, 75, 4.5, 8000},
		{"q35-9b", "qwen3.5", 16384, 30, 2.2, 18000},
		{"gemma3n-e4b", "gemma3", 4096, 60, 3.3, 9000},
	}
}

type session struct {
	model  string
	text   string // rendered prompt + generated assistant turn
	tokens int
	turns  int
	last   time.Time
}

// Server is the fake.
type Server struct {
	opt      Options
	models   map[string]ModelSpec
	npu      sync.Mutex // one request at a time
	mu       sync.Mutex // guards the fields below
	loaded   string
	resident map[string]string // model -> session whose KV is resident
	sessions map[string]*session
	nreq     int
	nextSess int
}

// New builds a fake server.
func New(opt Options) *Server {
	if len(opt.Models) == 0 {
		opt.Models = DefaultModels()
	}
	if opt.RestorePerTok == 0 {
		opt.RestorePerTok = 0.05
	}
	s := &Server{opt: opt, models: map[string]ModelSpec{}, resident: map[string]string{}, sessions: map[string]*session{}}
	for _, m := range opt.Models {
		s.models[m.ID] = m
	}
	return s
}

// Handler returns the HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "ok") })
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/debug/state", s.handleState)
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	return mux
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var data []map[string]any
	for _, m := range s.opt.Models {
		d := map[string]any{"id": m.ID, "object": "model", "owned_by": "fake", "loaded": m.ID == s.loaded, "family": m.Family}
		if s.opt.ExposeContext {
			d["max_context"] = m.Context
		}
		data = append(data, d)
	}
	writeJSON(w, 200, map[string]any{"object": "list", "data": data})
}

func (s *Server) handleState(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var sess []map[string]any
	ids := make([]string, 0, len(s.sessions))
	for id := range s.sessions {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		x := s.sessions[id]
		sess = append(sess, map[string]any{"id": id, "model": x.model, "kv_tokens": x.tokens, "turns": x.turns, "resident": s.resident[x.model] == id})
	}
	writeJSON(w, 200, map[string]any{"loaded_model": s.loaded, "requests": s.nreq, "sessions": sess})
}

type chatReq struct {
	Model         string              `json:"model"`
	Messages      []scenarios.Message `json:"messages"`
	Tools         []json.RawMessage   `json:"tools"`
	MaxTokens     int                 `json:"max_tokens"`
	Stream        bool                `json:"stream"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	Cache *bool `json:"cache"`
}

// render is the fake chat template: deterministic, prefix-stable.
func render(tools []json.RawMessage, msgs []scenarios.Message) string {
	var b strings.Builder
	for _, t := range tools {
		b.Write(t)
	}
	for _, m := range msgs {
		b.WriteString(renderMsg(m))
	}
	b.WriteString("<|assistant|>")
	return b.String()
}

func renderMsg(m scenarios.Message) string {
	var b strings.Builder
	fmt.Fprintf(&b, "<|%s|>%s", m.Role, m.Content)
	for _, tc := range m.ToolCalls {
		fmt.Fprintf(&b, "<tool_call>%s %s</tool_call>", tc.Function.Name, tc.Function.Arguments)
	}
	b.WriteString("<|end|>")
	return b.String()
}

func toks(chars int) int { return (chars + 3) / 4 }

func lcp(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

func (s *Server) sleep(msv float64) {
	if s.opt.TimeScale > 0 && msv > 0 {
		time.Sleep(time.Duration(msv * s.opt.TimeScale * float64(time.Millisecond)))
	}
}

// reply picks the scripted assistant turn for this conversation.
func reply(msgs []scenarios.Message) scenarios.Reply {
	if r, ok := scenarios.ReferenceReply(msgs); ok {
		return r
	}
	return scenarios.Reply{Content: "OK."}
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, 405, map[string]any{"error": map[string]any{"message": "POST only"}})
		return
	}
	var req chatReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, 400, map[string]any{"error": map[string]any{"message": err.Error(), "type": "invalid_request_error"}})
		return
	}
	spec, ok := s.models[req.Model]
	if !ok {
		writeJSON(w, 404, map[string]any{"error": map[string]any{"message": "unknown model " + req.Model, "type": "invalid_request_error"}})
		return
	}
	cacheOn := req.Cache == nil || *req.Cache
	sid := r.Header.Get("X-Session-Id")

	tq := time.Now()
	s.npu.Lock()
	defer s.npu.Unlock()
	queueMs := float64(time.Since(tq).Microseconds()) / 1000

	s.mu.Lock()
	s.nreq++
	nreq := s.nreq
	if sid == "" {
		s.nextSess++
		sid = fmt.Sprintf("auto-%d", s.nextSess)
	}
	loadMs := 0.0
	if s.loaded != req.Model {
		loadMs = spec.LoadMs
		s.loaded = req.Model
		s.resident = map[string]string{}
	}
	prompt := render(req.Tools, req.Messages)
	promptTok := toks(len(prompt))
	cached, event, strategy, restoreMs := 0, "cold", "off", 0.0
	sess := s.sessions[sid]
	if cacheOn {
		event, strategy = "miss", "prefix"
		if sess != nil && sess.model == req.Model {
			common := lcp(sess.text, prompt)
			cached = common / 4
			if cached >= promptTok {
				cached = promptTok - 1 // always prefill at least one token
			}
			if cached > 0 {
				if common >= len(sess.text) {
					event, strategy = "hit", "append"
				} else {
					event = "partial"
				}
				if s.resident[req.Model] != sid {
					event = "restored"
					restoreMs = float64(sess.tokens) * s.opt.RestorePerTok
				}
			}
		}
	}
	s.mu.Unlock()

	rep := reply(req.Messages)
	malformed := s.opt.MalformedEvery > 0 && nreq%s.opt.MalformedEvery == 0 && len(rep.Calls) > 0
	msg := scenarios.Message{Role: "assistant", Content: rep.Content}
	if malformed {
		b, _ := json.Marshal(map[string]any{"name": rep.Calls[0].Name, "arguments": rep.Calls[0].Args})
		msg.Content = "<tool_call>" + string(b)[:len(b)/2]
	} else {
		for i, c := range rep.Calls {
			args, _ := json.Marshal(c.Args)
			msg.ToolCalls = append(msg.ToolCalls, scenarios.ToolCall{ID: fmt.Sprintf("call_%d_%d", nreq, i), Type: "function",
				Function: scenarios.FunctionCall{Name: c.Name, Arguments: string(args)}})
		}
	}
	gen := renderMsg(msg)
	genTok := toks(len(gen))
	prefillTok := promptTok - cached
	prefillMs := float64(prefillTok) / spec.PrefillTPS * 1000
	tokMs := 1000 / spec.DecodeTPS
	genMs := float64(genTok) * tokMs
	ttftMs := loadMs + restoreMs + prefillMs + tokMs

	s.mu.Lock()
	if cacheOn {
		s.sessions[sid] = &session{model: req.Model, text: strings.TrimSuffix(prompt, "<|assistant|>") + gen, tokens: promptTok + genTok,
			turns: turnsOf(sess) + 1, last: time.Now()}
		s.resident[req.Model] = sid
	} else {
		delete(s.sessions, sid)
		s.resident[req.Model] = ""
	}
	s.mu.Unlock()

	finish := "stop"
	if len(msg.ToolCalls) > 0 {
		finish = "tool_calls"
	}
	usage := map[string]any{
		"prompt_tokens": promptTok, "completion_tokens": genTok, "total_tokens": promptTok + genTok,
		"prompt_tokens_details": map[string]any{"cached_tokens": cached},
		"rkllm": map[string]any{
			"prefill_tokens": prefillTok, "cached_tokens": cached, "prefill_ms": prefillMs, "generate_tokens": genTok,
			"generate_ms": genMs, "ttft_ms": ttftMs, "queue_ms": queueMs, "model_load_ms": loadMs,
			"cache_strategy": strategy, "cache_event": event, "session_id": sid, "kv_cache_size": promptTok + genTok,
		},
	}
	w.Header().Set("X-Session-Id", sid)
	id := fmt.Sprintf("chatcmpl-fake-%d", nreq)
	s.sleep(loadMs + restoreMs + prefillMs)
	if !req.Stream {
		s.sleep(genMs)
		writeJSON(w, 200, map[string]any{"id": id, "object": "chat.completion", "model": req.Model,
			"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": finish}}, "usage": usage})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	fl, _ := w.(http.Flusher)
	send := func(v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if fl != nil {
			fl.Flush()
		}
	}
	chunk := func(delta map[string]any, fin any) map[string]any {
		return map[string]any{"id": id, "object": "chat.completion.chunk", "model": req.Model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": fin}}}
	}
	send(chunk(map[string]any{"role": "assistant"}, nil))
	for i := 0; i < len(msg.Content); i += 8 {
		j := min(i+8, len(msg.Content))
		s.sleep(tokMs * 2)
		send(chunk(map[string]any{"content": msg.Content[i:j]}, nil))
	}
	for i, tc := range msg.ToolCalls {
		s.sleep(float64(toks(len(tc.Function.Arguments)+len(tc.Function.Name))) * tokMs)
		args := tc.Function.Arguments
		half := len(args) / 2
		send(chunk(map[string]any{"tool_calls": []any{map[string]any{"index": i, "id": tc.ID, "type": "function",
			"function": map[string]any{"name": tc.Function.Name, "arguments": args[:half]}}}}, nil))
		send(chunk(map[string]any{"tool_calls": []any{map[string]any{"index": i, "function": map[string]any{"arguments": args[half:]}}}}, nil))
	}
	send(chunk(map[string]any{}, finish))
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		send(map[string]any{"id": id, "object": "chat.completion.chunk", "model": req.Model, "choices": []any{}, "usage": usage})
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	if fl != nil {
		fl.Flush()
	}
}

func turnsOf(s *session) int {
	if s == nil {
		return 0
	}
	return s.turns
}
