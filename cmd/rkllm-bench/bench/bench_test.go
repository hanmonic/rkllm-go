package bench

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"rkllm-go/cmd/rkllm-bench/fakeserver"
	"rkllm-go/cmd/rkllm-bench/scenarios"
)

type countingHandler struct {
	h     http.Handler
	chats atomic.Int64
}

func (c *countingHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/v1/chat/completions" {
		c.chats.Add(1)
	}
	c.h.ServeHTTP(w, r)
}

func startFake(t *testing.T, opt fakeserver.Options) (*httptest.Server, *countingHandler) {
	t.Helper()
	ch := &countingHandler{h: fakeserver.New(opt).Handler()}
	srv := httptest.NewServer(ch)
	t.Cleanup(srv.Close)
	return srv, ch
}

func runCfg(t *testing.T, cfg Config) {
	t.Helper()
	if cfg.Log == nil {
		cfg.Log = io.Discard
	}
	r, err := NewRunner(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestEndToEndFake(t *testing.T) {
	for _, stream := range []bool{true, false} {
		t.Run(map[bool]string{true: "stream", false: "nostream"}[stream], func(t *testing.T) {
			srv, ch := startFake(t, fakeserver.Options{})
			dir := t.TempDir()
			cfg := Config{Server: srv.URL, Models: []string{"qwen3-4b-16k", "qwen3-1.7b"}, Scenarios: scenarios.Names,
				Modes: []string{"on", "off"}, Out: dir, Stream: stream, Seed: 42, Warmup: true}
			runCfg(t, cfg)

			turns, tasks, err := LoadDir(dir, false)
			if err != nil {
				t.Fatal(err)
			}
			wantTasks := 2 * 2 * (4 + 3 + 4 + 3)
			if len(tasks) != wantTasks {
				t.Fatalf("tasks = %d want %d", len(tasks), wantTasks)
			}
			for _, tk := range tasks {
				if tk.Model == "qwen3-1.7b" && strings.HasSuffix(tk.Task, "two-bugs") {
					// 4096 window: the two-bug task cannot fit, it must end on the context budget
					if tk.StopReason != "ctx_exhausted" {
						t.Errorf("%s/%s/%s: want ctx_exhausted, got stop=%s", tk.Model, tk.Task, tk.Mode, tk.StopReason)
					}
					continue
				}
				if !tk.Pass || tk.TimeToSuccessMs == nil || tk.Malformed != 0 || tk.Errors != 0 {
					t.Errorf("%s/%s/%s/%s: pass=%v stop=%s detail=%v", tk.Model, tk.Scenario, tk.Task, tk.Mode, tk.Pass, tk.StopReason, tk.Detail)
				}
			}
			events := map[string]map[string]int{"on": {}, "off": {}}
			for _, tr := range turns {
				if tr.RK == nil {
					t.Fatal("missing usage.rkllm")
				}
				events[tr.Mode][tr.RK.CacheEvent]++
				if tr.RespSessionID != tr.SessionID {
					t.Errorf("session id not echoed: %s vs %s", tr.RespSessionID, tr.SessionID)
				}
				if stream && tr.ClientTTFTMs <= 0 {
					t.Error("no client TTFT while streaming")
				}
				if tr.Mode == "on" && tr.Turn > 0 && tr.Scenario != "interleaved" && tr.RK.CacheEvent != "hit" {
					t.Errorf("%s turn %d: event %s, history not echoed faithfully?", tr.Task, tr.Turn, tr.RK.CacheEvent)
				}
			}
			if len(events["off"]) != 1 || events["off"]["cold"] == 0 {
				t.Errorf("off-mode events: %v", events["off"])
			}
			if events["on"]["hit"] == 0 || events["on"]["restored"] == 0 {
				t.Errorf("on-mode events: %v", events["on"])
			}

			for _, f := range []string{"summary.json", "report.md", "all_turns.jsonl"} {
				if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
					t.Error(err)
				}
			}
			var sum Summary
			b, _ := os.ReadFile(filepath.Join(dir, "summary.json"))
			if err := json.Unmarshal(b, &sum); err != nil {
				t.Fatal(err)
			}
			if len(sum.Speedups) != 8 {
				t.Errorf("speedups = %d want 8", len(sum.Speedups))
			}
			for _, sp := range sum.Speedups {
				if sp.PrefillReduction <= 0.3 || sp.TTFTSpeedup <= 1 {
					t.Errorf("%s/%s: cache should help: %+v", sp.Model, sp.Scenario, sp)
				}
			}
			md, _ := os.ReadFile(filepath.Join(dir, "report.md"))
			for _, s := range []string{"## Cache on vs off", "## Time to useful result", "## qwen3-4b-16k", "## TTFT by cache event", "restored"} {
				if !strings.Contains(string(md), s) {
					t.Errorf("report.md missing %q", s)
				}
			}

			// resume: nothing left to do, no chat requests
			before := ch.chats.Load()
			runCfg(t, cfg)
			if ch.chats.Load() != before {
				t.Errorf("resume re-ran %d requests", ch.chats.Load()-before)
			}
		})
	}
}

func TestResumePartialCombo(t *testing.T) {
	srv, _ := startFake(t, fakeserver.Options{})
	dir := t.TempDir()
	cfg := Config{Server: srv.URL, Models: []string{"q35-2b"}, Scenarios: []string{"hard-multistep"}, Modes: []string{"on", "off"}, Out: dir, Stream: true, Smoke: true}
	runCfg(t, cfg)
	key := ComboKey("q35-2b", "hard-multistep", "off", true)
	// simulate a crash mid-combo: marker gone, torn output present
	os.Remove(donePath(dir, key))
	os.WriteFile(tasksPath(dir, key), []byte("{\"torn\":"), 0o644)
	runCfg(t, cfg)
	_, tasks, err := LoadDir(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(tasks) != 2 {
		t.Fatalf("tasks after resume = %d want 2", len(tasks))
	}
}

func TestMalformedCounted(t *testing.T) {
	srv, _ := startFake(t, fakeserver.Options{MalformedEvery: 1})
	dir := t.TempDir()
	runCfg(t, Config{Server: srv.URL, Models: []string{"qwen3-0.6b"}, Scenarios: []string{"house-batch"}, Modes: []string{"on"}, Out: dir, Stream: true, Smoke: true})
	_, tasks, _ := LoadDir(dir, false)
	if len(tasks) != 1 || tasks[0].Malformed == 0 || tasks[0].Pass {
		t.Fatalf("malformed not counted: %+v", tasks)
	}
}

func TestContextBudget(t *testing.T) {
	srv, _ := startFake(t, fakeserver.Options{})
	dir := t.TempDir()
	runCfg(t, Config{Server: srv.URL, Models: []string{"qwen3-0.6b"}, Scenarios: []string{"coding-agent"}, Modes: []string{"on"}, Out: dir,
		Stream: true, Smoke: true, CtxLimits: map[string]int{"qwen3-0.6b": 2600}})
	turns, tasks, _ := LoadDir(dir, false)
	if len(tasks) != 1 {
		t.Fatal(len(tasks))
	}
	trunc := false
	for _, tr := range turns {
		trunc = trunc || tr.Truncated
		if tr.PromptTokens+1024 > 2600+200 { // small slack: estimate vs fake tokenizer
			t.Errorf("turn %d prompt %d overflows the 2600 window", tr.Turn, tr.PromptTokens)
		}
	}
	if !trunc && tasks[0].StopReason != "ctx_exhausted" {
		t.Errorf("expected truncation or ctx_exhausted, got stop=%s", tasks[0].StopReason)
	}
}

func TestInterleavedParallel(t *testing.T) {
	srv, _ := startFake(t, fakeserver.Options{TimeScale: 0.001})
	dir := t.TempDir()
	runCfg(t, Config{Server: srv.URL, Models: []string{"q35-4b"}, Scenarios: []string{"interleaved"}, Modes: []string{"on"}, Out: dir,
		Stream: true, Smoke: true, Parallel: true, Timeout: time.Minute})
	_, tasks, _ := LoadDir(dir, false)
	if len(tasks) != 2 {
		t.Fatal(len(tasks))
	}
	for _, tk := range tasks {
		if !tk.Pass {
			t.Errorf("%s failed: %v", tk.Task, tk.Detail)
		}
	}
}

func TestStreamToolCallAssembly(t *testing.T) {
	// index-less deltas (id on first fragment only) and indexed parallel calls
	body := strings.Join([]string{
		`data: {"choices":[{"delta":{"role":"assistant"}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{\"x\""}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":":1}"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"id":"b","function":{"name":"g","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":7},"rkllm":{"cache_event":"hit","ttft_ms":12}}}`,
		`data: [DONE]`, ""}, "\n\n")
	res := &ChatResult{}
	if err := readStream(strings.NewReader(body), res, time.Now()); err != nil {
		t.Fatal(err)
	}
	tc := res.Message.ToolCalls
	if len(tc) != 2 || tc[0].Function.Arguments != `{"x":1}` || tc[1].Function.Name != "g" || tc[1].ID != "b" {
		t.Fatalf("calls: %+v", tc)
	}
	if res.FinishReason != "tool_calls" || res.Usage.CachedTokens() != 7 || res.Usage.RKLLM.CacheEvent != "hit" || res.TTFTMs <= 0 {
		t.Fatalf("result: %+v", res)
	}
}

func TestSummarizeSpeedupMath(t *testing.T) {
	f := func(v float64) *float64 { return &v }
	tasks := []TaskRecord{
		{Model: "m", Scenario: "s", Task: "a", Mode: "on", Pass: true, WallMs: 1000, TimeToSuccessMs: f(800), Turns: 2},
		{Model: "m", Scenario: "s", Task: "a", Mode: "off", Pass: true, WallMs: 3000, TimeToSuccessMs: f(2400), Turns: 2},
	}
	turns := []TurnRecord{
		{Model: "m", Scenario: "s", Mode: "on", PromptTokens: 100, CachedTokens: 80, RK: &RKUsage{TTFTMs: 100, PrefillTokens: 20, CacheEvent: "hit"}},
		{Model: "m", Scenario: "s", Mode: "off", PromptTokens: 100, RK: &RKUsage{TTFTMs: 400, PrefillTokens: 100, CacheEvent: "cold"}},
	}
	s := Summarize("x", turns, tasks)
	if len(s.Speedups) != 1 {
		t.Fatal(s.Speedups)
	}
	sp := s.Speedups[0]
	if sp.WallSpeedup != 3 || sp.TTFTSpeedup != 4 || sp.TTSSpeedup != 3 || sp.PrefillReduction != 0.8 || sp.PairedTasks != 1 {
		t.Errorf("%+v", sp)
	}
	if s.Combos[0].Mode != "on" || s.Combos[0].HitRatio != 0.8 {
		t.Errorf("combo order or hit ratio: %+v", s.Combos[0])
	}
}

func TestCtxHeuristic(t *testing.T) {
	for m, want := range map[string]int{"qwen3-4b-16k": 16384, "q35-2b": 16384, "qwen3-8b-allnpu": 16384, "qwen3-0.6b": 4096, "gemma3n-e4b": 4096} {
		if got := CtxHeuristic(m); got != want {
			t.Errorf("%s: %d want %d", m, got, want)
		}
	}
}
