package bench

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"rkllm-go/cmd/rkllm-bench/scenarios"
)

// Config controls a benchmark run.
type Config struct {
	Server     string
	Models     []string
	Scenarios  []string
	Modes      []string // "on", "off"
	Out        string
	MaxTurns   int           // 0 = per-scenario default
	Timeout    time.Duration // per request
	Smoke      bool
	Stream     bool
	Seed       int
	CtxLimits  map[string]int // per-model context window override
	DefaultCtx int            // 0 = heuristic from model name
	Warmup     bool
	Parallel   bool // run interleaved sessions concurrently instead of round-robin
	Log        io.Writer
}

// DefaultMaxTurns per scenario.
var DefaultMaxTurns = map[string]int{"house-batch": 12, "coding-agent": 20, "hard-multistep": 6, "interleaved": 20}

// CtxHeuristic guesses the context window from the model name.
func CtxHeuristic(model string) int {
	m := strings.ToLower(model)
	if strings.Contains(m, "16k") || strings.HasPrefix(m, "q35") || strings.Contains(m, "qwen3.5") || strings.Contains(m, "8b-allnpu") {
		return 16384
	}
	return 4096
}

// Runner executes combos.
type Runner struct {
	cfg    Config
	client *Client
	runID  string
	models map[string]ModelInfo
}

// NewRunner validates the config.
func NewRunner(cfg Config) (*Runner, error) {
	if cfg.Log == nil {
		cfg.Log = os.Stderr
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 30 * time.Minute
	}
	for _, m := range cfg.Modes {
		if m != "on" && m != "off" {
			return nil, fmt.Errorf("mode %q: want on or off", m)
		}
	}
	for _, s := range cfg.Scenarios {
		if _, err := scenarios.Specs(s, true); err != nil {
			return nil, err
		}
	}
	b := make([]byte, 3)
	_, _ = rand.Read(b)
	return &Runner{cfg: cfg, client: NewClient(cfg.Server), runID: time.Now().UTC().Format("0102T1504") + "-" + hex.EncodeToString(b)}, nil
}

func (r *Runner) logf(format string, a ...any) {
	fmt.Fprintf(r.cfg.Log, "%s "+format+"\n", append([]any{time.Now().Format("15:04:05")}, a...)...)
}

func (r *Runner) ctxLimit(model string) int {
	if v, ok := r.cfg.CtxLimits[model]; ok && v > 0 {
		return v
	}
	if mi, ok := r.models[model]; ok && mi.Context > 0 {
		return mi.Context
	}
	if r.cfg.DefaultCtx > 0 {
		return r.cfg.DefaultCtx
	}
	return CtxHeuristic(model)
}

// Run executes every pending combo, then writes the report.
func (r *Runner) Run(ctx context.Context) error {
	cfg := r.cfg
	if err := os.MkdirAll(cfg.Out, 0o755); err != nil {
		return err
	}
	hctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	err := r.client.GetJSON(hctx, "/healthz", nil)
	cancel()
	if err != nil {
		return fmt.Errorf("server not healthy at %s: %w", cfg.Server, err)
	}
	r.models = map[string]ModelInfo{}
	mctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	ms, err := r.client.Models(mctx)
	cancel()
	if err != nil {
		r.logf("warning: /v1/models failed: %v", err)
	}
	for _, m := range ms {
		r.models[m.ID] = m
	}
	_ = writeJSONAtomic(cfg.Out+"/run-"+r.runID+".json", map[string]any{"run_id": r.runID, "started": nowStr(), "config": cfg, "server_models": ms})
	r.logf("run %s -> %s", r.runID, cfg.Out)

	for _, model := range cfg.Models {
		if len(r.models) > 0 {
			if _, ok := r.models[model]; !ok {
				r.logf("skip model %s: not listed by /v1/models", model)
				continue
			}
		}
		var pending [][2]string
		for _, sc := range cfg.Scenarios {
			for _, mode := range cfg.Modes {
				if IsDone(cfg.Out, ComboKey(model, sc, mode, cfg.Smoke)) {
					r.logf("skip %s / %s / cache-%s: already done", model, sc, mode)
					continue
				}
				pending = append(pending, [2]string{sc, mode})
			}
		}
		if len(pending) == 0 {
			continue
		}
		if cfg.Warmup {
			r.warmup(ctx, model)
		}
		for _, p := range pending {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if err := r.runCombo(ctx, model, p[0], p[1]); err != nil {
				r.logf("combo %s/%s/%s: %v", model, p[0], p[1], err)
			}
		}
	}
	return WriteReport(cfg.Out)
}

func (r *Runner) warmup(ctx context.Context, model string) {
	off := false
	rctx, cancel := context.WithTimeout(ctx, r.cfg.Timeout)
	defer cancel()
	res, err := r.client.Chat(rctx, ChatRequest{Model: model, Messages: []scenarios.Message{{Role: "user", Content: "Reply with OK."}},
		Temperature: 0, MaxTokens: 8, Stream: r.cfg.Stream, Cache: &off}, "warmup-"+r.runID)
	if err != nil {
		r.logf("warmup %s: %v", model, err)
		return
	}
	load := 0.0
	if res.Usage.RKLLM != nil {
		load = res.Usage.RKLLM.ModelLoadMs
	}
	r.logf("warmup %s: %.1fs (model_load %.1fs)", model, res.WallMs/1000, load/1000)
}

// combo holds the writers of one (model, scenario, mode).
type combo struct {
	model, scenario, mode string
	turns, tasks          *jsonl
}

func (r *Runner) runCombo(ctx context.Context, model, sc, mode string) error {
	key := ComboKey(model, sc, mode, r.cfg.Smoke)
	specs, err := scenarios.Specs(sc, r.cfg.Smoke)
	if err != nil {
		return err
	}
	tw, err := createJSONL(turnsPath(r.cfg.Out, key))
	if err != nil {
		return err
	}
	defer tw.Close()
	kw, err := createJSONL(tasksPath(r.cfg.Out, key))
	if err != nil {
		return err
	}
	defer kw.Close()
	cb := &combo{model: model, scenario: sc, mode: mode, turns: tw, tasks: kw}
	r.logf("== %s / %s / cache-%s (%d tasks, ctx %d)", model, sc, mode, len(specs), r.ctxLimit(model))

	var runs []*taskRun
	for _, s := range specs {
		runs = append(runs, r.newTaskRun(cb, s))
	}
	switch {
	case scenarios.Interleaved(sc) && r.cfg.Parallel:
		var wg sync.WaitGroup
		for _, tr := range runs {
			wg.Add(1)
			go func(tr *taskRun) {
				defer wg.Done()
				for !tr.done && ctx.Err() == nil {
					r.step(ctx, tr)
				}
			}(tr)
		}
		wg.Wait()
	case scenarios.Interleaved(sc):
		for active := true; active && ctx.Err() == nil; {
			active = false
			for _, tr := range runs {
				if !tr.done {
					r.step(ctx, tr)
					active = true
				}
			}
		}
	default:
		for _, tr := range runs {
			for !tr.done && ctx.Err() == nil {
				r.step(ctx, tr)
			}
		}
	}
	if ctx.Err() != nil {
		return ctx.Err() // no done marker: the combo reruns on resume
	}
	for _, tr := range runs {
		if err := kw.Write(tr.finish()); err != nil {
			return err
		}
		r.logf("   %-28s %s score=%.2f turns=%d wall=%.1fs stop=%s", tr.task.ID(), passStr(tr.rec.Pass), tr.rec.Score, tr.rec.Turns, tr.rec.WallMs/1000, tr.rec.StopReason)
	}
	sctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	var state any
	if err := r.client.GetJSON(sctx, "/debug/state", &state); err == nil {
		_ = writeJSONAtomic(statePath(r.cfg.Out, key), state)
	}
	cancel()
	return writeJSONAtomic(donePath(r.cfg.Out, key), DoneMarker{Model: model, Scenario: sc, Mode: mode, Smoke: r.cfg.Smoke,
		RunID: r.runID, Tasks: len(runs), Finished: nowStr()})
}

func passStr(b bool) string {
	if b {
		return "PASS"
	}
	return "FAIL"
}

// taskRun is the in-flight state of one task (one session).
type taskRun struct {
	cb                         *combo
	task                       scenarios.Task
	names                      map[string]bool
	msgs                       []scenarios.Message
	session                    string
	turn                       int
	maxTurns                   int
	ctxLimit                   int
	done                       bool
	started                    time.Time
	lastEnd                    time.Time
	lastPrompt, lastCompletion int
	sentMsgs                   int // len(msgs) at the last request
	rec                        TaskRecord
}

func (r *Runner) newTaskRun(cb *combo, s scenarios.Spec) *taskRun {
	t := s.New()
	mt := r.cfg.MaxTurns
	if mt <= 0 {
		mt = DefaultMaxTurns[cb.scenario]
	}
	sess := safe(fmt.Sprintf("%s-%s-%s-%s-%s", r.runID, cb.model, cb.scenario, s.ID, cb.mode))
	names := map[string]bool{}
	for _, tl := range t.Tools() {
		names[tl.(map[string]any)["function"].(map[string]any)["name"].(string)] = true
	}
	tr := &taskRun{cb: cb, task: t, names: names, msgs: t.Initial(), session: sess, maxTurns: mt, ctxLimit: r.ctxLimit(cb.model)}
	tr.rec = TaskRecord{RunID: r.runID, Model: cb.model, Scenario: cb.scenario, Task: s.ID, Mode: cb.mode, Smoke: r.cfg.Smoke,
		SessionID: sess, SolvedTurn: -1, CtxLimit: tr.ctxLimit}
	return tr
}

const charsPerToken = 3.0

func msgChars(ms []scenarios.Message) int {
	n := 0
	for _, m := range ms {
		n += len(m.Content) + 16
		for _, tc := range m.ToolCalls {
			n += len(tc.Function.Name) + len(tc.Function.Arguments) + 24
		}
	}
	return n
}

// estTokens estimates the prompt size of the next request: the last reported
// prompt + completion plus whatever was appended since, or a char estimate.
func (tr *taskRun) estTokens() int {
	if tr.lastPrompt > 0 {
		return tr.lastPrompt + tr.lastCompletion + int(float64(msgChars(tr.msgs[tr.sentMsgs:]))/charsPerToken)
	}
	b, _ := json.Marshal(tr.task.Tools())
	return int(float64(msgChars(tr.msgs)+len(b)) / charsPerToken)
}

var leakRe = regexp.MustCompile(`<tool_call>|</tool_call>|<function=|<\|tool_call|"arguments"\s*:|\bcall:[A-Za-z_]+\{`)

// step performs one agent turn.
func (r *Runner) step(ctx context.Context, tr *taskRun) {
	if tr.turn >= tr.maxTurns {
		tr.stop("max_turns")
		return
	}
	maxTok := tr.task.MaxTokens()
	est := tr.estTokens()
	if est+maxTok > tr.ctxLimit {
		tr.stop("ctx_exhausted")
		return
	}
	req := ChatRequest{Model: tr.cb.model, Messages: tr.msgs, Tools: tr.task.Tools(), Temperature: 0, MaxTokens: maxTok, Stream: r.cfg.Stream}
	if r.cfg.Seed >= 0 {
		s := r.cfg.Seed
		req.Seed = &s
	}
	if tr.cb.mode == "off" {
		f := false
		req.Cache = &f
	}
	rec := TurnRecord{RunID: tr.rec.RunID, Model: tr.cb.model, Scenario: tr.cb.scenario, Task: tr.rec.Task, Mode: tr.cb.mode,
		Smoke: r.cfg.Smoke, SessionID: tr.session, Turn: tr.turn, EstContext: est}
	start := time.Now()
	if tr.started.IsZero() {
		tr.started = start
	}
	rec.Start = start.UTC().Format(time.RFC3339Nano)
	rctx, cancel := context.WithTimeout(ctx, r.cfg.Timeout)
	res, err := r.client.Chat(rctx, req, tr.session)
	cancel()
	tr.lastEnd = time.Now()
	tr.sentMsgs = len(tr.msgs)
	tr.turn++
	tr.rec.Turns = tr.turn
	if res != nil {
		rec.WallMs, rec.ClientTTFBMs, rec.ClientTTFTMs, rec.RespSessionID = res.WallMs, res.TTFBMs, res.TTFTMs, res.SessionID
	}
	if err != nil {
		rec.Error = err.Error()
		if rec.WallMs == 0 {
			rec.WallMs = ms(time.Since(start))
		}
		tr.rec.WallMs += rec.WallMs
		tr.rec.Errors++
		tr.rec.Error = rec.Error
		_ = tr.cb.turns.Write(rec)
		r.logf("   %s turn %d error: %v", tr.rec.Task, rec.Turn, err)
		tr.stop("error")
		return
	}
	u := res.Usage
	rec.PromptTokens, rec.CompletionTokens, rec.CachedTokens = u.PromptTokens, u.CompletionTokens, u.CachedTokens()
	if u.PromptTokens > 0 {
		rec.HitRatio = float64(rec.CachedTokens) / float64(u.PromptTokens)
	}
	rec.RK = u.RKLLM
	if u.RKLLM != nil && u.RKLLM.GenerateMs > 0 {
		rec.DecodeTPS = float64(u.RKLLM.GenerateTokens) / (u.RKLLM.GenerateMs / 1000)
	}
	rec.FinishReason = res.FinishReason
	tr.lastPrompt, tr.lastCompletion = u.PromptTokens, u.CompletionTokens
	if u.PromptTokens > tr.rec.MaxContext {
		tr.rec.MaxContext = u.PromptTokens
	}
	tr.rec.WallMs += res.WallMs
	tr.rec.PromptTokens += u.PromptTokens
	tr.rec.CachedTokens += rec.CachedTokens
	tr.rec.CompletionTokens += u.CompletionTokens
	if u.RKLLM != nil {
		tr.rec.PrefillTokens += u.RKLLM.PrefillTokens
	} else {
		tr.rec.PrefillTokens += u.PromptTokens - rec.CachedTokens
	}

	// assistant message goes back into history exactly as returned
	msg := res.Message
	msg.Role = "assistant"
	var calls []scenarios.Call
	for i := range msg.ToolCalls {
		tc := &msg.ToolCalls[i]
		if tc.ID == "" {
			tc.ID = fmt.Sprintf("call_%d_%d", tr.turn, i)
		}
		if tc.Type == "" {
			tc.Type = "function"
		}
		cl := scenarios.ParseCall(*tc)
		calls = append(calls, cl)
		rec.ToolCalls = append(rec.ToolCalls, cl.Name)
		if cl.Bad || cl.Name == "" {
			rec.Malformed++
		}
		if cl.Name != "" && !tr.names[cl.Name] {
			rec.UnknownTools++
		}
	}
	if len(calls) == 0 && leakRe.MatchString(msg.Content) {
		rec.Malformed++
	}
	rec.ContentChars = len(msg.Content)
	tr.rec.Malformed += rec.Malformed
	tr.rec.UnknownTools += rec.UnknownTools
	tr.msgs = append(tr.msgs, msg)

	stop := tr.task.OnAssistant(tr.turn-1, msg.Content, calls)
	if !stop && len(calls) > 0 {
		var results []scenarios.Message
		for _, cl := range calls {
			out := tr.task.Exec(cl)
			results = append(results, scenarios.Message{Role: "tool", ToolCallID: cl.ID, Name: cl.Name, Content: out})
		}
		rec.Truncated = tr.fitResults(results, maxTok)
		for _, m := range results {
			rec.ToolResultChars += len(m.Content)
		}
		tr.msgs = append(tr.msgs, results...)
	}
	rec.Solved = tr.task.Solved()
	if rec.Solved && tr.rec.SolvedTurn < 0 {
		tr.rec.SolvedTurn = rec.Turn
		v := tr.rec.WallMs
		tr.rec.TimeToSuccessMs = &v
	}
	_ = tr.cb.turns.Write(rec)
	switch {
	case stop:
		tr.stop("task_stop")
	case len(calls) == 0:
		tr.stop("done")
	}
}

// fitResults truncates tool outputs so the next prompt plus max_tokens fits
// the context window. Returns true when anything was cut.
func (tr *taskRun) fitResults(results []scenarios.Message, maxTok int) bool {
	base := tr.estTokens() // history incl. the assistant turn, excluding results
	budget := tr.ctxLimit - maxTok - base - 32*len(results)
	allowed := int(float64(budget) * charsPerToken)
	total := 0
	for _, m := range results {
		total += len(m.Content)
	}
	if total <= allowed {
		return false
	}
	if allowed < 0 {
		allowed = 0
	}
	per := allowed / len(results)
	for i := range results {
		c := results[i].Content
		if len(c) > per {
			results[i].Content = c[:per] + fmt.Sprintf("\n[output truncated: %d more characters, context window is full]", len(c)-per)
		}
	}
	return true
}

func (tr *taskRun) stop(reason string) {
	if tr.done {
		return
	}
	tr.done = true
	tr.rec.StopReason = reason
}

func (tr *taskRun) finish() TaskRecord {
	sc := tr.task.Score()
	tr.rec.Pass, tr.rec.Score, tr.rec.Detail = sc.Pass, sc.Score, sc.Detail
	if !tr.started.IsZero() {
		tr.rec.ElapsedMs = ms(tr.lastEnd.Sub(tr.started))
	}
	if !sc.Pass {
		tr.rec.TimeToSuccessMs = nil
	} else if tr.rec.TimeToSuccessMs == nil {
		v := tr.rec.WallMs // passed without the task's own "solved" signal (e.g. fixed but never re-ran tests)
		tr.rec.TimeToSuccessMs = &v
	}
	return tr.rec
}
