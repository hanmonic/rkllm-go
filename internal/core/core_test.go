package core

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"rkllm-go/internal/config"
	"rkllm-go/internal/engine/fake"
	"rkllm-go/internal/prompt"
)

// ---------- queue ----------

func newJob(ctx context.Context) *Job { return &Job{Ctx: ctx, Stream: NewStream()} }

func TestQueueBoundsAndOrder(t *testing.T) {
	q := NewQueue(2)
	a, b, c := newJob(context.Background()), newJob(context.Background()), newJob(context.Background())
	if q.Push(a) != nil || q.Push(b) != nil {
		t.Fatal("push failed")
	}
	if err := q.Push(c); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("want ErrQueueFull, got %v", err)
	}
	if q.Pop() != a {
		t.Fatal("not FIFO")
	}
	if q.Active() != 1 || q.Len() != 1 || q.Idle() {
		t.Fatal("bad accounting")
	}
	// one slot free again while a runs
	if err := q.Push(c); err != nil {
		t.Fatal(err)
	}
	q.Done()
	if q.Pop() != b || q.Pop() != c {
		t.Fatal("order")
	}
	q.Done()
	q.Done()
	if !q.Idle() {
		t.Fatal("should be idle")
	}
}

func TestQueueCancelledJobsFreeSlots(t *testing.T) {
	q := NewQueue(1)
	ctx, cancel := context.WithCancel(context.Background())
	a := newJob(ctx)
	if err := q.Push(a); err != nil {
		t.Fatal(err)
	}
	if err := q.Push(newJob(context.Background())); !errors.Is(err, ErrQueueFull) {
		t.Fatal("expected full")
	}
	cancel()
	b := newJob(context.Background())
	if err := q.Push(b); err != nil {
		t.Fatalf("cancelled job should not hold the slot: %v", err)
	}
	if q.Pop() != b {
		t.Fatal("cancelled job popped")
	}
}

func TestQueueRemoveAndClose(t *testing.T) {
	q := NewQueue(3)
	a, b := newJob(context.Background()), newJob(context.Background())
	_ = q.Push(a)
	_ = q.Push(b)
	if !q.Remove(a) || q.Remove(a) {
		t.Fatal("remove")
	}
	done := make(chan *Job)
	q2 := NewQueue(1)
	go func() { done <- q2.Pop() }()
	time.Sleep(10 * time.Millisecond)
	q2.Close()
	if j := <-done; j != nil {
		t.Fatal("pop after close should be nil")
	}
	if rest := q.Close(); len(rest) != 1 || rest[0] != b {
		t.Fatal("close should return waiting jobs")
	}
	if err := q.Push(a); !errors.Is(err, ErrQueueClosed) {
		t.Fatal("push after close")
	}
}

// ---------- LRU file store ----------

func TestFileStoreLRU(t *testing.T) {
	s := NewFileStore(100)
	var removed []string
	s.remove = func(p string) error { removed = append(removed, p); return nil }
	now := time.Unix(0, 0)
	s.now = func() time.Time { now = now.Add(time.Second); return now }
	s.AddSized("a", "session", 40)
	s.AddSized("b", "session", 40)
	s.Touch("a") // b is now LRU
	ok, ev := s.AddSized("c", "session", 40)
	if !ok || len(ev) != 1 || ev[0] != "b" {
		t.Fatalf("expected b evicted, got %v", ev)
	}
	if !s.Has("a") || s.Has("b") || !s.Has("c") || s.Total() != 80 {
		t.Fatal("bad state")
	}
	// re-adding a grown file re-accounts it
	_, ev = s.AddSized("c", "session", 70)
	if len(ev) != 1 || ev[0] != "a" || s.Total() != 70 {
		t.Fatalf("regrow: %v total %d", ev, s.Total())
	}
	// a file larger than the budget is rejected
	if ok, _ := s.AddSized("huge", "session", 101); ok || s.Has("huge") {
		t.Fatal("oversized file accepted")
	}
	s.Delete("c")
	if s.Total() != 0 {
		t.Fatal("delete accounting")
	}
}

// ---------- sessions ----------

func TestSessionResolve(t *testing.T) {
	ss := NewSessionStore(10)
	sys := strings.Repeat("S", 1000)
	a, created := ss.Resolve("m", "", sys+"A1")
	if !created {
		t.Fatal("expected new")
	}
	ss.Update(a, sys+"A1")
	b, _ := ss.Resolve("m", "", sys+"B1")
	ss.Update(b, sys+"B1")
	if a == b {
		t.Fatal("sessions sharing only a system prompt must not merge")
	}
	// continuation of a matches a by prefix
	if x, created := ss.Resolve("m", "", sys+"A1 reply A2"); x != a || created {
		t.Fatal("prefix match failed")
	}
	// explicit id wins
	if x, _ := ss.Resolve("m", "conv-7", sys+"A1 reply A2"); x == a || x.ID != "conv-7" {
		t.Fatal("explicit id ignored")
	}
	if x, created := ss.Resolve("m", "conv-7", "whatever"); x.ID != "conv-7" || created {
		t.Fatal("explicit id lookup")
	}
	// an edited history is a new conversation for the fallback matcher
	if x := ss.Match(sys + "A2"); x != nil {
		t.Fatal("partial match accepted")
	}
}

func TestSessionStoreBounded(t *testing.T) {
	ss := NewSessionStore(2)
	ss.Resolve("m", "x", "")
	ss.Resolve("m", "y", "")
	ss.Resolve("m", "z", "")
	if len(ss.List()) != 2 {
		t.Fatal("store not bounded")
	}
}

func TestFileSafe(t *testing.T) {
	if fileSafe("abc-1_2.x") != "abc-1_2.x" {
		t.Fatal("safe id changed")
	}
	for _, bad := range []string{"../etc", "a/b", "", ".hidden", strings.Repeat("x", 65)} {
		if f := fileSafe(bad); strings.ContainsAny(f, "/") || strings.HasPrefix(f, ".") || f == "" {
			t.Fatalf("unsafe %q -> %q", bad, f)
		}
	}
}

// ---------- delta computation ----------

func TestAppendDelta(t *testing.T) {
	cached := CachedText("", "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", []string{"Hel", "lo", ""})
	if cached != "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nHello" {
		t.Fatalf("cached %q", cached)
	}
	next := cached + "<|im_end|>\n<|im_start|>user\nmore<|im_end|>\n<|im_start|>assistant\n"
	d, ok := AppendDelta(cached, next)
	if !ok || !strings.HasPrefix(d, "<|im_end|>\n") {
		t.Fatalf("delta %q %v", d, ok)
	}
	// max_tokens cut: the last generated token is not cached
	cut := CachedText("P", "", []string{"a", "b", "c"})
	if cut != "Pab" {
		t.Fatalf("cut %q", cut)
	}
	if d, ok := AppendDelta(cut, "Pabc<|im_end|>\nX"); !ok || d != "c<|im_end|>\nX" {
		t.Fatalf("cut delta %q", d)
	}
	if _, ok := AppendDelta(cached, "something else"); ok {
		t.Fatal("diverged prompt accepted")
	}
	if _, ok := AppendDelta(cached, cached); ok {
		t.Fatal("empty delta accepted")
	}
}

func TestRawMap(t *testing.T) {
	r := NewRawMap(2)
	k1 := RawKey("hi", nil)
	k2 := RawKey("", []RawCall{{"get", `{"a": 1}`}})
	r.Put(k1, "raw1")
	r.Put(k2, "raw2")
	r.Get(k1)
	r.Put("k3", "raw3") // evicts k2
	if _, ok := r.Get(k2); ok {
		t.Fatal("LRU")
	}
	if v, ok := r.Get(RawKey(" hi ", nil)); !ok || v != "raw1" {
		t.Fatal("trimmed key lookup")
	}
}

// ---------- end to end with the fake engine ----------

// conv renders a toy chat with qwen3 so the real templates are exercised.
type conv struct {
	tpl   prompt.Template
	msgs  []prompt.Message
	tools []byte
}

func (c *conv) prompt() string {
	return c.tpl.Format(c.msgs, nil, prompt.Options{})
}

func memoryReply(ctx string) string {
	// answers with every "fact:" word seen in the context, so recall is checkable
	var facts []string
	for _, f := range strings.Split(ctx, "fact:")[1:] {
		facts = append(facts, strings.Fields(f)[0])
	}
	return "I know " + strings.Join(facts, ",")
}

func testPool(t *testing.T, strat config.Strategy, noLCP bool, perRun bool) (*Pool, *fake.Engine) {
	t.Helper()
	fam := prompt.FamilyQwen3
	if noLCP {
		fam = prompt.FamilyQwen35
	}
	tpl, _ := prompt.Get(fam)
	cfg := &config.Config{Default: "m", Models: map[string]*config.Model{
		"m": {Name: "m", Path: "/nonexistent", Family: fam, Template: tpl, MaxContextLen: 1 << 20, Strategy: strat, MemMB: 100},
	}}
	eng := &fake.Engine{NoLCP: noLCP, Reply: memoryReply, PerRunMT: perRun}
	p := NewPool(cfg, eng, Options{QueueDepth: 4, SessionDir: filepath.Join(t.TempDir(), "s"), SessionBudget: 1 << 30, MemBudgetMB: 1000, PrefixCache: true, MinPrefixBytes: 64, MaxNewTokens: 512})
	if err := p.Preload("m"); err != nil {
		t.Fatal(err)
	}
	return p, eng
}

func run(t *testing.T, p *Pool, sess string, c *conv, static string, noCache bool) Result {
	t.Helper()
	j := &Job{Ctx: context.Background(), Model: "m", Prompt: c.prompt(), StaticPrefix: static, SessionID: sess, NoCache: noCache, Start: time.Now(), Stream: NewStream()}
	if err := p.Submit(j); err != nil {
		t.Fatal(err)
	}
	for {
		_, _, _, done, res := j.Stream.Snapshot(0)
		if done {
			if res.Err != nil {
				t.Fatal(res.Err)
			}
			c.msgs = append(c.msgs, prompt.Message{Role: "assistant", Content: res.Text, Raw: res.Text})
			return res
		}
		<-j.Stream.Notify()
	}
}

func newConv(fam prompt.Family, sys string) *conv {
	tpl, _ := prompt.Get(fam)
	return &conv{tpl: tpl, msgs: []prompt.Message{{Role: "system", Content: sys}}}
}

func (c *conv) user(s string) *conv {
	c.msgs = append(c.msgs, prompt.Message{Role: "user", Content: s})
	return c
}

func TestPrefixStrategyReuseAndParkRestore(t *testing.T) {
	for _, perRun := range []bool{true, false} { // 1.3.1 and 1.2.3 save paths
		t.Run(fmt.Sprint("perRunMaxTokens=", perRun), func(t *testing.T) {
			p, eng := testPool(t, config.StrategyPrefix, false, perRun)
			sys := strings.Repeat("You are a test bot. ", 20)
			a := newConv(prompt.FamilyQwen3, sys)
			prev := a.user("fact:apple").prompt()
			r := run(t, p, "A", a, "", false)
			if r.Usage.CacheEvent != EventCold || r.Usage.CachedTokens != 0 {
				t.Fatalf("turn1 %+v", r.Usage)
			}
			r = run(t, p, "A", a.user("fact:banana"), "", false)
			if r.Usage.CacheEvent != EventHit || r.Usage.CachedTokens != len(prev) {
				t.Fatalf("turn2 %+v", r.Usage)
			}
			if !strings.Contains(r.Text, "apple") || !strings.Contains(r.Text, "banana") {
				t.Fatalf("recall %q", r.Text)
			}
			// session B interleaves
			b := newConv(prompt.FamilyQwen3, sys)
			r = run(t, p, "B", b.user("fact:cherry"), "", false)
			if r.Usage.CacheEvent != EventCold || r.Usage.ParkMs == 0 && perRun {
				t.Fatalf("B1 %+v", r.Usage)
			}
			if perRun && r.Usage.CachedTokens < len(sys) {
				t.Fatalf("B1 should reuse the shared system prompt via the prefix record: %+v", r.Usage)
			}
			// back to A: restored from the parked file
			aPrompt := a.user("fact:date").prompt()
			r = run(t, p, "A", a, "", false)
			if r.Usage.CacheEvent != EventRestored {
				t.Fatalf("A3 %+v", r.Usage)
			}
			if r.Usage.CachedTokens < len(aPrompt)-200 {
				t.Fatalf("A3 restored too little: cached %d of %d", r.Usage.CachedTokens, len(aPrompt))
			}
			if !strings.Contains(r.Text, "banana") || strings.Contains(r.Text, "cherry") {
				t.Fatalf("A3 recall %q", r.Text)
			}
			r = run(t, p, "B", b.user("fact:elder"), "", false)
			if r.Usage.CacheEvent != EventRestored || !strings.Contains(r.Text, "cherry") || strings.Contains(r.Text, "apple") {
				t.Fatalf("B2 %+v %q", r.Usage, r.Text)
			}
			if eng.Handles[0].Corrupted {
				t.Fatal("keep_history modes mixed")
			}
			// cache off: cold, session marked lost, parked file kept
			r = run(t, p, "B", b.user("fact:fig"), "", true)
			if r.Usage.CachedTokens != 0 || r.Usage.CacheStrategy != "off" || r.Usage.CacheEvent != EventCold {
				t.Fatalf("off %+v", r.Usage)
			}
			r = run(t, p, "B", b.user("fact:grape"), "", false)
			if r.Usage.CacheEvent != EventMiss && r.Usage.CacheEvent != EventRestored {
				t.Fatalf("after off %+v", r.Usage)
			}
		})
	}
}

func TestAppendStrategy(t *testing.T) {
	p, eng := testPool(t, config.StrategyAppend, true, true)
	a := newConv(prompt.FamilyQwen35, "sys")
	run(t, p, "A", a.user("fact:apple"), "", false)
	for i, f := range []string{"banana", "cherry", "date"} {
		full := a.user("fact:" + f).prompt()
		r := run(t, p, "A", a, "", false)
		if r.Usage.CacheEvent != EventHit {
			t.Fatalf("turn %d %+v", i+2, r.Usage)
		}
		h := eng.Handles[0]
		if h.LastContext != full {
			t.Fatalf("context mismatch:\n%q\n%q", h.LastContext, full)
		}
		if !strings.HasPrefix(full[len(full)-h.LastPrefill:], "<|im_end|>\n") {
			t.Fatalf("delta does not start with the stop token: %q", full[len(full)-h.LastPrefill:])
		}
		if r.Usage.CachedTokens == 0 || r.Usage.PrefillTokens != h.LastPrefill {
			t.Fatalf("usage %+v", r.Usage)
		}
	}
	// diverging history (edited message): miss, full clear + full prompt
	a.msgs[1].Content = "fact:avocado"
	r := run(t, p, "A", a.user("fact:x"), "", false)
	if r.Usage.CacheEvent != EventMiss || r.Usage.CachedTokens != 0 {
		t.Fatalf("diverged %+v", r.Usage)
	}
	// another session takes the handle: cold; A comes back: cold re-prefill
	b := newConv(prompt.FamilyQwen35, "sys")
	if r := run(t, p, "B", b.user("fact:b"), "", false); r.Usage.CacheEvent != EventCold {
		t.Fatalf("B %+v", r.Usage)
	}
	r = run(t, p, "A", a.user("fact:y"), "", false)
	if r.Usage.CacheEvent != EventCold || !strings.Contains(r.Text, "avocado") {
		t.Fatalf("A back %+v %q", r.Usage, r.Text)
	}
	if eng.Handles[0].Corrupted {
		t.Fatal("modes mixed")
	}
}

func TestSnapshotStrategyInterleaved(t *testing.T) {
	p, eng := testPool(t, config.StrategySnapshot, true, true)
	a := newConv(prompt.FamilyQwen35, "sys")
	b := newConv(prompt.FamilyQwen35, "sys")
	run(t, p, "A", a.user("fact:a1"), "", false)
	run(t, p, "B", b.user("fact:b1"), "", false)
	for i := 0; i < 3; i++ {
		for _, x := range []struct {
			id string
			c  *conv
		}{{"A", a}, {"B", b}} {
			r := run(t, p, x.id, x.c.user(fmt.Sprintf("fact:%s%d", strings.ToLower(x.id), i+2)), "", false)
			if r.Usage.CacheEvent != EventRestored || r.Usage.CachedTokens == 0 {
				t.Fatalf("%s turn %d %+v", x.id, i+2, r.Usage)
			}
			other := map[string]string{"A": "b1", "B": "a1"}[x.id]
			if strings.Contains(r.Text, other) {
				t.Fatalf("cross-talk %q", r.Text)
			}
		}
	}
	if eng.Handles[0].Corrupted {
		t.Fatal("modes mixed")
	}
}

func TestSharedPrefixCache(t *testing.T) {
	p, eng := testPool(t, config.StrategySnapshot, true, true)
	sys := strings.Repeat("tools tools tools ", 20)
	static := func(c *conv) string {
		x := newConv(prompt.FamilyQwen35, sys)
		p1 := x.tpl.Format(append(x.msgs, prompt.Message{Role: "user", Content: "\x01"}), nil, prompt.Options{})
		p2 := x.tpl.Format(append(x.msgs, prompt.Message{Role: "user", Content: "\x02"}), nil, prompt.Options{})
		return p1[:CommonPrefixLen(p1, p2)]
	}
	a := newConv(prompt.FamilyQwen35, sys)
	r := run(t, p, "A", a.user("fact:a"), static(a), false)
	if r.Usage.PrefixCache != "created" {
		t.Fatalf("A %+v", r.Usage)
	}
	b := newConv(prompt.FamilyQwen35, sys)
	r = run(t, p, "B", b.user("fact:b"), static(b), false)
	if r.Usage.PrefixCache != "loaded" || r.Usage.CachedTokens < len(sys) {
		t.Fatalf("B %+v", r.Usage)
	}
	_ = eng
	files, _ := os.ReadDir(filepath.Join(p.opts.SessionDir, "m"))
	if len(files) < 3 {
		t.Fatalf("expected prefix + 2 session files, got %d", len(files))
	}
}

func TestQueueFullAndCancel(t *testing.T) {
	p, _ := testPool(t, config.StrategyPrefix, false, true)
	// hold the NPU so jobs queue up
	p.npu.Lock()
	c := newConv(prompt.FamilyQwen3, "s").user("hi")
	var jobs []*Job
	var err error
	ctx, cancel := context.WithCancel(context.Background())
	for i := 0; i < 6; i++ {
		j := &Job{Ctx: ctx, Model: "m", Prompt: c.prompt(), Start: time.Now(), Stream: NewStream()}
		if err = p.Submit(j); err != nil {
			break
		}
		jobs = append(jobs, j)
		time.Sleep(10 * time.Millisecond) // let the worker pick up the first job
	}
	// 1 popped by the worker (blocked on the NPU lock) + 4 waiting
	if !errors.Is(err, ErrQueueFull) || len(jobs) != 5 {
		t.Fatalf("expected queue full after 5, got %d (%v)", len(jobs), err)
	}
	cancel()
	p.npu.Unlock()
	j := &Job{Ctx: context.Background(), Model: "m", Prompt: c.prompt(), Start: time.Now(), Stream: NewStream()}
	if err := p.Submit(j); err != nil {
		t.Fatalf("cancelled jobs should free the queue: %v", err)
	}
	for {
		_, _, _, done, res := j.Stream.Snapshot(0)
		if done {
			if res.Err != nil {
				t.Fatal(res.Err)
			}
			break
		}
		<-j.Stream.Notify()
	}
}
