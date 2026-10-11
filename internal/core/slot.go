package core

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"rkllm-go/internal/config"
	"rkllm-go/internal/engine"
	"rkllm-go/internal/prompt"
)

// Cache events reported in usage.rkllm.cache_event.
const (
	EventHit      = "hit"      // live session, the cache held everything before the new input
	EventPartial  = "partial"  // live session, history diverged: reused up to the first difference
	EventMiss     = "miss"     // live session, nothing reusable (state lost or diverged under append)
	EventRestored = "restored" // session restored from a parked / snapshot file
	EventCold     = "cold"     // new session (or no file to restore), or cache off
)

// handleState is what the server knows about the KV cache of a handle.
type handleState struct {
	mode   int  // -1 clean (after clear), 0 keep_history=0 used, 1 keep_history=1 used
	dirty  bool // unknown state (aborted / failed / stopped run): full clear before next use
	live   *Session
	lost   *Session // session whose live state was discarded by a cache-off request
	lcp    string   // prompt of the last keep_history=0 run = the runtime's prefix-match record
	cached string   // keep_history=1: the text in the KV cache
	kv     int
}

// Slot is one loaded model handle with its queue and worker.
type Slot struct {
	pool  *Pool
	m     *config.Model
	q     *Queue
	h     engine.Handle
	noLCP bool // the runtime does no keep_history=0 prefix reuse (Qwen3.5)

	ready     chan struct{}
	loadErr   error
	loadMs    float64
	memMB     int
	domains   []int
	preloaded bool

	mu       sync.Mutex // guards st (for debug reads), lastUsed, running
	st       handleState
	lastUsed time.Time
	running  *Job
}

func (s *Slot) isReady() bool {
	select {
	case <-s.ready:
		return true
	default:
		return false
	}
}

func (s *Slot) worker() {
	for {
		j := s.q.Pop()
		if j == nil {
			return
		}
		s.pool.npu.Lock()
		s.mu.Lock()
		s.running = j
		s.mu.Unlock()
		s.exec(j)
		s.mu.Lock()
		s.running = nil
		s.lastUsed = time.Now()
		s.mu.Unlock()
		s.pool.npu.Unlock()
		s.q.Done()
	}
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func (s *Slot) sessionPath(x *Session) string {
	return filepath.Join(s.pool.opts.SessionDir, s.m.Name, fileSafe(x.ID)+".cache")
}

func (s *Slot) prefixPath(static string) string {
	return filepath.Join(s.pool.opts.SessionDir, s.m.Name, "prefix-"+hashString(static)+".cache")
}

// exec runs one job. Called with the NPU lock held.
func (s *Slot) exec(j *Job) {
	u := Usage{QueueMs: ms(time.Since(j.enqueued))}
	if j.loadWait {
		u.ModelLoadMs = s.loadMs
	}
	strat := s.m.Strategy
	if j.NoCache || s.pool.opts.CacheOff {
		strat = config.StrategyOff
	}
	u.CacheStrategy = string(strat)
	sess, _ := s.pool.Sessions(s.m.Name).Resolve(s.m.Name, j.SessionID, j.Prompt)
	u.SessionID = sess.ID
	j.Stream.Start(sess.ID)
	if err := j.Ctx.Err(); err != nil {
		j.Stream.Finish(Result{Err: err, Usage: u})
		return
	}
	if est := len(j.Prompt) / 4; est >= s.m.MaxContextLen {
		j.Stream.Finish(Result{Err: fmt.Errorf("%w: prompt is ~%d tokens, model %s max_context_len is %d", ErrContextLength, est, s.m.Name, s.m.MaxContextLen), Usage: u})
		return
	}

	var res Result
	switch strat {
	case config.StrategyOff:
		res = s.runOff(j, sess, &u)
	case config.StrategyAppend:
		res = s.runAppend(j, sess, &u)
	case config.StrategySnapshot:
		res = s.runSnapshot(j, sess, &u)
	default:
		res = s.runPrefix(j, sess, &u)
	}
	if res.Err == nil {
		s.pool.Sessions(s.m.Name).Update(sess, j.Prompt)
	}
	j.Stream.Finish(res)
}

// ErrContextLength is returned for prompts that cannot fit.
var ErrContextLength = errors.New("context length exceeded")

func (s *Slot) setLive(x *Session) {
	s.mu.Lock()
	if x != nil && s.st.lost == x {
		s.st.lost = nil
	}
	s.st.live = x
	s.mu.Unlock()
}

// clear is a full KV clear; it also resets the runtime's prefix record and
// any loaded prompt cache. The live session's state is lost.
func (s *Slot) clear() {
	if err := s.h.ClearKV(); err != nil {
		log.Printf("[%s] clear: %v", s.m.Name, err)
	}
	s.mu.Lock()
	s.st = handleState{mode: -1, lost: s.st.lost}
	s.mu.Unlock()
}

func (s *Slot) ensureClean() {
	if s.st.dirty {
		s.clear()
	}
}

func (s *Slot) load(path string, u *Usage) bool {
	t := time.Now()
	err := s.h.LoadPromptCache(path)
	if u != nil {
		u.RestoreMs += ms(time.Since(t))
	}
	if err != nil {
		log.Printf("[%s] load %s: %v", s.m.Name, path, err)
		s.pool.store.Forget(path)
		return false
	}
	s.pool.store.Touch(path)
	return true
}

// saveRun writes a prompt-cache file for prompt with a keep_history=0 run
// that generates a single token. Afterwards the runtime's prefix record is
// prompt (1.3.1), or the handle is dirty (1.2.3: run stopped with rkllm_abort
// after one token; the caller full-clears before the next use).
func (s *Slot) saveRun(promptText, path string, kind string) bool {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		log.Printf("[%s] save: %v", s.m.Name, err)
		return false
	}
	req := engine.RunRequest{Prompt: promptText, SaveCachePath: path}
	if s.pool.eng.PerRunMaxTokens() {
		req.MaxNewTokens = 1
	} else {
		req.StopAfter = 1
	}
	res, err := s.h.Run(context.Background(), req)
	s.mu.Lock()
	defer s.mu.Unlock()
	if err != nil || res.Aborted || res.Stopped || !res.Finished {
		s.st.dirty = true
	} else {
		s.st.mode, s.st.lcp = 0, promptText
		s.st.kv = s.h.KVSize()
	}
	if err != nil {
		log.Printf("[%s] save run: %v", s.m.Name, err)
		return false
	}
	if ok, ev := s.pool.store.Add(path, kind); ok {
		s.pool.logEvicted(ev)
		return true
	}
	return false
}

// park preserves the live session before another session takes the handle.
// prefix: lazy save of the last prompt (the runtime's prefix record), so the
// hot path never pays for writing files. snapshot: the file is always
// current, nothing to do. append: keep_history=1 state cannot be saved in a
// reusable form (probe Q4b), so the session will re-prefill when it returns.
func (s *Slot) park(u *Usage) {
	live := s.st.live
	s.setLive(nil)
	if live == nil || s.st.dirty || s.pool.opts.SessionBudget <= 0 {
		return
	}
	if s.st.mode != 0 || s.st.lcp == "" || s.m.Strategy == config.StrategySnapshot || s.noLCP {
		return
	}
	ss := s.pool.Sessions(s.m.Name)
	if f, covers := ss.GetFile(live); f != "" && covers == s.st.lcp && s.pool.store.Has(f) {
		return
	}
	path := s.sessionPath(live)
	t := time.Now()
	if s.saveRun(s.st.lcp, path, "session") {
		ss.SetFile(live, path, s.st.lcp)
	}
	if u != nil {
		u.ParkMs = ms(time.Since(t))
	}
}

// usePrefix seeds the cache with the shared system+tools prefix for a new
// session if the handle's current state does not already start with it.
func (s *Slot) usePrefix(j *Job, u *Usage) {
	static := j.StaticPrefix
	if !s.pool.opts.PrefixCache || s.pool.opts.SessionBudget <= 0 || len(static) < s.pool.opts.MinPrefixBytes || !strings.HasPrefix(j.Prompt, static) || len(static) >= len(j.Prompt) {
		return
	}
	if !s.noLCP && !s.st.dirty && s.st.mode == 0 && strings.HasPrefix(s.st.lcp, static) {
		return // the runtime's prefix match covers it already
	}
	path := s.prefixPath(static)
	if s.pool.store.Has(path) {
		s.ensureClean()
		if s.load(path, u) {
			u.PrefixCache = "loaded"
		}
		return
	}
	s.ensureClean()
	if s.st.mode == 1 {
		s.clear()
	}
	t := time.Now()
	if !s.saveRun(static, path, "prefix") {
		return
	}
	u.SaveMs += ms(time.Since(t))
	u.PrefixCache = "created"
	if s.st.dirty || s.noLCP {
		s.ensureClean()
		s.load(path, u)
	}
}

// generate runs the model and fills usage. sent is the text passed to
// rkllm_run (the full prompt or an append delta).
func (s *Slot) generate(j *Job, req engine.RunRequest, u *Usage) Result {
	return s.generateStrip(j, req, u, 0)
}

// generateStrip is generate that hides up to strip leading "\n" bytes of the
// output (the snapshot strategy lets the model generate the generation
// prompt's trailing newlines itself, see runSnapshot).
func (s *Slot) generateStrip(j *Job, req engine.RunRequest, u *Usage, strip int) Result {
	first := true
	cut := func(text string) string {
		for strip > 0 && len(text) > 0 && text[0] == '\n' {
			text, strip = text[1:], strip-1
		}
		if text != "" {
			strip = 0
		}
		return text
	}
	var visible []string
	req.OnToken = func(t engine.Token) {
		if first {
			first = false
			u.TTFTMs = ms(time.Since(j.Start))
		}
		txt := cut(t.Text)
		visible = append(visible, txt)
		j.Stream.Token(txt)
	}
	if s.pool.eng.PerRunMaxTokens() {
		if j.MaxNewTokens > 0 {
			req.MaxNewTokens = j.MaxNewTokens
		}
		req.Sampling = j.Sampling
	}
	res, err := s.h.Run(j.Ctx, req)
	texts := make([]string, len(res.Tokens))
	var all strings.Builder
	for i := range res.Tokens {
		if i < len(visible) {
			texts[i] = visible[i]
		}
		all.WriteString(texts[i])
	}
	out := Result{Text: all.String(), Tokens: texts}
	if err != nil || res.Aborted || !res.Finished {
		s.mu.Lock()
		s.st.dirty = true
		s.mu.Unlock()
		switch {
		case err != nil:
			out.Err = err
		case res.Aborted:
			out.Err = context.Canceled
			if e := j.Ctx.Err(); e != nil {
				out.Err = e
			}
		default:
			out.Err = fmt.Errorf("%w: no FINISH callback", engine.ErrRunFailed)
		}
		out.Usage = *u
		return out
	}
	kv := s.h.KVSize()
	n := len(res.Tokens)
	promptTok := kv
	if n > 0 {
		promptTok = kv - (n - 1)
	}
	if res.Perf != nil {
		u.PrefillTokens = res.Perf.PrefillTokens
		u.PrefillMs = res.Perf.PrefillMs
		u.GenerateTokens = res.Perf.GenerateTokens
		u.GenerateMs = res.Perf.GenerateMs
	}
	u.PromptTokens = promptTok
	u.CachedTokens = promptTok - u.PrefillTokens
	if u.CachedTokens < 0 {
		u.CachedTokens = 0
	}
	if u.PromptTokens < u.PrefillTokens {
		u.PromptTokens = u.PrefillTokens
	}
	u.KVCacheSize = kv
	u.CompletionTokens = n
	u.FinishReason = "length"
	if n > 0 && !res.Tokens[n-1].Waiting && res.Tokens[n-1].Text == "" {
		u.FinishReason = "stop"
	}
	s.mu.Lock()
	s.st.kv = kv
	s.mu.Unlock()
	out.Usage = *u
	return out
}

// ---- strategies ----

func (s *Slot) runOff(j *Job, sess *Session, u *Usage) Result {
	u.CacheEvent = EventCold
	// The live session's state is discarded, not parked; the request's own
	// session (if any) is marked so its next turn reports a miss. Parked
	// files are kept.
	s.setLive(nil)
	s.clear()
	s.mu.Lock()
	s.st.lost = sess
	s.mu.Unlock()
	res := s.generate(j, engine.RunRequest{Prompt: j.Prompt}, u)
	if res.Err == nil {
		s.mu.Lock()
		s.st.mode, s.st.lcp = 0, j.Prompt
		s.mu.Unlock()
	}
	return res
}

func (s *Slot) runPrefix(j *Job, sess *Session, u *Usage) Result {
	if s.st.mode == 1 { // never mix keep_history modes without a clear
		s.clear()
	}
	if s.st.live == sess && !s.st.dirty {
		switch {
		case strings.HasPrefix(j.Prompt, s.st.lcp):
			u.CacheEvent = EventHit
		case CommonPrefixLen(j.Prompt, s.st.lcp) > 0:
			u.CacheEvent = EventPartial
		default:
			u.CacheEvent = EventMiss
		}
	} else {
		wasLive := s.st.live == sess || s.st.lost == sess
		s.park(u)
		s.ensureClean()
		if s.st.lost == sess {
			// a cache-off request invalidated this session's live state: do
			// not let the runtime's prefix record (the cache-off prompt)
			// silently serve it; restore from its parked file or re-prefill.
			s.clear()
		}
		f, covers := s.pool.Sessions(s.m.Name).GetFile(sess)
		switch {
		case f != "" && s.pool.store.Has(f) && CommonPrefixLen(j.Prompt, covers) > CommonPrefixLen(j.Prompt, s.st.lcp) && s.load(f, u):
			u.CacheEvent = EventRestored
		case wasLive:
			u.CacheEvent = EventMiss // state lost (aborted run)
			s.usePrefix(j, u)
		default:
			u.CacheEvent = EventCold
			s.usePrefix(j, u)
		}
	}
	s.ensureClean()
	s.setLive(sess)
	res := s.generate(j, engine.RunRequest{Prompt: j.Prompt}, u)
	if res.Err == nil {
		s.mu.Lock()
		s.st.mode, s.st.lcp = 0, j.Prompt
		s.mu.Unlock()
	}
	return res
}

// approxTokens is a deliberately generous token estimate (no tokenizer).
func approxTokens(s string) int { return len(s)/3 + 1 }

func (s *Slot) runAppend(j *Job, sess *Session, u *Usage) Result {
	if s.st.dirty || s.st.mode == 0 {
		s.clear()
	}
	if s.st.live == sess && s.st.mode == 1 {
		if delta, ok := AppendDelta(s.st.cached, j.Prompt); ok {
			maxNew := j.MaxNewTokens
			if maxNew <= 0 {
				maxNew = s.pool.opts.MaxNewTokens
			}
			if s.st.kv+approxTokens(delta)+maxNew < s.m.MaxContextLen {
				u.CacheEvent = EventHit
				before := s.st.cached
				res := s.generate(j, engine.RunRequest{Prompt: delta, KeepHistory: true}, u)
				if res.Err == nil {
					s.mu.Lock()
					s.st.cached = CachedText(before, delta, res.Tokens)
					s.mu.Unlock()
				}
				return res
			}
		}
		u.CacheEvent = EventMiss
	} else {
		u.CacheEvent = EventCold
		if s.st.lost == sess {
			u.CacheEvent = EventMiss
		}
		s.park(u) // nothing restorable in keep_history=1 mode
	}
	if s.st.mode != -1 {
		s.clear()
	}
	s.setLive(sess)
	res := s.generate(j, engine.RunRequest{Prompt: j.Prompt, KeepHistory: true}, u)
	if res.Err == nil {
		s.mu.Lock()
		s.st.mode = 1
		s.st.cached = CachedText("", j.Prompt, res.Tokens)
		s.mu.Unlock()
	}
	return res
}

// snapshotAppend serves a live Qwen3.5 session with keep_history=1 deltas so
// the previous reply is not re-prefilled (the snapshot file only covers the
// prompt of the turn it was saved on). Turn N+1 after a snapshot turn loads
// the snapshot and sends the rest of the prompt as a kh=1 delta (probe Q7: the
// loaded state is attended by a kh=1 run); later turns append to the live
// cache. ok=false means the caller takes the normal snapshot path.
func (s *Slot) snapshotAppend(j *Job, sess *Session, u *Usage) (Result, bool) {
	if s.st.dirty || s.st.live != sess || s.pool.opts.NoLiveAppend {
		return Result{}, false
	}
	maxNew := j.MaxNewTokens
	if maxNew <= 0 {
		maxNew = s.pool.opts.MaxNewTokens
	}
	var base, delta string
	switch s.st.mode {
	case 1:
		d, ok := AppendDelta(s.st.cached, j.Prompt)
		if !ok || s.st.kv+approxTokens(d)+maxNew >= s.m.MaxContextLen {
			if !ok {
				n := CommonPrefixLen(j.Prompt, s.st.cached)
				log.Printf("[%s] live-append miss: cached %dB prompt %dB common %dB; cached[..]=%q prompt[..]=%q", s.m.Name, len(s.st.cached), len(j.Prompt), n, tailAt(s.st.cached, n), tailAt(j.Prompt, n))
			}
			return Result{}, false
		}
		base, delta = s.st.cached, d
		u.CacheEvent = EventHit
	case 0:
		f, covers := s.pool.Sessions(s.m.Name).GetFile(sess)
		if f == "" || !s.pool.store.Has(f) || len(j.Prompt) <= len(covers) || !strings.HasPrefix(j.Prompt, covers) {
			return Result{}, false
		}
		s.clear() // also drops the live marker
		s.setLive(sess)
		if !s.load(f, u) {
			return Result{}, false
		}
		base, delta = covers, j.Prompt[len(covers):]
		u.CacheEvent = EventRestored
	default:
		return Result{}, false
	}
	res := s.generate(j, engine.RunRequest{Prompt: delta, KeepHistory: true}, u)
	if res.Err == nil {
		s.mu.Lock()
		s.st.mode = 1
		s.st.cached = CachedText(base, delta, res.Tokens)
		s.mu.Unlock()
	}
	return res, true
}

func (s *Slot) runSnapshot(j *Job, sess *Session, u *Usage) Result {
	if s.noLCP {
		if res, ok := s.snapshotAppend(j, sess, u); ok {
			return res
		}
	}
	if s.st.mode == 1 {
		s.clear()
	}
	ss := s.pool.Sessions(s.m.Name)
	live := s.st.live == sess && !s.st.dirty
	if !live {
		s.park(u)
	}
	s.ensureClean()
	f, covers := ss.GetFile(sess)
	haveFile := f != "" && s.pool.store.Has(f) && strings.HasPrefix(j.Prompt, covers)
	switch {
	case live && !s.noLCP:
		u.CacheEvent = EventHit // runtime prefix match, no load needed
		if !strings.HasPrefix(j.Prompt, s.st.lcp) {
			u.CacheEvent = EventPartial
		}
	case haveFile && s.load(f, u):
		u.CacheEvent = EventRestored
		if live {
			u.CacheEvent = EventHit
		}
	default:
		u.CacheEvent = EventCold
		if live || s.st.lost == sess {
			u.CacheEvent = EventMiss
		}
		s.usePrefix(j, u)
	}
	s.ensureClean()
	s.setLive(sess)
	path := s.sessionPath(sess)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		log.Printf("[%s] %v", s.m.Name, err)
	}
	// Qwen3.5 can only resume from a snapshot whose tokens are an exact
	// prefix of the next prompt (no partial reuse of recurrent state). The
	// generation prompt ends in "</think>\n\n", tokenized as one "\n\n"
	// token at the end of a prompt but as "\n","\n" when the next prompt
	// continues with the reply text, so a snapshot of the full prompt never
	// matches. Send the prompt without its trailing newlines (the snapshot
	// then ends at the special token) and let the model generate them; they
	// are stripped from the output.
	sent := j.Prompt
	strip := 0
	if s.noLCP {
		sent = strings.TrimRight(j.Prompt, "\n")
		strip = len(j.Prompt) - len(sent)
	}
	res := s.generateStrip(j, engine.RunRequest{Prompt: sent, SaveCachePath: path}, u, strip)
	if res.Err == nil {
		s.mu.Lock()
		s.st.mode, s.st.lcp = 0, sent
		s.mu.Unlock()
		if ok, ev := s.pool.store.Add(path, "snapshot"); ok {
			s.pool.logEvicted(ev)
			ss.SetFile(sess, path, sent)
		}
	}
	return res
}

// templateFamilyNoLCP reports families on which the runtime's
// keep_history=0 prefix reuse does not work (probe Q2d).
func templateFamilyNoLCP(f prompt.Family) bool { return f == prompt.FamilyQwen35 }

// tailAt returns a short window of x around byte offset n (for mismatch logs).
func tailAt(x string, n int) string {
	a, b := n-30, n+40
	if a < 0 {
		a = 0
	}
	if b > len(x) {
		b = len(x)
	}
	if a > b {
		a = b
	}
	return x[a:b]
}
