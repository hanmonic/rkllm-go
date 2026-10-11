package core

import (
	"errors"
	"fmt"
	"log"
	"os"
	"sort"
	"sync"
	"time"

	"rkllm-go/internal/config"
	"rkllm-go/internal/engine"
)

// NumDomains is the number of RKNPU IOMMU domains (ids 0..15; driver 0.9.8
// rejects 16+). Each has its own ~4 GiB IOVA space, shared by all processes.
const NumDomains = 16

// domainBytes is the planning size of one domain (4 GiB IOVA minus slack).
const domainMB = 3500

// Options configure the pool.
type Options struct {
	QueueDepth     int
	SessionDir     string
	SessionBudget  int64 // bytes of prompt-cache files (parked sessions, snapshots, prefixes)
	MemBudgetMB    int   // resident model memory budget
	CacheOff       bool
	NoLiveAppend   bool // Qwen3.5: disable kh=1 appends on live sessions (snapshot only)
	PrefixCache    bool
	MinPrefixBytes int
	MaxNewTokens   int
	TopK           int
	TopP           float32
	Temperature    float32
	RepeatPenalty  float32
}

var (
	ErrUnknownModel = errors.New("unknown model")
	ErrNoCapacity   = errors.New("model cannot be loaded within the memory budget")
)

// Pool owns the loaded models.
type Pool struct {
	cfg  *config.Config
	eng  engine.Engine
	opts Options

	mu       sync.Mutex
	slots    map[string]*Slot
	domains  [NumDomains]string
	sessions map[string]*SessionStore
	raws     map[string]*RawMap

	npu   sync.Mutex // one NPU job at a time across all handles
	store *FileStore
}

func NewPool(cfg *config.Config, eng engine.Engine, opts Options) *Pool {
	if opts.MinPrefixBytes <= 0 {
		opts.MinPrefixBytes = 1024
	}
	if opts.SessionDir != "" {
		// files from a previous process are not tracked; start clean
		_ = os.RemoveAll(opts.SessionDir)
		_ = os.MkdirAll(opts.SessionDir, 0o755)
	}
	return &Pool{
		cfg: cfg, eng: eng, opts: opts,
		slots:    map[string]*Slot{},
		sessions: map[string]*SessionStore{},
		raws:     map[string]*RawMap{},
		store:    NewFileStore(opts.SessionBudget),
	}
}

func (p *Pool) Config() *config.Config { return p.cfg }
func (p *Pool) Engine() engine.Engine  { return p.eng }

// Sessions returns the session store of a model.
func (p *Pool) Sessions(model string) *SessionStore {
	p.mu.Lock()
	defer p.mu.Unlock()
	s, ok := p.sessions[model]
	if !ok {
		s = NewSessionStore(1000)
		p.sessions[model] = s
	}
	return s
}

// Raw returns the generated-text map of a model.
func (p *Pool) Raw(model string) *RawMap {
	p.mu.Lock()
	defer p.mu.Unlock()
	r, ok := p.raws[model]
	if !ok {
		r = NewRawMap(4096)
		p.raws[model] = r
	}
	return r
}

func (p *Pool) logEvicted(paths []string) {
	for _, e := range paths {
		log.Printf("cache budget: evicted %s", e)
	}
}

// Loaded reports whether a model's handle is ready.
func (p *Pool) Loaded(model string) bool {
	p.mu.Lock()
	s := p.slots[model]
	p.mu.Unlock()
	return s != nil && s.isReady() && s.loadErr == nil
}

// estimateMB is the expected resident memory of a handle.
func estimateMB(m *config.Model) int {
	if m.MemMB > 0 {
		return m.MemMB
	}
	st, err := os.Stat(m.Path)
	if err != nil {
		return 4096
	}
	// weights + fp16 KV for max_context_len (~0.12 MB/token for 1.7B-8B Qwen3)
	return int(st.Size()>>20) + m.MaxContextLen*120/1000 + 300
}

// Submit enqueues a job, loading the model if needed.
func (p *Pool) Submit(j *Job) error {
	m, ok := p.cfg.Models[j.Model]
	if !ok {
		return fmt.Errorf("%w: %q", ErrUnknownModel, j.Model)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	s := p.slots[j.Model]
	if s == nil {
		s = p.newSlotLocked(m, false)
	}
	j.enqueued = time.Now()
	j.loadWait = !s.isReady()
	return s.q.Push(j)
}

// Preload loads a model and waits for it.
func (p *Pool) Preload(model string) error {
	m, ok := p.cfg.Models[model]
	if !ok {
		return fmt.Errorf("%w: %q", ErrUnknownModel, model)
	}
	p.mu.Lock()
	s := p.slots[model]
	if s == nil {
		s = p.newSlotLocked(m, true)
	}
	p.mu.Unlock()
	<-s.ready
	return s.loadErr
}

func (p *Pool) newSlotLocked(m *config.Model, preload bool) *Slot {
	s := &Slot{
		pool: p, m: m, q: NewQueue(p.opts.QueueDepth),
		noLCP:     templateFamilyNoLCP(m.Family),
		ready:     make(chan struct{}),
		memMB:     estimateMB(m),
		preloaded: preload,
		st:        handleState{mode: -1},
	}
	p.slots[m.Name] = s
	go s.loadLoop()
	return s
}

// residentMBLocked sums the memory of slots other than except.
func (p *Pool) residentMBLocked(except *Slot) int {
	t := 0
	for _, s := range p.slots {
		if s != except && (len(s.domains) > 0 || s.isReady()) {
			t += s.memMB
		}
	}
	return t
}

// freeDomainsLocked finds a run of n free domains starting at or after from.
func (p *Pool) freeDomainsLocked(n, from int) int {
	for b := from; b+n <= NumDomains; b++ {
		ok := true
		for d := b; d < b+n; d++ {
			if p.domains[d] != "" {
				ok = false
				break
			}
		}
		if ok {
			return b
		}
	}
	return -1
}

// evictableLocked returns the least recently used idle loaded slot.
func (p *Pool) evictableLocked(except *Slot) *Slot {
	var v *Slot
	for _, s := range p.slots {
		if s == except || !s.isReady() || s.loadErr != nil || !s.q.Idle() {
			continue
		}
		s.mu.Lock()
		lu := s.lastUsed
		s.mu.Unlock()
		if v == nil {
			v = s
			continue
		}
		v.mu.Lock()
		vl := v.lastUsed
		v.mu.Unlock()
		if lu.Before(vl) {
			v = s
		}
	}
	return v
}

func (s *Slot) span() int {
	n := (s.memMB + domainMB - 1) / domainMB
	if n < 1 {
		n = 1
	}
	return n
}

// reserve claims memory and domains, evicting idle LRU models. It returns
// the base domain, or -1 if it must wait.
func (p *Pool) reserve(s *Slot, extra int) (int, error) {
	for {
		p.mu.Lock()
		if s.memMB > p.opts.MemBudgetMB && p.residentMBLocked(s) == 0 {
			// a single model larger than the budget: allow it alone
		} else if s.memMB > p.opts.MemBudgetMB*2 {
			p.mu.Unlock()
			return -1, ErrNoCapacity
		}
		need := s.span() + extra
		fits := p.residentMBLocked(s)+s.memMB <= p.opts.MemBudgetMB || p.residentMBLocked(s) == 0
		base := p.freeDomainsLocked(need, 0)
		if fits && base >= 0 {
			for d := base; d < base+need; d++ {
				p.domains[d] = s.m.Name
			}
			s.domains = nil
			for d := base; d < base+need; d++ {
				s.domains = append(s.domains, d)
			}
			p.mu.Unlock()
			return base, nil
		}
		v := p.evictableLocked(s)
		if v != nil {
			p.removeSlotLocked(v)
		}
		p.mu.Unlock()
		if v != nil {
			v.shutdown("swap for " + s.m.Name)
			continue
		}
		// nothing idle to evict: wait for a running model to finish
		if !s.preloaded && s.q.Idle() {
			return -1, context_canceled
		}
		time.Sleep(200 * time.Millisecond)
	}
}

var context_canceled = errors.New("all waiting requests cancelled")

func (p *Pool) removeSlotLocked(s *Slot) {
	if p.slots[s.m.Name] == s {
		delete(p.slots, s.m.Name)
	}
}

func (p *Pool) releaseDomainsLocked(s *Slot) {
	for _, d := range s.domains {
		if p.domains[d] == s.m.Name {
			p.domains[d] = ""
		}
	}
	s.domains = nil
}

func (s *Slot) loadLoop() {
	p := s.pool
	defer close(s.ready)
	fail := func(err error) {
		s.loadErr = err
		p.mu.Lock()
		p.releaseDomainsLocked(s)
		p.removeSlotLocked(s)
		p.mu.Unlock()
		for _, j := range s.q.Close() {
			j.Stream.Finish(Result{Err: fmt.Errorf("load %s: %w", s.m.Name, err)})
		}
		log.Printf("[%s] load failed: %v", s.m.Name, err)
	}
	t := time.Now()
	var h engine.Handle
	var err error
	for extra := 0; extra < 2; extra++ {
		var base int
		base, err = p.reserve(s, extra)
		if err != nil {
			fail(err)
			return
		}
		lt := time.Now()
		// rkllm_init does not touch the NPU queue, but serialize with runs anyway
		// to keep load timings clean.
		p.npu.Lock()
		h, err = p.eng.Load(engine.Params{
			ModelPath: s.m.Path, MaxContextLen: s.m.MaxContextLen, MaxNewTokens: p.opts.MaxNewTokens,
			TopK: p.opts.TopK, TopP: p.opts.TopP, Temperature: p.opts.Temperature, RepeatPenalty: p.opts.RepeatPenalty,
			BaseDomainID: base, SkipSpecial: true,
		})
		if err == nil {
			// raw mode: the server renders prompts itself (also clears the cache)
			err = h.SetChatTemplate("", "", "")
		}
		p.npu.Unlock()
		if err == nil {
			log.Printf("[%s] loaded in %v (domains %v, ~%d MB, strategy %s)", s.m.Name, time.Since(lt).Round(time.Millisecond), s.domains, s.memMB, s.m.Strategy)
			break
		}
		log.Printf("[%s] init on domains %v failed: %v", s.m.Name, s.domains, err)
		p.mu.Lock()
		p.releaseDomainsLocked(s)
		p.mu.Unlock()
	}
	if err != nil {
		fail(err)
		return
	}
	s.h = h
	s.loadMs = ms(time.Since(t))
	s.lastUsed = time.Now()
	go s.worker()
}

// shutdown parks the live session and destroys the handle (slot already
// removed from the pool and idle).
func (s *Slot) shutdown(why string) {
	for _, j := range s.q.Close() {
		j.Stream.Finish(Result{Err: fmt.Errorf("model %s unloaded", s.m.Name)})
	}
	p := s.pool
	p.npu.Lock()
	t := time.Now()
	s.park(nil)
	s.h.Destroy()
	p.npu.Unlock()
	p.mu.Lock()
	p.releaseDomainsLocked(s)
	p.mu.Unlock()
	log.Printf("[%s] unloaded (%s) in %v", s.m.Name, why, time.Since(t).Round(time.Millisecond))
}

// Close unloads everything.
func (p *Pool) Close() {
	p.mu.Lock()
	var ss []*Slot
	for _, s := range p.slots {
		ss = append(ss, s)
	}
	p.slots = map[string]*Slot{}
	p.mu.Unlock()
	for _, s := range ss {
		<-s.ready
		if s.loadErr == nil {
			for !s.q.Idle() {
				time.Sleep(50 * time.Millisecond)
			}
			s.shutdown("exit")
		}
	}
}

// ---- debug ----

type HandleInfo struct {
	Model         string   `json:"model"`
	Family        string   `json:"family"`
	Strategy      string   `json:"strategy"`
	Loaded        bool     `json:"loaded"`
	LoadMs        float64  `json:"load_ms"`
	MemMB         int      `json:"mem_mb_estimate"`
	Domains       []int    `json:"iommu_domains"`
	MaxContextLen int      `json:"max_context_len"`
	LiveSession   string   `json:"live_session"`
	KVMode        string   `json:"kv_mode"`
	Dirty         bool     `json:"dirty"`
	KVCacheSize   int      `json:"kv_cache_size"`
	LCPBytes      int      `json:"prefix_record_bytes"`
	CachedBytes   int      `json:"append_cached_bytes"`
	QueueWaiting  int      `json:"queue_waiting"`
	QueueActive   int      `json:"queue_active"`
	QueueDepth    int      `json:"queue_depth"`
	Running       string   `json:"running_session,omitempty"`
	Sessions      []string `json:"-"`
}

type State struct {
	Runtime       string                   `json:"runtime"`
	CacheOff      bool                     `json:"cache_off"`
	PrefixCache   bool                     `json:"prefix_cache"`
	MemBudgetMB   int                      `json:"mem_budget_mb"`
	ResidentMB    int                      `json:"resident_mb_estimate"`
	Handles       []HandleInfo             `json:"handles"`
	Files         []fileEntry              `json:"cache_files"`
	FilesBytes    int64                    `json:"cache_files_bytes"`
	FilesBudget   int64                    `json:"cache_files_budget"`
	Sessions      map[string][]SessionInfo `json:"sessions"`
	DomainsOwners [NumDomains]string       `json:"iommu_domain_owners"`
}

func (p *Pool) State() State {
	p.mu.Lock()
	st := State{Runtime: p.eng.Version(), CacheOff: p.opts.CacheOff, PrefixCache: p.opts.PrefixCache, MemBudgetMB: p.opts.MemBudgetMB, DomainsOwners: p.domains, Sessions: map[string][]SessionInfo{}}
	st.ResidentMB = p.residentMBLocked(nil)
	var slots []*Slot
	for _, s := range p.slots {
		slots = append(slots, s)
	}
	stores := map[string]*SessionStore{}
	for k, v := range p.sessions {
		stores[k] = v
	}
	p.mu.Unlock()
	for _, s := range slots {
		hi := HandleInfo{Model: s.m.Name, Family: string(s.m.Family), Strategy: string(s.m.Strategy), Loaded: s.isReady() && s.loadErr == nil,
			MemMB: s.memMB, Domains: s.domains, MaxContextLen: s.m.MaxContextLen,
			QueueWaiting: s.q.Len(), QueueActive: s.q.Active(), QueueDepth: p.opts.QueueDepth}
		if hi.Loaded {
			hi.LoadMs = s.loadMs
		}
		s.mu.Lock()
		if s.st.live != nil {
			hi.LiveSession = s.st.live.ID
		}
		hi.KVMode = map[int]string{-1: "clean", 0: "keep_history=0", 1: "keep_history=1"}[s.st.mode]
		hi.Dirty, hi.KVCacheSize, hi.LCPBytes, hi.CachedBytes = s.st.dirty, s.st.kv, len(s.st.lcp), len(s.st.cached)
		if s.running != nil {
			hi.Running = s.running.SessionID
		}
		s.mu.Unlock()
		st.Handles = append(st.Handles, hi)
	}
	sort.Slice(st.Handles, func(i, j int) bool { return st.Handles[i].Model < st.Handles[j].Model })
	for k, v := range stores {
		st.Sessions[k] = v.List()
	}
	st.Files = p.store.List()
	st.FilesBytes = p.store.Total()
	st.FilesBudget = p.store.Budget()
	return st
}
