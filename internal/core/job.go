package core

import (
	"context"
	"sync"
	"time"

	"rkllm-go/internal/engine"
)

// Job is one chat completion request for a model.
type Job struct {
	Ctx   context.Context
	Model string
	// Prompt is the full formatted prompt ending with the generation prompt.
	Prompt string
	// StaticPrefix is the append-stable system + tools block of Prompt ("" if
	// none); used for the shared-prefix cache.
	StaticPrefix string
	// SessionID from X-Session-Id / the OpenAI "user" field ("" = match by prefix).
	SessionID string
	// NoCache: full clear before the run and keep_history=0 (true cold prefill).
	NoCache      bool
	MaxNewTokens int
	// Sampling per request (nil = server default; honoured on 1.3.1 only).
	Sampling *engine.Sampling
	// Start is when the HTTP request arrived (ttft is measured from it).
	Start time.Time

	enqueued time.Time
	loadWait bool // the model was not loaded when the job was submitted
	Stream   *Stream
}

// Usage is the rkllm-specific usage block (usage.rkllm in responses).
type Usage struct {
	PromptTokens   int     `json:"prompt_tokens"`
	PrefillTokens  int     `json:"prefill_tokens"`
	CachedTokens   int     `json:"cached_tokens"`
	PrefillMs      float64 `json:"prefill_ms"`
	GenerateTokens int     `json:"generate_tokens"`
	GenerateMs     float64 `json:"generate_ms"`
	TTFTMs         float64 `json:"ttft_ms"`
	QueueMs        float64 `json:"queue_ms"`
	ModelLoadMs    float64 `json:"model_load_ms"`
	CacheStrategy  string  `json:"cache_strategy"`
	CacheEvent     string  `json:"cache_event"`
	SessionID      string  `json:"session_id"`
	KVCacheSize    int     `json:"kv_cache_size"`
	// extras (not part of the benchmark contract)
	CompletionTokens int     `json:"completion_tokens"`
	ParkMs           float64 `json:"park_ms,omitempty"`
	RestoreMs        float64 `json:"restore_ms,omitempty"`
	SaveMs           float64 `json:"save_ms,omitempty"`
	PrefixCache      string  `json:"prefix_cache,omitempty"` // "loaded" | "created"
	FinishReason     string  `json:"-"`
}

// Result is the outcome of a job.
type Result struct {
	Text   string
	Tokens []string // per-token text (the final one is usually the empty stop token)
	Usage  Usage
	Err    error
}

// Stream carries the worker's progress to the HTTP handler without ever
// blocking the runtime callback.
type Stream struct {
	mu      sync.Mutex
	notify  chan struct{}
	started bool
	session string
	texts   []string
	done    bool
	result  Result
}

func NewStream() *Stream { return &Stream{notify: make(chan struct{}, 1)} }

func (s *Stream) signal() {
	select {
	case s.notify <- struct{}{}:
	default:
	}
}

// Start reports the resolved session (headers can be written now).
func (s *Stream) Start(session string) {
	s.mu.Lock()
	s.started, s.session = true, session
	s.mu.Unlock()
	s.signal()
}

// Token appends generated text.
func (s *Stream) Token(text string) {
	if text == "" {
		return
	}
	s.mu.Lock()
	s.texts = append(s.texts, text)
	s.mu.Unlock()
	s.signal()
}

// Finish ends the stream.
func (s *Stream) Finish(r Result) {
	s.mu.Lock()
	s.done, s.result = true, r
	s.mu.Unlock()
	s.signal()
}

// Notify fires whenever something new is available.
func (s *Stream) Notify() <-chan struct{} { return s.notify }

// Snapshot returns the current state; texts from index `from` on.
func (s *Stream) Snapshot(from int) (started bool, session string, texts []string, done bool, res Result) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if from < len(s.texts) {
		texts = append(texts, s.texts[from:]...)
	}
	return s.started, s.session, texts, s.done, s.result
}
