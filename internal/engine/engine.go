// Package engine is the runtime-neutral interface to an RKLLM model handle.
// The cgo implementations live in internal/engine/rk (build tag rk131, the
// default, or rk123); everything above this interface (cache strategies,
// sessions, queueing, HTTP) is shared and testable with the fake in
// internal/engine/fake.
package engine

import (
	"context"
	"errors"
)

// Params are init-time parameters of one handle.
type Params struct {
	ModelPath     string
	MaxContextLen int
	MaxNewTokens  int
	TopK          int
	TopP          float32
	Temperature   float32
	RepeatPenalty float32
	// BaseDomainID selects the RKNPU IOMMU domain the handle's NPU buffers are
	// mapped into (RKLLMExtendParam.base_domain_id). Each domain has its own
	// ~4 GiB IOVA space; see docs/v2-design.md, step 0.
	BaseDomainID int
	// SkipSpecial drops special-token text from callbacks (stop tokens arrive
	// with empty text). Qwen's <think>/<tool_call> are not special.
	SkipSpecial bool
}

// Token is one generation callback (NORMAL or WAITING).
type Token struct {
	ID      int32
	Text    string // empty for WAITING (incomplete UTF-8) and special tokens
	Waiting bool   // RKLLM_RUN_WAITING: bytes follow with a later token
}

// Perf mirrors RKLLMPerfStat (only delivered with FINISH).
type Perf struct {
	PrefillMs      float64
	PrefillTokens  int
	GenerateMs     float64
	GenerateTokens int
	MemoryMB       float64
}

// RunRequest is one rkllm_run call with a text prompt.
type RunRequest struct {
	Prompt      string
	KeepHistory bool
	// MaxNewTokens overrides the init value for this run (1.3.1 only; 0 = init value).
	MaxNewTokens int
	// SaveCachePath, if set, saves the prompt cache of this run to the file.
	SaveCachePath string
	// StopAfter > 0 stops the run with rkllm_abort after that many tokens.
	// Used on 1.2.3, which has no per-run max_new_tokens, to end prompt-cache
	// save runs. Callers should full-clear the KV cache afterwards. (Pausing
	// by returning 1 from the callback is not used: on 1.2.3 the paused state
	// survives rkllm_clear_kv_cache and makes the next rkllm_load_prompt_cache
	// a no-op, see docs/v2-design.md.)
	StopAfter int
	// Sampling overrides the init-time sampling for this run (1.3.1 only).
	Sampling *Sampling
	// OnToken is called synchronously from the runtime callback for every
	// NORMAL/WAITING token. It must not block for long.
	OnToken func(Token)
}

// Sampling is a per-run sampling override (RKLLMSamplingParam).
type Sampling struct {
	TopK        int
	TopP        float32
	Temperature float32
}

// RunResult describes a finished (or aborted / failed) run.
type RunResult struct {
	Perf     *Perf // nil unless FINISH arrived
	Finished bool  // FINISH callback seen
	Aborted  bool  // ctx was cancelled and rkllm_abort called
	Stopped  bool  // StopAfter triggered (rkllm_abort); Aborted stays false
	Tokens   []Token
	// FirstTokenAt is the wall time from the rkllm_run call to the first
	// NORMAL/WAITING callback (0 if none).
	FirstTokenMs float64
	WallMs       float64
}

// ErrRunFailed is returned when rkllm_run returns non-zero. The runtime
// reports most failures this way, without an ERROR callback.
var ErrRunFailed = errors.New("rkllm_run failed")

// ErrUnsupported is returned by stub builds.
var ErrUnsupported = errors.New("rkllm engine not available in this build")

// Handle is one loaded model (rkllm_init). Calls must be serialized by the
// caller; Abort may be called concurrently with Run.
type Handle interface {
	// Run blocks until the run ends. Cancelling ctx aborts it (rkllm_abort).
	Run(ctx context.Context, req RunRequest) (RunResult, error)
	ClearKV() error
	KVSize() int
	LoadPromptCache(path string) error
	ReleasePromptCache() error
	SetChatTemplate(system, prefix, postfix string) error
	Destroy()
}

// Engine creates handles.
type Engine interface {
	Load(p Params) (Handle, error)
	// Version is "1.3.1" or "1.2.3".
	Version() string
	// PerRunMaxTokens reports whether RunRequest.MaxNewTokens is honoured.
	PerRunMaxTokens() bool
}
