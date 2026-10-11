package rkllm

import "errors"

type CallState int

const (
	StateNormal  CallState = 0
	StateWaiting CallState = 1 // UTF-8 incomplete char — skip, do not emit
	StateFinish  CallState = 2
	StateError   CallState = 3
)

// Config holds startup parameters for the RKLLM engine.
// Sampling params are fixed at init time — rkllm_run does not accept per-call overrides.
type Config struct {
	ModelPath     string
	MaxContextLen int32
	MaxNewTokens  int32
	TopK          int32
	TopP          float32
	Temperature   float32
	RepeatPenalty float32
	NKeep         int32 // -1 = keep all tokens in sliding window
}

// RunInput is the input to a single inference call.
type RunInput struct {
	Prompt         string
	Role           string // "system" when passing a full ChatML-formatted history
	EnableThinking bool   // Qwen3: true enables <think> chain-of-thought output
}

// TokenEvent is sent on the channel returned by Engine.Run for each callback firing.
type TokenEvent struct {
	Text      string
	State     CallState
	Perf      *PerfStat  // non-nil on StateFinish for generation
	Embedding []float32  // non-nil on StateFinish for embedding mode
}

// PerfStat mirrors RKLLMPerfStat from the runtime, populated on StateFinish.
type PerfStat struct {
	PrefillTimeMs  float32
	PrefillTokens  int
	GenerateTimeMs float32
	GenerateTokens int
	MemoryMB       float32
}

var ErrBusy = errors.New("inference already in progress")

// ErrEmbedNotSupported is returned when the loaded model was not converted with
// hidden-state extraction enabled (RKLLM_INFER_GET_LAST_HIDDEN_LAYER).
// Re-convert the model with the rkllm-toolkit export_hidden_states option.
var ErrEmbedNotSupported = errors.New("model does not support embedding extraction")
