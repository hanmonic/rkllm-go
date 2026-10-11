package rkllm

/*
#include "rkllm_rt.h"
*/
import "C"
import (
	"runtime/cgo"
	"unsafe"
)

// rkllmGoCallback is the single global C callback registered with rkllm_init.
// Per-request routing uses cgo.Handle stored in userdata to find the caller's channel.
//
// CGo rule: files with //export must not define C functions in their preamble.
// C helper definitions live in rkllm.go.
//
//export rkllmGoCallback
func rkllmGoCallback(result *C.RKLLMResult, userdata unsafe.Pointer, state C.LLMCallState) C.int {
	if userdata == nil {
		return 0
	}
	h := cgo.Handle(uintptr(userdata))
	ch, ok := h.Value().(chan TokenEvent)
	if !ok {
		return 0
	}

	switch state {
	case C.RKLLM_RUN_NORMAL:
		if result != nil && result.text != nil {
			ch <- TokenEvent{Text: C.GoString(result.text), State: StateNormal}
		}
	case C.RKLLM_RUN_WAITING:
		// Runtime is buffering an incomplete UTF-8 sequence. Do not emit — next
		// RKLLM_RUN_NORMAL will carry the completed character.
	case C.RKLLM_RUN_FINISH:
		ev := TokenEvent{State: StateFinish}
		if result != nil {
			ev.Perf = &PerfStat{
				PrefillTimeMs:  float32(result.perf.prefill_time_ms),
				PrefillTokens:  int(result.perf.prefill_tokens),
				GenerateTimeMs: float32(result.perf.generate_time_ms),
				GenerateTokens: int(result.perf.generate_tokens),
				MemoryMB:       float32(result.perf.memory_usage_mb),
			}
			// Embedding mode: extract hidden states for the first token.
			// Layout is [num_tokens][embd_size]; we take token 0 only,
			// matching the reference Python implementation.
			hl := result.last_hidden_layer
			if hl.embd_size > 0 && hl.hidden_states != nil {
				n := int(hl.embd_size)
				src := unsafe.Slice((*float32)(unsafe.Pointer(hl.hidden_states)), n)
				ev.Embedding = make([]float32, n)
				copy(ev.Embedding, src)
			}
		}
		ch <- ev
	case C.RKLLM_RUN_ERROR:
		ch <- TokenEvent{State: StateError}
	}
	return 0
}
