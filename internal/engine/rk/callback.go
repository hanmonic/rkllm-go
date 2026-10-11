//go:build linux && cgo

package rk

/*
#include "rkllm.h"
*/
import "C"

import (
	"runtime/cgo"
	"time"
	"unsafe"

	"rkllm-go/internal/engine"
)

// Call states, identical in 1.2.3 and 1.3.1 (verified by the probe).
const (
	stNormal  = 0
	stWaiting = 1
	stFinish  = 2
	stError   = 3
)

// CGo rule: a file with //export must not define C functions in its preamble.

//export rkGoCallback
func rkGoCallback(r *C.RKLLMResult, ud unsafe.Pointer, st C.LLMCallState) C.int {
	if ud == nil {
		return 0
	}
	rs, ok := cgo.Handle(uintptr(ud)).Value().(*runState)
	if !ok {
		return 0
	}
	switch int(st) {
	case stNormal, stWaiting:
		var t engine.Token
		if r != nil {
			t.ID = int32(r.token_id)
			t.Waiting = int(st) == stWaiting
			if int(st) == stNormal && r.text != nil {
				t.Text = C.GoString(r.text)
			}
		}
		if len(rs.res.Tokens) == 0 {
			rs.res.FirstTokenMs = float64(time.Since(rs.start).Microseconds()) / 1000
		}
		rs.res.Tokens = append(rs.res.Tokens, t)
		if rs.onToken != nil {
			rs.onToken(t)
		}
		if rs.stopAfter > 0 && len(rs.res.Tokens) >= rs.stopAfter && !rs.res.Stopped {
			// abort asynchronously (rkllm_abort from inside the callback is
			// not documented as safe); a few more tokens may arrive
			rs.res.Stopped = true
			rs.stop()
		}
	case stFinish:
		rs.res.Finished = true
		if r != nil {
			rs.res.Perf = &engine.Perf{
				PrefillMs:      float64(r.perf.prefill_time_ms),
				PrefillTokens:  int(r.perf.prefill_tokens),
				GenerateMs:     float64(r.perf.generate_time_ms),
				GenerateTokens: int(r.perf.generate_tokens),
				MemoryMB:       float64(r.perf.memory_usage_mb),
			}
		}
	case stError:
	}
	return 0
}
