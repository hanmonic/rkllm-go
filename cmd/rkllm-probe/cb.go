//go:build linux && cgo

package main

/*
#include "rkllm.h"
*/
import "C"

import (
	"runtime/cgo"
	"time"
	"unsafe"
)

// CGo rule: a file with //export must not define C functions in its preamble.

//export probeGoCallback
func probeGoCallback(r *C.RKLLMResult, ud unsafe.Pointer, st C.LLMCallState) C.int {
	if ud == nil {
		return 0
	}
	rs, ok := cgo.Handle(uintptr(ud)).Value().(*runState)
	if !ok {
		return 0
	}
	at := time.Since(rs.start)
	var ev Event
	ev.State = int(st)
	ev.AtMs = float64(at.Microseconds()) / 1000
	var p Perf
	if r != nil {
		ev.Tok = int32(r.token_id)
		if r.text != nil {
			ev.Text = C.GoString(r.text)
		}
		p = Perf{
			PrefillMs:   float64(r.perf.prefill_time_ms),
			PrefillTok:  int(r.perf.prefill_tokens),
			GenerateMs:  float64(r.perf.generate_time_ms),
			GenerateTok: int(r.perf.generate_tokens),
			MemMB:       float64(r.perf.memory_usage_mb),
		}
		if p != (Perf{}) {
			rs.perfAt = append(rs.perfAt, int(st))
			pp := p
			ev.Perf = &pp
		}
	}
	if rs.keepEvents {
		rs.events = append(rs.events, ev)
	}
	ret := C.int(0)
	switch int(st) {
	case stNormal:
		if rs.firstTok < 0 {
			rs.firstTok = at
		}
		rs.nNormal++
		rs.text = append(rs.text, ev.Text...)
		rs.ids = append(rs.ids, ev.Tok)
		if rs.pauseAfter > 0 && rs.nNormal >= rs.pauseAfter {
			rs.paused = true
			ret = 1
		}
		for _, id := range rs.pauseOn {
			if ev.Tok == id {
				rs.paused = true
				ret = 1
			}
		}
	case stWaiting:
		if rs.firstTok < 0 {
			rs.firstTok = at
		}
		rs.nWaiting++
		rs.ids = append(rs.ids, ev.Tok)
	case stFinish:
		rs.nFinish++
		if p != (Perf{}) {
			rs.perf = p
		}
	case stError:
		rs.nError++
	}
	if (st == stNormal || st == stWaiting) && p != (Perf{}) && rs.perf == (Perf{}) {
		rs.perf = p
	}
	return ret
}
