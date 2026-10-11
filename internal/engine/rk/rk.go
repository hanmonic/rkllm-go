//go:build linux && cgo

// Package rk is the cgo binding to librkllmrt.so, for runtime 1.3.1 (default,
// build tag rk131 or none) or 1.2.3 (build tag rk123). The C helpers below are
// shared; RKLLM_V131 (set in flags_131.go) selects the API differences:
// rkllm_init takes an RKLLMCallback struct, and RKLLMInferParam carries
// per-run max_new_tokens / sampling in 1.3.1.
package rk

/*
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "rkllm.h"

extern int rkGoCallback(RKLLMResult* result, void* userdata, LLMCallState state);

static int rk_cb(RKLLMResult* r, void* ud, LLMCallState st) { return rkGoCallback(r, ud, st); }

static int rk_init(LLMHandle* h, const char* path, int ctx, int maxnew, int topk, float topp,
                   float temp, float rep, int skip_special, int domain) {
	RKLLMParam p = rkllm_createDefaultParam();
	p.model_path = path;
	p.max_context_len = ctx;
	p.max_new_tokens = maxnew;
	p.top_k = topk;
	p.top_p = topp;
	p.temperature = temp;
	p.repeat_penalty = rep;
	p.frequency_penalty = 0.0f;
	p.presence_penalty = 0.0f;
	p.mirostat = 0;
	p.n_keep = -1;
	p.skip_special_token = skip_special ? true : false;
	p.is_async = false;
	p.extend_param.base_domain_id = domain;
	p.extend_param.embed_flash = 1;
	p.extend_param.enabled_cpus_num = 4;
	p.extend_param.enabled_cpus_mask = 0xF0; // A76 big cores
	p.extend_param.n_batch = 1;
	p.extend_param.use_cross_attn = 0;
#ifdef RKLLM_V131
	// Heap-allocated and intentionally leaked (one per handle): we do not know
	// whether the runtime copies the struct or keeps the pointer.
	RKLLMCallback* cb = (RKLLMCallback*)calloc(1, sizeof(RKLLMCallback));
	cb->result_callback = rk_cb;
	return rkllm_init(h, &p, cb);
#else
	return rkllm_init(h, &p, rk_cb);
#endif
}

static int rk_run(LLMHandle h, const char* prompt, int keep, int maxnew, const char* cpath,
                  int samp, int topk, float topp, float temp, float rep, uintptr_t ud) {
	RKLLMInput in;
	memset(&in, 0, sizeof(in));
	in.role = "user";
	in.enable_thinking = false;
	in.input_type = RKLLM_INPUT_PROMPT;
	in.prompt_input = prompt;
	RKLLMInferParam ip;
	memset(&ip, 0, sizeof(ip));
	ip.mode = RKLLM_INFER_GENERATE;
	ip.keep_history = keep;
	// Always pass a prompt-cache param struct, zeroed (save_prompt_cache=0)
	// when not saving. Runtime 1.2.3 dereferences ip.prompt_cache_params on
	// every run once any run on the handle has saved a prompt cache, so a
	// NULL there after a save run segfaults at address 0 (reproduced with
	// cmd/rkllm-probe: save run, then any run with NULL params). A zeroed
	// struct is accepted by 1.2.3 and 1.3.1 and saves nothing.
	RKLLMPromptCacheParam pc;
	memset(&pc, 0, sizeof(pc));
	pc.prompt_cache_path = "";
	if (cpath != NULL) {
		pc.save_prompt_cache = 1;
		pc.prompt_cache_path = cpath;
	}
	ip.prompt_cache_params = &pc;
#ifdef RKLLM_V131
	ip.max_new_tokens = maxnew;
	RKLLMSamplingParam sp;
	memset(&sp, 0, sizeof(sp));
	if (samp) {
		sp.top_k = topk;
		sp.top_p = topp;
		sp.temperature = temp;
		sp.repeat_penalty = rep;
		ip.sampling_params = &sp;
	}
#endif
	return rkllm_run(h, &in, &ip, (void*)ud);
}

static int rk_kv(LLMHandle h) {
	int sz[8];
	memset(sz, 0, sizeof(sz));
	if (rkllm_get_kv_cache_size(h, sz) != 0) return -1;
	return sz[0];
}

static int rk_clear(LLMHandle h) { return rkllm_clear_kv_cache(h, 0, NULL, NULL); }

static int rk_v131(void) {
#ifdef RKLLM_V131
	return 1;
#else
	return 0;
#endif
}
*/
import "C"

import (
	"context"
	"fmt"
	"runtime/cgo"
	"sync"
	"time"
	"unsafe"

	"rkllm-go/internal/engine"
)

type rkEngine struct{}

// New returns the engine for this build's runtime.
func New() engine.Engine { return rkEngine{} }

func (rkEngine) Version() string       { return runtimeVersion }
func (rkEngine) PerRunMaxTokens() bool { return C.rk_v131() == 1 }

type handle struct {
	h      C.LLMHandle
	mu     sync.Mutex // guards destroyed
	repeat float32
}

func (rkEngine) Load(p engine.Params) (engine.Handle, error) {
	cp := C.CString(p.ModelPath) // not freed: runtime may retain it
	skip := 0
	if p.SkipSpecial {
		skip = 1
	}
	h := &handle{repeat: p.RepeatPenalty}
	ret := C.rk_init(&h.h, cp, C.int(p.MaxContextLen), C.int(p.MaxNewTokens), C.int(p.TopK), C.float(p.TopP),
		C.float(p.Temperature), C.float(p.RepeatPenalty), C.int(skip), C.int(p.BaseDomainID))
	if ret != 0 {
		return nil, fmt.Errorf("rkllm_init(%s, domain %d) returned %d", p.ModelPath, p.BaseDomainID, int(ret))
	}
	return h, nil
}

// runState is reached from the C callback through a cgo.Handle in userdata.
type runState struct {
	start     time.Time
	onToken   func(engine.Token)
	stopAfter int
	stop      context.CancelFunc
	res       engine.RunResult
}

func (h *handle) Run(ctx context.Context, req engine.RunRequest) (engine.RunResult, error) {
	runCtx, stop := context.WithCancel(ctx)
	defer stop()
	rs := &runState{onToken: req.OnToken, stopAfter: req.StopAfter, stop: stop}
	ud := cgo.NewHandle(rs)
	defer ud.Delete()

	cPrompt := C.CString(req.Prompt)
	defer C.free(unsafe.Pointer(cPrompt))
	var cPath *C.char
	if req.SaveCachePath != "" {
		cPath = C.CString(req.SaveCachePath)
		defer C.free(unsafe.Pointer(cPath))
	}
	keep := 0
	if req.KeepHistory {
		keep = 1
	}

	done := make(chan struct{})
	aborted := make(chan bool, 1)
	go func() {
		select {
		case <-runCtx.Done():
			C.rkllm_abort(h.h)
			aborted <- true
		case <-done:
			aborted <- false
		}
	}()

	rs.start = time.Now()
	samp, topk, topp, temp := 0, 1, float32(1), float32(0)
	if sp := req.Sampling; sp != nil {
		samp, topk, topp, temp = 1, sp.TopK, sp.TopP, sp.Temperature
	}
	ret := C.rk_run(h.h, cPrompt, C.int(keep), C.int(req.MaxNewTokens), cPath,
		C.int(samp), C.int(topk), C.float(topp), C.float(temp), C.float(h.repeat), C.uintptr_t(ud))
	close(done)
	rs.res.Aborted = <-aborted && ctx.Err() != nil
	rs.res.WallMs = float64(time.Since(rs.start).Microseconds()) / 1000
	if ret != 0 && !rs.res.Aborted {
		return rs.res, fmt.Errorf("%w: ret %d", engine.ErrRunFailed, int(ret))
	}
	return rs.res, nil
}

func (h *handle) ClearKV() error {
	if r := C.rk_clear(h.h); r != 0 {
		return fmt.Errorf("rkllm_clear_kv_cache: %d", int(r))
	}
	return nil
}

func (h *handle) KVSize() int { return int(C.rk_kv(h.h)) }

func (h *handle) LoadPromptCache(path string) error {
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	if r := C.rkllm_load_prompt_cache(h.h, p); r != 0 {
		return fmt.Errorf("rkllm_load_prompt_cache(%s): %d", path, int(r))
	}
	return nil
}

func (h *handle) ReleasePromptCache() error {
	if r := C.rkllm_release_prompt_cache(h.h); r != 0 {
		return fmt.Errorf("rkllm_release_prompt_cache: %d", int(r))
	}
	return nil
}

func (h *handle) SetChatTemplate(system, prefix, postfix string) error {
	a, b, c := C.CString(system), C.CString(prefix), C.CString(postfix) // not freed: runtime may retain
	if r := C.rkllm_set_chat_template(h.h, a, b, c); r != 0 {
		return fmt.Errorf("rkllm_set_chat_template: %d", int(r))
	}
	return nil
}

func (h *handle) Destroy() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.h != nil {
		C.rkllm_destroy(h.h)
		h.h = nil
	}
}
