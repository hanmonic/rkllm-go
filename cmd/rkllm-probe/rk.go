//go:build linux && cgo

package main

/*
// Build against 1.3.1: CGO_CFLAGS="-I<dir with 1.3.1 rkllm.h> -DRKLLM_V131" CGO_LDFLAGS="-L<1.3.1 lib dir>"
// Build against 1.2.3: CGO_CFLAGS="-I/opt/rkllm/include"                  CGO_LDFLAGS="-L/opt/rkllm/runtime"
#cgo CFLAGS: -I${SRCDIR}/compat
#cgo LDFLAGS: -lrkllmrt

#include <stdlib.h>
#include <string.h>
#include "rkllm.h"

extern int probeGoCallback(RKLLMResult* result, void* userdata, LLMCallState state);

static int probe_cb(RKLLMResult* r, void* ud, LLMCallState st) { return probeGoCallback(r, ud, st); }

static int probe_version131(void) {
#ifdef RKLLM_V131
	return 1;
#else
	return 0;
#endif
}

static int probe_init(LLMHandle* h, const char* path, int ctx, int maxnew, int nkeep, float temp, int skip_special) {
	RKLLMParam p = rkllm_createDefaultParam();
	p.model_path = path;
	p.max_context_len = ctx;
	p.max_new_tokens = maxnew;
	p.top_k = 1;
	p.top_p = 1.0f;
	p.temperature = temp;
	p.repeat_penalty = 1.0f;
	p.frequency_penalty = 0.0f;
	p.presence_penalty = 0.0f;
	p.mirostat = 0;
	p.n_keep = nkeep;
	p.skip_special_token = skip_special ? true : false;
	p.is_async = false;
	p.extend_param.base_domain_id = 0;
	p.extend_param.embed_flash = 1;
	p.extend_param.enabled_cpus_num = 4;
	p.extend_param.enabled_cpus_mask = 0xF0; // A76 big cores
	p.extend_param.n_batch = 1;
	p.extend_param.use_cross_attn = 0;
#ifdef RKLLM_V131
	// Heap-allocated and intentionally leaked: we do not know whether the
	// runtime copies the struct or keeps the pointer.
	RKLLMCallback* cb = (RKLLMCallback*)calloc(1, sizeof(RKLLMCallback));
	cb->result_callback = probe_cb;
	return rkllm_init(h, &p, cb);
#else
	return rkllm_init(h, &p, probe_cb);
#endif
}

static int probe_run(LLMHandle h, const char* role, int think, int itype, const char* prompt,
                     int32_t* toks, size_t ntok, int keep, int maxnew, int save, const char* cpath,
                     float temp, void* ud) {
	RKLLMInput in;
	memset(&in, 0, sizeof(in));
	in.role = role;
	in.enable_thinking = think ? true : false;
	in.input_type = (RKLLMInputType)itype;
	if (itype == RKLLM_INPUT_TOKEN) {
		in.token_input.input_ids = toks;
		in.token_input.n_tokens = ntok;
	} else {
		in.prompt_input = prompt;
	}
	RKLLMInferParam ip;
	memset(&ip, 0, sizeof(ip));
	ip.mode = RKLLM_INFER_GENERATE;
	ip.keep_history = keep;
	RKLLMPromptCacheParam pc;
	memset(&pc, 0, sizeof(pc));
	if (save) {
		pc.save_prompt_cache = 1;
		pc.prompt_cache_path = cpath;
		ip.prompt_cache_params = &pc;
	}
#ifdef RKLLM_V131
	RKLLMSamplingParam sp;
	memset(&sp, 0, sizeof(sp));
	sp.top_k = 1;
	sp.top_p = 1.0f;
	sp.temperature = temp;
	sp.repeat_penalty = 1.0f;
	ip.sampling_params = &sp;
	ip.max_new_tokens = maxnew;
#endif
	return rkllm_run(h, &in, &ip, ud);
}

static int probe_kv(LLMHandle h, int* ret) {
	int sz[8];
	memset(sz, 0, sizeof(sz));
	*ret = rkllm_get_kv_cache_size(h, sz);
	return sz[0];
}

static int probe_clear(LLMHandle h, int keep_sys, int use_range, int start, int end) {
	if (!use_range) return rkllm_clear_kv_cache(h, keep_sys, NULL, NULL);
	int s[1] = {start};
	int e[1] = {end};
	return rkllm_clear_kv_cache(h, keep_sys, s, e);
}
*/
import "C"

import (
	"runtime/cgo"
	"time"
	"unsafe"
)

const (
	stNormal  = 0
	stWaiting = 1
	stFinish  = 2
	stError   = 3
)

func is131() bool { return C.probe_version131() == 1 }

type Perf struct {
	PrefillMs   float64 `json:"prefill_ms"`
	PrefillTok  int     `json:"prefill_tokens"`
	GenerateMs  float64 `json:"generate_ms"`
	GenerateTok int     `json:"generate_tokens"`
	MemMB       float64 `json:"mem_mb"`
}

type Event struct {
	State int     `json:"st"`
	Tok   int32   `json:"id"`
	Text  string  `json:"t,omitempty"`
	Perf  *Perf   `json:"perf,omitempty"`
	AtMs  float64 `json:"at_ms"`
}

type runState struct {
	start      time.Time
	firstTok   time.Duration
	pauseAfter int
	pauseOn    []int32
	nNormal    int
	nWaiting   int
	nFinish    int
	nError     int
	text       []byte
	ids        []int32
	perf       Perf
	perfAt     []int // states in which non-zero perf was observed
	events     []Event
	keepEvents bool
	paused     bool
}

type Model struct {
	h    C.LLMHandle
	path string
}

func Init(path string, ctx, maxNew, nKeep int, temp float64, skipSpecial bool) (*Model, int) {
	ss := 0
	if skipSpecial {
		ss = 1
	}
	m := &Model{path: path}
	cp := C.CString(path)
	// path deliberately not freed: runtime may retain it
	ret := C.probe_init(&m.h, cp, C.int(ctx), C.int(maxNew), C.int(nKeep), C.float(temp), C.int(ss))
	return m, int(ret)
}

type RunArgs struct {
	Role       string
	NilRole    bool
	Think      bool
	Prompt     string
	Tokens     []int32
	UseTokens  bool
	KeepHist   int
	MaxNew     int
	SavePath   string
	PauseAfter int
	PauseOn    []int32
	Temp       float64
	KeepEvents bool
}

func (m *Model) Run(a RunArgs) (*runState, int, time.Duration) {
	rs := &runState{pauseAfter: a.PauseAfter, pauseOn: a.PauseOn, keepEvents: a.KeepEvents, firstTok: -1}
	h := cgo.NewHandle(rs)
	defer h.Delete()

	var cRole *C.char
	if !a.NilRole {
		cRole = C.CString(a.Role)
		defer C.free(unsafe.Pointer(cRole))
	}
	cPrompt := C.CString(a.Prompt)
	defer C.free(unsafe.Pointer(cPrompt))
	var cTok *C.int32_t
	itype := 0
	if a.UseTokens {
		itype = 1
		n := len(a.Tokens)
		if n == 0 {
			n = 1
		}
		cTok = (*C.int32_t)(C.malloc(C.size_t(4 * n)))
		defer C.free(unsafe.Pointer(cTok))
		dst := unsafe.Slice((*int32)(unsafe.Pointer(cTok)), n)
		copy(dst, a.Tokens)
	}
	save := 0
	var cPath *C.char
	if a.SavePath != "" {
		save = 1
		cPath = C.CString(a.SavePath)
		defer C.free(unsafe.Pointer(cPath))
	}
	think := 0
	if a.Think {
		think = 1
	}
	rs.start = time.Now()
	ret := C.probe_run(m.h, cRole, C.int(think), C.int(itype), cPrompt, cTok, C.size_t(len(a.Tokens)),
		C.int(a.KeepHist), C.int(a.MaxNew), C.int(save), cPath, C.float(a.Temp), unsafe.Pointer(uintptr(h)))
	return rs, int(ret), time.Since(rs.start)
}

func (m *Model) KV() (int, int) {
	var ret C.int
	n := C.probe_kv(m.h, &ret)
	return int(n), int(ret)
}

func (m *Model) Clear(keepSys int, useRange bool, start, end int) int {
	ur := 0
	if useRange {
		ur = 1
	}
	return int(C.probe_clear(m.h, C.int(keepSys), C.int(ur), C.int(start), C.int(end)))
}

func (m *Model) SetTemplate(sys, pre, post string) int {
	a, b, c := C.CString(sys), C.CString(pre), C.CString(post)
	// not freed: runtime may retain
	return int(C.rkllm_set_chat_template(m.h, a, b, c))
}

func (m *Model) SetFunctionTools(sys, tools, toolResp string) int {
	a, b, c := C.CString(sys), C.CString(tools), C.CString(toolResp)
	// not freed: runtime may retain
	return int(C.rkllm_set_function_tools(m.h, a, b, c))
}

func (m *Model) LoadCache(path string) int {
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	return int(C.rkllm_load_prompt_cache(m.h, p))
}

func (m *Model) ReleaseCache() int { return int(C.rkllm_release_prompt_cache(m.h)) }
func (m *Model) Abort() int        { return int(C.rkllm_abort(m.h)) }
func (m *Model) IsRunning() int    { return int(C.rkllm_is_running(m.h)) }
func (m *Model) Destroy() int      { return int(C.rkllm_destroy(m.h)) }
