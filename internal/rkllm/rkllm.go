package rkllm

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -lrkllmrt -Wl,-rpath,/opt/rkllm/runtime

#include "rkllm_rt.h"
#include <string.h>
#include <stdlib.h>

extern int rkllmGoCallback(RKLLMResult* result, void* userdata, LLMCallState state);

static void initPromptInput(RKLLMInput* inp, const char* role, _Bool enable_thinking, const char* prompt) {
    memset(inp, 0, sizeof(*inp));
    inp->role            = role;
    inp->enable_thinking = enable_thinking;
    inp->input_type      = RKLLM_INPUT_PROMPT;
    inp->prompt_input    = prompt;
}

// mode: 0=RKLLM_INFER_GENERATE, 1=RKLLM_INFER_GET_LAST_HIDDEN_LAYER
static void initInferParam(RKLLMInferParam* p, int mode, int keep_history) {
    memset(p, 0, sizeof(*p));
    p->mode         = mode;
    p->keep_history = keep_history;
}
*/
import "C"
import (
	"context"
	"fmt"
	"log"
	"runtime/cgo"
	"sync"
	"unsafe"
)

// Engine owns a single RKLLM model handle and serialises all inference.
// Models are hot-swapped on demand: if a request names a different model than
// the one currently loaded, the engine destroys the current handle and inits
// the new one before running inference. Only one operation at a time.
type Engine struct {
	handle     C.LLMHandle
	loadedPath string
	baseCfg    Config
	mu         sync.Mutex
}

// NewEngine creates an Engine with the given sampling config.
// If cfg.ModelPath is non-empty the model is loaded immediately (warmup);
// otherwise the first request triggers loading.
func NewEngine(cfg Config) (*Engine, error) {
	e := &Engine{baseCfg: cfg}
	if cfg.ModelPath != "" {
		if err := e.loadLocked(cfg.ModelPath); err != nil {
			return nil, err
		}
	}
	return e, nil
}

// Destroy releases the model handle. Call during graceful shutdown.
// Safe to call concurrently — acquires the mutex.
func (e *Engine) Destroy() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle != nil {
		C.rkllm_destroy(e.handle)
		e.handle = nil
		e.loadedPath = ""
	}
}

// LoadedModel returns the path of the currently loaded model (empty if none).
func (e *Engine) LoadedModel() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.loadedPath
}

// Run starts generation in a background goroutine and returns a channel of
// token events. The channel is closed when inference completes.
// Returns ErrBusy if another operation is in progress.
// Cancelling ctx aborts the ongoing inference.
func (e *Engine) Run(ctx context.Context, inp RunInput, modelPath string) (<-chan TokenEvent, error) {
	if !e.mu.TryLock() {
		return nil, ErrBusy
	}

	ch := make(chan TokenEvent, 64)
	h := cgo.NewHandle(ch)

	go func() {
		defer e.mu.Unlock()
		defer h.Delete()
		defer close(ch)

		if err := e.ensureModel(modelPath); err != nil {
			ch <- TokenEvent{State: StateError}
			return
		}

		cPrompt := C.CString(inp.Prompt)
		defer C.free(unsafe.Pointer(cPrompt))
		cRole := C.CString(inp.Role)
		defer C.free(unsafe.Pointer(cRole))

		var input C.RKLLMInput
		C.initPromptInput(&input, cRole, C.bool(inp.EnableThinking), cPrompt)

		var inferParam C.RKLLMInferParam
		C.initInferParam(&inferParam, 0, 0) // GENERATE, keep_history=0

		abortDone := make(chan struct{})
		go func() {
			select {
			case <-ctx.Done():
				C.rkllm_abort(e.handle)
			case <-abortDone:
			}
		}()
		defer close(abortDone)

		C.rkllm_run(e.handle, &input, &inferParam, unsafe.Pointer(uintptr(h)))
	}()

	return ch, nil
}

// Embed runs inference in GET_LAST_HIDDEN_LAYER mode and returns the hidden-state
// vector for the input text. The returned slice has length embd_size (the model's
// embedding dimension); it corresponds to the first token's representation, matching
// the reference Python implementation.
// Returns ErrBusy if another operation is in progress.
func (e *Engine) Embed(ctx context.Context, text, modelPath string) ([]float32, error) {
	if !e.mu.TryLock() {
		return nil, ErrBusy
	}

	ch := make(chan TokenEvent, 4)
	h := cgo.NewHandle(ch)

	go func() {
		defer e.mu.Unlock()
		defer h.Delete()
		defer close(ch)

		if err := e.ensureModel(modelPath); err != nil {
			ch <- TokenEvent{State: StateError}
			return
		}

		cText := C.CString(text)
		defer C.free(unsafe.Pointer(cText))
		cRole := C.CString("user")
		defer C.free(unsafe.Pointer(cRole))

		var input C.RKLLMInput
		C.initPromptInput(&input, cRole, C.bool(false), cText)

		var inferParam C.RKLLMInferParam
		C.initInferParam(&inferParam, 1, 0) // GET_LAST_HIDDEN_LAYER, keep_history=0

		abortDone := make(chan struct{})
		go func() {
			select {
			case <-ctx.Done():
				C.rkllm_abort(e.handle)
			case <-abortDone:
			}
		}()
		defer close(abortDone)

		C.rkllm_run(e.handle, &input, &inferParam, unsafe.Pointer(uintptr(h)))
	}()

	var embedding []float32
	var inferErr error
	for ev := range ch {
		switch ev.State {
		case StateFinish:
			embedding = ev.Embedding
		case StateError:
			inferErr = fmt.Errorf("embedding inference failed")
		}
	}
	if inferErr != nil {
		return nil, inferErr
	}
	if len(embedding) == 0 {
		return nil, ErrEmbedNotSupported
	}
	return embedding, nil
}

// ensureModel hot-swaps the model if path differs from the currently loaded one.
// Must be called with e.mu held.
func (e *Engine) ensureModel(path string) error {
	if e.loadedPath == path {
		return nil
	}
	log.Printf("loading model %s (was: %q)", path, e.loadedPath)
	return e.loadLocked(path)
}

// loadLocked destroys any existing handle and inits a new one for path.
// Must be called with e.mu held (or from constructor before engine is shared).
func (e *Engine) loadLocked(path string) error {
	if e.handle != nil {
		C.rkllm_destroy(e.handle)
		e.handle = nil
		e.loadedPath = ""
	}

	param := C.rkllm_createDefaultParam()

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	param.model_path = cPath

	param.max_context_len    = C.int32_t(e.baseCfg.MaxContextLen)
	param.max_new_tokens     = C.int32_t(e.baseCfg.MaxNewTokens)
	param.top_k              = C.int32_t(e.baseCfg.TopK)
	param.top_p              = C.float(e.baseCfg.TopP)
	param.temperature        = C.float(e.baseCfg.Temperature)
	param.repeat_penalty     = C.float(e.baseCfg.RepeatPenalty)
	param.n_keep             = C.int32_t(e.baseCfg.NKeep)
	param.skip_special_token = C.bool(true)
	param.is_async           = C.bool(false)

	param.extend_param.base_domain_id    = 0
	param.extend_param.embed_flash       = 1
	param.extend_param.enabled_cpus_num  = 4
	param.extend_param.enabled_cpus_mask = C.uint32_t(0xF0)
	param.extend_param.n_batch           = 1
	param.extend_param.use_cross_attn    = 0

	ret := C.rkllm_init(&e.handle, &param, C.LLMResultCallback(C.rkllmGoCallback))
	if ret != 0 {
		return fmt.Errorf("rkllm_init failed for %s: %d", path, int(ret))
	}
	e.loadedPath = path
	return nil
}
