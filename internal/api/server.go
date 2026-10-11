package api

import (
	"encoding/json"
	"net/http"
	"sort"
	"time"

	"rkllm-go/internal/core"
)

// Defaults are server-wide sampling defaults applied when a request does not
// override them.
type Defaults struct {
	TopK        int
	TopP        float32
	Temperature float32
}

// Handler holds shared state for all HTTP handlers.
type Handler struct {
	pool     *core.Pool
	defaults Defaults
	started  time.Time
}

func NewHandler(pool *core.Pool, d Defaults) *Handler {
	return &Handler{pool: pool, defaults: d, started: time.Now()}
}

// Register wires all routes into mux.
func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/chat/completions", h.handleChat)
	mux.HandleFunc("GET /v1/models", h.handleModels)
	mux.HandleFunc("POST /v1/embeddings", h.handleEmbeddings)
	mux.HandleFunc("GET /healthz", h.handleHealth)
	mux.HandleFunc("GET /debug/state", h.handleState)
}

func (h *Handler) handleModels(w http.ResponseWriter, r *http.Request) {
	cfg := h.pool.Config()
	now := h.started.Unix()
	var out []ModelInfo
	for _, name := range cfg.Names() {
		m := cfg.Models[name]
		out = append(out, ModelInfo{ID: name, Object: "model", OwnedBy: "rkllm", Created: now,
			Loaded: h.pool.Loaded(name), Family: string(m.Family), ContextLength: m.MaxContextLen, CacheStrategy: string(m.Strategy)})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	writeJSON(w, http.StatusOK, ModelsResponse{Object: "list", Data: out})
}

func (h *Handler) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "runtime": h.pool.Engine().Version(), "uptime_s": int(time.Since(h.started).Seconds())})
}

func (h *Handler) handleState(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, h.pool.State())
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg, errType, code string) {
	writeJSON(w, status, ErrorResponse{Error: APIError{Message: msg, Type: errType, Code: code}})
}
