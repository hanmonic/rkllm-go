package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"rkllm-go/internal/config"
	"rkllm-go/internal/core"
	"rkllm-go/internal/engine"
	"rkllm-go/internal/prompt"
	"rkllm-go/internal/toolparse"
)

// convertMessages maps API messages to prompt messages. Assistant messages
// whose content + tool calls equal a response this server generated are
// rendered from the stored raw generation (prompt.Message.Raw), so the
// prompt stays byte-identical to what is in the KV cache.
func convertMessages(in []Message, raw *core.RawMap) []prompt.Message {
	out := make([]prompt.Message, len(in))
	for i, m := range in {
		pm := prompt.Message{Role: m.Role, Content: string(m.Content), Reasoning: m.ReasoningContent, ToolCallID: m.ToolCallID, Name: m.Name}
		var rc []core.RawCall
		for _, tc := range m.ToolCalls {
			pm.ToolCalls = append(pm.ToolCalls, prompt.ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
			rc = append(rc, core.RawCall{Name: tc.Function.Name, Arguments: tc.Function.Arguments})
		}
		if m.Role == "assistant" && raw != nil {
			if r, ok := raw.Get(core.RawKey(string(m.Content), rc)); ok {
				pm.Raw = r
			}
		}
		out[i] = pm
	}
	return out
}

// staticPrefix returns the part of the prompt fixed by the leading system
// message and the tools: the common prefix of two renders that differ only in
// the first user message.
func staticPrefix(t prompt.Template, msgs []prompt.Message, tools []json.RawMessage, o prompt.Options) string {
	k := 0
	if len(msgs) > 0 && (msgs[0].Role == "system" || msgs[0].Role == "developer") {
		k = 1
	}
	if k == 0 && len(tools) == 0 {
		return ""
	}
	base := append([]prompt.Message(nil), msgs[:k]...)
	a := t.Format(append(base, prompt.Message{Role: "user", Content: "\x01"}), tools, o)
	b := t.Format(append(base, prompt.Message{Role: "user", Content: "\x02"}), tools, o)
	return a[:core.CommonPrefixLen(a, b)]
}

type output struct {
	content, reasoning string
	calls              []ToolCall
}

func parseOutput(fam prompt.Family, text string, tools []json.RawMessage, thinking bool) output {
	o := output{content: text}
	if len(tools) > 0 {
		content, parsed, err := toolparse.Parse(string(fam), text, toolparse.SchemasFromTools(tools))
		if err != nil {
			log.Printf("tool call parse: %v", err)
		}
		o.content = content
		for _, c := range parsed {
			o.calls = append(o.calls, ToolCall{ID: c.ID, Type: c.Type, Function: ToolCallFunction{Name: c.Function.Name, Arguments: c.Function.Arguments}})
		}
	}
	if thinking || strings.Contains(o.content, "</think>") {
		o.reasoning, o.content = toolparse.SplitReasoning(o.content)
	}
	return o
}

// splitLive splits a partial generation into reasoning and content for
// streaming, holding back a trailing partial "</think>" and whitespace.
func splitLive(text string, thinking bool) (reasoning, content string) {
	if !thinking {
		return "", text
	}
	if i := strings.Index(text, "</think>"); i >= 0 {
		r := strings.TrimSpace(strings.TrimPrefix(strings.TrimLeft(text[:i], "\n"), "<think>"))
		return r, strings.TrimLeft(text[i+len("</think>"):], "\n")
	}
	r := strings.TrimLeft(strings.TrimPrefix(strings.TrimLeft(text, "\n"), "<think>"), "\n")
	for n := len("</think>") - 1; n > 0; n-- {
		if strings.HasSuffix(r, "</think>"[:n]) {
			r = r[:len(r)-n]
			break
		}
	}
	return strings.TrimRight(r, " \n\t"), ""
}

func toUsage(u core.Usage) *Usage {
	return &Usage{PromptTokens: u.PromptTokens, CompletionTokens: u.CompletionTokens, TotalTokens: u.PromptTokens + u.CompletionTokens,
		PromptTokensDetails: PromptTokensDetails{CachedTokens: u.CachedTokens}, RKLLM: &u}
}

func (h *Handler) sampling(req *ChatCompletionRequest) *engine.Sampling {
	if req.Temperature == nil && req.TopK == nil && req.TopP == nil {
		return nil
	}
	s := &engine.Sampling{TopK: h.defaults.TopK, TopP: h.defaults.TopP, Temperature: h.defaults.Temperature}
	if req.Temperature != nil {
		s.Temperature = *req.Temperature
	}
	if req.TopK != nil {
		s.TopK = *req.TopK
	}
	if req.TopP != nil {
		s.TopP = *req.TopP
	}
	if s.Temperature <= 0 {
		s.Temperature, s.TopK = 0, 1 // greedy
	}
	if s.TopP <= 0 {
		s.TopP = 1
	}
	return s
}

func (h *Handler) handleChat(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error(), "invalid_request_error", "bad_request")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages must not be empty", "invalid_request_error", "bad_request")
		return
	}
	cfg := h.pool.Config()
	name := req.Model
	if name == "" {
		name = cfg.Default
	}
	m, ok := cfg.Models[name]
	if !ok {
		writeError(w, http.StatusNotFound, fmt.Sprintf("model %q not found; available: %v", name, cfg.Names()), "invalid_request_error", "model_not_found")
		return
	}
	thinking := false
	if req.EnableThinking != nil {
		thinking = *req.EnableThinking
	} else if req.ChatTemplateKwargs != nil && req.ChatTemplateKwargs.EnableThinking != nil {
		thinking = *req.ChatTemplateKwargs.EnableThinking
	}
	opts := prompt.Options{EnableThinking: thinking}
	raw := h.pool.Raw(name)
	msgs := convertMessages(req.Messages, raw)
	tools := make([]json.RawMessage, 0, len(req.Tools))
	for _, t := range req.Tools {
		if b, err := json.Marshal(t); err == nil {
			tools = append(tools, b)
		}
	}
	full := m.Template.Format(msgs, tools, opts)
	sessID := r.Header.Get("X-Session-Id")
	if sessID == "" {
		sessID = req.User
	}
	maxNew := 0
	if req.MaxCompletionTokens != nil {
		maxNew = *req.MaxCompletionTokens
	} else if req.MaxTokens != nil {
		maxNew = *req.MaxTokens
	}
	job := &core.Job{
		Ctx: r.Context(), Model: name, Prompt: full, StaticPrefix: staticPrefix(m.Template, msgs, tools, opts),
		SessionID: sessID, NoCache: req.Cache != nil && !*req.Cache, MaxNewTokens: maxNew,
		Sampling: h.sampling(&req), Start: start, Stream: core.NewStream(),
	}
	if err := h.pool.Submit(job); err != nil {
		switch {
		case errors.Is(err, core.ErrQueueFull):
			writeError(w, http.StatusServiceUnavailable, "queue full for model "+name, "server_error", "server_busy")
		case errors.Is(err, core.ErrUnknownModel):
			writeError(w, http.StatusNotFound, err.Error(), "invalid_request_error", "model_not_found")
		default:
			writeError(w, http.StatusServiceUnavailable, err.Error(), "server_error", "unavailable")
		}
		return
	}

	// wait for the worker to pick the job up and resolve the session
	var sess string
	for {
		started, s, _, done, res := job.Stream.Snapshot(1 << 30)
		if done && !started {
			h.fail(w, res.Err)
			return
		}
		if started {
			sess = s
			break
		}
		select {
		case <-job.Stream.Notify():
		case <-r.Context().Done():
			return
		}
	}
	w.Header().Set("X-Session-Id", sess)
	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	if req.Stream {
		h.stream(w, r, job, &req, m, tools, thinking, raw, id)
	} else {
		h.buffered(w, r, job, m, tools, thinking, raw, id)
	}
}

func (h *Handler) fail(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, core.ErrContextLength):
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "context_length_exceeded")
	default:
		writeError(w, http.StatusInternalServerError, fmt.Sprint(err), "server_error", "inference_error")
	}
}

func waitDone(r *http.Request, job *core.Job) (core.Result, bool) {
	for {
		_, _, _, done, res := job.Stream.Snapshot(1 << 30)
		if done {
			return res, true
		}
		select {
		case <-job.Stream.Notify():
		case <-r.Context().Done():
			return core.Result{}, false
		}
	}
}

func (h *Handler) remember(raw *core.RawMap, o output, text string) {
	var rc []core.RawCall
	for _, c := range o.calls {
		rc = append(rc, core.RawCall{Name: c.Function.Name, Arguments: c.Function.Arguments})
	}
	raw.Put(core.RawKey(o.content, rc), text)
}

func (h *Handler) buffered(w http.ResponseWriter, r *http.Request, job *core.Job, m *config.Model, tools []json.RawMessage, thinking bool, raw *core.RawMap, id string) {
	res, ok := waitDone(r, job)
	if !ok {
		return
	}
	if res.Err != nil {
		h.fail(w, res.Err)
		return
	}
	o := parseOutput(m.Family, res.Text, tools, thinking)
	h.remember(raw, o, res.Text)
	msg := Message{Role: "assistant", Content: Content(o.content), ReasoningContent: o.reasoning, ToolCalls: o.calls}
	finish := res.Usage.FinishReason
	if len(o.calls) > 0 {
		finish = "tool_calls"
	}
	writeJSON(w, http.StatusOK, ChatCompletionResponse{
		ID: id, Object: "chat.completion", Created: time.Now().Unix(), Model: m.Name,
		Choices: []Choice{{Index: 0, Message: msg, FinishReason: finish}},
		Usage:   toUsage(res.Usage),
	})
}

func (h *Handler) stream(w http.ResponseWriter, r *http.Request, job *core.Job, req *ChatCompletionRequest, m *config.Model, tools []json.RawMessage, thinking bool, raw *core.RawMap, id string) {
	flusher, _ := w.(http.Flusher)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	created := time.Now().Unix()
	send := func(c ChatCompletionChunk) {
		c.ID, c.Object, c.Created, c.Model = id, "chat.completion.chunk", created, m.Name
		if c.Choices == nil {
			c.Choices = []ChunkChoice{}
		}
		b, _ := json.Marshal(c)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if flusher != nil {
			flusher.Flush()
		}
	}
	delta := func(d Delta) { send(ChatCompletionChunk{Choices: []ChunkChoice{{Delta: d}}}) }
	delta(Delta{Role: "assistant"})

	var text strings.Builder
	n, sentC, sentR := 0, 0, 0
	for {
		_, _, texts, done, res := job.Stream.Snapshot(n)
		n += len(texts)
		for _, t := range texts {
			text.WriteString(t)
		}
		if done {
			if res.Err != nil {
				b, _ := json.Marshal(ErrorResponse{Error: APIError{Message: res.Err.Error(), Type: "server_error", Code: "inference_error"}})
				fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", b)
				if flusher != nil {
					flusher.Flush()
				}
				return
			}
			o := parseOutput(m.Family, res.Text, tools, thinking)
			h.remember(raw, o, res.Text)
			_, liveC := splitLive(res.Text, thinking)
			if len(o.reasoning) > sentR {
				delta(Delta{ReasoningContent: o.reasoning[sentR:]})
			}
			if sentC <= len(liveC) && strings.HasPrefix(o.content, liveC[:sentC]) && len(o.content) > sentC {
				delta(Delta{Content: o.content[sentC:]})
			}
			finish := res.Usage.FinishReason
			if len(o.calls) > 0 {
				finish = "tool_calls"
				calls := make([]ToolCall, len(o.calls))
				for i, c := range o.calls {
					idx := i
					c.Index = &idx
					calls[i] = c
				}
				delta(Delta{ToolCalls: calls})
			}
			send(ChatCompletionChunk{Choices: []ChunkChoice{{Delta: Delta{}, FinishReason: &finish}}})
			if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
				send(ChatCompletionChunk{Choices: []ChunkChoice{}, Usage: toUsage(res.Usage)})
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
			if flusher != nil {
				flusher.Flush()
			}
			return
		}
		if len(texts) > 0 {
			rs, cs := splitLive(text.String(), thinking)
			if len(rs) > sentR {
				delta(Delta{ReasoningContent: rs[sentR:]})
				sentR = len(rs)
			}
			safe := len(cs)
			if len(tools) > 0 {
				safe, _ = toolparse.Holdback(string(m.Family), cs)
			}
			if safe > sentC {
				delta(Delta{Content: cs[sentC:safe]})
				sentC = safe
			}
		}
		select {
		case <-job.Stream.Notify():
		case <-r.Context().Done():
			return
		}
	}
}
