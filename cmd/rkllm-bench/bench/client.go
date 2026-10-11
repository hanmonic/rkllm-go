// Package bench drives multi-turn agent tasks against an OpenAI-compatible
// rkllm-go server over HTTP, records per-turn metrics, and builds reports.
package bench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"rkllm-go/cmd/rkllm-bench/scenarios"
)

// RKUsage is the rkllm-go usage extension (usage.rkllm).
type RKUsage struct {
	PrefillTokens  int     `json:"prefill_tokens"`
	CachedTokens   int     `json:"cached_tokens"`
	PrefillMs      float64 `json:"prefill_ms"`
	GenerateTokens int     `json:"generate_tokens"`
	GenerateMs     float64 `json:"generate_ms"`
	TTFTMs         float64 `json:"ttft_ms"`
	QueueMs        float64 `json:"queue_ms"`
	ModelLoadMs    float64 `json:"model_load_ms"`
	CacheStrategy  string  `json:"cache_strategy"`
	CacheEvent     string  `json:"cache_event"`
	SessionID      string  `json:"session_id"`
	KVCacheSize    int     `json:"kv_cache_size"`
}

// Usage is the OpenAI usage object plus the rkllm extension.
type Usage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details,omitempty"`
	RKLLM *RKUsage `json:"rkllm,omitempty"`
}

// CachedTokens prefers the OpenAI field, falling back to the rkllm one.
func (u Usage) CachedTokens() int {
	if u.PromptTokensDetails != nil {
		return u.PromptTokensDetails.CachedTokens
	}
	if u.RKLLM != nil {
		return u.RKLLM.CachedTokens
	}
	return 0
}

// ChatRequest is the request body we send.
type ChatRequest struct {
	Model         string              `json:"model"`
	Messages      []scenarios.Message `json:"messages"`
	Tools         []any               `json:"tools,omitempty"`
	Temperature   float64             `json:"temperature"`
	Seed          *int                `json:"seed,omitempty"`
	MaxTokens     int                 `json:"max_tokens,omitempty"`
	Stream        bool                `json:"stream"`
	StreamOptions *StreamOptions      `json:"stream_options,omitempty"`
	Cache         *bool               `json:"cache,omitempty"`
}

// StreamOptions is the OpenAI stream_options object.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// ChatResult is one completed request as seen by the client.
type ChatResult struct {
	Message      scenarios.Message
	FinishReason string
	Usage        Usage
	HasUsage     bool
	SessionID    string  // X-Session-Id response header
	WallMs       float64 // request start to last byte
	TTFBMs       float64 // first response byte (headers)
	TTFTMs       float64 // first non-empty content/tool delta (streaming only)
}

// Client talks to one server.
type Client struct {
	Base string
	HTTP *http.Client
}

// NewClient builds a client with no global timeout (per-request contexts are used).
func NewClient(base string) *Client {
	return &Client{Base: strings.TrimRight(base, "/"), HTTP: &http.Client{}}
}

// GetJSON fetches a JSON document.
func (c *Client) GetJSON(ctx context.Context, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.Base+path, nil)
	if err != nil {
		return err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return fmt.Errorf("GET %s: HTTP %d: %s", path, resp.StatusCode, snippet(b))
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(b, out)
}

// ModelInfo is one /v1/models entry.
type ModelInfo struct {
	ID      string         `json:"id"`
	Loaded  bool           `json:"loaded"`
	Family  string         `json:"family"`
	Context int            `json:"-"`
	Raw     map[string]any `json:"-"`
}

// Models lists /v1/models. A context length is picked up from any of a few
// plausible field names if the server exposes one.
func (c *Client) Models(ctx context.Context) ([]ModelInfo, error) {
	var body struct {
		Data []map[string]any `json:"data"`
	}
	if err := c.GetJSON(ctx, "/v1/models", &body); err != nil {
		return nil, err
	}
	var out []ModelInfo
	for _, d := range body.Data {
		m := ModelInfo{Raw: d}
		m.ID, _ = d["id"].(string)
		m.Loaded, _ = d["loaded"].(bool)
		m.Family, _ = d["family"].(string)
		for _, k := range []string{"max_context", "context_length", "max_context_len", "ctx_len", "max_model_len"} {
			if f, ok := d[k].(float64); ok && f > 0 {
				m.Context = int(f)
				break
			}
		}
		out = append(out, m)
	}
	return out, nil
}

func snippet(b []byte) string {
	s := strings.TrimSpace(string(b))
	if len(s) > 300 {
		s = s[:300] + "..."
	}
	return s
}

// Chat sends one chat completion and returns the assembled result.
func (c *Client) Chat(ctx context.Context, req ChatRequest, sessionID string) (*ChatResult, error) {
	if req.Stream {
		req.StreamOptions = &StreamOptions{IncludeUsage: true}
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	hreq, err := http.NewRequestWithContext(ctx, "POST", c.Base+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	if req.Stream {
		hreq.Header.Set("Accept", "text/event-stream")
	}
	if sessionID != "" {
		hreq.Header.Set("X-Session-Id", sessionID)
	}
	t0 := time.Now()
	resp, err := c.HTTP.Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	res := &ChatResult{TTFBMs: ms(time.Since(t0)), SessionID: resp.Header.Get("X-Session-Id")}
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return res, fmt.Errorf("HTTP %d: %s", resp.StatusCode, snippet(b))
	}
	ct := resp.Header.Get("Content-Type")
	if req.Stream && strings.Contains(ct, "text/event-stream") {
		err = readStream(resp.Body, res, t0)
	} else {
		err = readJSON(resp.Body, res)
	}
	res.WallMs = ms(time.Since(t0))
	return res, err
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func readJSON(r io.Reader, res *ChatResult) error {
	var d struct {
		Choices []struct {
			Message      scenarios.Message `json:"message"`
			FinishReason string            `json:"finish_reason"`
		} `json:"choices"`
		Usage *Usage `json:"usage"`
	}
	b, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, &d); err != nil {
		return fmt.Errorf("decode response: %v: %s", err, snippet(b))
	}
	if len(d.Choices) == 0 {
		return fmt.Errorf("response has no choices: %s", snippet(b))
	}
	res.Message = d.Choices[0].Message
	res.FinishReason = d.Choices[0].FinishReason
	if d.Usage != nil {
		res.Usage, res.HasUsage = *d.Usage, true
	}
	return nil
}

type streamChunk struct {
	Choices []struct {
		Delta struct {
			Role             string `json:"role"`
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
			ToolCalls        []struct {
				Index    *int   `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *Usage `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func readStream(r io.Reader, res *ChatResult, t0 time.Time) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 8*1024*1024)
	var content strings.Builder
	var calls []scenarios.ToolCall
	byIndex := map[int]int{}
	res.Message.Role = "assistant"
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var ch streamChunk
		if err := json.Unmarshal([]byte(data), &ch); err != nil {
			return fmt.Errorf("bad SSE chunk: %v: %s", err, snippet([]byte(data)))
		}
		if ch.Error != nil {
			return fmt.Errorf("stream error: %s", ch.Error.Message)
		}
		if ch.Usage != nil {
			res.Usage, res.HasUsage = *ch.Usage, true
		}
		for _, choice := range ch.Choices {
			d := choice.Delta
			if res.TTFTMs == 0 && (d.Content != "" || d.ReasoningContent != "" || len(d.ToolCalls) > 0) {
				res.TTFTMs = ms(time.Since(t0))
			}
			content.WriteString(d.Content)
			for _, tc := range d.ToolCalls {
				slot := -1
				if tc.Index != nil {
					if s, ok := byIndex[*tc.Index]; ok {
						slot = s
					}
				} else if len(calls) > 0 && (tc.ID == "" || tc.ID == calls[len(calls)-1].ID) {
					slot = len(calls) - 1
				}
				if slot < 0 {
					calls = append(calls, scenarios.ToolCall{ID: tc.ID, Type: tc.Type})
					slot = len(calls) - 1
					if tc.Index != nil {
						byIndex[*tc.Index] = slot
					}
				}
				cl := &calls[slot]
				if tc.ID != "" {
					cl.ID = tc.ID
				}
				if tc.Type != "" {
					cl.Type = tc.Type
				}
				cl.Function.Name += tc.Function.Name
				cl.Function.Arguments += tc.Function.Arguments
			}
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				res.FinishReason = *choice.FinishReason
			}
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	res.Message.Content = content.String()
	res.Message.ToolCalls = calls
	return nil
}
