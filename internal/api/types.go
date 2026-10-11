package api

import (
	"encoding/json"
	"strings"

	"rkllm-go/internal/core"
)

// Tool represents a function tool in the OpenAI tools array.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction is the function definition within a Tool.
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ToolCall is a single tool invocation produced by the model. Index is set
// only in streaming deltas.
type ToolCall struct {
	Index    *int             `json:"index,omitempty"`
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction carries the name and JSON-encoded arguments.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Content accepts a string, null, or an array of content parts (text parts
// are concatenated) and always marshals as a string.
type Content string

func (c *Content) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		*c = Content(s)
		return nil
	}
	if strings.TrimSpace(string(b)) == "null" {
		*c = ""
		return nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(b, &parts); err != nil {
		return err
	}
	var sb strings.Builder
	for _, p := range parts {
		if p.Type == "text" || p.Type == "" {
			sb.WriteString(p.Text)
		}
	}
	*c = Content(sb.String())
	return nil
}

// Message is an OpenAI-style chat message.
type Message struct {
	Role             string     `json:"role"`
	Content          Content    `json:"content"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
	Name             string     `json:"name,omitempty"`
}

// StreamOptions is the OpenAI stream_options object.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// ChatCompletionRequest mirrors the OpenAI /v1/chat/completions request body
// plus rkllm-go extensions (enable_thinking, top_k, cache).
type ChatCompletionRequest struct {
	Model               string         `json:"model"`
	Messages            []Message      `json:"messages"`
	Stream              bool           `json:"stream"`
	StreamOptions       *StreamOptions `json:"stream_options,omitempty"`
	Temperature         *float32       `json:"temperature,omitempty"`
	TopP                *float32       `json:"top_p,omitempty"`
	TopK                *int           `json:"top_k,omitempty"`
	MaxTokens           *int           `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int           `json:"max_completion_tokens,omitempty"`
	EnableThinking      *bool          `json:"enable_thinking,omitempty"`
	ChatTemplateKwargs  *struct {
		EnableThinking *bool `json:"enable_thinking,omitempty"`
	} `json:"chat_template_kwargs,omitempty"`
	Tools      []Tool `json:"tools,omitempty"`
	ToolChoice any    `json:"tool_choice,omitempty"`
	User       string `json:"user,omitempty"`
	// Cache=false: full clear + cold prefill for this request (benchmarking).
	Cache *bool `json:"cache,omitempty"`
}

// ChatCompletionResponse is the non-streaming response body.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

// Choice is a single candidate in a non-streaming response.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// ChatCompletionChunk is one SSE event in a streaming response.
type ChatCompletionChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
	Usage   *Usage        `json:"usage,omitempty"`
}

// ChunkChoice is a single delta in a streaming chunk.
type ChunkChoice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// Delta carries the incremental content for a streaming chunk.
type Delta struct {
	Role             string     `json:"role,omitempty"`
	Content          string     `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

// PromptTokensDetails is the OpenAI prompt_tokens_details object.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// Usage is the OpenAI usage object plus the rkllm block.
type Usage struct {
	PromptTokens        int                 `json:"prompt_tokens"`
	CompletionTokens    int                 `json:"completion_tokens"`
	TotalTokens         int                 `json:"total_tokens"`
	PromptTokensDetails PromptTokensDetails `json:"prompt_tokens_details"`
	RKLLM               *core.Usage         `json:"rkllm,omitempty"`
}

// ModelsResponse is the body for GET /v1/models.
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// ModelInfo is a single entry in the models list.
type ModelInfo struct {
	ID            string `json:"id"`
	Object        string `json:"object"`
	OwnedBy       string `json:"owned_by"`
	Created       int64  `json:"created"`
	Loaded        bool   `json:"loaded"`
	Family        string `json:"family"`
	ContextLength int    `json:"context_length"`
	CacheStrategy string `json:"cache_strategy"`
}

// ErrorResponse wraps an API error in the OpenAI error envelope.
type ErrorResponse struct {
	Error APIError `json:"error"`
}

// APIError is the inner error object.
type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}
