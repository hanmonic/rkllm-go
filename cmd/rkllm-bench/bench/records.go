package bench

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// TurnRecord is one request/response of an agent loop (one JSONL line).
type TurnRecord struct {
	RunID            string   `json:"run_id"`
	Model            string   `json:"model"`
	Scenario         string   `json:"scenario"`
	Task             string   `json:"task"`
	Mode             string   `json:"mode"` // "on" | "off"
	Smoke            bool     `json:"smoke,omitempty"`
	SessionID        string   `json:"session_id"`
	RespSessionID    string   `json:"resp_session_id,omitempty"`
	Turn             int      `json:"turn"` // 0-based
	Start            string   `json:"start"`
	WallMs           float64  `json:"wall_ms"`
	ClientTTFBMs     float64  `json:"client_ttfb_ms"`
	ClientTTFTMs     float64  `json:"client_ttft_ms"` // 0 when not streaming
	PromptTokens     int      `json:"prompt_tokens"`
	CompletionTokens int      `json:"completion_tokens"`
	CachedTokens     int      `json:"cached_tokens"`
	HitRatio         float64  `json:"hit_ratio"`  // cached / prompt
	DecodeTPS        float64  `json:"decode_tps"` // generate_tokens / generate_ms
	RK               *RKUsage `json:"rkllm,omitempty"`
	FinishReason     string   `json:"finish_reason,omitempty"`
	ToolCalls        []string `json:"tool_calls,omitempty"`
	Malformed        int      `json:"malformed"`
	UnknownTools     int      `json:"unknown_tools"`
	ContentChars     int      `json:"content_chars"`
	ToolResultChars  int      `json:"tool_result_chars"`
	Truncated        bool     `json:"tool_output_truncated,omitempty"`
	EstContext       int      `json:"est_context_tokens"`
	Solved           bool     `json:"solved_after_turn"`
	Error            string   `json:"error,omitempty"`
}

// TaskRecord is the outcome of one task run (one JSONL line).
type TaskRecord struct {
	RunID            string         `json:"run_id"`
	Model            string         `json:"model"`
	Scenario         string         `json:"scenario"`
	Task             string         `json:"task"`
	Mode             string         `json:"mode"`
	Smoke            bool           `json:"smoke,omitempty"`
	SessionID        string         `json:"session_id"`
	Turns            int            `json:"turns"`
	Pass             bool           `json:"pass"`
	Score            float64        `json:"score"`
	Detail           map[string]any `json:"detail,omitempty"`
	StopReason       string         `json:"stop_reason"` // done|task_stop|max_turns|ctx_exhausted|error
	WallMs           float64        `json:"wall_ms"`     // sum of this task's request wall times
	ElapsedMs        float64        `json:"elapsed_ms"`  // first request start to last response end
	TimeToSuccessMs  *float64       `json:"time_to_success_ms"`
	SolvedTurn       int            `json:"solved_turn"` // -1 if never
	PromptTokens     int            `json:"prompt_tokens"`
	PrefillTokens    int            `json:"prefill_tokens"`
	CachedTokens     int            `json:"cached_tokens"`
	CompletionTokens int            `json:"completion_tokens"`
	MaxContext       int            `json:"max_context_tokens"`
	CtxLimit         int            `json:"ctx_limit"`
	Malformed        int            `json:"malformed"`
	UnknownTools     int            `json:"unknown_tools"`
	Errors           int            `json:"errors"`
	Error            string         `json:"error,omitempty"`
}

// DoneMarker is written when a (model, scenario, mode) combo completes.
type DoneMarker struct {
	Model    string `json:"model"`
	Scenario string `json:"scenario"`
	Mode     string `json:"mode"`
	Smoke    bool   `json:"smoke"`
	RunID    string `json:"run_id"`
	Tasks    int    `json:"tasks"`
	Finished string `json:"finished"`
}

var unsafeRe = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

func safe(s string) string { return unsafeRe.ReplaceAllString(s, "_") }

// ComboKey is the file stem of a (model, scenario, mode) combo.
func ComboKey(model, scenario, mode string, smoke bool) string {
	k := safe(model) + "__" + safe(scenario) + "__" + safe(mode)
	if smoke {
		k += "__smoke"
	}
	return k
}

// Layout of an output directory.
func turnsPath(dir, key string) string { return filepath.Join(dir, "turns", key+".jsonl") }
func tasksPath(dir, key string) string { return filepath.Join(dir, "tasks", key+".jsonl") }
func donePath(dir, key string) string  { return filepath.Join(dir, "done", key+".json") }
func statePath(dir, key string) string { return filepath.Join(dir, "state", key+".json") }

// IsDone reports whether a combo already completed in dir.
func IsDone(dir, key string) bool {
	_, err := os.Stat(donePath(dir, key))
	return err == nil
}

func writeJSONAtomic(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(b, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// jsonl is a mutex-guarded append-only JSONL writer that fsyncs per line, so
// an interrupted multi-hour run loses at most the in-flight request.
type jsonl struct {
	mu sync.Mutex
	f  *os.File
}

func createJSONL(path string) (*jsonl, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	f, err := os.Create(path) // truncates partial output of an interrupted combo
	if err != nil {
		return nil, err
	}
	return &jsonl{f: f}, nil
}

func (j *jsonl) Write(v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	if _, err := j.f.Write(append(b, '\n')); err != nil {
		return err
	}
	return j.f.Sync()
}

func (j *jsonl) Close() error { return j.f.Close() }

func readJSONL[T any](path string) ([]T, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []T
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 16*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var v T
		if err := json.Unmarshal([]byte(line), &v); err != nil {
			continue // tolerate a torn last line
		}
		out = append(out, v)
	}
	return out, sc.Err()
}

// LoadDir reads every turn and task record in dir. Only completed combos
// (with a done marker) are included unless includePartial is set.
func LoadDir(dir string, includePartial bool) ([]TurnRecord, []TaskRecord, error) {
	files, err := filepath.Glob(filepath.Join(dir, "tasks", "*.jsonl"))
	if err != nil {
		return nil, nil, err
	}
	sort.Strings(files)
	var turns []TurnRecord
	var tasks []TaskRecord
	for _, f := range files {
		key := strings.TrimSuffix(filepath.Base(f), ".jsonl")
		if !includePartial && !IsDone(dir, key) {
			continue
		}
		ts, err := readJSONL[TaskRecord](f)
		if err != nil {
			return nil, nil, err
		}
		tasks = append(tasks, ts...)
		tr, err := readJSONL[TurnRecord](turnsPath(dir, key))
		if err != nil && !os.IsNotExist(err) {
			return nil, nil, err
		}
		turns = append(turns, tr...)
	}
	return turns, tasks, nil
}

func nowStr() string { return time.Now().UTC().Format(time.RFC3339Nano) }
