package core

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"strings"
	"sync"
	"time"
)

// Session is one conversation on one model.
type Session struct {
	ID    string
	Model string
	// LastPrompt is the full prompt of the session's last request.
	LastPrompt string
	// File is the session's prompt-cache file (parked or snapshot), "" if none.
	// FilePrompt is the prompt text it covers.
	File       string
	FilePrompt string
	LastUsed   time.Time
	Turns      int
}

// SessionStore holds the sessions of one model.
type SessionStore struct {
	mu   sync.Mutex
	max  int
	byID map[string]*Session
	now  func() time.Time
}

func NewSessionStore(max int) *SessionStore {
	if max <= 0 {
		max = 1000
	}
	return &SessionStore{max: max, byID: map[string]*Session{}, now: time.Now}
}

// CommonPrefixLen returns the length in bytes of the common prefix of a and b.
func CommonPrefixLen(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

// Resolve returns the session for a request. An explicit id wins (created if
// new). Otherwise the session whose last prompt is the longest prefix of
// prompt is chosen (a continuation: append-stable templates guarantee the
// previous request's prompt is a prefix of the next one); ties go to the most
// recently used. A partial match is not enough: two conversations that share
// a long system prompt + tools would merge. Otherwise a new session with a
// generated id is created. created reports a new session.
func (s *SessionStore) Resolve(model, explicitID, prompt string) (sess *Session, created bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if explicitID != "" {
		if x, ok := s.byID[explicitID]; ok {
			return x, false
		}
		return s.createLocked(model, explicitID), true
	}
	if m := s.matchLocked(prompt); m != nil {
		return m, false
	}
	return s.createLocked(model, newSessionID()), true
}

// Match returns the best prefix-matching session or nil (no creation).
func (s *SessionStore) Match(prompt string) *Session {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.matchLocked(prompt)
}

func (s *SessionStore) matchLocked(prompt string) *Session {
	var best *Session
	bestLen := -1
	for _, x := range s.byID {
		if x.LastPrompt == "" {
			continue
		}
		if !strings.HasPrefix(prompt, x.LastPrompt) {
			continue
		}
		l := len(x.LastPrompt)
		if l > bestLen || (l == bestLen && x.LastUsed.After(best.LastUsed)) {
			best, bestLen = x, l
		}
	}
	return best
}

func (s *SessionStore) createLocked(model, id string) *Session {
	if len(s.byID) >= s.max {
		// drop the least recently used session record (its file, if any, is
		// left to the FileStore's LRU)
		var old *Session
		for _, x := range s.byID {
			if old == nil || x.LastUsed.Before(old.LastUsed) {
				old = x
			}
		}
		delete(s.byID, old.ID)
	}
	x := &Session{ID: id, Model: model, LastUsed: s.now()}
	s.byID[id] = x
	return x
}

// Update records a finished request (called by the slot worker).
func (s *SessionStore) Update(x *Session, prompt string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	x.LastPrompt = prompt
	x.LastUsed = s.now()
	x.Turns++
}

// SetFile records the session's cache file.
func (s *SessionStore) SetFile(x *Session, file, covers string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	x.File, x.FilePrompt = file, covers
}

// GetFile returns the session's cache file and the prompt it covers.
func (s *SessionStore) GetFile(x *Session) (string, string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return x.File, x.FilePrompt
}

// SessionInfo is a snapshot for /debug/state.
type SessionInfo struct {
	ID          string    `json:"id"`
	Turns       int       `json:"turns"`
	PromptBytes int       `json:"prompt_bytes"`
	File        string    `json:"file,omitempty"`
	LastUsed    time.Time `json:"last_used"`
}

// List returns all sessions, most recent first.
func (s *SessionStore) List() []SessionInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]SessionInfo, 0, len(s.byID))
	for _, x := range s.byID {
		out = append(out, SessionInfo{ID: x.ID, Turns: x.Turns, PromptBytes: len(x.LastPrompt), File: x.File, LastUsed: x.LastUsed})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].LastUsed.After(out[j].LastUsed) })
	return out
}

func newSessionID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return "s-" + hex.EncodeToString(b[:])
}

// fileSafe maps a session id to a file-name-safe token.
func fileSafe(id string) string {
	ok := len(id) > 0 && len(id) <= 64
	for i := 0; ok && i < len(id); i++ {
		c := id[i]
		ok = c == '-' || c == '_' || c == '.' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
	}
	if ok && !strings.HasPrefix(id, ".") {
		return id
	}
	h := sha256.Sum256([]byte(id))
	return "h-" + hex.EncodeToString(h[:12])
}

// hashString is a short stable hash for cache file names.
func hashString(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:12])
}

// AppendDelta decides the keep_history=1 input: if prompt extends the text
// that is in the KV cache, it returns the remainder (which starts with the
// stop token + "\n" of the previous assistant turn, e.g. "<|im_end|>\n").
func AppendDelta(cached, prompt string) (delta string, ok bool) {
	if cached == "" || len(prompt) <= len(cached) || !strings.HasPrefix(prompt, cached) {
		return "", false
	}
	return prompt[len(cached):], true
}

// CachedText is the text that ends up in the KV cache after a run: the input
// that was sent plus the generated tokens except the last one (the runtime
// never feeds the final token, normally the stop token, back into the cache).
func CachedText(before, sent string, tokens []string) string {
	var b strings.Builder
	b.WriteString(before)
	b.WriteString(sent)
	for i := 0; i+1 < len(tokens); i++ {
		b.WriteString(tokens[i])
	}
	return b.String()
}
