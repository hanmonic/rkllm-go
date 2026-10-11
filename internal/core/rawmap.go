package core

import (
	"container/list"
	"strings"
	"sync"
)

// RawMap remembers the exact generated text behind each assistant response
// the server returned, keyed by what the client will send back (content +
// tool calls). Rendering history with prompt.Message.Raw makes the prompt
// byte-identical to the KV cache contents, which the append strategy needs
// and which keeps prefix reuse exact.
type RawMap struct {
	mu  sync.Mutex
	max int
	ll  *list.List
	m   map[string]*list.Element
}

type rawEntry struct{ key, raw string }

func NewRawMap(max int) *RawMap {
	return &RawMap{max: max, ll: list.New(), m: map[string]*list.Element{}}
}

// RawCall is the part of a tool call that identifies it.
type RawCall struct{ Name, Arguments string }

// RawKey builds the lookup key for an assistant message.
func RawKey(content string, calls []RawCall) string {
	var b strings.Builder
	b.WriteString(strings.TrimSpace(content))
	for _, c := range calls {
		b.WriteString("\x00")
		b.WriteString(c.Name)
		b.WriteString("\x00")
		b.WriteString(strings.TrimSpace(c.Arguments))
	}
	return b.String()
}

func (r *RawMap) Put(key, raw string) {
	if raw == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if e, ok := r.m[key]; ok {
		e.Value.(*rawEntry).raw = raw
		r.ll.MoveToFront(e)
		return
	}
	r.m[key] = r.ll.PushFront(&rawEntry{key, raw})
	for r.ll.Len() > r.max {
		e := r.ll.Back()
		r.ll.Remove(e)
		delete(r.m, e.Value.(*rawEntry).key)
	}
}

func (r *RawMap) Get(key string) (string, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if e, ok := r.m[key]; ok {
		r.ll.MoveToFront(e)
		return e.Value.(*rawEntry).raw, true
	}
	return "", false
}
