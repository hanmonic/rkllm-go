package core

import (
	"os"
	"sort"
	"sync"
	"time"
)

// FileStore tracks prompt-cache files (parked sessions, snapshots, shared
// prefixes) under a byte budget and evicts least-recently-used files.
type FileStore struct {
	mu      sync.Mutex
	budget  int64
	total   int64
	entries map[string]*fileEntry
	now     func() time.Time
	remove  func(string) error
}

type fileEntry struct {
	Path     string    `json:"path"`
	Size     int64     `json:"size"`
	LastUsed time.Time `json:"last_used"`
	Kind     string    `json:"kind"` // session | snapshot | prefix
}

func NewFileStore(budget int64) *FileStore {
	return &FileStore{budget: budget, entries: map[string]*fileEntry{}, now: time.Now, remove: os.Remove}
}

// Add registers (or re-registers after an overwrite) path with its current
// size and evicts LRU files, never path itself, until the total fits. It
// returns the evicted paths. If path alone exceeds the budget it is removed
// too and Add returns false.
func (s *FileStore) Add(path, kind string) (ok bool, evicted []string) {
	st, err := os.Stat(path)
	if err != nil {
		s.Forget(path)
		return false, nil
	}
	return s.AddSized(path, kind, st.Size())
}

// AddSized is Add with a known size (for tests).
func (s *FileStore) AddSized(path, kind string, size int64) (bool, []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.entries[path]; ok {
		s.total -= e.Size
	}
	if size > s.budget {
		delete(s.entries, path)
		_ = s.remove(path)
		return false, []string{path}
	}
	s.entries[path] = &fileEntry{Path: path, Size: size, LastUsed: s.now(), Kind: kind}
	s.total += size
	var evicted []string
	for s.total > s.budget {
		var victim *fileEntry
		for _, e := range s.entries {
			if e.Path == path {
				continue
			}
			if victim == nil || e.LastUsed.Before(victim.LastUsed) {
				victim = e
			}
		}
		if victim == nil {
			break
		}
		s.total -= victim.Size
		delete(s.entries, victim.Path)
		_ = s.remove(victim.Path)
		evicted = append(evicted, victim.Path)
	}
	return true, evicted
}

// Has reports whether path is registered (i.e. not evicted).
func (s *FileStore) Has(path string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.entries[path]
	return ok
}

// Touch marks path as recently used.
func (s *FileStore) Touch(path string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.entries[path]; ok {
		e.LastUsed = s.now()
	}
}

// Delete removes path from disk and the index.
func (s *FileStore) Delete(path string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.entries[path]; ok {
		s.total -= e.Size
		delete(s.entries, path)
	}
	_ = s.remove(path)
}

// Forget drops path from the index without touching disk.
func (s *FileStore) Forget(path string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.entries[path]; ok {
		s.total -= e.Size
		delete(s.entries, path)
	}
}

// Total bytes registered.
func (s *FileStore) Total() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.total
}

// Budget in bytes.
func (s *FileStore) Budget() int64 { return s.budget }

// List returns entries, most recently used first.
func (s *FileStore) List() []fileEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]fileEntry, 0, len(s.entries))
	for _, e := range s.entries {
		out = append(out, *e)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].LastUsed.After(out[j].LastUsed) })
	return out
}
