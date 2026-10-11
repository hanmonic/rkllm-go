package core

import (
	"errors"
	"sync"
)

// ErrQueueFull is returned by Queue.Push when depth jobs are already waiting.
var ErrQueueFull = errors.New("queue full")

// ErrQueueClosed is returned by Queue.Push after Close.
var ErrQueueClosed = errors.New("queue closed")

// Queue is a bounded FIFO of jobs for one model handle. depth counts waiting
// jobs only; the job being executed (between Pop and Done) is "active".
// Jobs whose context is cancelled while waiting are skipped by Pop and can
// be removed early with Remove.
type Queue struct {
	mu     sync.Mutex
	cond   *sync.Cond
	items  []*Job
	depth  int
	active int
	closed bool
}

func NewQueue(depth int) *Queue {
	if depth < 0 {
		depth = 0
	}
	q := &Queue{depth: depth}
	q.cond = sync.NewCond(&q.mu)
	return q
}

// Push enqueues j or fails with ErrQueueFull / ErrQueueClosed.
func (q *Queue) Push(j *Job) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return ErrQueueClosed
	}
	q.purgeLocked()
	if len(q.items) >= q.depth {
		return ErrQueueFull
	}
	q.items = append(q.items, j)
	q.cond.Signal()
	return nil
}

// purgeLocked drops cancelled jobs so they do not hold queue slots.
func (q *Queue) purgeLocked() {
	out := q.items[:0]
	for _, j := range q.items {
		if j.Ctx.Err() == nil {
			out = append(out, j)
		}
	}
	for i := len(out); i < len(q.items); i++ {
		q.items[i] = nil
	}
	q.items = out
}

// Pop blocks until a live job is available (marking it active) or the queue
// is closed and empty (returns nil).
func (q *Queue) Pop() *Job {
	q.mu.Lock()
	defer q.mu.Unlock()
	for {
		for len(q.items) > 0 {
			j := q.items[0]
			q.items[0] = nil
			q.items = q.items[1:]
			if j.Ctx.Err() != nil {
				continue
			}
			q.active++
			return j
		}
		if q.closed {
			return nil
		}
		q.cond.Wait()
	}
}

// Done marks the active job finished.
func (q *Queue) Done() {
	q.mu.Lock()
	q.active--
	q.mu.Unlock()
}

// Remove drops j if it is still waiting. It reports whether it was found.
func (q *Queue) Remove(j *Job) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	for i, x := range q.items {
		if x == j {
			q.items = append(q.items[:i], q.items[i+1:]...)
			return true
		}
	}
	return false
}

// Len is the number of waiting jobs (cancelled ones included until purged).
func (q *Queue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.purgeLocked()
	return len(q.items)
}

// Active is the number of jobs between Pop and Done (0 or 1 with one worker).
func (q *Queue) Active() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.active
}

// Idle reports no waiting and no active jobs.
func (q *Queue) Idle() bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.purgeLocked()
	return len(q.items) == 0 && q.active == 0
}

// Close wakes Pop; remaining jobs are returned so the caller can fail them.
func (q *Queue) Close() []*Job {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.closed = true
	rest := q.items
	q.items = nil
	q.cond.Broadcast()
	return rest
}
