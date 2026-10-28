package server

import (
	"context"
	"sync"
)

// contextMutex keeps ordinary mutex semantics for short critical sections, and
// lets request waiters leave a long model load immediately when cancelled. A
// single notification channel serves all waiters; no helper goroutine or poll
// is needed for each abandoned request.
type contextMutex struct {
	sync.Mutex
	waitMu sync.Mutex
	waitCh chan struct{}
}

func (m *contextMutex) Unlock() {
	m.waitMu.Lock()
	m.Mutex.Unlock()
	if m.waitCh != nil {
		close(m.waitCh)
		m.waitCh = nil
	}
	m.waitMu.Unlock()
}

func (m *contextMutex) lockContext(ctx context.Context, other context.Context) bool {
	for {
		if ctx.Err() != nil || other.Err() != nil {
			return false
		}
		if m.TryLock() {
			if ctx.Err() == nil && other.Err() == nil {
				return true
			}
			m.Unlock()
			return false
		}
		m.waitMu.Lock()
		if m.waitCh == nil {
			m.waitCh = make(chan struct{})
		}
		wait := m.waitCh
		m.waitMu.Unlock()
		// An unlock may have preceded channel registration. Retry before
		// waiting so that notification cannot be missed.
		if m.TryLock() {
			if ctx.Err() == nil && other.Err() == nil {
				return true
			}
			m.Unlock()
			return false
		}
		select {
		case <-ctx.Done():
			return false
		case <-other.Done():
			return false
		case <-wait:
		}
	}
}
