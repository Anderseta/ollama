package server

import (
	"testing"
	"time"
)

func TestResidencyValuePreservesExpensiveHotRunner(t *testing.T) {
	now := time.Unix(10000, 0)
	hot := &runnerRef{loadDuration: time.Minute, lastUsed: now, useCount: 20, totalSize: 10 << 30, sessionDuration: -1}
	cold := &runnerRef{loadDuration: time.Second, lastUsed: now.Add(-5 * time.Minute), useCount: 1, totalSize: 1 << 30, sessionDuration: -1}
	if runnerEvictionValue(hot, now) <= runnerEvictionValue(cold, now) {
		t.Fatal("hot expensive runner should be retained")
	}
	hot.sessionDuration = 0
	if runnerEvictionValue(hot, now) != 0 {
		t.Fatal("explicit unload must remain immediate")
	}
	if runnerEvictionValue(&runnerRef{}, now) != -1 {
		t.Fatal("unknown runners retain the legacy comparison")
	}
}
