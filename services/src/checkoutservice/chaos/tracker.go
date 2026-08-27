package chaos

import (
	"sync"
	"time"
)

type RPSTracker struct {
	mu      sync.Mutex
	buckets [5]int64
	currIdx int
	total   int64
}

func NewRPSTracker() *RPSTracker {
	t := &RPSTracker{}
	go t.rotateLoop()
	return t
}

func (t *RPSTracker) RecordRequest() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buckets[t.currIdx]++
}

func (t *RPSTracker) GetRPS() float64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	
	// Calculate sum of all 5 buckets
	var sum int64
	for i := 0; i < 5; i++ {
		sum += t.buckets[i]
	}
	
	return float64(sum) / 5.0
}

func (t *RPSTracker) rotateLoop() {
	ticker := time.NewTicker(1 * time.Second)
	for range ticker.C {
		t.mu.Lock()
		t.currIdx = (t.currIdx + 1) % 5
		t.buckets[t.currIdx] = 0 // Clear the new bucket
		t.mu.Unlock()
	}
}
