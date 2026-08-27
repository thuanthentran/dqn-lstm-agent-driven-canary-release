package chaos

import (
	"runtime"
	"time"
)

type ResourceEngine struct {
	cpuEval    *PatternEvaluator
	memEval    *PatternEvaluator
	rpsTracker *RPSTracker
	startTime  time.Time
}

func NewResourceEngine(cpuEval, memEval *PatternEvaluator, tracker *RPSTracker) *ResourceEngine {
	return &ResourceEngine{
		cpuEval:    cpuEval,
		memEval:    memEval,
		rpsTracker: tracker,
		startTime:  time.Now(),
	}
}

func (r *ResourceEngine) Start() {
	if r.cpuEval != nil {
		go r.cpuLoop()
	}
	if r.memEval != nil {
		go r.memLoop()
	}
}

func (r *ResourceEngine) cpuLoop() {
	runtime.LockOSThread()
	cycleDuration := 10 * time.Millisecond

	for {
		uptime := time.Since(r.startTime).Seconds()
		rps := r.rpsTracker.GetRPS()

		val, _ := r.cpuEval.Evaluate(uptime, 0, rps)
		if val < 0 {
			val = 0
		}
		if val > 100 {
			val = 100
		}

		workFraction := val / 100.0
		workDuration := time.Duration(float64(cycleDuration) * workFraction)
		sleepDuration := cycleDuration - workDuration

		deadline := time.Now().Add(workDuration)
		for time.Now().Before(deadline) {
			// busy-loop
		}
		if sleepDuration > 0 {
			time.Sleep(sleepDuration)
		}
	}
}

func (r *ResourceEngine) memLoop() {
	var memoryChunks [][]byte
	currentMB := 0.0

	for {
		uptime := time.Since(r.startTime).Seconds()
		rps := r.rpsTracker.GetRPS()

		val, release := r.memEval.Evaluate(uptime, 0, rps)

		if val > currentMB {
			// Allocate difference
			diffMB := int(val - currentMB)
			if diffMB > 0 {
				chunk := make([]byte, diffMB*1024*1024)
				memoryChunks = append(memoryChunks, chunk)
				currentMB += float64(diffMB)
			}
		} else if release && val < currentMB {
			// Free memory to reach val (Safe Mode)
			keepMB := int(val)
			var newChunks [][]byte
			accumulated := 0
			for _, chunk := range memoryChunks {
				chunkSize := len(chunk) / (1024 * 1024)
				if accumulated+chunkSize <= keepMB {
					newChunks = append(newChunks, chunk)
					accumulated += chunkSize
				} else {
					break
				}
			}
			memoryChunks = newChunks
			currentMB = float64(accumulated)
			runtime.GC()
		}

		time.Sleep(1 * time.Second)
	}
}
