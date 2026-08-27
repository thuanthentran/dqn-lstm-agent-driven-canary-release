package chaos

import (
	"encoding/json"
	"os"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

type GroundTruthLog struct {
	Timestamp    string  `json:"timestamp"`
	RunID        string  `json:"run_id"`
	TargetSignal string  `json:"target_signal"`
	Action       string  `json:"action"` // e.g. "delay_injected", "error_injected", "phase_switch", "start"
	Value        float64 `json:"value"`  // e.g. 50.5 (ms), 1.0 (error), 100 (cpu)
	Details      string  `json:"details"`
	NodeName     string  `json:"node_name"`
	GitCommit    string  `json:"git_commit"`
	ProcessStart string  `json:"process_start_time"`
}

type Logger struct {
	file         *os.File
	encoder      *json.Encoder
	mu           sync.Mutex
	runID        string
	nodeName     string
	gitCommit    string
	processStart string
}

func NewLogger(runID string) *Logger {
	logPath := "/var/log/chaos/ground_truth.jsonl"
	
	// Create directory if not exists
	os.MkdirAll("/var/log/chaos", 0755)

	f, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		logrus.WithError(err).Warnf("chaos: cannot open ground truth log file at %s", logPath)
		// We still return a valid object, just with nil file, to avoid panics
		f = nil
	}

	return &Logger{
		file:         f,
		encoder:      json.NewEncoder(f),
		runID:        runID,
		nodeName:     os.Getenv("NODE_NAME"),
		gitCommit:    os.Getenv("GIT_COMMIT"),
		processStart: time.Now().UTC().Format(time.RFC3339Nano),
	}
}

func (l *Logger) Log(signal, action string, value float64, details string) {
	if l.file == nil {
		return
	}

	entry := GroundTruthLog{
		Timestamp:    time.Now().UTC().Format(time.RFC3339Nano),
		RunID:        l.runID,
		TargetSignal: signal,
		Action:       action,
		Value:        value,
		Details:      details,
		NodeName:     l.nodeName,
		GitCommit:    l.gitCommit,
		ProcessStart: l.processStart,
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	_ = l.encoder.Encode(entry)
}

func (l *Logger) Close() {
	if l.file != nil {
		l.file.Close()
	}
}
