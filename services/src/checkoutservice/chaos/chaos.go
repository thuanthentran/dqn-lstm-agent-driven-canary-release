package chaos

import (
	"context"
	"math/rand"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type Module struct {
	config     *ChaosConfig
	logger     *Logger
	rpsTracker *RPSTracker
	startTime  time.Time
	reqCount   int64

	rng   *rand.Rand
	rngMu sync.Mutex

	latEval *PatternEvaluator
	errEval *PatternEvaluator
}

func Init() (*Module, grpc.UnaryServerInterceptor) {
	config := LoadConfig()
	if !config.Enabled {
		return nil, func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
			return handler(ctx, req)
		}
	}

	rng := rand.New(rand.NewSource(config.Seed))
	logger := NewLogger(config.RunID)
	tracker := NewRPSTracker()

	m := &Module{
		config:     config,
		logger:     logger,
		rpsTracker: tracker,
		startTime:  time.Now(),
		rng:        rng,
	}

	if sig, ok := config.Signals["latency"]; ok {
		m.latEval = NewPatternEvaluator(sig, rng, logger, "latency", config.ResourceSafe)
	}
	if sig, ok := config.Signals["error"]; ok {
		m.errEval = NewPatternEvaluator(sig, rng, logger, "error", config.ResourceSafe)
	}

	var cpuEval, memEval *PatternEvaluator
	if sig, ok := config.Signals["cpu"]; ok {
		cpuEval = NewPatternEvaluator(sig, rng, logger, "cpu", config.ResourceSafe)
	}
	if sig, ok := config.Signals["mem"]; ok {
		memEval = NewPatternEvaluator(sig, rng, logger, "mem", config.ResourceSafe)
	}

	if cpuEval != nil || memEval != nil {
		engine := NewResourceEngine(cpuEval, memEval, tracker)
		engine.Start()
	}

	logger.Log("system", "start", 0, "Chaos module initialized")
	logrus.Infof("chaos: module enabled for run %s", config.RunID)

	return m, m.interceptor()
}

func (m *Module) interceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		if strings.HasPrefix(info.FullMethod, "/grpc.health.v1.Health") {
			return handler(ctx, req)
		}

		m.rpsTracker.RecordRequest()
		count := atomic.AddInt64(&m.reqCount, 1)
		uptime := time.Since(m.startTime).Seconds()
		rps := m.rpsTracker.GetRPS()

		// 1. Latency
		if m.latEval != nil {
			delayMs, _ := m.latEval.Evaluate(uptime, count, rps)
			if delayMs > 0 {
				m.logger.Log("latency", "delay_injected", delayMs, info.FullMethod)
				select {
				case <-ctx.Done():
					return nil, status.FromContextError(ctx.Err()).Err()
				case <-time.After(time.Duration(delayMs * float64(time.Millisecond))):
				}
			}
		}

		// 2. Error rate
		if m.errEval != nil {
			errRate, _ := m.errEval.Evaluate(uptime, count, rps)
			if errRate > 0 {
				m.rngMu.Lock()
				roll := m.rng.Float64()
				m.rngMu.Unlock()
				if roll < errRate {
					m.logger.Log("error", "error_injected", 1.0, info.FullMethod)
					return nil, status.Errorf(codes.Unavailable, "chaos: injected fault (rate=%.2f)", errRate)
				}
			}
		}

		return handler(ctx, req)
	}
}

func (m *Module) Close() {
	if m != nil && m.logger != nil {
		m.logger.Close()
	}
}
