// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package chaos provides fault injection capabilities for the checkoutservice.
//
// All behaviour is controlled exclusively via environment variables, making it
// transparent, reproducible, and suitable for academic benchmarking.
//
// # Environment variables
//
//	CHAOS_ENABLED          – "true" to activate fault injection (default: "false")
//	CHAOS_RANDOM_SEED      – int64 seed for the RNG; fixes the random sequence
//	                          so that two controllers see identical fault inputs
//	                          (default: 42)
//
//	CHAOS_LATENCY_MS_MIN   – lower bound of injected latency in ms (default: 0)
//	CHAOS_LATENCY_MS_MAX   – upper bound; 0 means disabled (default: 0)
//	CHAOS_LATENCY_DIST     – distribution to sample from:
//	                            "none"     – disabled
//	                            "fixed"    – constant value = (min+max)/2
//	                            "uniform"  – Uniform[min, max]
//	                            "normal"   – Normal(μ=(min+max)/2, σ=(max-min)/6)
//	                            "lognormal"– LogNormal matched to [min,max] range
//	                          (default: "lognormal")
//
//	CHAOS_ERROR_RATE       – probability [0.0, 1.0] that a request returns a
//	                          gRPC UNAVAILABLE error (default: 0.0)
//	CHAOS_ERROR_CODE       – gRPC status code string to return on injected error
//	                          e.g. "UNAVAILABLE", "INTERNAL", "RESOURCE_EXHAUSTED"
//	                          (default: "UNAVAILABLE")
//
//	CHAOS_CPU_PERCENT      – target CPU load from a background busy-loop goroutine,
//	                          0–100 (default: 0, disabled)
//
//	CHAOS_MEM_ALLOC_MB     – megabytes to hold in memory; allocated once on startup
//	                          and never freed (simulates a slow memory leak baseline)
//	                          (default: 0, disabled)
package chaos

import (
	"context"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var log = logrus.New()

// Config holds all parsed chaos parameters.
type Config struct {
	Enabled      bool
	LatencyMin   float64 // ms
	LatencyMax   float64 // ms
	LatencyDist  string  // none | fixed | uniform | normal | lognormal
	ErrorRate    float64 // [0.0, 1.0]
	ErrorCode    codes.Code
	CPUPercent   int // 0-100
	MemAllocMB   int // MB
	rng          *rand.Rand
}

// LoadFromEnv reads all CHAOS_* environment variables and returns a Config.
// It also starts background goroutines for CPU load and memory allocation.
func LoadFromEnv() *Config {
	cfg := &Config{}

	cfg.Enabled = strings.ToLower(getEnv("CHAOS_ENABLED", "false")) == "true"
	if !cfg.Enabled {
		log.Info("chaos: disabled (CHAOS_ENABLED != true)")
		return cfg
	}

	seed := parseInt64Env("CHAOS_RANDOM_SEED", 42)
	cfg.rng = rand.New(rand.NewSource(seed)) //nolint:gosec // intentional for reproducibility

	cfg.LatencyMin = parseFloat64Env("CHAOS_LATENCY_MS_MIN", 0)
	cfg.LatencyMax = parseFloat64Env("CHAOS_LATENCY_MS_MAX", 0)
	cfg.LatencyDist = strings.ToLower(getEnv("CHAOS_LATENCY_DIST", "lognormal"))

	cfg.ErrorRate = parseFloat64Env("CHAOS_ERROR_RATE", 0.0)
	cfg.ErrorCode = parseGRPCCode(getEnv("CHAOS_ERROR_CODE", "UNAVAILABLE"))

	cfg.CPUPercent = parseIntEnv("CHAOS_CPU_PERCENT", 0)
	cfg.MemAllocMB = parseIntEnv("CHAOS_MEM_ALLOC_MB", 0)

	log.WithFields(logrus.Fields{
		"seed":         seed,
		"latency_dist": cfg.LatencyDist,
		"latency_min":  cfg.LatencyMin,
		"latency_max":  cfg.LatencyMax,
		"error_rate":   cfg.ErrorRate,
		"error_code":   cfg.ErrorCode,
		"cpu_percent":  cfg.CPUPercent,
		"mem_alloc_mb": cfg.MemAllocMB,
	}).Warn("chaos: ENABLED — fault injection is active")

	// Start background side-effects
	if cfg.CPUPercent > 0 {
		go cfg.runCPULoad()
	}
	if cfg.MemAllocMB > 0 {
		go cfg.allocMemory()
	}

	return cfg
}

// Apply executes the chaos logic for a single incoming request.
// It should be called at the very beginning of each gRPC handler (via interceptor).
func (c *Config) Apply(ctx context.Context) error {
	if !c.Enabled {
		return nil
	}

	// 1. Latency injection
	if c.LatencyMax > 0 && c.LatencyDist != "none" {
		delay := c.sampleLatency()
		if delay > 0 {
			select {
			case <-ctx.Done():
				return status.FromContextError(ctx.Err()).Err()
			case <-time.After(time.Duration(delay * float64(time.Millisecond))):
			}
		}
	}

	// 2. Error injection (Bernoulli trial)
	if c.ErrorRate > 0 && c.rng.Float64() < c.ErrorRate {
		return status.Errorf(c.ErrorCode, "chaos: injected fault (rate=%.2f)", c.ErrorRate)
	}

	return nil
}

// UnaryServerInterceptor returns a gRPC interceptor that applies chaos to
// every unary RPC. Register it with grpc.ChainUnaryInterceptor.
func (c *Config) UnaryServerInterceptor() grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		if err := c.Apply(ctx); err != nil {
			log.WithField("method", info.FullMethod).Warnf("chaos: injecting fault: %v", err)
			return nil, err
		}
		return handler(ctx, req)
	}
}

// ─── Internal helpers ────────────────────────────────────────────────────────

// sampleLatency returns a latency value in ms drawn from the configured distribution.
func (c *Config) sampleLatency() float64 {
	lo, hi := c.LatencyMin, c.LatencyMax
	if lo > hi {
		lo, hi = hi, lo
	}
	mid := (lo + hi) / 2.0

	switch c.LatencyDist {
	case "fixed":
		return mid

	case "uniform":
		// Uniform[lo, hi]
		return lo + c.rng.Float64()*(hi-lo)

	case "normal":
		// Normal(μ=mid, σ=(hi-lo)/6) — ~99.7% of samples within [lo, hi]
		sigma := (hi - lo) / 6.0
		if sigma == 0 {
			return mid
		}
		v := mid + c.rng.NormFloat64()*sigma
		return math.Max(lo, math.Min(hi, v)) // clamp to [lo, hi]

	case "lognormal":
		// Fit a log-normal so that the 5th and 95th percentiles map to lo and hi.
		// This produces a realistic heavy-tailed latency distribution.
		if lo <= 0 {
			lo = 1.0 // log-normal requires positive support
		}
		logLo := math.Log(lo)
		logHi := math.Log(hi)
		mu := (logLo + logHi) / 2.0
		sigma := (logHi - logLo) / (2.0 * 1.645) // 90th-percentile spread
		if sigma <= 0 {
			return math.Exp(mu)
		}
		sample := math.Exp(mu + c.rng.NormFloat64()*sigma)
		return math.Max(lo, math.Min(hi*2, sample)) // soft cap at 2×hi for tail

	default:
		return 0
	}
}

// runCPULoad keeps one goroutine busy for CPUPercent/100 of the time.
// Uses a work/sleep cycle pinned to a single OS thread.
func (c *Config) runCPULoad() {
	runtime.LockOSThread()
	workFraction := float64(c.CPUPercent) / 100.0
	cycleDuration := 10 * time.Millisecond
	workDuration := time.Duration(float64(cycleDuration) * workFraction)
	sleepDuration := cycleDuration - workDuration

	log.Warnf("chaos: CPU load goroutine started at %d%%", c.CPUPercent)
	for {
		deadline := time.Now().Add(workDuration)
		for time.Now().Before(deadline) {
			// busy-loop
		}
		if sleepDuration > 0 {
			time.Sleep(sleepDuration)
		}
	}
}

// allocMemory allocates MemAllocMB megabytes and holds them forever,
// simulating a static memory-leak baseline.
func (c *Config) allocMemory() {
	mb := c.MemAllocMB
	log.Warnf("chaos: allocating %d MB and holding (memory leak simulation)", mb)
	_ = make([]byte, mb*1024*1024)
	// Block forever so GC cannot reclaim the allocation.
	select {}
}

// ─── Env-parsing utilities ───────────────────────────────────────────────────

func getEnv(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}

func parseFloat64Env(key string, defaultVal float64) float64 {
	v := os.Getenv(key)
	if v == "" {
		return defaultVal
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		log.Warnf("chaos: invalid float for %s=%q, using default %.2f", key, v, defaultVal)
		return defaultVal
	}
	return f
}

func parseInt64Env(key string, defaultVal int64) int64 {
	v := os.Getenv(key)
	if v == "" {
		return defaultVal
	}
	i, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		log.Warnf("chaos: invalid int64 for %s=%q, using default %d", key, v, defaultVal)
		return defaultVal
	}
	return i
}

func parseIntEnv(key string, defaultVal int) int {
	return int(parseInt64Env(key, int64(defaultVal)))
}

func parseGRPCCode(s string) codes.Code {
	switch strings.ToUpper(s) {
	case "OK":
		return codes.OK
	case "CANCELLED":
		return codes.Canceled
	case "UNKNOWN":
		return codes.Unknown
	case "INVALID_ARGUMENT":
		return codes.InvalidArgument
	case "DEADLINE_EXCEEDED":
		return codes.DeadlineExceeded
	case "NOT_FOUND":
		return codes.NotFound
	case "ALREADY_EXISTS":
		return codes.AlreadyExists
	case "PERMISSION_DENIED":
		return codes.PermissionDenied
	case "RESOURCE_EXHAUSTED":
		return codes.ResourceExhausted
	case "FAILED_PRECONDITION":
		return codes.FailedPrecondition
	case "ABORTED":
		return codes.Aborted
	case "OUT_OF_RANGE":
		return codes.OutOfRange
	case "UNIMPLEMENTED":
		return codes.Unimplemented
	case "INTERNAL":
		return codes.Internal
	case "UNAVAILABLE":
		return codes.Unavailable
	case "DATA_LOSS":
		return codes.DataLoss
	case "UNAUTHENTICATED":
		return codes.Unauthenticated
	default:
		log.Warnf("chaos: unknown gRPC code %q, defaulting to UNAVAILABLE", s)
		return codes.Unavailable
	}
}
