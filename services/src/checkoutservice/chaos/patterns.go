package chaos

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

type PatternEvaluator struct {
	config     SignalConfig
	rng        *rand.Rand
	logger     *Logger
	signalName string
	safeMode   bool

	mu        sync.Mutex
	lastPhase string
}

func NewPatternEvaluator(config SignalConfig, rng *rand.Rand, logger *Logger, signalName string, safeMode bool) *PatternEvaluator {
	return &PatternEvaluator{
		config:     config,
		rng:        rng,
		logger:     logger,
		signalName: signalName,
		safeMode:   safeMode,
	}
}

// Evaluate returns the current fault value and a boolean requesting resource release (for safe mode).
func (p *PatternEvaluator) Evaluate(uptimeSec float64, reqCount int64, rps float64) (float64, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()

	switch p.config.Pattern {
	case "static":
		return getFloatParam(p.config.Params, "value", 0), false

	case "stochastic":
		min := getFloatParam(p.config.Params, "min", 0)
		max := getFloatParam(p.config.Params, "max", 0)
		dist := getStringParam(p.config.Params, "dist", "uniform")
		meanShift := getBoolParam(p.config.Params, "mean_shift", true)

		if min > max {
			min, max = max, min
		}

		var val float64
		if dist == "lognormal" {
			if min <= 0 {
				min = 1
			}
			logLo := math.Log(min)
			logHi := math.Log(max)
			mu := (logLo + logHi) / 2.0
			sigma := (logHi - logLo) / (2.0 * 1.645)

			sample := math.Exp(mu + p.rng.NormFloat64()*sigma)
			if !meanShift {
				val = sample - math.Exp(mu)
				if val < 0 {
					val = 0
				}
			} else {
				val = sample
			}
			val = math.Min(max*2, val)
		} else {
			// uniform
			if !meanShift {
				val = math.Max(0, p.rng.Float64()*(max-min))
			} else {
				val = min + p.rng.Float64()*(max-min)
			}
		}
		return val, false

	case "linear":
		start := getFloatParam(p.config.Params, "start_value", 0)
		growth := getFloatParam(p.config.Params, "growth_rate_per_sec", 0)
		ceiling := getFloatParam(p.config.Params, "ceiling", 0)

		val := start + growth*uptimeSec
		if ceiling > 0 && val > ceiling {
			val = ceiling
		}
		return val, false

	case "step_change":
		before := getFloatParam(p.config.Params, "before_value", 0)
		after := getFloatParam(p.config.Params, "after_value", 0)
		triggerType := getStringParam(p.config.Params, "trigger_type", "uptime_sec")
		triggerValue := getFloatParam(p.config.Params, "trigger_value", 0)

		val := before
		phase := "before"

		if triggerType == "uptime_sec" {
			if uptimeSec >= triggerValue {
				val = after
				phase = "after"
			}
		} else if triggerType == "request_count" {
			if float64(reqCount) >= triggerValue {
				val = after
				phase = "after"
			}
		}

		if phase != p.lastPhase {
			if p.lastPhase != "" && p.logger != nil {
				p.logger.Log(p.signalName, "phase_switch", val, phase)
			}
			p.lastPhase = phase
		}

		return val, false

	case "load_dependent":
		rpsThreshold := getFloatParam(p.config.Params, "rps_threshold", 0)
		valBelow := getFloatParam(p.config.Params, "value_below", 0)
		valAbove := getFloatParam(p.config.Params, "value_above", 0)

		val := valBelow
		phase := "below"
		if rps >= rpsThreshold {
			val = valAbove
			phase = "above"
		}

		if phase != p.lastPhase {
			if p.lastPhase != "" && p.logger != nil {
				p.logger.Log(p.signalName, "phase_switch", val, fmt.Sprintf("rps=%.1f threshold=%.1f phase=%s", rps, rpsThreshold, phase))
			}
			p.lastPhase = phase
		}

		return val, false

	case "cyclic":
		valLow := getFloatParam(p.config.Params, "value_low", 0)
		valHigh := getFloatParam(p.config.Params, "value_high", 0)
		period := getFloatParam(p.config.Params, "period_sec", 60)
		dutyCycle := getFloatParam(p.config.Params, "duty_cycle", 0.5)

		if period <= 0 {
			period = 60
		}

		cyclePos := math.Mod(uptimeSec, period) / period // 0.0 to 1.0
		val := valLow
		phase := "low"
		release := false

		if cyclePos < dutyCycle {
			val = valHigh
			phase = "high"
		}

		if phase != p.lastPhase {
			if p.lastPhase != "" && p.logger != nil {
				p.logger.Log(p.signalName, "phase_switch", val, phase)
			}
			if p.lastPhase != "" {
				if phase == "low" && p.safeMode {
					release = true
				}
			}
			p.lastPhase = phase
		}

		return val, release

	default:
		return 0, false
	}
}
