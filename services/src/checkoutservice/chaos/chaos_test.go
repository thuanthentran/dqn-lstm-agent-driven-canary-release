package chaos

import (
	"math/rand"
	"testing"
)

func TestPatternEvaluator_Static(t *testing.T) {
	config := SignalConfig{
		Pattern: "static",
		Params:  map[string]interface{}{"value": 42.0},
	}
	p := NewPatternEvaluator(config, rand.New(rand.NewSource(0)), nil, "test", false)
	val, _ := p.Evaluate(0, 0, 0)
	if val != 42.0 {
		t.Errorf("expected 42, got %v", val)
	}
}

func TestPatternEvaluator_Linear(t *testing.T) {
	config := SignalConfig{
		Pattern: "linear",
		Params: map[string]interface{}{
			"start_value":         10.0,
			"growth_rate_per_sec": 5.0,
			"ceiling":             25.0,
		},
	}
	p := NewPatternEvaluator(config, rand.New(rand.NewSource(0)), nil, "test", false)

	// uptime=1s -> 10 + 5*1 = 15
	val, _ := p.Evaluate(1, 0, 0)
	if val != 15.0 {
		t.Errorf("expected 15, got %v", val)
	}

	// uptime=5s -> 10 + 5*5 = 35 -> ceiling 25
	val, _ = p.Evaluate(5, 0, 0)
	if val != 25.0 {
		t.Errorf("expected 25 (ceiling), got %v", val)
	}
}

func TestPatternEvaluator_Cyclic(t *testing.T) {
	config := SignalConfig{
		Pattern: "cyclic",
		Params: map[string]interface{}{
			"value_low":  10.0,
			"value_high": 90.0,
			"period_sec": 10.0,
			"duty_cycle": 0.4,
		},
	}
	p := NewPatternEvaluator(config, rand.New(rand.NewSource(0)), nil, "test", true)

	// t=0 (cyclePos=0 < 0.4 -> high)
	val, release := p.Evaluate(0, 0, 0)
	if val != 90.0 {
		t.Errorf("expected 90, got %v", val)
	}
	if release {
		t.Errorf("expected release=false, got true")
	}

	// t=3.99 (cyclePos=0.399 < 0.4 -> high)
	val, release = p.Evaluate(3.99, 0, 0)
	if val != 90.0 {
		t.Errorf("expected 90, got %v", val)
	}
	if release {
		t.Errorf("expected release=false, got true")
	}

	// t=4.01 (cyclePos=0.401 > 0.4 -> low). Since phase changes to low and safeMode is true, release should be true!
	val, release = p.Evaluate(4.01, 0, 0)
	if val != 10.0 {
		t.Errorf("expected 10, got %v", val)
	}
	if !release {
		t.Errorf("expected release=true on phase change to low, got false")
	}

	// t=5 (still low, but phase didn't change this step, so release should be false)
	val, release = p.Evaluate(5, 0, 0)
	if val != 10.0 {
		t.Errorf("expected 10, got %v", val)
	}
	if release {
		t.Errorf("expected release=false, got true")
	}
}

func TestPatternEvaluator_StepChange(t *testing.T) {
	config := SignalConfig{
		Pattern: "step_change",
		Params: map[string]interface{}{
			"before_value": 0.0,
			"after_value":  1.0,
			"trigger_type": "request_count",
			"trigger_value": 100.0,
		},
	}
	p := NewPatternEvaluator(config, rand.New(rand.NewSource(0)), nil, "test", false)

	// count=99 -> before
	val, _ := p.Evaluate(0, 99, 0)
	if val != 0.0 {
		t.Errorf("expected 0, got %v", val)
	}

	// count=100 -> after
	val, _ = p.Evaluate(0, 100, 0)
	if val != 1.0 {
		t.Errorf("expected 1.0, got %v", val)
	}
}
