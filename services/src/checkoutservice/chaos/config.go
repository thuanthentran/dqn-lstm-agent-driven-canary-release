package chaos

import (
	"encoding/json"
	"os"

	"github.com/sirupsen/logrus"
)

type SignalConfig struct {
	Pattern string                 `json:"pattern"`
	Params  map[string]interface{} `json:"params"`
}

type ChaosConfig struct {
	Enabled      bool                    `json:"enabled"`
	RunID        string                  `json:"run_id"`
	Seed         int64                   `json:"seed"`
	ResourceSafe bool                    `json:"resource_safe_mode"`
	Signals      map[string]SignalConfig `json:"signals"`
}

func LoadConfig() *ChaosConfig {
	envStr := os.Getenv("CHAOS_CONFIG")
	if envStr == "" {
		return &ChaosConfig{Enabled: false}
	}

	var cfg ChaosConfig
	if err := json.Unmarshal([]byte(envStr), &cfg); err != nil {
		logrus.WithError(err).Warn("chaos: failed to parse CHAOS_CONFIG JSON, disabling chaos")
		return &ChaosConfig{Enabled: false}
	}

	if cfg.Seed == 0 {
		cfg.Seed = 42
	}

	return &cfg
}

func getFloatParam(params map[string]interface{}, key string, def float64) float64 {
	if val, ok := params[key]; ok {
		if f, ok := val.(float64); ok {
			return f
		}
	}
	return def
}

func getStringParam(params map[string]interface{}, key string, def string) string {
	if val, ok := params[key]; ok {
		if s, ok := val.(string); ok {
			return s
		}
	}
	return def
}

func getBoolParam(params map[string]interface{}, key string, def bool) bool {
	if val, ok := params[key]; ok {
		if b, ok := val.(bool); ok {
			return b
		}
	}
	return def
}
