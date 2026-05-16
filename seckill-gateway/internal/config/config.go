package config

import (
	"log"
	"os"
	"sync"

	"github.com/fsnotify/fsnotify"
	"gopkg.in/yaml.v3"
)

type Config struct {
	Server  ServerConfig  `yaml:"server"`
	Backend BackendConfig `yaml:"backend"`
	Redis   RedisConfig   `yaml:"redis"`
	Plugins PluginsConfig `yaml:"plugins"`
}

type ServerConfig struct {
	Port      int `yaml:"port"`
	AdminPort int `yaml:"admin_port"`
}

type BackendConfig struct {
	GRPCAddr string `yaml:"grpc_addr"`
}

type RedisConfig struct {
	Addr     string `yaml:"addr"`
	PoolSize int    `yaml:"pool_size"`
}

type PluginsConfig struct {
	Blacklist      BlacklistConfig      `yaml:"blacklist"`
	RateLimit      RateLimitConfig      `yaml:"ratelimit"`
	Auth           AuthConfig           `yaml:"auth"`
	CircuitBreaker CircuitBreakerConfig `yaml:"circuitbreaker"`
	SeckillGuard   SeckillGuardConfig   `yaml:"seckillguard"`
	Metrics        MetricsConfig        `yaml:"metrics"`
}

type BlacklistConfig struct {
	Enabled bool     `yaml:"enabled"`
	IPs     []string `yaml:"ips"`
}

type RateLimitConfig struct {
	Enabled        bool    `yaml:"enabled"`
	GlobalRate     float64 `yaml:"global_rate"`
	GlobalCapacity float64 `yaml:"global_capacity"`
	IPRate         float64 `yaml:"ip_rate"`
	IPCapacity     float64 `yaml:"ip_capacity"`
}

type AuthConfig struct {
	Enabled        bool     `yaml:"enabled"`
	JWTSecret      string   `yaml:"jwt_secret"`
	ProtectedPaths []string `yaml:"protected_paths"`
	PublicPaths    []string `yaml:"public_paths"`
}

type CircuitBreakerConfig struct {
	Enabled        bool `yaml:"enabled"`
	Threshold      int  `yaml:"threshold"`
	TimeoutSeconds int  `yaml:"timeout_seconds"`
	MaxHalfOpen    int  `yaml:"max_half_open"`
}

type SeckillGuardConfig struct {
	Enabled             bool     `yaml:"enabled"`
	Paths               []string `yaml:"paths"`
	LocalTTLSeconds     int      `yaml:"local_ttl_seconds"`
	RedisPollIntervalMs int      `yaml:"redis_poll_interval_ms"`
	PubSubChannel       string   `yaml:"pubsub_channel"`
}

type MetricsConfig struct {
	Enabled bool   `yaml:"enabled"`
	Path    string `yaml:"path"`
}

var (
	globalCfg Config
	cfgMu     sync.RWMutex
	onChange  []func(*Config)
)

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	cfgMu.Lock()
	globalCfg = cfg
	cfgMu.Unlock()
	return &cfg, nil
}

func Get() Config {
	cfgMu.RLock()
	defer cfgMu.RUnlock()
	return globalCfg
}

func OnChange(fn func(*Config)) {
	onChange = append(onChange, fn)
}

func Watch(path string) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		log.Printf("[config] watcher create failed: %v", err)
		return
	}
	go func() {
		defer watcher.Close()
		for {
			select {
			case event := <-watcher.Events:
				if event.Op&fsnotify.Write == fsnotify.Write {
					cfg, err := Load(path)
					if err != nil {
						log.Printf("[config] hot-reload failed: %v", err)
						continue
					}
					log.Printf("[config] ♻️ reloaded successfully")
					for _, fn := range onChange {
						fn(cfg)
					}
				}
			case err := <-watcher.Errors:
				log.Printf("[config] watcher error: %v", err)
			}
		}
	}()
	if err := watcher.Add(path); err != nil {
		log.Printf("[config] watch add failed: %v", err)
	}
	log.Printf("[config] watching %s for changes", path)
}
