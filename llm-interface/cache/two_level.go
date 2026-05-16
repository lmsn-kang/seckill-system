package cache

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"log"
	"time"
)

type CacheResult struct {
	IsSafe     bool             `json:"is_safe"`
	Label      string           `json:"label"`
	Confidence float32          `json:"confidence"`
	AllScores  map[string]float32 `json:"all_scores"`
}

type Config struct {
	L1MaxItems    int64
	L1TTL         time.Duration

	RedisAddr     string
	RedisPassword string
	RedisDB       int
	L2TTL         time.Duration
}

type TwoLevelCache struct {
	l1   *L1Cache
	l2   *L2Cache
	cfg  *Config
	ctx  context.Context
}

func MakeKey(text string) string {
	h := md5.Sum([]byte(text))
	return hex.EncodeToString(h[:])
}

func NewTwoLevelCache(cfg *Config) *TwoLevelCache {
	l1 := NewL1Cache(cfg.L1MaxItems, cfg.L1TTL)

	var l2 *L2Cache
	if cfg.RedisAddr != "" {
		l2 = NewL2Cache(cfg.RedisAddr, cfg.RedisPassword, cfg.RedisDB, cfg.L2TTL)
	}

	if l2 != nil {
		log.Printf("[Cache] 两级缓存就绪 | L1=Ristretto(%d) L2=Redis(%s)", cfg.L1MaxItems, cfg.RedisAddr)
	} else {
		log.Printf("[Cache] 仅 L1 缓存就绪 | L1=Ristretto(%d) | Redis 不可用", cfg.L1MaxItems)
	}

	return &TwoLevelCache{
		l1:  l1,
		l2:  l2,
		cfg: cfg,
		ctx: context.Background(),
	}
}

func (c *TwoLevelCache) Get(text string) (*CacheResult, bool) {
	if text == "" {
		return nil, false
	}

	key := MakeKey(text)

	if result, ok := c.l1.Get(key); ok {
		return result, true
	}

	if c.l2 != nil {
		if result, ok := c.l2.Get(c.ctx, key); ok {
			c.l1.Set(key, result)
			return result, true
		}
	}

	return nil, false
}

func (c *TwoLevelCache) Set(text string, result *CacheResult) {
	if text == "" {
		return
	}

	key := MakeKey(text)

	c.l1.Set(key, result)

	if c.l2 != nil {
		c.l2.Set(c.ctx, key, result)
	}
}

func (c *TwoLevelCache) Close() {
	if c.l1 != nil {
		c.l1.Close()
	}
	if c.l2 != nil {
		c.l2.Close()
	}
}

func (c *TwoLevelCache) Stats() map[string]interface{} {
	stats := map[string]interface{}{
		"l1": c.l1.Stats(),
	}
	if c.l2 != nil {
		stats["l2"] = c.l2.Stats()
	} else {
		stats["l2"] = "unavailable"
	}
	return stats
}
