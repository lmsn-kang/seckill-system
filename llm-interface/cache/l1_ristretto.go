package cache

import (

    "log"

    "sync/atomic"

    "time"

    "github.com/dgraph-io/ristretto"

)

type L1Cache struct {

    cache *ristretto.Cache

    ttl   time.Duration

    hits   atomic.Int64

    misses atomic.Int64

}

func NewL1Cache(maxItems int64, ttl time.Duration) *L1Cache {

    cache, err := ristretto.NewCache(&ristretto.Config{

        NumCounters: maxItems * 10,

        MaxCost: maxItems,

        BufferItems: 64,

    })

    if err != nil {

        log.Fatalf("Ristretto 初始化失败: %v", err)

    }

    log.Printf("[L1] Ristretto 就绪 | 容量:%d | TTL:%v", maxItems, ttl)

    return &L1Cache{cache: cache, ttl: ttl}

}

func (c *L1Cache) Get(key string) (*CacheResult, bool) {

    val, found := c.cache.Get(key)

    if !found {

        c.misses.Add(1)

        return nil, false

    }

    result, ok := val.(*CacheResult)

    if !ok {

        c.misses.Add(1)

        return nil, false

    }

    c.hits.Add(1)

    return result, true

}

func (c *L1Cache) Set(key string, result *CacheResult) {

    c.cache.SetWithTTL(key, result, 1, c.ttl)

}

func (c *L1Cache) Stats() map[string]interface{} {

    hits := c.hits.Load()

    misses := c.misses.Load()

    total := hits + misses

    rate := float64(0)

    if total > 0 {

        rate = float64(hits) / float64(total)

    }

    return map[string]interface{}{

        "type":     "ristretto",

        "hits":     hits,

        "misses":   misses,

        "hit_rate": rate,

    }

}

func (c *L1Cache) Close() {

    c.cache.Close()

}

