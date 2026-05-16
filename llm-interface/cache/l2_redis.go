package cache

import (

    "context"

    "encoding/json"

    "log"

    "sync/atomic"

    "time"

    "github.com/redis/go-redis/v9"

)

type L2Cache struct {

    client    *redis.Client

    ttl       time.Duration

    keyPrefix string

    hits   atomic.Int64

    misses atomic.Int64

}

func NewL2Cache(addr, password string, db int, ttl time.Duration) *L2Cache {

    client := redis.NewClient(&redis.Options{

        Addr:     addr,

        Password: password,

        DB:       db,

        PoolSize:     20,

        MinIdleConns: 5,

        ReadTimeout:  2 * time.Second,

        WriteTimeout: 2 * time.Second,

    })

    ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)

    defer cancel()

    if err := client.Ping(ctx).Err(); err != nil {

        log.Printf("[L2] ⚠️ Redis 连接失败: %v（将仅使用 L1 缓存）", err)

        return nil

    }

    log.Printf("[L2] Redis 就绪 | addr:%s | TTL:%v", addr, ttl)

    return &L2Cache{

        client:    client,

        ttl:       ttl,

        keyPrefix: "mod:v1:",

    }

}

func (c *L2Cache) fullKey(key string) string {

    return c.keyPrefix + key

}

func (c *L2Cache) Get(ctx context.Context, key string) (*CacheResult, bool) {

    data, err := c.client.Get(ctx, c.fullKey(key)).Bytes()

    if err != nil {

        c.misses.Add(1)

        return nil, false

    }

    var result CacheResult

    if err := json.Unmarshal(data, &result); err != nil {

        c.misses.Add(1)

        return nil, false

    }

    c.hits.Add(1)

    return &result, true

}

func (c *L2Cache) Set(ctx context.Context, key string, result *CacheResult) {

    data, err := json.Marshal(result)

    if err != nil {

        return

    }

    c.client.SetEx(ctx, c.fullKey(key), data, c.ttl)

}

func (c *L2Cache) Stats() map[string]interface{} {

    hits := c.hits.Load()

    misses := c.misses.Load()

    total := hits + misses

    rate := float64(0)

    if total > 0 {

        rate = float64(hits) / float64(total)

    }

    return map[string]interface{}{

        "type":     "redis",

        "hits":     hits,

        "misses":   misses,

        "hit_rate": rate,

    }

}

func (c *L2Cache) Close() {

    if c.client != nil {

        c.client.Close()

    }

}