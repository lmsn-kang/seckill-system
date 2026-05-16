package plugins

import (
	"context"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
)

const tokenBucketLuaScript = `
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local tk = key .. ":tokens"
local ts = key .. ":timestamp"

local cur = tonumber(redis.call("GET", tk))
local last = tonumber(redis.call("GET", ts))
if cur == nil then cur = capacity; last = now end

local elapsed = math.max(0, now - last)
cur = math.min(capacity, cur + elapsed * rate)

local ok = 0
if cur >= requested then
    cur = cur - requested
    ok = 1
end

redis.call("SETEX", tk, 60, tostring(cur))
redis.call("SETEX", ts, 60, tostring(now))

return { ok, math.floor(cur) }
`

type RateLimitPlugin struct {
	rdb            *redis.Client
	sha            string
	globalRate     float64
	globalCapacity float64
	ipRate         float64
	ipCapacity     float64
}

func NewRateLimit(rdb *redis.Client, globalRate, globalCap, ipRate, ipCap float64) *RateLimitPlugin {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	sha, err := rdb.ScriptLoad(ctx, tokenBucketLuaScript).Result()
	if err != nil {
		log.Fatalf("[ratelimit] load lua script failed: %v", err)
	}

	return &RateLimitPlugin{
		rdb: rdb, sha: sha,
		globalRate: globalRate, globalCapacity: globalCap,
		ipRate: ipRate, ipCapacity: ipCap,
	}
}

func (p *RateLimitPlugin) Name() string  { return "distributed-rate-limit" }
func (p *RateLimitPlugin) Priority() int { return 20 }

func (p *RateLimitPlugin) Handler() gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()

		if ok, _ := p.allow(ctx, "rl:global", p.globalRate, p.globalCapacity); !ok {
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"code": 429, "message": "system busy, please retry",
			})
			return
		}

		ip := ClientIP(c.Request)
		if ok, _ := p.allow(ctx, "rl:ip:"+ip, p.ipRate, p.ipCapacity); !ok {
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"code": 429, "message": "too many requests from your IP",
			})
			return
		}

		c.Next()
	}
}

func (p *RateLimitPlugin) allow(ctx context.Context, key string, r, cap float64) (bool, int64) {
	now := float64(time.Now().UnixNano()) / 1e9

	res, err := p.rdb.EvalSha(ctx, p.sha, []string{key}, r, cap, now, 1).Int64Slice()
	if err != nil {
		log.Printf("[ratelimit] redis error (allowing): %v", err)
		return true, 0
	}
	return res[0] == 1, res[1]
}

func (p *RateLimitPlugin) Reload(globalRate, globalCap, ipRate, ipCap float64) {
	p.globalRate = globalRate
	p.globalCapacity = globalCap
	p.ipRate = ipRate
	p.ipCapacity = ipCap
	log.Printf("[ratelimit] ♻️ reloaded: global=%.0f/s ip=%.0f/s", globalRate, ipRate)
}
