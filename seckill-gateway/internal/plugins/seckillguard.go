package plugins

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"

	"seckill-gateway/internal/cache"
)

type SeckillGuardPlugin struct {
	cache       *cache.StockCache
	paths       map[string]bool
	intercepted int64
	passed      int64
}

func NewSeckillGuard(
	rdb *redis.Client,
	paths []string,
	localTTLSec int,
	pollIntervalMs int,
	pubsubChannel string,
) *SeckillGuardPlugin {
	stockCache := cache.New(rdb, pubsubChannel)

	m := make(map[string]bool, len(paths))
	for _, p := range paths {
		m[p] = true
	}

	g := &SeckillGuardPlugin{
		cache: stockCache,
		paths: m,
	}

	ctx := context.Background()

	stockCache.SubscribeSoldOut(ctx)

	go func() {
		interval := time.Duration(pollIntervalMs) * time.Millisecond
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				g.pollKnownGoods(ctx)
			}
		}
	}()

	if localTTLSec > 0 {
		go stockCache.CleanExpired(ctx, time.Duration(localTTLSec)*time.Second)
	}

	go g.logStats()

	return g
}

func (p *SeckillGuardPlugin) Name() string  { return "seckill-guard" }
func (p *SeckillGuardPlugin) Priority() int { return 40 }

func (p *SeckillGuardPlugin) Handler() gin.HandlerFunc {
	return func(c *gin.Context) {
		if !p.paths[c.Request.URL.Path] {
			c.Next()
			return
		}

		goodsID := extractGoodsID(c)
		if goodsID == "" {
			c.Next()
			return
		}

		ctx := c.Request.Context()
		soldOut, source := p.cache.IsSoldOut(ctx, goodsID)

		if soldOut {
			atomic.AddInt64(&p.intercepted, 1)
			c.Header("X-Gateway-Cache", source)
			c.AbortWithStatusJSON(http.StatusOK, gin.H{
				"code":    1,
				"msg":     "sold out",
				"stock":   0,
				"source":  "gateway_cache:" + source,
			})
			return
		}

		atomic.AddInt64(&p.passed, 1)
		c.Next()

		p.checkResponseSoldOut(c, goodsID)
	}
}

func (p *SeckillGuardPlugin) checkResponseSoldOut(c *gin.Context, goodsID string) {
	if c.Writer.Header().Get("X-Sold-Out") == "true" {
		p.cache.MarkSoldOut(goodsID)
	}
}

func (p *SeckillGuardPlugin) pollKnownGoods(ctx context.Context) {
	stats := p.cache.Stats()
	goodsStatus, ok := stats["goods_status"].(map[string]bool)
	if !ok {
		return
	}
	for goodsID, alreadySold := range goodsStatus {
		if alreadySold {
			continue
		}
		_, _ = p.cache.IsSoldOutRedis(ctx, goodsID)
	}
}

func (p *SeckillGuardPlugin) logStats() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		i := atomic.LoadInt64(&p.intercepted)
		pa := atomic.LoadInt64(&p.passed)
		total := i + pa
		if total == 0 {
			continue
		}
		stats := p.cache.Stats()
		log.Printf("[seckillguard] 📊 intercepted=%d(%.1f%%) passed=%d cache=%v",
			i, float64(i)/float64(total)*100, pa, stats)
	}
}

func (p *SeckillGuardPlugin) Cache() *cache.StockCache {
	return p.cache
}

func (p *SeckillGuardPlugin) Reset(ctx context.Context) {
	p.cache.ClearAll(ctx)
	atomic.StoreInt64(&p.intercepted, 0)
	atomic.StoreInt64(&p.passed, 0)
	log.Printf("[seckillguard] ♻️ all state reset")
}

func extractGoodsID(c *gin.Context) string {
	if gid := c.Query("goods_id"); gid != "" {
		return gid
	}

	if c.Request.Body == nil {
		return "default"
	}

	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		return "default"
	}
	c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

	if len(bodyBytes) == 0 {
		return "default"
	}

	var body struct {
		GoodsID string `json:"goods_id" form:"goods_id"`
	}
	if err := json.Unmarshal(bodyBytes, &body); err == nil && body.GoodsID != "" {
		return body.GoodsID
	}

	return "default"
}
