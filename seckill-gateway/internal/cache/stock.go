package cache

import (
	"context"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"
)

type StockCache struct {
	rdb     *redis.Client
	channel string

	local sync.Map

	l1Hits   int64
	l2Hits   int64
	l2Misses int64
}

type goodsEntry struct {
	soldOut   atomic.Bool
	markedAt  atomic.Int64
}

func New(rdb *redis.Client, channel string) *StockCache {
	sc := &StockCache{
		rdb:     rdb,
		channel: channel,
	}
	return sc
}

func (sc *StockCache) IsSoldOutLocal(goodsID string) (soldOut bool, exists bool) {
	v, ok := sc.local.Load(goodsID)
	if !ok {
		return false, false
	}
	entry := v.(*goodsEntry)
	if entry.soldOut.Load() {
		atomic.AddInt64(&sc.l1Hits, 1)
		return true, true
	}
	return false, true
}

func (sc *StockCache) IsSoldOutRedis(ctx context.Context, goodsID string) (bool, error) {
	key := fmt.Sprintf("sk:soldout:%s", goodsID)
	val, err := sc.rdb.Get(ctx, key).Result()
	if err == redis.Nil {
		atomic.AddInt64(&sc.l2Misses, 1)
		return false, nil
	}
	if err != nil {
		return false, err
	}

	if val == "1" {
		atomic.AddInt64(&sc.l2Hits, 1)
		sc.MarkSoldOut(goodsID)
		return true, nil
	}
	return false, nil
}

func (sc *StockCache) IsSoldOut(ctx context.Context, goodsID string) (soldOut bool, source string) {
	if sold, exists := sc.IsSoldOutLocal(goodsID); exists && sold {
		return true, "L1_local"
	}

	sold, err := sc.IsSoldOutRedis(ctx, goodsID)
	if err != nil {
		log.Printf("[cache] redis query error (pass through): %v", err)
		return false, "error_passthrough"
	}
	if sold {
		return true, "L2_redis"
	}

	return false, "miss"
}

func (sc *StockCache) MarkSoldOut(goodsID string) {
	entry := sc.getOrCreateEntry(goodsID)

	if entry.soldOut.CompareAndSwap(false, true) {
		entry.markedAt.Store(time.Now().Unix())
		log.Printf("[cache] 🔴 goods %s marked as SOLD OUT in L1", goodsID)
	}
}

func (sc *StockCache) ClearSoldOut(ctx context.Context, goodsID string) {
	if v, ok := sc.local.Load(goodsID); ok {
		entry := v.(*goodsEntry)
		entry.soldOut.Store(false)
		entry.markedAt.Store(0)
	}

	key := fmt.Sprintf("sk:soldout:%s", goodsID)
	sc.rdb.Del(ctx, key)

	log.Printf("[cache] 🟢 goods %s sold-out mark CLEARED", goodsID)
}

func (sc *StockCache) ClearAll(ctx context.Context) {
	sc.local.Range(func(key, value interface{}) bool {
		entry := value.(*goodsEntry)
		entry.soldOut.Store(false)
		entry.markedAt.Store(0)
		redisKey := fmt.Sprintf("sk:soldout:%s", key.(string))
		sc.rdb.Del(ctx, redisKey)
		return true
	})
	atomic.StoreInt64(&sc.l1Hits, 0)
	atomic.StoreInt64(&sc.l2Hits, 0)
	atomic.StoreInt64(&sc.l2Misses, 0)
	log.Printf("[cache] ♻️ all sold-out marks cleared")
}

func (sc *StockCache) SubscribeSoldOut(ctx context.Context) {
	pubsub := sc.rdb.Subscribe(ctx, sc.channel)

	go func() {
		defer pubsub.Close()
		log.Printf("[cache] 📡 subscribed to channel: %s", sc.channel)

		ch := pubsub.Channel()
		for {
			select {
			case <-ctx.Done():
				log.Printf("[cache] subscription stopped")
				return
			case msg, ok := <-ch:
				if !ok {
					log.Printf("[cache] subscription channel closed, reconnecting...")
					time.Sleep(time.Second)
					pubsub = sc.rdb.Subscribe(ctx, sc.channel)
					ch = pubsub.Channel()
					continue
				}
				goodsID := msg.Payload
				sc.MarkSoldOut(goodsID)
				log.Printf("[cache] 📡 received sold-out notification: goods=%s", goodsID)
			}
		}
	}()
}

func (sc *StockCache) PollRedis(ctx context.Context, goodsIDs []string, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	log.Printf("[cache] 🔄 polling Redis every %v for %d goods", interval, len(goodsIDs))

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			for _, gid := range goodsIDs {
				if sold, exists := sc.IsSoldOutLocal(gid); exists && sold {
					continue
				}
				key := fmt.Sprintf("sk:soldout:%s", gid)
				val, err := sc.rdb.Get(ctx, key).Result()
				if err != nil {
					continue
				}
				if val == "1" {
					sc.MarkSoldOut(gid)
				}
			}
		}
	}
}

func (sc *StockCache) CleanExpired(ctx context.Context, ttl time.Duration) {
	ticker := time.NewTicker(ttl / 2)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			now := time.Now().Unix()
			sc.local.Range(func(key, value interface{}) bool {
				entry := value.(*goodsEntry)
				markedAt := entry.markedAt.Load()
				if markedAt > 0 && now-markedAt > int64(ttl.Seconds()) {
					entry.soldOut.Store(false)
					entry.markedAt.Store(0)
					log.Printf("[cache] ⏰ L1 expired for goods %s (after %v)", key, ttl)
				}
				return true
			})
		}
	}
}

func (sc *StockCache) getOrCreateEntry(goodsID string) *goodsEntry {
	v, loaded := sc.local.LoadOrStore(goodsID, &goodsEntry{})
	entry := v.(*goodsEntry)
	if !loaded {
		log.Printf("[cache] created L1 entry for goods %s", goodsID)
	}
	return entry
}

func (sc *StockCache) Stats() map[string]interface{} {
	l1 := atomic.LoadInt64(&sc.l1Hits)
	l2 := atomic.LoadInt64(&sc.l2Hits)
	miss := atomic.LoadInt64(&sc.l2Misses)
	total := l1 + l2 + miss

	var hitRate float64
	if total > 0 {
		hitRate = float64(l1+l2) / float64(total) * 100
	}

	goods := make(map[string]bool)
	sc.local.Range(func(key, value interface{}) bool {
		entry := value.(*goodsEntry)
		goods[key.(string)] = entry.soldOut.Load()
		return true
	})

	return map[string]interface{}{
		"l1_hits":       l1,
		"l2_hits":       l2,
		"l2_misses":     miss,
		"total_queries": total,
		"hit_rate":      fmt.Sprintf("%.1f%%", hitRate),
		"goods_status":  goods,
	}
}
