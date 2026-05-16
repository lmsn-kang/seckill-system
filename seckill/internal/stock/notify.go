package stock

import (
	"context"
	"fmt"
	"log"

	"github.com/redis/go-redis/v9"
)

type Notifier struct {
	rdb     *redis.Client
	channel string
}

func NewNotifier(rdb *redis.Client, channel string) *Notifier {
	return &Notifier{rdb: rdb, channel: channel}
}

func (n *Notifier) NotifySoldOut(ctx context.Context, goodsID string) error {
	pipe := n.rdb.Pipeline()

	soldOutKey := fmt.Sprintf("sk:soldout:%s", goodsID)
	pipe.Set(ctx, soldOutKey, "1", 0)

	pipe.Publish(ctx, n.channel, goodsID)

	_, err := pipe.Exec(ctx)
	if err != nil {
		return fmt.Errorf("notify sold out: %w", err)
	}

	log.Printf("[notifier] 📡 sold-out notification sent: goods=%s", goodsID)
	return nil
}

func (n *Notifier) ClearSoldOut(ctx context.Context, goodsID string) error {
	soldOutKey := fmt.Sprintf("sk:soldout:%s", goodsID)
	_, err := n.rdb.Del(ctx, soldOutKey).Result()
	if err != nil {
		return fmt.Errorf("clear sold out: %w", err)
	}
	log.Printf("[notifier] 🟢 sold-out mark cleared: goods=%s", goodsID)
	return nil
}
