package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/segmentio/kafka-go"
	"go.opentelemetry.io/otel"

	"seckill/internal/order"
	"seckill/pkg/tracing"
)

type OrderMessage struct {
	OrderID int64  `json:"order_id"`
	UserID  string `json:"user_id"`
	GoodsID string `json:"goods_id"`
}

var (
	orderService *order.Service
	orderRepo    order.Repository
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	tp, _ := tracing.Init(ctx, "order-svc", env("JAEGER_ENDPOINT", "localhost:4317"))
	defer tp.Shutdown(context.Background())

	rdb := redis.NewClient(&redis.Options{Addr: env("REDIS_ADDR", "localhost:6379")})

	repo, err := order.NewRepository(env("MYSQL_DSN", ""))
	if err != nil {
		log.Printf("[order] MySQL连接失败（订单已通过主服务创建，此仅用于幂等确认）: %v", err)
	} else {
		orderRepo = repo
		orderService = order.NewService(repo)
		go orderService.StartExpireScanner(ctx, 1*time.Minute)
	}

	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:  []string{env("KAFKA_ADDR", "localhost:9092")},
		Topic:    "seckill-orders",
		GroupID:  "order-consumers",
		MinBytes: 1,
		MaxBytes: 10 << 20,
	})

	var eventConsumer *order.EventConsumer
	if orderRepo != nil {
		dlqWriter := &kafka.Writer{
			Addr:         kafka.TCP(env("KAFKA_ADDR", "localhost:9092")),
			Topic:        "order-status-events-dlq",
			Balancer:     &kafka.Hash{},
			RequiredAcks: kafka.RequireOne,
		}

		eventConsumer = order.NewEventConsumer(
			[]string{env("KAFKA_ADDR", "localhost:9092")},
			"order-status-consumers",
			orderRepo,
			rdb,
			dlqWriter,
		)
		go eventConsumer.Start(ctx)
		log.Println("[order] order-status event consumer started with DLQ")
	}

	log.Println("[order] 🚀 consumer started, waiting for messages...")

	go func() {
		for {
			msg, err := reader.FetchMessage(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				log.Printf("[order] fetch error: %v", err)
				time.Sleep(time.Second)
				continue
			}

			if err := processOrder(ctx, rdb, msg.Value); err != nil {
				log.Printf("[order] process failed: %v (will retry)", err)
				continue
			}
			_ = reader.CommitMessages(ctx, msg)
		}
	}()

	<-ctx.Done()
	_ = reader.Close()
	if eventConsumer != nil {
		_ = eventConsumer.Close()
	}
	if orderRepo != nil {
		_ = orderRepo.Close()
	}
	log.Println("[order] shutdown complete ✅")
}

func processOrder(ctx context.Context, rdb *redis.Client, data []byte) error {
	_, span := otel.Tracer("order-svc").Start(ctx, "processOrder")
	defer span.End()

	var orderMsg OrderMessage
	if err := json.Unmarshal(data, &orderMsg); err != nil {
		log.Printf("[order] bad message: %v", err)
		return nil
	}

	key := "order:done:" + strconv.FormatInt(orderMsg.OrderID, 10)
	if ok, _ := rdb.SetNX(ctx, key, 1, 24*time.Hour).Result(); !ok {
		log.Printf("[order] duplicate, skip: %s", orderMsg.OrderID)
		return nil
	}

	if orderService != nil {
		_, err := orderService.GetOrder(ctx, orderMsg.OrderID)
		if err != nil {
			log.Printf("[order] 订单不存在: %s (可能已被回滚)", orderMsg.OrderID)
			return nil
		}
	}

	log.Printf("[order] ✅ order confirmed: id=%d user=%s goods=%s",
		orderMsg.OrderID, orderMsg.UserID, orderMsg.GoodsID)
	return nil
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
