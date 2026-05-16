package order

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/segmentio/kafka-go"
)

type EventConsumer struct {
	reader     *kafka.Reader
	repo       Repository
	rdb        *redis.Client
	handlers   map[EventType]EventHandler

	dlqWriter  *kafka.Writer
	maxRetries int
}

type EventHandler func(ctx context.Context, event *OrderStatusEvent, repo Repository) error

type DLQMessage struct {
	OriginalEvent *OrderStatusEvent `json:"original_event"`
	Error         string            `json:"error"`
	RetryCount    int               `json:"retry_count"`
	FailedAt      time.Time         `json:"failed_at"`
	Topic         string            `json:"topic"`
	Partition     int               `json:"partition"`
	Offset        int64             `json:"offset"`
}

func NewEventConsumer(brokers []string, groupID string, repo Repository, rdb *redis.Client, dlqWriter *kafka.Writer) *EventConsumer {
	c := &EventConsumer{
		reader: kafka.NewReader(kafka.ReaderConfig{
			Brokers:  brokers,
			Topic:    StatusEventTopic,
			GroupID:  groupID,
			MinBytes: 1,
			MaxBytes: 10 << 20,
			MaxWait:  time.Second,
		}),
		repo:       repo,
		rdb:        rdb,
		handlers:   make(map[EventType]EventHandler),
		dlqWriter:  dlqWriter,
		maxRetries: 3,
	}

	c.handlers[EventOrderPaid] = handlePaid
	c.handlers[EventOrderShipped] = handleShipped
	c.handlers[EventOrderCompleted] = handleCompleted
	c.handlers[EventOrderCancelled] = handleCancelled
	c.handlers[EventOrderRefunded] = handleRefunded
	c.handlers[EventOrderExpired] = handleCancelled

	return c
}

func (c *EventConsumer) Start(ctx context.Context) {
	log.Println("[order-consumer] started, waiting for status events...")
	for {
		msg, err := c.reader.FetchMessage(ctx)
		if err != nil {
			if ctx.Err() != nil {
				log.Println("[order-consumer] shutdown")
				return
			}
			log.Printf("[order-consumer] fetch error: %v", err)
			time.Sleep(time.Second)
			continue
		}

		if err := c.processMessage(ctx, msg); err != nil {
			if c.shouldSendToDLQ(ctx, msg) {
				c.sendToDLQ(ctx, msg, err)
				_ = c.reader.CommitMessages(ctx, msg)
				continue
			}

			c.recordRetry(ctx, msg)
			log.Printf("[order-consumer] process failed (will retry): partition=%d offset=%d err=%v",
				msg.Partition, msg.Offset, err)
			continue
		}

		c.resetRetry(ctx, msg)
		_ = c.reader.CommitMessages(ctx, msg)
	}
}

func (c *EventConsumer) processMessage(ctx context.Context, msg kafka.Message) error {
	var event OrderStatusEvent
	if err := json.Unmarshal(msg.Value, &event); err != nil {
		log.Printf("[order-consumer] bad message: %v", err)
		return nil
	}

	if c.isProcessed(ctx, event.EventID) {
		log.Printf("[order-consumer] duplicate event skipped: %s", event.EventID)
		return nil
	}

	handler, ok := c.handlers[event.EventType]
	if !ok {
		log.Printf("[order-consumer] unknown event type: %s", event.EventType)
		return nil
	}

	if err := handler(ctx, &event, c.repo); err != nil {
		return fmt.Errorf("handler %s: %w", event.EventType, err)
	}

	c.markProcessed(ctx, event.EventID)

	log.Printf("[order-consumer] ✅ %s: order=%d %s->%s",
		event.EventType, event.OrderID, event.OldStatus, event.NewStatus)
	return nil
}

func (c *EventConsumer) isProcessed(ctx context.Context, eventID string) bool {
	key := fmt.Sprintf("order:event:%s", eventID)
	val, err := c.rdb.Get(ctx, key).Result()
	return err == nil && val == "1"
}

func (c *EventConsumer) markProcessed(ctx context.Context, eventID string) {
	key := fmt.Sprintf("order:event:%s", eventID)
	_ = c.rdb.Set(ctx, key, "1", 24*time.Hour).Err()
}

func (c *EventConsumer) Close() error {
	if c.dlqWriter != nil {
		_ = c.dlqWriter.Close()
	}
	return c.reader.Close()
}

func (c *EventConsumer) retryKey(msg kafka.Message) string {
	return fmt.Sprintf("consumer:retry:%s:%d:%d", msg.Topic, msg.Partition, msg.Offset)
}

func (c *EventConsumer) recordRetry(ctx context.Context, msg kafka.Message) {
	key := c.retryKey(msg)
	c.rdb.Incr(ctx, key)
	c.rdb.Expire(ctx, key, 24*time.Hour)
}

func (c *EventConsumer) resetRetry(ctx context.Context, msg kafka.Message) {
	c.rdb.Del(ctx, c.retryKey(msg))
}

func (c *EventConsumer) shouldSendToDLQ(ctx context.Context, msg kafka.Message) bool {
	if c.dlqWriter == nil || c.maxRetries <= 0 {
		return false
	}
	key := c.retryKey(msg)
	count, _ := c.rdb.Get(ctx, key).Int()
	return count >= c.maxRetries
}

func (c *EventConsumer) sendToDLQ(ctx context.Context, msg kafka.Message, processErr error) {
	if c.dlqWriter == nil {
		return
	}

	var event OrderStatusEvent
	if err := json.Unmarshal(msg.Value, &event); err != nil {
		log.Printf("[order-consumer] DLQ unmarshal failed: %v", err)
		return
	}

	dlq := DLQMessage{
		OriginalEvent: &event,
		Error:         processErr.Error(),
		RetryCount:    c.maxRetries,
		FailedAt:      time.Now(),
		Topic:         msg.Topic,
		Partition:     msg.Partition,
		Offset:        msg.Offset,
	}

	data, err := json.Marshal(dlq)
	if err != nil {
		log.Printf("[order-consumer] DLQ marshal failed: %v", err)
		return
	}

	if err := c.dlqWriter.WriteMessages(ctx, kafka.Message{
		Key:   []byte(event.PartitionKey()),
		Value: data,
		Headers: []kafka.Header{
			{Key: "dlq-source", Value: []byte(msg.Topic)},
			{Key: "dlq-reason", Value: []byte(processErr.Error())},
			{Key: "dlq-partition", Value: []byte(fmt.Sprintf("%d", msg.Partition))},
			{Key: "dlq-offset", Value: []byte(fmt.Sprintf("%d", msg.Offset))},
		},
	}); err != nil {
		log.Printf("[order-consumer] DLQ write failed: %v", err)
		return
	}

	c.resetRetry(ctx, msg)

	log.Printf("[order-consumer] ⚠️ DLQ: event=%s order=%d err=%s partition=%d offset=%d",
		event.EventType, event.OrderID, processErr.Error(), msg.Partition, msg.Offset)
}

func handlePaid(ctx context.Context, event *OrderStatusEvent, repo Repository) error {
	ok, err := repo.UpdatePay(ctx, event.OrderID, event.OldStatus, event.Version)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("concurrent modification or status mismatch")
	}
	return nil
}

func handleShipped(ctx context.Context, event *OrderStatusEvent, repo Repository) error {
	ok, err := repo.UpdateStatus(ctx, event.OrderID, event.OldStatus, StatusShipped)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("concurrent modification")
	}
	return nil
}

func handleCompleted(ctx context.Context, event *OrderStatusEvent, repo Repository) error {
	ok, err := repo.UpdateStatus(ctx, event.OrderID, event.OldStatus, StatusCompleted)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("concurrent modification")
	}
	return nil
}

func handleCancelled(ctx context.Context, event *OrderStatusEvent, repo Repository) error {
	ok, err := repo.UpdateStatus(ctx, event.OrderID, event.OldStatus, StatusCancelled)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("concurrent modification")
	}
	return nil
}

func handleRefunded(ctx context.Context, event *OrderStatusEvent, repo Repository) error {
	ok, err := repo.UpdateStatus(ctx, event.OrderID, event.OldStatus, StatusRefunded)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("concurrent modification")
	}
	return nil
}
