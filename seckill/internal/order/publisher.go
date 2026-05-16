package order

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/segmentio/kafka-go"
)

type EventPublisher struct {
	writer *kafka.Writer
}

func NewEventPublisher(brokers []string) *EventPublisher {
	return &EventPublisher{
		writer: &kafka.Writer{
			Addr:         kafka.TCP(brokers...),
			Topic:        StatusEventTopic,
			Balancer:     &kafka.Hash{},
			RequiredAcks: kafka.RequireOne,
			Async:        false,
		},
	}
}

func (p *EventPublisher) Publish(ctx context.Context, event *OrderStatusEvent) error {
	if event == nil {
		return fmt.Errorf("event is nil")
	}

	if !event.IsValidTransition() {
		return fmt.Errorf("invalid status transition: %s -> %s for event %s",
			event.OldStatus, event.NewStatus, event.EventType)
	}

	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("marshal event: %w", err)
	}

	msg := kafka.Message{
		Key:   []byte(event.PartitionKey()),
		Value: data,
		Headers: []kafka.Header{
			{Key: "event-type", Value: []byte(event.EventType)},
			{Key: "event-id", Value: []byte(event.EventID)},
		},
	}

	if err := p.writer.WriteMessages(ctx, msg); err != nil {
		return fmt.Errorf("kafka write: %w", err)
	}

	log.Printf("[order-event] published: %s order=%d %s->%s",
		event.EventType, event.OrderID, event.OldStatus, event.NewStatus)
	return nil
}

func (p *EventPublisher) PublishAsync(event *OrderStatusEvent) {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
		defer cancel()
		if err := p.Publish(ctx, event); err != nil {
			log.Printf("[order-event] async publish failed: %v", err)
		}
	}()
}

func (p *EventPublisher) Close() error {
	return p.writer.Close()
}

const defaultTimeout = 5 * time.Second
