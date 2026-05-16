package order

import (
	"fmt"
	"time"

	"github.com/google/uuid"
)

type EventType string

const (
	EventOrderPaid      EventType = "ORDER_PAID"
	EventOrderShipped   EventType = "ORDER_SHIPPED"
	EventOrderCompleted EventType = "ORDER_COMPLETED"
	EventOrderCancelled EventType = "ORDER_CANCELLED"
	EventOrderRefunded  EventType = "ORDER_REFUNDED"
	EventOrderExpired   EventType = "ORDER_EXPIRED"
)

type EventPayload struct {
	Transaction string `json:"transaction,omitempty"`
	Reason      string `json:"reason,omitempty"`
	OperatorID  string `json:"operator_id,omitempty"`
}

type OrderStatusEvent struct {
	EventID   string      `json:"event_id"`
	OrderID   int64       `json:"order_id"`
	EventType EventType   `json:"event_type"`
	OldStatus OrderStatus `json:"old_status"`
	NewStatus OrderStatus `json:"new_status"`
	Payload   EventPayload `json:"payload"`
	Version   int32       `json:"version"`
	Timestamp time.Time   `json:"timestamp"`
}

func NewOrderStatusEvent(orderID int64, eventType EventType, oldStatus, newStatus OrderStatus, version int32, payload EventPayload) *OrderStatusEvent {
	return &OrderStatusEvent{
		EventID:   uuid.New().String(),
		OrderID:   orderID,
		EventType: eventType,
		OldStatus: oldStatus,
		NewStatus: newStatus,
		Payload:   payload,
		Version:   version,
		Timestamp: time.Now(),
	}
}

const StatusEventTopic = "order-status-events"

func (e *OrderStatusEvent) PartitionKey() string {
	return fmt.Sprintf("order-%d", e.OrderID)
}

func (e *OrderStatusEvent) IsValidTransition() bool {
	switch e.EventType {
	case EventOrderPaid:
		return e.OldStatus == StatusCreated || e.OldStatus == StatusPending
	case EventOrderShipped:
		return e.OldStatus == StatusPaid
	case EventOrderCompleted:
		return e.OldStatus == StatusShipped
	case EventOrderCancelled:
		return e.OldStatus == StatusCreated || e.OldStatus == StatusPending
	case EventOrderRefunded:
		return e.OldStatus == StatusPaid || e.OldStatus == StatusShipped
	case EventOrderExpired:
		return e.OldStatus == StatusCreated || e.OldStatus == StatusPending
	default:
		return false
	}
}
