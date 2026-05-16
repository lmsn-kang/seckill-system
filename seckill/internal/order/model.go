package order

import (
	"time"
)

type OrderStatus int32

const (
	StatusCreated   OrderStatus = 0
	StatusPending   OrderStatus = 1
	StatusPaid      OrderStatus = 2
	StatusShipped   OrderStatus = 3
	StatusCompleted OrderStatus = 4
	StatusCancelled OrderStatus = 5
	StatusRefunded  OrderStatus = 6
)

func (s OrderStatus) String() string {
	switch s {
	case StatusCreated:
		return "CREATED"
	case StatusPending:
		return "PENDING"
	case StatusPaid:
		return "PAID"
	case StatusShipped:
		return "SHIPPED"
	case StatusCompleted:
		return "COMPLETED"
	case StatusCancelled:
		return "CANCELLED"
	case StatusRefunded:
		return "REFUNDED"
	default:
		return "UNKNOWN"
	}
}

type Order struct {
	ID          int64       `db:"id"`
	OrderID     int64       `db:"order_id"`
	UserID      string      `db:"user_id"`
	GoodsID     string      `db:"goods_id"`
	Status      OrderStatus `db:"status"`
	Amount      int64       `db:"amount"`
	ExpireTime  time.Time   `db:"expire_time"`
	CreatedAt   time.Time   `db:"created_at"`
	UpdatedAt   time.Time   `db:"updated_at"`
	Version     int32       `db:"version"`
}

type CreateOrderDTO struct {
	OrderID    int64
	UserID     string
	GoodsID    string
	Amount     int64
	ExpireTime time.Time
}

type PayOrderDTO struct {
	OrderID     int64
	PayAmount   int64
	Transaction string
}
