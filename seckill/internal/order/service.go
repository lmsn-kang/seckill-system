package order

import (
	"context"
	"fmt"
	"log"
	"strconv"
	"time"
)

type Service struct {
	repo      Repository
	publisher *EventPublisher
}

func NewService(repo Repository) *Service {
	return &Service{repo: repo}
}

func (s *Service) SetPublisher(p *EventPublisher) {
	s.publisher = p
}

func (s *Service) PublishTransition(ctx context.Context, orderID int64, eventType EventType, payload EventPayload) error {
	if s.publisher == nil {
		return fmt.Errorf("event publisher not initialized")
	}

	order, err := s.repo.GetByID(ctx, orderID)
	if err != nil {
		return fmt.Errorf("get order: %w", err)
	}

	var newStatus OrderStatus
	switch eventType {
	case EventOrderPaid:
		newStatus = StatusPaid
	case EventOrderShipped:
		newStatus = StatusShipped
	case EventOrderCompleted:
		newStatus = StatusCompleted
	case EventOrderCancelled:
		newStatus = StatusCancelled
	case EventOrderRefunded:
		newStatus = StatusRefunded
	case EventOrderExpired:
		newStatus = StatusCancelled
	default:
		return fmt.Errorf("unknown event type: %s", eventType)
	}

	event := NewOrderStatusEvent(orderID, eventType, order.Status, newStatus, order.Version, payload)

	if !event.IsValidTransition() {
		return fmt.Errorf("invalid status transition: %s -> %s for event %s",
			order.Status, newStatus, eventType)
	}

	if err := s.publisher.Publish(ctx, event); err != nil {
		return fmt.Errorf("publish event: %w", err)
	}

	log.Printf("[order] 事件已发布: %s order=%d %s->%s",
		eventType, orderID, order.Status, newStatus)
	return nil
}

func (s *Service) CreateOrder(ctx context.Context, dto *CreateOrderDTO) (*Order, error) {
	order, err := s.repo.Create(ctx, dto)
	if err != nil {
		return nil, fmt.Errorf("create order: %w", err)
	}

	log.Printf("[order] 创建订单成功: orderID=%d userID=%s goodsID=%s amount=%d status=%s",
		order.OrderID, order.UserID, order.GoodsID, order.Amount, order.Status)
	return order, nil
}

func (s *Service) PayOrder(ctx context.Context, orderID int64, transaction string) error {
	order, err := s.repo.GetByID(ctx, orderID)
	if err != nil {
		return fmt.Errorf("get order: %w", err)
	}

	if order.Status != StatusCreated && order.Status != StatusPending {
		return fmt.Errorf("订单状态不允许支付: 当前状态=%s", order.Status)
	}

	if time.Now().After(order.ExpireTime) {
		_, _ = s.CancelOrder(ctx, orderID)
		return fmt.Errorf("订单已过期")
	}

	ok, err := s.repo.UpdatePay(ctx, orderID, order.Status, order.Version)
	if err != nil {
		return fmt.Errorf("update pay: %w", err)
	}
	if !ok {
		return fmt.Errorf("支付失败，订单状态已变更或并发冲突")
	}

	log.Printf("[order] 支付成功: orderID=%d transaction=%s", orderID, transaction)
	return nil
}

func (s *Service) ShipOrder(ctx context.Context, orderID int64) error {
	order, err := s.repo.GetByID(ctx, orderID)
	if err != nil {
		return fmt.Errorf("get order: %w", err)
	}

	if order.Status != StatusPaid {
		return fmt.Errorf("订单状态不允许发货: 当前状态=%s", order.Status)
	}

	ok, err := s.repo.UpdateStatus(ctx, orderID, StatusPaid, StatusShipped)
	if err != nil {
		return fmt.Errorf("update status: %w", err)
	}
	if !ok {
		return fmt.Errorf("发货失败，订单状态已变更或并发冲突")
	}

	log.Printf("[order] 发货成功: orderID=%d", orderID)
	return nil
}

func (s *Service) CompleteOrder(ctx context.Context, orderID int64) error {
	order, err := s.repo.GetByID(ctx, orderID)
	if err != nil {
		return fmt.Errorf("get order: %w", err)
	}

	if order.Status != StatusShipped {
		return fmt.Errorf("订单状态不允许完成: 当前状态=%s", order.Status)
	}

	ok, err := s.repo.UpdateStatus(ctx, orderID, StatusShipped, StatusCompleted)
	if err != nil {
		return fmt.Errorf("update status: %w", err)
	}
	if !ok {
		return fmt.Errorf("完成订单失败，订单状态已变更或并发冲突")
	}

	log.Printf("[order] 订单完成: orderID=%d", orderID)
	return nil
}

func (s *Service) CancelOrder(ctx context.Context, orderID int64) (bool, error) {
	order, err := s.repo.GetByID(ctx, orderID)
	if err != nil {
		return false, fmt.Errorf("get order: %w", err)
	}

	if order.Status != StatusCreated && order.Status != StatusPending {
		return false, fmt.Errorf("订单状态不允许取消: 当前状态=%s", order.Status)
	}

	ok, err := s.repo.UpdateStatus(ctx, orderID, order.Status, StatusCancelled)
	if err != nil {
		return false, fmt.Errorf("update status: %w", err)
	}
	if !ok {
		return false, fmt.Errorf("取消订单失败，订单状态已变更或并发冲突")
	}

	log.Printf("[order] 订单已取消: orderID=%d", orderID)
	return true, nil
}

func (s *Service) RefundOrder(ctx context.Context, orderID int64) error {
	order, err := s.repo.GetByID(ctx, orderID)
	if err != nil {
		return fmt.Errorf("get order: %w", err)
	}

	var oldStatus OrderStatus
	switch order.Status {
	case StatusPaid, StatusShipped:
		oldStatus = order.Status
	default:
		return fmt.Errorf("订单状态不允许退款: 当前状态=%s", order.Status)
	}

	ok, err := s.repo.UpdateStatus(ctx, orderID, oldStatus, StatusRefunded)
	if err != nil {
		return fmt.Errorf("update status: %w", err)
	}
	if !ok {
		return fmt.Errorf("退款失败，订单状态已变更或并发冲突")
	}

	log.Printf("[order] 订单已退款: orderID=%d", orderID)
	return nil
}

func (s *Service) GetOrder(ctx context.Context, orderID int64) (*Order, error) {
	return s.repo.GetByID(ctx, orderID)
}

func (s *Service) GetUserOrders(ctx context.Context, userID string, page, pageSize int) ([]*Order, error) {
	offset := (page - 1) * pageSize
	return s.repo.GetByUserID(ctx, userID, pageSize, offset)
}

func (s *Service) ExpireOrders(ctx context.Context) ([]string, error) {
	orderIDs, err := s.repo.ExpireOrders(ctx, time.Now())
	if err != nil {
		return nil, fmt.Errorf("query expire orders: %w", err)
	}

	var expired []string
	for _, orderID := range orderIDs {
		id, err := strconv.ParseInt(orderID, 10, 64)
		if err != nil {
			log.Printf("[order] 过期订单ID解析失败: orderID=%s err=%v", orderID, err)
			continue
		}
		ok, err := s.CancelOrder(ctx, id)
		if err != nil {
			log.Printf("[order] 处理过期订单失败: orderID=%s err=%v", orderID, err)
			continue
		}
		if ok {
			expired = append(expired, orderID)
		}
	}

	log.Printf("[order] 处理过期订单: 总数=%d 成功=%d", len(orderIDs), len(expired))
	return expired, nil
}

func (s *Service) StartExpireScanner(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	log.Printf("[order] 启动过期订单扫描器: interval=%v", interval)

	for {
		select {
		case <-ctx.Done():
			log.Println("[order] 过期订单扫描器已停止")
			return
		case <-ticker.C:
			if _, err := s.ExpireOrders(ctx); err != nil {
				log.Printf("[order] 扫描过期订单失败: %v", err)
			}
		}
	}
}
