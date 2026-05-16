package service

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/bwmarrin/snowflake"
	"github.com/redis/go-redis/v9"
	"github.com/segmentio/kafka-go"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"

	pb "seckill/pb"
	"seckill/internal/order"
	"seckill/internal/stock"
)

var decrStockLua = redis.NewScript(`
local stock = tonumber(redis.call('GET', KEYS[1]))
if not stock or stock <= 0 then
    return -1
end
redis.call('DECR', KEYS[1])
return stock - 1
`)

type SeckillService struct {
	pb.UnimplementedSeckillServiceServer
	rdb         *redis.Client
	writer      *kafka.Writer
	notifier    *stock.Notifier
	orderService *order.Service
	sfNode      *snowflake.Node
}

func NewSeckillService(rdb *redis.Client, writer *kafka.Writer, notifier *stock.Notifier, orderService *order.Service, sfNode *snowflake.Node) *SeckillService {
	return &SeckillService{
		rdb:         rdb,
		writer:      writer,
		notifier:    notifier,
		orderService: orderService,
		sfNode:      sfNode,
	}
}

func (s *SeckillService) DoSeckill(ctx context.Context, req *pb.SeckillRequest) (*pb.SeckillResponse, error) {
	_, span := otel.Tracer("seckill-svc").Start(ctx, "DoSeckill")
	defer span.End()

	if req == nil {
		return &pb.SeckillResponse{Code: 3, Msg: "invalid request"}, nil
	}
	userID := req.UserId
	goodsID := req.GoodsId
	if goodsID == "" {
		goodsID = "default"
	}
	if userID == "" {
		return &pb.SeckillResponse{Code: 3, Msg: "user_id required"}, nil
	}
	if s.orderService == nil || s.sfNode == nil {
		return &pb.SeckillResponse{Code: 3, Msg: "order service unavailable"}, nil
	}

	span.SetAttributes(
		attribute.String("user_id", userID),
		attribute.String("goods_id", goodsID),
	)

	dedupKey := fmt.Sprintf("sk:dedup:%s:%s", goodsID, userID)
	dedupOK, err := s.rdb.SetNX(ctx, dedupKey, 1, 30*time.Minute).Result()
	if err != nil {
		return &pb.SeckillResponse{Code: 3, Msg: "system error"}, nil
	}
	if !dedupOK {
		return &pb.SeckillResponse{Code: 2, Msg: "already purchased"}, nil
	}

	stockKey := fmt.Sprintf("sk:stock:%s", goodsID)
	remaining, err := decrStockLua.Run(ctx, s.rdb, []string{stockKey}).Int64()
	if err != nil {
		s.rdb.Del(ctx, dedupKey)
		return &pb.SeckillResponse{Code: 3, Msg: "system error"}, nil
	}
	if remaining == -1 {
		s.rdb.Del(ctx, dedupKey)

		_ = s.notifier.NotifySoldOut(ctx, goodsID)

		return &pb.SeckillResponse{Code: 1, Msg: "sold out", Stock: 0}, nil
	}

	orderID := s.sfNode.Generate().Int64()
	_, err = s.orderService.CreateOrder(ctx, &order.CreateOrderDTO{
		OrderID:    orderID,
		UserID:     userID,
		GoodsID:    goodsID,
		Amount:     100,
		ExpireTime: time.Now().Add(15 * time.Minute),
	})
	if err != nil {
		s.rdb.Incr(ctx, stockKey)
		s.rdb.Del(ctx, dedupKey)
		log.Printf("[seckill] create order failed, rolled back: %v", err)
		return &pb.SeckillResponse{Code: 3, Msg: "system error"}, nil
	}

	if remaining == 0 {
		_ = s.notifier.NotifySoldOut(ctx, goodsID)
	}

	msg, _ := json.Marshal(map[string]any{
		"order_id": orderID,
		"user_id":  userID,
		"goods_id": goodsID,
	})
	if err := s.writer.WriteMessages(ctx, kafka.Message{
		Key:   []byte(userID),
		Value: msg,
	}); err != nil {
		log.Printf("[seckill] kafka write failed (order already created): %v", err)
	}

	return &pb.SeckillResponse{
		Code:        0,
		OrderId:     orderID,
		Msg:         "success, order created",
		Stock:       remaining,
		OrderStatus: pb.OrderStatus(order.StatusCreated),
	}, nil
}

func (s *SeckillService) GetStock(ctx context.Context, req *pb.StockRequest) (*pb.StockResponse, error) {
	goodsID := "default"
	if req != nil && req.GoodsId != "" {
		goodsID = req.GoodsId
	}

	stockKey := fmt.Sprintf("sk:stock:%s", goodsID)
	val, err := s.rdb.Get(ctx, stockKey).Int64()
	if err == redis.Nil {
		return &pb.StockResponse{GoodsId: goodsID, Stock: 0, SoldOut: true}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("get stock: %w", err)
	}

	return &pb.StockResponse{
		GoodsId: goodsID,
		Stock:   val,
		SoldOut: val <= 0,
	}, nil
}
