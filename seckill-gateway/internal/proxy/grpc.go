package proxy

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "seckill-gateway/pb"
)

type GRPCProxy struct {
	Client pb.SeckillServiceClient
	conn   *grpc.ClientConn
}

func NewGRPCProxy(target string) *GRPCProxy {
	conn, err := grpc.Dial(target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		log.Fatalf("[grpc-proxy] connect failed: %v", err)
	}
	log.Printf("[grpc-proxy] connected to backend: %s", target)

	return &GRPCProxy{
		Client: pb.NewSeckillServiceClient(conn),
		conn:   conn,
	}
}

func (p *GRPCProxy) Close() {
	if p.conn != nil {
		p.conn.Close()
	}
}

func (p *GRPCProxy) RegisterRoutes(r *gin.Engine) {
	r.POST("/api/seckill", p.handleSeckill)
	r.GET("/api/stock", p.handleStock)
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})
}

func (p *GRPCProxy) handleSeckill(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 3*time.Second)
	defer cancel()

	var req struct {
		UserID  string `json:"user_id"`
		GoodsID string `json:"goods_id"`
	}

	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": 1, "msg": "invalid request body"})
		return
	}
	if len(body) > 0 {
		if err := json.Unmarshal(body, &req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"code": 1, "msg": "invalid json body"})
			return
		}
	}
	if req.UserID == "" {
		req.UserID = c.GetHeader("X-User-ID")
	}
	if req.GoodsID == "" {
		req.GoodsID = c.DefaultQuery("goods_id", "default")
	}

	resp, err := p.Client.DoSeckill(ctx, &pb.SeckillRequest{
		UserId:  req.UserID,
		GoodsId: req.GoodsID,
	})
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"code": 3, "msg": err.Error()})
		return
	}

	if resp.Code == 1 {
		c.Header("X-Sold-Out", "true")
	}

	c.JSON(http.StatusOK, gin.H{
		"code":     resp.Code,
		"msg":      resp.Msg,
		"stock":    resp.Stock,
		"order_id": resp.OrderId,
	})
}

func (p *GRPCProxy) handleStock(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 2*time.Second)
	defer cancel()

	goodsID := c.DefaultQuery("goods_id", "default")

	resp, err := p.Client.GetStock(ctx, &pb.StockRequest{
		GoodsId: goodsID,
	})
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"code": 3, "msg": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"goods_id": resp.GoodsId,
		"stock":    resp.Stock,
		"sold_out": resp.SoldOut,
	})
}
