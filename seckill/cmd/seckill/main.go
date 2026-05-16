package main

import (
	"context"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	"github.com/segmentio/kafka-go"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"google.golang.org/grpc"

	pb "seckill/pb"
	"github.com/bwmarrin/snowflake"
	"seckill/internal/comment"
	"seckill/internal/order"
	"seckill/internal/service"
	"seckill/internal/stock"
	"seckill/pkg/discovery"
	"seckill/pkg/tracing"
)

var (
	orderService   *order.Service
	orderRepo      order.Repository
	commentService *comment.Service
	commentRepo    comment.Repository
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	tp, _ := tracing.Init(ctx, "seckill-svc", env("JAEGER_ENDPOINT", "localhost:4317"))
	defer tp.Shutdown(context.Background())

	rdb := redis.NewClient(&redis.Options{
		Addr:     env("REDIS_ADDR", "localhost:6379"),
		PoolSize: 200,
	})
	rdb.Set(ctx, "sk:stock:goods_001", 100, 0)
	rdb.Set(ctx, "sk:stock:default", 100, 0)
	log.Println("[seckill] stock initialized: goods_001=100, default=100")

	repo, err := order.NewRepository(env("MYSQL_DSN", ""))
	if err != nil {
		log.Printf("[seckill] MySQL连接失败: %v", err)
	} else {
		orderRepo = repo
		orderService = order.NewService(repo)
		go orderService.StartExpireScanner(ctx, 5*time.Minute)
		log.Println("[seckill] order service started with expire scanner")
	}

	if orderRepo != nil {
		commentRepo, err = comment.NewRepository(env("MYSQL_DSN", ""))
		if err != nil {
			log.Printf("[seckill] 评论仓储初始化失败: %v", err)
		} else {
			commentService = comment.NewService(commentRepo)
			modClient := comment.NewModerationClient(env("MODERATION_ADDR", "http://localhost:9090"))
			commentService.SetModerationClient(modClient)
			log.Println("[seckill] comment service started, moderation client injected")
		}
	}

	notifier := stock.NewNotifier(rdb, env("SOLDOUT_CHANNEL", "sk:soldout"))

	writer := &kafka.Writer{
		Addr:         kafka.TCP(env("KAFKA_ADDR", "localhost:9092")),
		Topic:        "seckill-orders",
		Balancer:     &kafka.Hash{},
		BatchTimeout: 5 * time.Millisecond,
		RequiredAcks: kafka.RequireOne,
	}

	orderEventPublisher := order.NewEventPublisher([]string{env("KAFKA_ADDR", "localhost:9092")})
	if orderService != nil {
		orderService.SetPublisher(orderEventPublisher)
		log.Println("[seckill] order event publisher injected")
	}

	var sfNode *snowflake.Node
	workerID := int64(1)
	if w := os.Getenv("SNOWFLAKE_WORKER_ID"); w != "" {
		if wid, _ := strconv.ParseInt(w, 10, 64); wid > 0 {
			workerID = wid
		}
	}
	sfNode, err = snowflake.NewNode(workerID)
	if err != nil {
		log.Fatalf("雪花算法初始化失败: %v", err)
	}
	log.Printf("[seckill] 雪花算法初始化成功: worker_id=%d", workerID)

	listenAddr := env("LISTEN_ADDR", ":9090")
	advertiseAddr := env("ADVERTISE_ADDR", "localhost:9090")

	lis, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatal(err)
	}

	srv := grpc.NewServer(grpc.StatsHandler(otelgrpc.NewServerHandler()))

	seckillSvc := service.NewSeckillService(rdb, writer, notifier, orderService, sfNode)
	pb.RegisterSeckillServiceServer(srv, seckillSvc)

	reg, err := discovery.NewRegistry([]string{env("ETCD_ADDR", "localhost:2379")})
	if err != nil {
		log.Printf("[seckill] etcd registry skipped: %v", err)
	} else {
		if err := reg.Register(ctx, "seckill-svc", advertiseAddr, 10); err != nil {
			log.Printf("[seckill] register failed: %v", err)
		}
	}

	go func() {
		log.Printf("[seckill] 🚀 serving on %s (advertise: %s)", listenAddr, advertiseAddr)
		if err := srv.Serve(lis); err != nil {
			log.Fatal(err)
		}
	}()

	go startHTTPServer(ctx, env("HTTP_ADDR", ":8080"))

	<-ctx.Done()
	srv.GracefulStop()
	if reg != nil {
		_ = reg.Deregister(context.Background())
	}
	_ = writer.Close()
	if orderEventPublisher != nil {
		_ = orderEventPublisher.Close()
	}
	if commentRepo != nil {
		_ = commentRepo.Close()
	}
	if orderRepo != nil {
		_ = orderRepo.Close()
	}
	log.Println("[seckill] shutdown complete ✅")
}

type PayCallbackRequest struct {
	OrderID     int64  `json:"order_id"`
	Transaction string `json:"transaction"`
	Amount      int64  `json:"amount"`
	Status      string `json:"status"`
}

func startHTTPServer(ctx context.Context, addr string) {
	router := gin.Default()

	router.POST("/pay/callback", func(c *gin.Context) {
		var req PayCallbackRequest
		if err := c.BindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"code": 1, "msg": "invalid request"})
			return
		}

		if req.Status != "success" {
			c.JSON(http.StatusOK, gin.H{"code": 0, "msg": "payment failed, ignored"})
			return
		}

		if orderService == nil {
			c.JSON(http.StatusInternalServerError, gin.H{"code": 1, "msg": "order service not available"})
			return
		}

		if err := orderService.PublishTransition(c.Request.Context(), req.OrderID, order.EventOrderPaid, order.EventPayload{
			Transaction: req.Transaction,
		}); err != nil {
			log.Printf("[pay] 发布支付事件失败: orderID=%d err=%v", req.OrderID, err)
			c.JSON(http.StatusOK, gin.H{"code": 1, "msg": err.Error()})
			return
		}

		log.Printf("[pay] 支付事件已发布: orderID=%d transaction=%s", req.OrderID, req.Transaction)
		c.JSON(http.StatusOK, gin.H{"code": 0, "msg": "payment event published"})
	})

	router.POST("/order/:order_id/ship", func(c *gin.Context) {
		orderIDStr := c.Param("order_id")
		orderID, _ := strconv.ParseInt(orderIDStr, 10, 64)
		if orderID == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"code": 1, "msg": "invalid order_id"})
			return
		}
		if orderService == nil {
			c.JSON(http.StatusInternalServerError, gin.H{"code": 1, "msg": "order service not available"})
			return
		}
		if err := orderService.PublishTransition(c.Request.Context(), orderID, order.EventOrderShipped, order.EventPayload{}); err != nil {
			c.JSON(http.StatusOK, gin.H{"code": 1, "msg": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"code": 0, "msg": "ship event published"})
	})

	router.POST("/order/:order_id/complete", func(c *gin.Context) {
		orderIDStr := c.Param("order_id")
		orderID, _ := strconv.ParseInt(orderIDStr, 10, 64)
		if orderID == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"code": 1, "msg": "invalid order_id"})
			return
		}
		if orderService == nil {
			c.JSON(http.StatusInternalServerError, gin.H{"code": 1, "msg": "order service not available"})
			return
		}
		if err := orderService.PublishTransition(c.Request.Context(), orderID, order.EventOrderCompleted, order.EventPayload{}); err != nil {
			c.JSON(http.StatusOK, gin.H{"code": 1, "msg": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"code": 0, "msg": "complete event published"})
	})

	router.POST("/order/:order_id/cancel", func(c *gin.Context) {
		orderIDStr := c.Param("order_id")
		orderID, _ := strconv.ParseInt(orderIDStr, 10, 64)
		if orderID == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"code": 1, "msg": "invalid order_id"})
			return
		}
		if orderService == nil {
			c.JSON(http.StatusInternalServerError, gin.H{"code": 1, "msg": "order service not available"})
			return
		}
		if err := orderService.PublishTransition(c.Request.Context(), orderID, order.EventOrderCancelled, order.EventPayload{}); err != nil {
			c.JSON(http.StatusOK, gin.H{"code": 1, "msg": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"code": 0, "msg": "cancel event published"})
	})

	router.POST("/order/:order_id/refund", func(c *gin.Context) {
		orderIDStr := c.Param("order_id")
		orderID, _ := strconv.ParseInt(orderIDStr, 10, 64)
		if orderID == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"code": 1, "msg": "invalid order_id"})
			return
		}
		if orderService == nil {
			c.JSON(http.StatusInternalServerError, gin.H{"code": 1, "msg": "order service not available"})
			return
		}
		if err := orderService.PublishTransition(c.Request.Context(), orderID, order.EventOrderRefunded, order.EventPayload{}); err != nil {
			c.JSON(http.StatusOK, gin.H{"code": 1, "msg": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"code": 0, "msg": "refund event published"})
	})

	router.GET("/order/:order_id", func(c *gin.Context) {
		orderIDStr := c.Param("order_id")
		orderID, _ := strconv.ParseInt(orderIDStr, 10, 64)

		if orderService == nil {
			c.JSON(http.StatusInternalServerError, gin.H{"code": 1, "msg": "order service not available"})
			return
		}

		o, err := orderService.GetOrder(ctx, orderID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"code": 1, "msg": "order not found"})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"code": 0,
			"data": gin.H{
				"order_id":    o.OrderID,
				"user_id":     o.UserID,
				"goods_id":    o.GoodsID,
				"status":      o.Status.String(),
				"amount":      o.Amount,
				"expire_time": o.ExpireTime,
				"created_at":  o.CreatedAt,
			},
		})
	})

	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	router.POST("/comment", func(c *gin.Context) {
		var dto comment.CreateCommentDTO
		if err := c.BindJSON(&dto); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"code": 1, "msg": "invalid request"})
			return
		}
		if commentService == nil {
			c.JSON(http.StatusInternalServerError, gin.H{"code": 1, "msg": "comment service not available"})
			return
		}
		comm, err := commentService.CreateComment(c.Request.Context(), &dto)
		if err != nil {
			c.JSON(http.StatusOK, gin.H{"code": 1, "msg": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{
			"code": 0,
			"msg":  "comment submitted",
			"data": gin.H{
				"comment_id": comm.ID,
				"status":     comm.Status.String(),
				"content":    comm.Content,
			},
		})
	})

	router.GET("/comments", func(c *gin.Context) {
		goodsID := c.Query("goods_id")
		if goodsID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"code": 1, "msg": "goods_id is required"})
			return
		}
		page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
		pageSize, _ := strconv.Atoi(c.DefaultQuery("page_size", "20"))
		if commentService == nil {
			c.JSON(http.StatusInternalServerError, gin.H{"code": 1, "msg": "comment service not available"})
			return
		}
		comments, err := commentService.GetGoodsComments(c.Request.Context(), goodsID, page, pageSize)
		if err != nil {
			c.JSON(http.StatusOK, gin.H{"code": 1, "msg": err.Error()})
			return
		}
		var list []gin.H
		for _, comm := range comments {
			list = append(list, gin.H{
				"comment_id": comm.ID,
				"user_id":    comm.UserID,
				"content":    comm.Content,
				"label":      comm.Label,
				"confidence": comm.Confidence,
				"created_at": comm.CreatedAt,
			})
		}
		c.JSON(http.StatusOK, gin.H{"code": 0, "data": list})
	})

	router.GET("/comment/:comment_id", func(c *gin.Context) {
		idStr := c.Param("comment_id")
		id, _ := strconv.ParseInt(idStr, 10, 64)
		if id == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"code": 1, "msg": "invalid comment_id"})
			return
		}
		if commentService == nil {
			c.JSON(http.StatusInternalServerError, gin.H{"code": 1, "msg": "comment service not available"})
			return
		}
		comm, err := commentService.GetComment(c.Request.Context(), id)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"code": 1, "msg": "comment not found"})
			return
		}
		c.JSON(http.StatusOK, gin.H{
			"code": 0,
			"data": gin.H{
				"comment_id": comm.ID,
				"order_id":   comm.OrderID,
				"user_id":    comm.UserID,
				"goods_id":   comm.GoodsID,
				"content":    comm.Content,
				"status":     comm.Status.String(),
				"label":      comm.Label,
				"confidence": comm.Confidence,
				"created_at": comm.CreatedAt,
				"updated_at": comm.UpdatedAt,
			},
		})
	})

	log.Printf("[seckill] HTTP server started on %s", addr)
	if err := router.Run(addr); err != nil {
		log.Printf("[seckill] HTTP server error: %v", err)
	}
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
