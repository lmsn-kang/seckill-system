package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	pb "seckill/llm-interface/pb"
	"seckill/llm-interface/cache"
	"google.golang.org/grpc"
)

type moderationServer struct {
	pb.UnimplementedModerationServiceServer

	cache   *cache.TwoLevelCache
	batcher *Batcher
	metrics *Metrics
}

func (s *moderationServer) Moderate(
	ctx context.Context, req *pb.ModerationRequest,
) (*pb.ModerationResponse, error) {

	start := time.Now()

	if req.TextContent != "" && len(req.ImageData) == 0 {
		if cached, ok := s.cache.Get(req.TextContent); ok {
			ms := time.Since(start).Milliseconds()
			s.metrics.RecordCacheHit(ms)
			return &pb.ModerationResponse{
				RequestId:  req.RequestId,
				IsSafe:     cached.IsSafe,
				Label:      cached.Label,
				Confidence: cached.Confidence,
				AllScores:  cached.AllScores,
				LatencyMs:  ms,
				FromCache:  true,
			}, nil
		}
	}
	item := &InferItem{
		Text:     req.TextContent,
		Image:    req.ImageData,
		ResultCh: make(chan *InferResult, 1),
	}

	if !s.batcher.Submit(item) {
		ms := time.Since(start).Milliseconds()
		s.metrics.RecordError(ms)
		return &pb.ModerationResponse{
			RequestId: req.RequestId,
			Error:     "服务过载，队列已满",
			LatencyMs: ms,
		}, nil
	}

	select {
	case result := <-item.ResultCh:
		ms := time.Since(start).Milliseconds()

		if result.Error != nil {
			s.metrics.RecordError(ms)
			return &pb.ModerationResponse{
				RequestId: req.RequestId,
				Error:     result.Error.Error(),
				LatencyMs: ms,
			}, nil
		}

		if req.TextContent != "" && len(req.ImageData) == 0 {
			s.cache.Set(req.TextContent, &cache.CacheResult{
				IsSafe:     result.IsSafe,
				Label:      result.Label,
				Confidence: result.Confidence,
				AllScores:  result.AllScores,
			})
		}

		s.metrics.RecordSuccess(ms)
		return &pb.ModerationResponse{
			RequestId:  req.RequestId,
			IsSafe:     result.IsSafe,
			Label:      result.Label,
			Confidence: result.Confidence,
			AllScores:  result.AllScores,
			LatencyMs:  ms,
		}, nil

	case <-ctx.Done():
		ms := time.Since(start).Milliseconds()
		s.metrics.RecordError(ms)
		return &pb.ModerationResponse{
			RequestId: req.RequestId,
			Error:     "请求超时",
			LatencyMs: ms,
		}, nil
	}
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	grpcPort := getEnv("GRPC_PORT", "50051")
	pythonAddr := getEnv("PYTHON_ADDR", "localhost:50052")
	monitorPort := getEnv("MONITOR_PORT", "9090")

	l1Max, _ := strconv.ParseInt(getEnv("CACHE_L1_MAX", "10000"), 10, 64)
	l1TTL, _ := time.ParseDuration(getEnv("CACHE_L1_TTL", "30m"))
	l2TTL, _ := time.ParseDuration(getEnv("CACHE_L2_TTL", "1h"))
	redisDB, _ := strconv.Atoi(getEnv("REDIS_DB", "0"))

	cacheCfg := &cache.Config{
		L1MaxItems:    l1Max,
		L1TTL:         l1TTL,
		RedisAddr:     getEnv("REDIS_ADDR", "localhost:6379"),
		RedisPassword: getEnv("REDIS_PASSWORD", ""),
		RedisDB:       redisDB,
		L2TTL:         l2TTL,
	}

	bridge := NewPythonBridge(pythonAddr)
	defer bridge.Close()

	twoLevelCache := cache.NewTwoLevelCache(cacheCfg)
	defer twoLevelCache.Close()
	metrics := NewMetrics()

	batcher := NewBatcher(bridge, 500, 8, 50*time.Millisecond)
	go batcher.Run()

	svc := &moderationServer{
		cache:   twoLevelCache,
		batcher: batcher,
		metrics: metrics,
	}

	lis, err := net.Listen("tcp", ":"+grpcPort)
	if err != nil {
		log.Fatalf("监听失败: %v", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterModerationServiceServer(grpcServer, svc)

	go func() {
		mux := http.NewServeMux()
		mux.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
			data := map[string]interface{}{
				"metrics": metrics.Snapshot(),
				"cache":   twoLevelCache.Stats(),
				"batcher": batcher.Stats(),
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(data)
		})
		mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprintln(w, `{"status":"ok"}`)
		})

		mux.HandleFunc("/moderate", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				w.WriteHeader(http.StatusMethodNotAllowed)
				return
			}

			var req struct {
				RequestID   string `json:"request_id"`
				TextContent string `json:"text_content"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				json.NewEncoder(w).Encode(map[string]interface{}{
					"error": "invalid json: " + err.Error(),
				})
				return
			}

			resp, err := svc.Moderate(r.Context(), &pb.ModerationRequest{
				RequestId:   req.RequestID,
				TextContent: req.TextContent,
			})
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				json.NewEncoder(w).Encode(map[string]interface{}{
					"error": err.Error(),
				})
				return
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"request_id":  resp.RequestId,
				"is_safe":     resp.IsSafe,
				"label":       resp.Label,
				"confidence":  resp.Confidence,
				"all_scores":  resp.AllScores,
				"latency_ms":  resp.LatencyMs,
				"from_cache":  resp.FromCache,
				"error":       resp.Error,
			})
		})

		log.Printf("📊 监控面板: http://localhost:%s/stats", monitorPort)
		log.Printf("🔐 Moderate API: http://localhost:%s/moderate", monitorPort)
		http.ListenAndServe(":"+monitorPort, mux)
	}()

	go func() {
		log.Printf("🚀 Go 推理调度服务启动")
		log.Printf("   gRPC:    :%s（你的后端调这个）", grpcPort)
		log.Printf("   Python:  %s（内部通信）", pythonAddr)
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("gRPC 服务失败: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("正在关闭...")
	grpcServer.GracefulStop()
	log.Println("✅ 已关闭")
}

func getEnv(k, v string) string {
	if e := os.Getenv(k); e != "" {
		return e
	}
	return v
}