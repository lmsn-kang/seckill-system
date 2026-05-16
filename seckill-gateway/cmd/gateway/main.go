package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"

	"seckill-gateway/internal/config"
	"seckill-gateway/internal/middleware"
	"seckill-gateway/internal/plugin"
	"seckill-gateway/internal/plugins"
	"seckill-gateway/internal/proxy"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	cfgPath := envOr("CONFIG_PATH", "configs/gateway.yaml")
	cfg, err := config.Load(cfgPath)
	if err != nil {
		log.Fatalf("load config: %v", err)
	}
	if v := os.Getenv("BACKEND_GRPC_ADDR"); v != "" {
		cfg.Backend.GRPCAddr = v
	}
	if v := os.Getenv("REDIS_ADDR"); v != "" {
		cfg.Redis.Addr = v
	}

	rdb := redis.NewClient(&redis.Options{
		Addr:     cfg.Redis.Addr,
		PoolSize: cfg.Redis.PoolSize,
	})
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Fatalf("redis connect failed: %v", err)
	}
	log.Printf("[main] Redis connected: %s", cfg.Redis.Addr)

	grpcProxy := proxy.NewGRPCProxy(cfg.Backend.GRPCAddr)
	defer grpcProxy.Close()

	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(middleware.Recovery())

	chain := plugin.NewChain()

	var (
		metricsPlugin  *plugins.MetricsPlugin
		blacklistPl    *plugins.BlacklistPlugin
		rateLimitPl    *plugins.RateLimitPlugin
		authPlugin     *plugins.AuthPlugin
		breakerPlugin  *plugins.CircuitBreakerPlugin
		guardPlugin    *plugins.SeckillGuardPlugin
	)

	if cfg.Plugins.Metrics.Enabled {
		metricsPlugin = plugins.NewMetrics(cfg.Plugins.Metrics.Path)
		chain.Use(metricsPlugin)
	}

	if cfg.Plugins.Blacklist.Enabled {
		blacklistPl = plugins.NewBlacklist(cfg.Plugins.Blacklist.IPs)
		chain.Use(blacklistPl)
	}

	if cfg.Plugins.RateLimit.Enabled {
		rl := cfg.Plugins.RateLimit
		rateLimitPl = plugins.NewRateLimit(rdb, rl.GlobalRate, rl.GlobalCapacity, rl.IPRate, rl.IPCapacity)
		chain.Use(rateLimitPl)
	}

	if cfg.Plugins.Auth.Enabled {
		a := cfg.Plugins.Auth
		authPlugin = plugins.NewAuth(a.JWTSecret, a.ProtectedPaths, a.PublicPaths)
		chain.Use(authPlugin)
	}

	if cfg.Plugins.CircuitBreaker.Enabled {
		cb := cfg.Plugins.CircuitBreaker
		breakerPlugin = plugins.NewCircuitBreaker(cb.Threshold, cb.TimeoutSeconds, cb.MaxHalfOpen)
		chain.Use(breakerPlugin)
	}

	if cfg.Plugins.SeckillGuard.Enabled {
		sg := cfg.Plugins.SeckillGuard
		guardPlugin = plugins.NewSeckillGuard(
			rdb,
			sg.Paths,
			sg.LocalTTLSeconds,
			sg.RedisPollIntervalMs,
			sg.PubSubChannel,
		)
		chain.Use(guardPlugin)
	}

	chain.Apply(r)

	grpcProxy.RegisterRoutes(r)

	config.OnChange(func(newCfg *config.Config) {
		if blacklistPl != nil {
			blacklistPl.Reload(newCfg.Plugins.Blacklist.IPs)
		}
		if rateLimitPl != nil {
			rl := newCfg.Plugins.RateLimit
			rateLimitPl.Reload(rl.GlobalRate, rl.GlobalCapacity, rl.IPRate, rl.IPCapacity)
		}
		log.Printf("[main] ♻️ hot-reload applied")
	})
	config.Watch(cfgPath)

	go startAdminServer(cfg.Server.AdminPort, authPlugin, blacklistPl, guardPlugin, breakerPlugin)

	addr := fmt.Sprintf(":%d", cfg.Server.Port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      r,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	go func() {
		log.Printf("🚀 seckill-gateway started on %s", addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server error: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("[main] shutting down...")
	shutCtx, shutCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutCancel()
	_ = srv.Shutdown(shutCtx)
	log.Println("[main] shutdown complete ✅")
}

func startAdminServer(
	port int,
	auth *plugins.AuthPlugin,
	bl *plugins.BlacklistPlugin,
	guard *plugins.SeckillGuardPlugin,
	breaker *plugins.CircuitBreakerPlugin,
) {
	mux := http.NewServeMux()

	if auth != nil {
		mux.HandleFunc("/admin/token", func(w http.ResponseWriter, r *http.Request) {
			userID := r.URL.Query().Get("user")
			if userID == "" {
				userID = "testuser"
			}
			token, err := auth.GenerateToken(userID)
			if err != nil {
				http.Error(w, err.Error(), 500)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"token":"%s","user_id":"%s"}`, token, userID)
		})
	}

	if bl != nil {
		mux.HandleFunc("/admin/blacklist/add", func(w http.ResponseWriter, r *http.Request) {
			ip := r.URL.Query().Get("ip")
			if ip == "" {
				http.Error(w, "ip required", 400)
				return
			}
			bl.Add(ip)
			fmt.Fprintf(w, `{"added":"%s"}`, ip)
		})
	}

	if guard != nil {
		mux.HandleFunc("/admin/guard/reset", func(w http.ResponseWriter, r *http.Request) {
			guard.Reset(r.Context())
			fmt.Fprint(w, `{"status":"reset"}`)
		})
		mux.HandleFunc("/admin/guard/stats", func(w http.ResponseWriter, r *http.Request) {
			stats := guard.Cache().Stats()
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, "%v", stats)
		})
	}

	if breaker != nil {
		mux.HandleFunc("/admin/breaker/state", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"state":"%s"}`, breaker.State())
		})
	}

	addr := fmt.Sprintf(":%d", port)
	log.Printf("[admin] management API on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Printf("[admin] server error: %v", err)
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
