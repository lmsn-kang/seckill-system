package plugins

import (
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sony/gobreaker/v2"
)

type CircuitBreakerPlugin struct {
	cb *gobreaker.CircuitBreaker[struct{}]
}

func NewCircuitBreaker(threshold, timeoutSec, maxHalfOpen int) *CircuitBreakerPlugin {
	settings := gobreaker.Settings{
		Name:        "backend-breaker",
		MaxRequests: uint32(maxHalfOpen),
		Interval:    10 * time.Second,
		Timeout:     time.Duration(timeoutSec) * time.Second,

		ReadyToTrip: func(counts gobreaker.Counts) bool {
			return int(counts.ConsecutiveFailures) >= threshold
		},

		OnStateChange: func(name string, from, to gobreaker.State) {
			log.Printf("[circuitbreaker] ⚡ %s: %s → %s", name, from, to)
		},
	}

	return &CircuitBreakerPlugin{
		cb: gobreaker.NewCircuitBreaker[struct{}](settings),
	}
}

func (p *CircuitBreakerPlugin) Name() string  { return "circuit-breaker" }
func (p *CircuitBreakerPlugin) Priority() int { return 35 }

func (p *CircuitBreakerPlugin) Handler() gin.HandlerFunc {
	return func(c *gin.Context) {
		_, err := p.cb.Execute(func() (struct{}, error) {
			c.Next()
			if c.Writer.Status() >= http.StatusInternalServerError {
				return struct{}{}, fmt.Errorf("backend error: %d", c.Writer.Status())
			}
			return struct{}{}, nil
		})

		if err != nil {
			if err == gobreaker.ErrOpenState || err == gobreaker.ErrTooManyRequests {
				c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{
					"code":    503,
					"message": "service temporarily unavailable (circuit open)",
				})
			}
		}
	}
}

func (p *CircuitBreakerPlugin) State() string {
	return p.cb.State().String()
}
