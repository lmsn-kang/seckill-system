package plugins

import (
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	httpRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gateway_http_requests_total",
			Help: "Total HTTP requests",
		},
		[]string{"path", "status"},
	)

	httpRequestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "gateway_http_request_duration_seconds",
			Help:    "HTTP request duration",
			Buckets: []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1},
		},
		[]string{"path"},
	)

	cacheHitsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gateway_cache_hits_total",
			Help: "Cache hits by level",
		},
		[]string{"level"},
	)
)

func init() {
	prometheus.MustRegister(httpRequestsTotal, httpRequestDuration, cacheHitsTotal)
}

type MetricsPlugin struct {
	metricsPath string
}

func NewMetrics(path string) *MetricsPlugin {
	return &MetricsPlugin{metricsPath: path}
}

func (p *MetricsPlugin) Name() string  { return "prometheus-metrics" }
func (p *MetricsPlugin) Priority() int { return 5 }

func (p *MetricsPlugin) Handler() gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.URL.Path == p.metricsPath {
			promhttp.Handler().ServeHTTP(c.Writer, c.Request)
			c.Abort()
			return
		}

		start := time.Now()
		c.Next()
		duration := time.Since(start).Seconds()

		path := c.Request.URL.Path
		status := strconv.Itoa(c.Writer.Status())
		httpRequestsTotal.WithLabelValues(path, status).Inc()
		httpRequestDuration.WithLabelValues(path).Observe(duration)

		if source := c.Writer.Header().Get("X-Gateway-Cache"); source != "" {
			cacheHitsTotal.WithLabelValues(source).Inc()
		}
	}
}

func RecordCacheHit(level string) {
	cacheHitsTotal.WithLabelValues(level).Inc()
}
