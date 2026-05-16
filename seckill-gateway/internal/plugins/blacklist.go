package plugins

import (
	"net"
	"net/http"
	"strings"
	"sync"

	"github.com/gin-gonic/gin"
)

type BlacklistPlugin struct {
	mu   sync.RWMutex
	list map[string]bool
}

func NewBlacklist(ips []string) *BlacklistPlugin {
	m := make(map[string]bool, len(ips))
	for _, ip := range ips {
		m[ip] = true
	}
	return &BlacklistPlugin{list: m}
}

func (p *BlacklistPlugin) Name() string  { return "ip-blacklist" }
func (p *BlacklistPlugin) Priority() int { return 10 }

func (p *BlacklistPlugin) Handler() gin.HandlerFunc {
	return func(c *gin.Context) {
		ip := ClientIP(c.Request)

		p.mu.RLock()
		blocked := p.list[ip]
		p.mu.RUnlock()

		if blocked {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"code": 403, "message": "forbidden",
			})
			return
		}
		c.Next()
	}
}

func (p *BlacklistPlugin) Reload(ips []string) {
	m := make(map[string]bool, len(ips))
	for _, ip := range ips {
		m[ip] = true
	}
	p.mu.Lock()
	p.list = m
	p.mu.Unlock()
}

func (p *BlacklistPlugin) Add(ip string) {
	p.mu.Lock()
	p.list[ip] = true
	p.mu.Unlock()
}

func ClientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		return strings.TrimSpace(strings.Split(xff, ",")[0])
	}
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}
	ip, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return ip
}
