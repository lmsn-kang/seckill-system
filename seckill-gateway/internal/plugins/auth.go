package plugins

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
)

type AuthPlugin struct {
	secret         []byte
	protectedPaths map[string]bool
	publicPaths    map[string]bool
}

func NewAuth(secret string, protected, public []string) *AuthPlugin {
	pp := make(map[string]bool, len(protected))
	for _, p := range protected {
		pp[p] = true
	}
	pub := make(map[string]bool, len(public))
	for _, p := range public {
		pub[p] = true
	}
	return &AuthPlugin{
		secret:         []byte(secret),
		protectedPaths: pp,
		publicPaths:    pub,
	}
}

func (p *AuthPlugin) Name() string  { return "jwt-auth" }
func (p *AuthPlugin) Priority() int { return 30 }

func (p *AuthPlugin) Handler() gin.HandlerFunc {
	return func(c *gin.Context) {
		path := c.Request.URL.Path

		if p.publicPaths[path] {
			c.Next()
			return
		}
		if !p.protectedPaths[path] {
			c.Next()
			return
		}

		tokenStr := extractToken(c.Request)
		if tokenStr == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"code": 401, "message": "authorization token required",
			})
			return
		}

		claims, err := p.parseJWT(tokenStr)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"code": 401, "message": "invalid token: " + err.Error(),
			})
			return
		}

		c.Set("user_id", claims.UserID)
		c.Request.Header.Set("X-User-ID", claims.UserID)
		c.Next()
	}
}

type Claims struct {
	UserID string `json:"user_id"`
	jwt.RegisteredClaims
}

func (p *AuthPlugin) parseJWT(tokenStr string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenStr, &Claims{}, func(t *jwt.Token) (interface{}, error) {
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", t.Header["alg"])
		}
		return p.secret, nil
	})
	if err != nil {
		return nil, err
	}
	claims, ok := token.Claims.(*Claims)
	if !ok || !token.Valid {
		return nil, fmt.Errorf("invalid token")
	}
	return claims, nil
}

func (p *AuthPlugin) GenerateToken(userID string) (string, error) {
	claims := Claims{
		UserID: userID,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(24 * time.Hour)),
		},
	}
	return jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString(p.secret)
}

func extractToken(r *http.Request) string {
	if auth := r.Header.Get("Authorization"); strings.HasPrefix(auth, "Bearer ") {
		return strings.TrimPrefix(auth, "Bearer ")
	}
	return r.URL.Query().Get("token")
}
