package middleware

import (
	"log"
	"net/http"
	"runtime/debug"

	"github.com/gin-gonic/gin"
)

func Recovery() gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			if err := recover(); err != nil {
				log.Printf("[recovery] panic: %v\n%s", err, debug.Stack())
				c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{
					"code": 500, "message": "internal server error",
				})
			}
		}()
		c.Next()
	}
}
