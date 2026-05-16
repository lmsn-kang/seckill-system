package plugin

import (
	"fmt"
	"sort"

	"github.com/gin-gonic/gin"
)

type Plugin interface {
	Name() string
	Priority() int
}

type GinPlugin interface {
	Plugin
	Handler() gin.HandlerFunc
}

type Chain struct {
	plugins []GinPlugin
}

func NewChain() *Chain {
	return &Chain{}
}

func (c *Chain) Use(p GinPlugin) {
	c.plugins = append(c.plugins, p)
	sort.Slice(c.plugins, func(i, j int) bool {
		return c.plugins[i].Priority() < c.plugins[j].Priority()
	})
}

func (c *Chain) Apply(r *gin.Engine) {
	for _, p := range c.plugins {
		r.Use(p.Handler())
	}
	c.Print()
}

func (c *Chain) Print() {
	fmt.Println("╔═══════════════════════════════════════════╗")
	fmt.Println("║            Plugin Chain Order             ║")
	fmt.Println("╠═══════════════════════════════════════════╣")
	for i, p := range c.plugins {
		fmt.Printf("║  [%d] %-28s P=%-4d  ║\n", i+1, p.Name(), p.Priority())
	}
	fmt.Println("╚═══════════════════════════════════════════╝")
}
