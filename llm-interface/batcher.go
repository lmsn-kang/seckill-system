package main

import (
	"fmt"
	"log"
	"sync/atomic"
	"time"
)

type InferItem struct {
	Text     string
	Image    []byte
	ResultCh chan *InferResult
}

type InferResult struct {
	IsSafe     bool
	Label      string
	Confidence float32
	AllScores  map[string]float32
	Error      error
}

type Batcher struct {
	inputCh chan *InferItem

	bridge *PythonBridge

	maxBatchSize int
	maxWait      time.Duration

	totalBatches atomic.Int64
	totalItems   atomic.Int64
}

func NewBatcher(bridge *PythonBridge, queueSize, maxBatch int, maxWait time.Duration) *Batcher {
	return &Batcher{
		inputCh:      make(chan *InferItem, queueSize),
		bridge:       bridge,
		maxBatchSize: maxBatch,
		maxWait:      maxWait,
	}
}

func (b *Batcher) Submit(item *InferItem) bool {
	select {
	case b.inputCh <- item:
		return true
	default:
		return false
	}
}

func (b *Batcher) Run() {
	log.Printf("[Batcher] 启动 | batch_size=%d | wait=%v | queue=%d",
		b.maxBatchSize, b.maxWait, cap(b.inputCh))

	for {
		first, ok := <-b.inputCh
		if !ok {
			log.Println("[Batcher] channel 关闭，退出")
			return
		}

		batch := make([]*InferItem, 0, b.maxBatchSize)
		batch = append(batch, first)

		timer := time.NewTimer(b.maxWait)

	collect:
		for len(batch) < b.maxBatchSize {
			select {
			case item, ok := <-b.inputCh:
				if !ok {
					break collect
				}
				batch = append(batch, item)

			case <-timer.C:
				break collect
			}
		}
		timer.Stop()

		b.processBatch(batch)
	}
}

func (b *Batcher) processBatch(batch []*InferItem) {
	batchSize := len(batch)
	start := time.Now()

	results, err := b.bridge.RunBatch(batch)

	elapsed := time.Since(start)

	if err != nil {
		log.Printf("[Batcher] ❌ batch 失败: %v", err)
		for _, item := range batch {
			item.ResultCh <- &InferResult{Error: err}
		}
		return
	}

	for i, item := range batch {
		if i < len(results) {
			item.ResultCh <- results[i]
		} else {
			item.ResultCh <- &InferResult{Error: fmt.Errorf("结果数量不匹配")}
		}
	}

	b.totalBatches.Add(1)
	b.totalItems.Add(int64(batchSize))

	log.Printf("[Batcher] ✅ batch=%d | %v | avg=%v/item",
		batchSize, elapsed, elapsed/time.Duration(batchSize))
}

func (b *Batcher) Stats() map[string]interface{} {
	batches := b.totalBatches.Load()
	items := b.totalItems.Load()
	avgBatch := float64(0)
	if batches > 0 {
		avgBatch = float64(items) / float64(batches)
	}
	return map[string]interface{}{
		"pending":        len(b.inputCh),
		"capacity":       cap(b.inputCh),
		"total_batches":  batches,
		"total_items":    items,
		"avg_batch_size": avgBatch,
	}
}