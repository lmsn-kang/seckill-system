package main

import (
	"math"
	"sync/atomic"
)

type Metrics struct {
	successCount  atomic.Int64
	errorCount    atomic.Int64
	cacheHitCount atomic.Int64

	successLatencySumMs atomic.Int64
	errorLatencySumMs   atomic.Int64
	cacheHitLatSumMs    atomic.Int64
}

func NewMetrics() *Metrics {
	return &Metrics{}
}

func (m *Metrics) RecordSuccess(ms int64) {
	m.successCount.Add(1)
	m.successLatencySumMs.Add(ms)
}

func (m *Metrics) RecordError(ms int64) {
	m.errorCount.Add(1)
	m.errorLatencySumMs.Add(ms)
}

func (m *Metrics) RecordCacheHit(ms int64) {
	m.cacheHitCount.Add(1)
	m.cacheHitLatSumMs.Add(ms)
}

func (m *Metrics) Snapshot() map[string]interface{} {
	sc := m.successCount.Load()
	ec := m.errorCount.Load()
	chc := m.cacheHitCount.Load()

	sLat := m.successLatencySumMs.Load()
	eLat := m.errorLatencySumMs.Load()
	chLat := m.cacheHitLatSumMs.Load()

	var avgSuccess, avgError, avgCache float64
	if sc > 0 {
		avgSuccess = float64(sLat) / float64(sc)
	}
	if ec > 0 {
		avgError = float64(eLat) / float64(ec)
	}
	if chc > 0 {
		avgCache = float64(chLat) / float64(chc)
	}

	total := sc + ec
	var successRate float64
	if total > 0 {
		successRate = float64(sc) / float64(total)
	}

	_ = math.Max

	return map[string]interface{}{
		"success_count":    sc,
		"error_count":      ec,
		"cache_hit_count":  chc,
		"avg_success_ms":   avgSuccess,
		"avg_error_ms":     avgError,
		"avg_cache_hit_ms": avgCache,
		"success_rate":     successRate,
	}
}