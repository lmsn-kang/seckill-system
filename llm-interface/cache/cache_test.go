package cache

import (

    "fmt"

    "sync"

    "testing"

    "time"

)

func TestTwoLevelCache_Basic(t *testing.T) {

    cfg := &Config{

        L1MaxItems:    100,

        L1TTL:         5 * time.Second,

        RedisAddr:     "localhost:6379",

        RedisPassword: "",

        RedisDB:       15,

        L2TTL:         10 * time.Second,

    }

    c := NewTwoLevelCache(cfg)

    defer c.Close()

    time.Sleep(100 * time.Millisecond)

    result := &CacheResult{

        IsSafe:     false,

        Label:      "谩骂引战",

        Confidence: 0.92,

        AllScores:  map[string]float32{"安全": 0.03, "谩骂引战": 0.92},

    }

    c.Set("你脑子有病", result)

    time.Sleep(50 * time.Millisecond)

    got, ok := c.Get("你脑子有病")

    if !ok {

        t.Fatal("应该命中缓存")

    }

    if got.Label != "谩骂引战" {

        t.Fatalf("label 不匹配: got %s", got.Label)

    }

    _, ok = c.Get("今天天气真好")

    if ok {

        t.Fatal("不应该命中")

    }

    t.Logf("Stats: %+v", c.Stats())

}

func TestTwoLevelCache_L2Backfill(t *testing.T) {

    cfg := &Config{

        L1MaxItems:    100,

        L1TTL:         2 * time.Second,

        RedisAddr:     "localhost:6379",

        RedisPassword: "",

        RedisDB:       15,

        L2TTL:         30 * time.Second,

    }

    c := NewTwoLevelCache(cfg)

    defer c.Close()

    result := &CacheResult{Label: "广告欺诈", Confidence: 0.88}

    c.Set("加微信发财", result)

    time.Sleep(100 * time.Millisecond)

    _, ok := c.Get("加微信发财")

    if !ok {

        t.Fatal("L1 应该命中")

    }

    time.Sleep(3 * time.Second)

    got, ok := c.Get("加微信发财")

    if !ok {

        t.Fatal("L2 应该命中并回填 L1")

    }

    if got.Label != "广告欺诈" {

        t.Fatalf("回填数据不对: %s", got.Label)

    }

    time.Sleep(50 * time.Millisecond)

    _, ok = c.Get("加微信发财")

    if !ok {

        t.Fatal("回填后 L1 应该命中")

    }

    t.Logf("Stats: %+v", c.Stats())

}

func TestTwoLevelCache_Concurrent(t *testing.T) {

    cfg := &Config{

        L1MaxItems: 1000,

        L1TTL:      10 * time.Second,

        RedisAddr:  "localhost:6379",

        RedisDB:    15,

        L2TTL:      30 * time.Second,

    }

    c := NewTwoLevelCache(cfg)

    defer c.Close()

    var wg sync.WaitGroup

    for i := 0; i < 100; i++ {

        wg.Add(1)

        go func(id int) {

            defer wg.Done()

            text := fmt.Sprintf("测试文本_%d", id%20)

            c.Set(text, &CacheResult{

                Label:      fmt.Sprintf("标签_%d", id%5),

                Confidence: float32(id%100) / 100.0,

            })

            time.Sleep(10 * time.Millisecond)

            for j := 0; j < 10; j++ {

                c.Get(text)

            }

        }(i)

    }

    wg.Wait()

    stats := c.Stats()

    t.Logf("并发测试完成 | Stats: %+v", stats)

}

func BenchmarkL1Get(b *testing.B) {

    l1 := NewL1Cache(10000, 10*time.Minute)

    defer l1.Close()

    for i := 0; i < 1000; i++ {

        key := MakeKey(fmt.Sprintf("text_%d", i))

        l1.Set(key, &CacheResult{Label: "安全"})

    }

    time.Sleep(100 * time.Millisecond)

    b.ResetTimer()

    b.RunParallel(func(pb *testing.PB) {

        i := 0

        for pb.Next() {

            key := MakeKey(fmt.Sprintf("text_%d", i%1000))

            l1.Get(key)

            i++

        }

    })

}