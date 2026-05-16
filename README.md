# 分布式秒杀系统

## 项目结构

```
seckill-system/
├── seckill-gateway/          # 网关层：流量治理、协议转换、缓存拦截
│   ├── go.mod
│   ├── Makefile
│   ├── configs/
│   │   ├── gateway.yaml
│   │   └── nginx.conf
│   ├── proto/
│   │   └── seckill/v1/
│   │       └── seckill.proto
│   ├── pb/                   # protoc 生成
│   ├── cmd/
│   │   └── gateway/
│   │       └── main.go
│   ├── internal/
│   │   ├── config/
│   │   │   └── config.go
│   │   ├── plugin/
│   │   │   └── chain.go
│   │   ├── plugins/
│   │   │   ├── blacklist.go
│   │   │   ├── ratelimit.go
│   │   │   ├── auth.go
│   │   │   ├── circuitbreaker.go
│   │   │   ├── seckillguard.go
│   │   │   └── metrics.go
│   │   ├── proxy/
│   │   │   └── grpc.go
│   │   ├── cache/
│   │   │   └── stock.go
│   │   └── middleware/
│   │       └── recovery.go
│   ├── scripts/
│   │   ├── bench.sh
│   │   └── token_bucket.lua
│   └── Dockerfile
│
├── seckill/                  # 业务层：秒杀核心逻辑、订单处理
│   ├── go.mod
│   ├── Makefile
│   ├── proto/
│   │   └── seckill/v1/
│   │       └── seckill.proto
│   ├── pb/
│   ├── pkg/
│   │   ├── discovery/
│   │   │   └── etcd.go
│   │   └── tracing/
│   │       └── otel.go
│   ├── cmd/
│   │   ├── seckill/
│   │   │   └── main.go
│   │   └── order/
│   │       └── main.go
│   ├── internal/
│   │   ├── service/
│   │   │   └── seckill.go
│   │   └── stock/
│   │       └── notify.go
│   └── Dockerfile
│
├── docker-compose.yml
└── prometheus.yml
```

## 启动方式

### 使用 Docker Compose（推荐）

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f gateway
docker-compose logs -f seckill-svc
```

### 本地开发

```bash
# 1. 启动基础设施
docker-compose up -d redis etcd kafka jaeger

# 2. 生成 protobuf 代码
cd seckill-gateway && make proto && cd ..
cd seckill && make proto && cd ..

# 3. 启动业务层
cd seckill
go run ./cmd/seckill
# 另一个终端
go run ./cmd/order

# 4. 启动网关层
cd seckill-gateway
go run ./cmd/gateway
```

## 测试

```bash
# 1. 获取测试 Token
TOKEN=$(curl -s http://localhost:8090/admin/token?user=user_001 | grep -o '"token":"[^"]*"' | cut -d'"' -f4)

# 2. 查询库存
curl -s http://localhost:8080/api/stock?goods_id=goods_001 | jq .

# 3. 执行秒杀
curl -s -X POST http://localhost:8080/api/seckill \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user_001","goods_id":"goods_001"}' | jq .

# 4. 查看缓存统计
curl -s http://localhost:8090/admin/guard/stats | jq .

# 5. 查看链路追踪
open http://localhost:16686
```

## 核心特性

### 网关层

1. **多级限流**
   - Nginx 全局限流 + 单 IP 限流
   - Redis Lua 令牌桶（分布式精确限流）

2. **二级缓存**
   - L1: 本地 atomic 标记（零开销）
   - L2: Redis GET（兜底）
   - Redis Pub/Sub 实时同步售罄状态

3. **熔断器**
   - sony/gobreaker 三态熔断
   - 自动恢复、半开试探

4. **JWT 鉴权**
   - 基于 golang-jwt/jwt
   - 管理接口动态生成 Token

### 业务层

1. **Redis Lua 原子扣库存**
   - 检查库存 + 扣减一次完成
   - 无并发超卖

2. **一人一单去重**
   - Redis SETNX 原子操作

3. **Kafka 异步削峰**
   - 秒杀请求写入 Kafka
   - 订单消费者异步处理

4. **库存售罄通知**
   - Redis SET 持久化标记
   - Redis PUBLISH 实时推送到网关

## 监控

- **Prometheus**: http://localhost:9091
- **Jaeger**: http://localhost:16686
- **网关管理 API**: http://localhost:8090
  - `/admin/token?user=xxx` - 生成测试 Token
  - `/admin/guard/stats` - 缓存统计
  - `/admin/guard/reset` - 重置售罄标记
  - `/admin/breaker/state` - 熔断器状态
