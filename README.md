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


## 多模态内容审核模型

`llm-interface` 提供独立的多模态内容审核能力，用于对评论、图片以及图文混合内容做风险分类。它和秒杀主业务解耦：Go 进程负责 gRPC 接口、批处理、缓存和路由，Python worker 负责 Qwen2.5-VL 模型推理。

### 核心能力

- 支持纯文本、纯图像和图文混合输入，适合评论、动态、商品图文等审核场景。
- 分类标签覆盖 `安全`、`暴力`、`色情`、`广告欺诈`、`谩骂引战` 五类业务风险。
- 推理阶段不走逐字生成，而是直接输出分类 logits、置信度和各类别概率，接口响应更稳定。
- 训练和推理均支持 4-bit NF4 量化，配合 LoRA 降低显存占用。

### 模型结构

传统的 `Prompt + 文本生成` 审核方案存在输出格式不稳定、推理链路偏慢、置信度难以直接使用的问题。这里将 Qwen2.5-VL 改造成判别式分类模型，保留视觉语言理解能力，同时让输出适配后端审核接口。

- **Backbone**：使用 `Qwen2.5-VL-3B-Instruct` 作为图文特征提取器，在 Attention 和 MLP 相关投影层注入 LoRA，训练时主要更新适配器参数。
- **Pooling**：取每个样本最后一个有效 token 的 hidden state 作为图文融合后的全局表征，维度为 2048。
- **Classifier Head**：在 backbone 输出后接三层 MLP，结构为 `2048 -> 512 -> 128 -> 5`，使用 `GELU` 和 `Dropout` 控制非线性表达与过拟合。
- **Learning Rate**：LoRA 参数使用 `2e-5`，分类头使用 `1e-3`，让新初始化的分类层更快收敛，同时避免破坏预训练模型表征。
- **Class Balance**：实验数据按类别做均衡采样，每类固定训练样本和验证样本数量，降低多数类对 Macro F1 的影响。

### 训练入口

训练脚本位于 `llm-interface/train_classifier.py`。它负责加载本地处理后的 `train_samples.pkl`、`val_samples.pkl`，构建 Qwen2.5-VL + LoRA + MLP 分类头，并保存最佳验证集模型。

```bash
python llm-interface/train_classifier.py \
  --model-path /path/to/Qwen2.5-VL-3B-Instruct \
  --processed-dir /path/to/processed_data \
  --checkpoint-dir /path/to/checkpoints
```

训练产物包括 LoRA adapter 和 `classifier_head.pt`。模型权重、处理后的数据和中间 checkpoint 不提交到 Git 仓库，部署时通过本地路径或模型仓库挂载。

### 推理入口

推理逻辑位于 `llm-interface/worker.py`。默认 `USE_MOCK=1`，便于在没有模型权重的环境下启动接口；真实模型推理需要设置：

```bash
USE_MOCK=0 \
MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
BEST_MODEL_PATH=/path/to/checkpoints/best_model \
python llm-interface/worker.py
```

`BEST_MODEL_PATH` 中应包含 LoRA adapter 文件和 `classifier_head.pt`。推理结果返回 `is_safe`、`label`、`confidence` 和 `all_scores`，可直接接入业务风控或人工复核流程。

### 模型表现

在平衡验证集上，当前实验结果为：

- Accuracy: `97.5%`
- Macro F1-Score: `0.9745`

该指标用于说明当前训练配置下的分类效果。线上使用时仍建议结合业务阈值、黑白名单规则和人工复核策略。

## 监控

- **Prometheus**: http://localhost:9091
- **Jaeger**: http://localhost:16686
- **网关管理 API**: http://localhost:8090
  - `/admin/token?user=xxx` - 生成测试 Token
  - `/admin/guard/stats` - 缓存统计
  - `/admin/guard/reset` - 重置售罄标记
  - `/admin/breaker/state` - 熔断器状态
