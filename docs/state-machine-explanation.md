# 秒杀系统订单状态机详解

## 1. 状态存储在哪里？

状态存储在 **MySQL 数据库的 orders 表**中，通过 `status` 字段（int32）持久化：

```
┌─────────────────────────────────────────────┐
│              MySQL: orders 表                │
├──────────┬──────────┬────────┬───────────────┤
│   id     │ order_id │ status │   version     │
├──────────┼──────────┼────────┼───────────────┤
│    1     │  100001  │   1    │      3        │
│    2     │  100002  │   2    │      1        │
└──────────┴──────────┴────────┴───────────────┘
                              ↑          ↑
                           当前状态    乐观锁版本号
```

对应代码定义（`internal/order/model.go`）：
```go
type OrderStatus int32

const (
    StatusCreated   OrderStatus = 0 // 已创建
    StatusPending   OrderStatus = 1 // 待支付
    StatusPaid      OrderStatus = 2 // 已支付
    StatusShipped   OrderStatus = 3 // 已发货
    StatusCompleted OrderStatus = 4 // 已完成
    StatusCancelled OrderStatus = 5 // 已取消
    StatusRefunded  OrderStatus = 6 // 已退款
)

type Order struct {
    OrderID  int64       `db:"order_id"`
    Status   OrderStatus `db:"status"`    // 状态字段
    Version  int32       `db:"version"`   // 乐观锁
    // ...
}
```

**关键点**：状态不是存在内存中，而是直接存在数据库。每次操作都从数据库读取当前状态，确保分布式环境下一致。

## 2. 状态如何流转？

状态流转通过 **Service 层的方法** 控制，每个方法是一个"状态转换函数"：

```
                    ┌─────────┐
                    │ CREATED │ (0)  已创建
                    └────┬────┘
                         │ 用户确认支付
                    ┌────┴────┐
                    │ PENDING │ (1)  待支付
                    └────┬────┘
                         │ 支付成功
                    ┌────┴────┐
              ┌────▶│  PAID   │ (2)  已支付
              │     └────┬────┘
              │          │ 发货
              │     ┌────┴────┐
              │     │ SHIPPED │ (3)  已发货
              │     └────┬────┘
              │          │ 确认收货
              │     ┌────┴─────┐
              │     │COMPLETED │ (4)  已完成
              │     └──────────┘
              │
         取消/过期           退款
              │               │
         ┌────┴────┐     ┌────┴────┐
         │CANCELLED│     │ REFUNDED│
         │  (5)    │     │   (6)   │
         └─────────┘     └─────────┘
```

### 转换规则（硬编码在 Service 方法中）

| 方法 | 允许的前置状态 | 目标状态 | 业务含义 |
|------|--------------|---------|---------|
| `CreateOrder` | 无（新建） | PENDING | 创建订单 |
| `PayOrder` | CREATED, PENDING | PAID | 用户支付 |
| `ShipOrder` | PAID | SHIPPED | 商家发货 |
| `CompleteOrder` | SHIPPED | COMPLETED | 用户确认收货 |
| `CancelOrder` | CREATED, PENDING | CANCELLED | 取消订单 |
| `RefundOrder` | PAID, SHIPPED | REFUNDED | 申请退款 |
| `ExpireOrders` | CREATED, PENDING（且超时） | CANCELLED | 定时任务取消过期订单 |

## 3. 到哪里修改状态？

**唯一修改入口**：`internal/order/repository.go` 中的三个更新方法

```go
// Repository 接口定义
type Repository interface {
    Create(ctx context.Context, dto *CreateOrderDTO) (*Order, error)
    GetByID(ctx context.Context, id int64) (*Order, error)
    
    // ── 以下三个是状态修改的唯一入口 ──
    UpdatePay(ctx context.Context, orderID int64, oldStatus OrderStatus, version int32) (bool, error)
    UpdateStatus(ctx context.Context, orderID int64, oldStatus, newStatus OrderStatus) (bool, error)
    ExpireOrders(ctx context.Context, now time.Time) ([]int64, error)
}
```

Repository 实现中用 **乐观锁** 保证并发安全：

```sql
-- UpdatePay 对应的 SQL
UPDATE orders 
SET status = 2,           -- PAID
    version = version + 1,
    updated_at = NOW()
WHERE order_id = ? 
  AND status = ?          -- 必须是期望的旧状态
  AND version = ?         -- 必须是期望的版本号
```

如果 `WHERE` 条件不匹配（说明期间有其他请求改了状态），`rows affected = 0`，返回 `ok=false`。

## 4. 修改的是链路上的哪一环？

以 `PayOrder` 支付流程为例，展示完整的调用链路：

```
用户请求
  │
  ▼
┌─────────────────────────────────────────┐
│           HTTP Handler (api/order.go)    │
│  - 参数校验                                │
│  - 调用 service.PayOrder()               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Service 层 (service.go) ← 状态机核心     │
│                                          │
│  1. repo.GetByID()     ← 从 DB 读取当前状态│
│  2. 状态检查:                             │
│     if status != CREATED && status !=    │
│        PENDING { return err }            │
│  3. 过期检查:                             │
│     if now > ExpireTime { cancel }       │
│  4. repo.UpdatePay()   ← 带乐观锁的更新   │
│     if !ok { return "并发冲突" }         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Repository 层 (repository.go)           │
│  - 执行 SQL:                             │
│    UPDATE orders SET status=2,           │
│      version=version+1                   │
│    WHERE order_id=? AND status=1         │
│      AND version=?                       │
│  - 返回 affected rows > 0 ?              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│          MySQL 数据库                     │
│  - 行级锁保证原子性                        │
│  - 乐观锁 version 防并发                  │
└─────────────────────────────────────────┘
```

### 核心设计思想

```
┌──────────┐    读状态     ┌──────────┐    改状态     ┌──────────┐
│ Service  │ ───────────▶ │   DB     │ ◀─────────── │ Service  │
│ (状态检查) │             │(orders表) │              │ (乐观锁更新)│
└──────────┘              └──────────┘              └──────────┘
     ↑                                                  ↑
     │  业务规则判断                                     │  WHERE status = oldStatus
     │  (哪个状态能做什么)                                │  AND version = ?
     └──────────────────────────────────────────────────┘
                    闭环：读-验-改 三步原子操作
```

**Service 层**：只管"什么状态下允许做什么操作"（状态守卫）
**Repository 层**：只管"怎么读数据库、怎么写数据库"（数据访问）
**MySQL 层**：通过行锁 + 乐观锁保证并发安全

如果将来换数据库（MySQL → PostgreSQL），只需改 Repository 层的 SQL 实现，Service 层的状态机逻辑完全不需要变动。这就是分层的核心价值。
