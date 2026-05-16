package order

import (
	"context"
	"database/sql"
	"fmt"
	"strconv"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

const (
	defaultDSN = "root:root123@tcp(127.0.0.1:3306)/seckill?parseTime=true&loc=Local&timeout=5s&readTimeout=5s&writeTimeout=5s"
)

type Repository interface {
	Create(ctx context.Context, dto *CreateOrderDTO) (*Order, error)
	GetByID(ctx context.Context, orderID int64) (*Order, error)
	GetByUserID(ctx context.Context, userID string, limit, offset int) ([]*Order, error)
	UpdateStatus(ctx context.Context, orderID int64, oldStatus, newStatus OrderStatus) (bool, error)
	UpdatePay(ctx context.Context, orderID int64, oldStatus OrderStatus, version int32) (bool, error)
	ExpireOrders(ctx context.Context, expireTime time.Time) ([]string, error)
	Close() error
}

type repository struct {
	db *sql.DB
}

func NewRepository(dsn string) (Repository, error) {
	if dsn == "" {
		dsn = defaultDSN
	}

	db, err := sql.Open("mysql", dsn)
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}

	db.SetMaxOpenConns(100)
	db.SetMaxIdleConns(10)
	db.SetConnMaxLifetime(time.Hour)

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("ping database: %w", err)
	}

	r := &repository{db: db}
	if err := r.initSchema(); err != nil {
		return nil, fmt.Errorf("init schema: %w", err)
	}

	return r, nil
}

func (r *repository) initSchema() error {
	schema := `
	CREATE TABLE IF NOT EXISTS orders (
		id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键',
		order_id BIGINT NOT NULL UNIQUE COMMENT '订单ID',
		user_id VARCHAR(64) NOT NULL COMMENT '用户ID',
		goods_id VARCHAR(64) NOT NULL COMMENT '商品ID',
		status TINYINT NOT NULL DEFAULT 0 COMMENT '订单状态: 0已创建 1待支付 2已支付 3已发货 4已完成 5已取消 6已退款',
		amount BIGINT NOT NULL DEFAULT 0 COMMENT '金额(分)',
		expire_time DATETIME NOT NULL COMMENT '支付过期时间',
		created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
		updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
		version INT NOT NULL DEFAULT 0 COMMENT '乐观锁版本号',
		INDEX idx_user_id (user_id) COMMENT '用户ID索引',
		INDEX idx_status (status) COMMENT '状态索引',
		INDEX idx_expire_time (expire_time) COMMENT '过期时间索引',
		INDEX idx_created_at (created_at) COMMENT '创建时间索引'
	) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='秒杀订单表';
	`

	if _, err := r.db.Exec(schema); err != nil {
		return fmt.Errorf("create table: %w", err)
	}
	return nil
}

func (r *repository) Create(ctx context.Context, dto *CreateOrderDTO) (*Order, error) {
	query := `
		INSERT INTO orders (order_id, user_id, goods_id, status, amount, expire_time)
		VALUES (?, ?, ?, ?, ?, ?)
	`

	result, err := r.db.ExecContext(ctx, query,
		dto.OrderID, dto.UserID, dto.GoodsID, StatusCreated, dto.Amount, dto.ExpireTime,
	)
	if err != nil {
		return nil, fmt.Errorf("insert order: %w", err)
	}

	id, err := result.LastInsertId()
	if err != nil {
		return nil, fmt.Errorf("last insert id: %w", err)
	}

	return &Order{
		ID:         id,
		OrderID:    dto.OrderID,
		UserID:     dto.UserID,
		GoodsID:    dto.GoodsID,
		Status:     StatusCreated,
		Amount:     dto.Amount,
		ExpireTime: dto.ExpireTime,
		Version:    0,
	}, nil
}

func (r *repository) GetByID(ctx context.Context, orderID int64) (*Order, error) {
	query := `
		SELECT id, order_id, user_id, goods_id, status, amount, expire_time, created_at, updated_at, version
		FROM orders WHERE order_id = ?
	`

	var order Order
	err := r.db.QueryRowContext(ctx, query, orderID).Scan(
		&order.ID, &order.OrderID, &order.UserID, &order.GoodsID,
		&order.Status, &order.Amount, &order.ExpireTime,
		&order.CreatedAt, &order.UpdatedAt, &order.Version,
	)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("order not found")
	}
	if err != nil {
		return nil, fmt.Errorf("query order: %w", err)
	}

	return &order, nil
}

func (r *repository) GetByUserID(ctx context.Context, userID string, limit, offset int) ([]*Order, error) {
	query := `
		SELECT id, order_id, user_id, goods_id, status, amount, expire_time, created_at, updated_at, version
		FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?
	`

	rows, err := r.db.QueryContext(ctx, query, userID, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("query orders: %w", err)
	}
	defer rows.Close()

	var orders []*Order
	for rows.Next() {
		var order Order
		if err := rows.Scan(
			&order.ID, &order.OrderID, &order.UserID, &order.GoodsID,
			&order.Status, &order.Amount, &order.ExpireTime,
			&order.CreatedAt, &order.UpdatedAt, &order.Version,
		); err != nil {
			return nil, fmt.Errorf("scan order: %w", err)
		}
		orders = append(orders, &order)
	}

	return orders, nil
}

func (r *repository) UpdateStatus(ctx context.Context, orderID int64, oldStatus, newStatus OrderStatus) (bool, error) {
	query := `
		UPDATE orders
		SET status = ?, version = version + 1
		WHERE order_id = ? AND status = ? AND version = ?
	`

	order, err := r.GetByID(ctx, orderID)
	if err != nil {
		return false, err
	}

	result, err := r.db.ExecContext(ctx, query, newStatus, orderID, oldStatus, order.Version)
	if err != nil {
		return false, fmt.Errorf("update status: %w", err)
	}

	affected, _ := result.RowsAffected()
	return affected > 0, nil
}

func (r *repository) UpdatePay(ctx context.Context, orderID int64, oldStatus OrderStatus, version int32) (bool, error) {
	query := `
		UPDATE orders
		SET status = ?, version = version + 1
		WHERE order_id = ? AND status = ? AND version = ?
	`

	result, err := r.db.ExecContext(ctx, query, StatusPaid, orderID, oldStatus, version)
	if err != nil {
		return false, fmt.Errorf("update pay: %w", err)
	}

	affected, _ := result.RowsAffected()
	return affected > 0, nil
}

func (r *repository) ExpireOrders(ctx context.Context, expireTime time.Time) ([]string, error) {
	query := `
		SELECT order_id FROM orders
		WHERE status IN (?, ?) AND expire_time < ?
		LIMIT 1000
	`

	rows, err := r.db.QueryContext(ctx, query, StatusCreated, StatusPending, expireTime)
	if err != nil {
		return nil, fmt.Errorf("query expire orders: %w", err)
	}
	defer rows.Close()

	var orderIDs []string
	for rows.Next() {
		var orderID int64
		if err := rows.Scan(&orderID); err != nil {
			return nil, fmt.Errorf("scan order_id: %w", err)
		}
		orderIDs = append(orderIDs, strconv.FormatInt(orderID, 10))
	}

	return orderIDs, nil
}

func (r *repository) Close() error {
	return r.db.Close()
}
