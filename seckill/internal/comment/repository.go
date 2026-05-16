package comment

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

const defaultDSN = "root:root123@tcp(127.0.0.1:3306)/seckill?parseTime=true&loc=Local&timeout=5s&readTimeout=5s&writeTimeout=5s"

type Repository interface {
	Create(ctx context.Context, dto *CreateCommentDTO) (*Comment, error)
	GetByID(ctx context.Context, id int64) (*Comment, error)
	GetByGoodsID(ctx context.Context, goodsID string, limit, offset int) ([]*Comment, error)
	Delete(ctx context.Context, id int64) error
	UpdateStatus(ctx context.Context, id int64, status CommentStatus, label string, confidence float32) error
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
	db.SetMaxOpenConns(50)
	db.SetMaxIdleConns(5)
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
	CREATE TABLE IF NOT EXISTS comments (
		id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键',
		order_id BIGINT NOT NULL COMMENT '订单ID',
		user_id VARCHAR(64) NOT NULL COMMENT '用户ID',
		goods_id VARCHAR(64) NOT NULL COMMENT '商品ID',
		content TEXT NOT NULL COMMENT '评论内容',
		status TINYINT NOT NULL DEFAULT 0 COMMENT '评论状态: 0待审核 1已通过 2已删除',
		label VARCHAR(32) COMMENT '审核标签',
		confidence FLOAT COMMENT '审核置信度',
		created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
		updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
		INDEX idx_order_id (order_id) COMMENT '订单ID索引',
		INDEX idx_goods_id (goods_id) COMMENT '商品ID索引',
		INDEX idx_user_id (user_id) COMMENT '用户ID索引',
		INDEX idx_status (status) COMMENT '状态索引'
	) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户评论表';
	`
	if _, err := r.db.Exec(schema); err != nil {
		return fmt.Errorf("create table: %w", err)
	}
	return nil
}

func (r *repository) Create(ctx context.Context, dto *CreateCommentDTO) (*Comment, error) {
	query := `
		INSERT INTO comments (order_id, user_id, goods_id, content, status)
		VALUES (?, ?, ?, ?, ?)
	`
	result, err := r.db.ExecContext(ctx, query, dto.OrderID, dto.UserID, dto.GoodsID, dto.Content, StatusPending)
	if err != nil {
		return nil, fmt.Errorf("insert comment: %w", err)
	}
	id, err := result.LastInsertId()
	if err != nil {
		return nil, fmt.Errorf("last insert id: %w", err)
	}
	return &Comment{
		ID:        id,
		OrderID:   dto.OrderID,
		UserID:    dto.UserID,
		GoodsID:   dto.GoodsID,
		Content:   dto.Content,
		Status:    StatusPending,
		CreatedAt: time.Now(),
	}, nil
}

func (r *repository) GetByID(ctx context.Context, id int64) (*Comment, error) {
	query := `
		SELECT id, order_id, user_id, goods_id, content, status, COALESCE(label, ''), COALESCE(confidence, 0), created_at, updated_at
		FROM comments WHERE id = ?
	`
	var c Comment
	err := r.db.QueryRowContext(ctx, query, id).Scan(
		&c.ID, &c.OrderID, &c.UserID, &c.GoodsID, &c.Content,
		&c.Status, &c.Label, &c.Confidence, &c.CreatedAt, &c.UpdatedAt,
	)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("comment not found")
	}
	if err != nil {
		return nil, fmt.Errorf("query comment: %w", err)
	}
	return &c, nil
}

func (r *repository) GetByGoodsID(ctx context.Context, goodsID string, limit, offset int) ([]*Comment, error) {
	query := `
		SELECT id, order_id, user_id, goods_id, content, status, COALESCE(label, ''), COALESCE(confidence, 0), created_at, updated_at
		FROM comments WHERE goods_id = ? AND status = ?
		ORDER BY created_at DESC LIMIT ? OFFSET ?
	`
	rows, err := r.db.QueryContext(ctx, query, goodsID, StatusApproved, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("query comments: %w", err)
	}
	defer rows.Close()

	var comments []*Comment
	for rows.Next() {
		var c Comment
		if err := rows.Scan(
			&c.ID, &c.OrderID, &c.UserID, &c.GoodsID, &c.Content,
			&c.Status, &c.Label, &c.Confidence, &c.CreatedAt, &c.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan comment: %w", err)
		}
		comments = append(comments, &c)
	}
	return comments, nil
}

func (r *repository) Delete(ctx context.Context, id int64) error {
	query := `DELETE FROM comments WHERE id = ?`
	result, err := r.db.ExecContext(ctx, query, id)
	if err != nil {
		return fmt.Errorf("delete comment: %w", err)
	}
	affected, _ := result.RowsAffected()
	if affected == 0 {
		return fmt.Errorf("comment not found")
	}
	return nil
}

func (r *repository) UpdateStatus(ctx context.Context, id int64, status CommentStatus, label string, confidence float32) error {
	query := `UPDATE comments SET status = ?, label = ?, confidence = ? WHERE id = ?`
	_, err := r.db.ExecContext(ctx, query, status, label, confidence, id)
	if err != nil {
		return fmt.Errorf("update comment status: %w", err)
	}
	return nil
}

func (r *repository) Close() error {
	return r.db.Close()
}
