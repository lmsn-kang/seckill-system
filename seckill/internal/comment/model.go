package comment

import "time"

type CommentStatus int8

const (
	StatusPending  CommentStatus = 0
	StatusApproved CommentStatus = 1
	StatusRejected CommentStatus = 2
)

func (s CommentStatus) String() string {
	switch s {
	case StatusPending:
		return "PENDING"
	case StatusApproved:
		return "APPROVED"
	case StatusRejected:
		return "REJECTED"
	default:
		return "UNKNOWN"
	}
}

type Comment struct {
	ID         int64         `db:"id"`
	OrderID    int64         `db:"order_id"`
	UserID     string        `db:"user_id"`
	GoodsID    string        `db:"goods_id"`
	Content    string        `db:"content"`
	Status     CommentStatus `db:"status"`
	Label      string        `db:"label"`
	Confidence float32       `db:"confidence"`
	CreatedAt  time.Time     `db:"created_at"`
	UpdatedAt  time.Time     `db:"updated_at"`
}

type CreateCommentDTO struct {
	OrderID int64  `json:"order_id"`
	UserID  string `json:"user_id"`
	GoodsID string `json:"goods_id"`
	Content string `json:"content"`
}
