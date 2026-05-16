package comment

import (
	"context"
	"fmt"
	"log"

	"github.com/google/uuid"
)

type Service struct {
	repo       Repository
	moderation *ModerationClient
}

func NewService(repo Repository) *Service {
	return &Service{repo: repo}
}

func (s *Service) SetModerationClient(c *ModerationClient) {
	s.moderation = c
}

func (s *Service) CreateComment(ctx context.Context, dto *CreateCommentDTO) (*Comment, error) {
	comment, err := s.repo.Create(ctx, dto)
	if err != nil {
		return nil, fmt.Errorf("create comment: %w", err)
	}

	log.Printf("[comment] 已创建: id=%d order=%d user=%s goods=%s",
		comment.ID, comment.OrderID, comment.UserID, comment.GoodsID)

	if s.moderation != nil {
		go s.reviewComment(context.Background(), comment)
	} else {
		log.Printf("[comment] 警告: moderation client 未配置，评论 id=%d 将始终保持 PENDING", comment.ID)
	}

	return comment, nil
}

func (s *Service) reviewComment(ctx context.Context, comment *Comment) {
	if s.moderation == nil {
		return
	}

	reqID := fmt.Sprintf("comment-%d-%s", comment.ID, uuid.New().String())
	result, err := s.moderation.Moderate(ctx, reqID, comment.Content)
	if err != nil {
		log.Printf("[comment] 审核调用失败: comment=%d err=%v", comment.ID, err)
		return
	}

	if result.Error != "" {
		log.Printf("[comment] 审核接口返回错误: comment=%d err=%s", comment.ID, result.Error)
		return
	}

	if result.IsSafe {
		if err := s.repo.UpdateStatus(ctx, comment.ID, StatusApproved, result.Label, result.Confidence); err != nil {
			log.Printf("[comment] 更新状态失败: comment=%d err=%v", comment.ID, err)
			return
		}
		log.Printf("[comment] 审核通过: comment=%d label=%s confidence=%.2f",
			comment.ID, result.Label, result.Confidence)
	} else {
		if err := s.repo.Delete(ctx, comment.ID); err != nil {
			log.Printf("[comment] 删除失败: comment=%d err=%v", comment.ID, err)
			return
		}
		log.Printf("[comment] 审核不通过已删除: comment=%d label=%s confidence=%.2f reason=%s",
			comment.ID, result.Label, result.Confidence, result.Label)
	}
}

func (s *Service) GetComment(ctx context.Context, id int64) (*Comment, error) {
	return s.repo.GetByID(ctx, id)
}

func (s *Service) GetGoodsComments(ctx context.Context, goodsID string, page, pageSize int) ([]*Comment, error) {
	if page < 1 {
		page = 1
	}
	if pageSize < 1 || pageSize > 100 {
		pageSize = 20
	}
	offset := (page - 1) * pageSize
	return s.repo.GetByGoodsID(ctx, goodsID, pageSize, offset)
}
