package comment

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type ModerationClient struct {
	endpoint string
	client   *http.Client
}

type ModerationResult struct {
	IsSafe     bool               `json:"is_safe"`
	Label      string             `json:"label"`
	Confidence float32            `json:"confidence"`
	AllScores  map[string]float32 `json:"all_scores"`
	Error      string             `json:"error"`
}

func NewModerationClient(endpoint string) *ModerationClient {
	if endpoint == "" {
		endpoint = "http://localhost:9090"
	}
	return &ModerationClient{
		endpoint: endpoint,
		client:   &http.Client{Timeout: 10 * time.Second},
	}
}

func (c *ModerationClient) Moderate(ctx context.Context, requestID, text string) (*ModerationResult, error) {
	reqBody, err := json.Marshal(map[string]string{
		"request_id":   requestID,
		"text_content": text,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+"/moderate", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http do: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}

	var result ModerationResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &result, nil
}
