package main

import (
	"context"
	"fmt"
	"log"
	"time"

	pb "seckill/llm-interface/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
)

type PythonBridge struct {
	client pb.InferenceWorkerClient
	conn   *grpc.ClientConn
}

func NewPythonBridge(addr string) *PythonBridge {
	conn, err := grpc.Dial(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(50*1024*1024),
			grpc.MaxCallSendMsgSize(50*1024*1024),
		),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                20 * time.Second,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}),
	)
	if err != nil {
		log.Fatalf("连接 Python Worker 失败: %v", err)
	}

	log.Printf("[Bridge] 已连接 Python Worker: %s", addr)
	return &PythonBridge{
		client: pb.NewInferenceWorkerClient(conn),
		conn:   conn,
	}
}

func (b *PythonBridge) RunBatch(items []*InferItem) ([]*InferResult, error) {
	req := &pb.BatchRequest{
		Items: make([]*pb.InferenceItem, len(items)),
	}
	for i, item := range items {
		req.Items[i] = &pb.InferenceItem{
			TextContent: item.Text,
			ImageData:   item.Image,
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := b.client.RunBatch(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("Python 调用失败: %w", err)
	}

	results := make([]*InferResult, len(resp.Results))
	for i, r := range resp.Results {
		if r.Error != "" {
			results[i] = &InferResult{Error: fmt.Errorf(r.Error)}
		} else {
			results[i] = &InferResult{
				IsSafe:     r.IsSafe,
				Label:      r.Label,
				Confidence: r.Confidence,
				AllScores:  r.AllScores,
			}
		}
	}
	return results, nil
}

func (b *PythonBridge) Close() {
	if b.conn != nil {
		b.conn.Close()
	}
}