.PHONY: proto build

proto:
	protoc --go_out=. --go_opt=module=seckill \
	       --go-grpc_out=. --go-grpc_opt=module=seckill \
	       proto/seckill/v1/seckill.proto
	protoc --go_out=. --go_opt=module=seckill \
	       --go-grpc_out=. --go-grpc_opt=module=seckill \
	       proto/moderation/v1/moderation.proto

build:
	go build -o bin/seckill ./cmd/seckill
	go build -o bin/order   ./cmd/order
