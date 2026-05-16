.PHONY: proto build run

proto:
	protoc --go_out=. --go_opt=module=seckill-gateway \
	       --go-grpc_out=. --go-grpc_opt=module=seckill-gateway \
	       proto/seckill/v1/seckill.proto

build:
	go build -o bin/gateway ./cmd/gateway

run: build
	./bin/gateway
