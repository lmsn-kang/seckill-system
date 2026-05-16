.PHONY: proto build run

proto:
	protoc --go_out=. --go_opt=module=seckill/llm-interface \
	       --go-grpc_out=. --go-grpc_opt=module=seckill/llm-interface \
	       proto/worker.proto

build:
	go build -o bin/llm-interface .

run: build
	go run .
