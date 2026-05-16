# PB 代码生成说明

本目录需要存放 `protoc` 生成的 Go protobuf 代码。
当前仓库中未提交生成的 `.pb.go` 文件，需要手动生成。

## 依赖安装

```bash
# 安装 protoc
sudo apt install -y protobuf-compiler

# 安装 Go 插件
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
```

## 生成命令

```bash
cd /home/lmsn/seckill/llm-interface
make proto
```

或者手动执行：

```bash
protoc --go_out=. --go_opt=module=seckill/llm-interface \
       --go-grpc_out=. --go-grpc_opt=module=seckill/llm-interface \
       proto/worker.proto
```

生成后会在本目录下产生 `worker.pb.go` 和 `worker_grpc.pb.go`。
