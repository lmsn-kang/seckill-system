module seckill

go 1.22

require (
	github.com/bwmarrin/snowflake v0.3.0
	github.com/gin-gonic/gin v1.10.0
	github.com/go-sql-driver/mysql v1.8.1
	github.com/google/uuid v1.6.0
	github.com/redis/go-redis/v9 v9.7.0
	github.com/segmentio/kafka-go v0.4.47
	go.etcd.io/etcd/client/v3 v3.5.12
	go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc v0.49.0
	go.opentelemetry.io/otel v1.24.0
	go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc v1.24.0
	go.opentelemetry.io/otel/sdk v1.24.0
	google.golang.org/grpc v1.62.1
	google.golang.org/protobuf v1.33.0
)

replace google.golang.org/genproto/googleapis/rpc => google.golang.org/genproto/googleapis/rpc v0.0.0-20241202173237-19429a94021a
