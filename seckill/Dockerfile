FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /app/seckill ./cmd/seckill/
RUN CGO_ENABLED=0 go build -o /app/order   ./cmd/order/

FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY --from=builder /app/seckill /app/seckill
COPY --from=builder /app/order   /app/order
WORKDIR /app
