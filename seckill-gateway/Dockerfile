FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /app/gateway ./cmd/gateway/

FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY --from=builder /app/gateway /app/gateway
COPY configs/ /app/configs/
WORKDIR /app
ENTRYPOINT ["/app/gateway"]
