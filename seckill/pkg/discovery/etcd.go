package discovery

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc/resolver"
)

const svcPrefix = "/seckill/svc/"

type Registry struct {
	client  *clientv3.Client
	leaseID clientv3.LeaseID
	key     string
}

func NewRegistry(endpoints []string) (*Registry, error) {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("etcd connect: %w", err)
	}
	return &Registry{client: cli}, nil
}

func (r *Registry) Register(ctx context.Context, name, addr string, ttl int64) error {
	grant, err := r.client.Grant(ctx, ttl)
	if err != nil {
		return err
	}
	r.leaseID = grant.ID
	r.key = fmt.Sprintf("%s%s/%s", svcPrefix, name, addr)

	if _, err = r.client.Put(ctx, r.key, addr, clientv3.WithLease(r.leaseID)); err != nil {
		return err
	}

	ch, err := r.client.KeepAlive(ctx, r.leaseID)
	if err != nil {
		return err
	}
	go func() {
		for range ch {
		}
		log.Printf("[registry] lease expired: %s", r.key)
	}()
	return nil
}

func (r *Registry) Deregister(ctx context.Context) error {
	_, err := r.client.Revoke(ctx, r.leaseID)
	return err
}

type ResolverBuilder struct {
	client *clientv3.Client
}

func NewResolverBuilder(cli *clientv3.Client) *ResolverBuilder {
	return &ResolverBuilder{client: cli}
}

func (b *ResolverBuilder) Scheme() string { return "etcd" }

func (b *ResolverBuilder) Build(target resolver.Target, cc resolver.ClientConn, _ resolver.BuildOptions) (resolver.Resolver, error) {
	r := &etcdResolver{
		client:  b.client,
		cc:      cc,
		svcName: strings.TrimPrefix(target.URL.Path, "/"),
		closeCh: make(chan struct{}),
	}
	r.resolve()
	go r.watch()
	return r, nil
}

type etcdResolver struct {
	client  *clientv3.Client
	cc      resolver.ClientConn
	svcName string
	closeCh chan struct{}
}

func (r *etcdResolver) ResolveNow(_ resolver.ResolveNowOptions) { r.resolve() }
func (r *etcdResolver) Close()                                  { close(r.closeCh) }

func (r *etcdResolver) resolve() {
	prefix := svcPrefix + r.svcName + "/"
	resp, err := r.client.Get(context.Background(), prefix, clientv3.WithPrefix())
	if err != nil {
		log.Printf("[resolver] get failed: %v", err)
		return
	}
	addrs := make([]resolver.Address, 0, len(resp.Kvs))
	for _, kv := range resp.Kvs {
		addrs = append(addrs, resolver.Address{Addr: string(kv.Value)})
	}
	r.cc.UpdateState(resolver.State{Addresses: addrs})
}

func (r *etcdResolver) watch() {
	prefix := svcPrefix + r.svcName + "/"
	wch := r.client.Watch(context.Background(), prefix, clientv3.WithPrefix())
	for {
		select {
		case <-r.closeCh:
			return
		case _, ok := <-wch:
			if !ok {
				return
			}
			r.resolve()
		}
	}
}
