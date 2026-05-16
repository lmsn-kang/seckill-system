-- seckill-gateway/scripts/token_bucket.lua
-- Redis 令牌桶限流：原子操作，多网关实例共享同一个桶
local key = KEYS[1]
local rate = tonumber(ARGV[1])       -- 每秒产生的令牌数
local capacity = tonumber(ARGV[2])   -- 桶容量
local now = tonumber(ARGV[3])        -- 当前时间戳（秒，浮点）
local requested = tonumber(ARGV[4])  -- 本次请求的令牌数

local tk = key .. ":tokens"
local ts = key .. ":timestamp"

-- 读取当前令牌数和上次填充时间
local cur = tonumber(redis.call("GET", tk))
local last = tonumber(redis.call("GET", ts))
if cur == nil then cur = capacity; last = now end

-- 按时间差补充令牌
local elapsed = math.max(0, now - last)
cur = math.min(capacity, cur + elapsed * rate)

-- 判断是否放行
local ok = 0
if cur >= requested then
    cur = cur - requested
    ok = 1
end

-- 写回（60秒过期，防止key残留）
redis.call("SETEX", tk, 60, tostring(cur))
redis.call("SETEX", ts, 60, tostring(now))

return { ok, math.floor(cur) }
