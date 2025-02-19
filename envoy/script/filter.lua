-- 加载 redis 库
local redis = require 'redis'

-- 处理请求时执行
function envoy_on_request(request_handle)
    -- 获取源 IP
    local source_ip = request_handle:streamInfo():downstreamRemoteAddress():ip():address()
    -- 获取目的 IP
    local destination_ip = request_handle:streamInfo():upstreamHost():address():ip():address()
    -- 获取时间戳
    local timestamp = os.time()
    -- 获取请求方法
    local method = request_handle:headers():get(":method")
    -- 获取请求路径
    local path = request_handle:headers():get(":path")

    -- 连接 Redis
    local red = redis:new()
    red:set_timeout(1000) -- 1 秒超时

    local ok, err = red:connect("127.0.0.1", 6379)
    if not ok then
        request_handle:logErr("Failed to connect to Redis: " .. err)
        return
    end

    -- 订阅结果消息队列
    local res, err = red:subscribe("result_queue")
    if not res then
        request_handle:logErr("Failed to subscribe to result queue: " .. err)
        return
    end

    while true do
        local message = red:read_reply()
        if message and message[1] == "message" then
            if message[3] == "Normal" then
                -- 正常流量，放行
                request_handle:headers():add("X-Custom-Header", "Normal")
                request_handle:continue()
            else
                -- 异常流量，拒绝
                request_handle:headers():add("X-Custom-Header", "Abnormal")
                request_handle:respond({ [":status"] = "403" }, "Forbidden")
            end
        end
    end

    -- 保持 Redis 连接以供后续使用
    red:set_keepalive(10000, 100)
end
