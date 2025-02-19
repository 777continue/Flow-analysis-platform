-- 加载 redis 库
local redis = require 'redis'
local data_source = {
    state = "ESTABLISHED",
    ct_state_ttl = 10,
    attack_cat = "Normal",
    sbytes = 1024,
    smeansz = 512,
    Sload = 0.5,
    dmeansz = 256,
    Dpkts = 20,
    Dload = 0.2,
    dttl = 64,
    dur = 10.5,
    dbytes = 2048,
    sport = 8080,
    ct_srv_dst = 5,
    Dintpkt = 0.1,
    other_feature = "Some value" -- 这个特征不会被收集
}
-- 处理请求时执行
function envoy_on_request(request_handle)
    -- 收集模拟数据源 data_source 的数据信息
    for key, value in pairs(data_source) do
        if key ~= "other_feature" then -- 跳过不收集的特征
            data = data .. "|" .. tostring(value)
        end
    end

    -- 连接 Redis
    local redis = redis:new()
    redis:set_timeout(1000) -- 1 秒超时

    local ok, err = redis:connect("127.0.0.1", 6379)
    if not ok then
        request_handle:logErr("Failed to connect to Redis: " .. err)
        return
    end

    -- 将数据存储到 Redis
    local res, err = redis:rpush("flow_data", data)
    if not res then
        request_handle:logErr("Failed to push data to Redis: " .. err)
    end

    -- 保持 Redis 连接以供后续使用
    redis:set_keepalive(10000, 100)
end
