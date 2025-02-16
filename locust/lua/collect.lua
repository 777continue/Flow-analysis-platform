ngx.log(ngx.INFO, "Package path set successfully")

-- 获取源 IP、目的 IP 和时间戳
local src_ip = ngx.var.remote_addr
local dst_ip = ngx.var.server_addr
-- 使用 ngx.now() 获取精确到毫秒的时间戳
local timestamp = ngx.now()

ngx.log(ngx.INFO, "Source IP: " .. src_ip)
ngx.log(ngx.INFO, "Destination IP: " .. dst_ip)
ngx.log(ngx.INFO, "Timestamp: " .. timestamp)

-- 格式化要输出的信息
local log_message = string.format("src_ip: %s, dst_ip: %s, timestamp: %.3f\n", src_ip, dst_ip, timestamp)

-- 输出信息到终端（在 OpenResty 中可以使用 ngx.say 输出到响应，类似输出到终端的效果）
ngx.say(log_message)
