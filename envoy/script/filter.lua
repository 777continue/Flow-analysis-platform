local redis = require 'redis';
local client = redis.connect('redis', 6379);

if client then
    print('Connected to Redis');
    client:set('test_key', 'test_value');
    local value = client:get('test_key');
    if value then
        print('Value retrieved: ' .. value);
    end;
else
    print('Failed to connect to Redis');
end

-- 处理请求的函数，输出请求相关日志
function envoy_on_request(request_handle)
    local headers = request_handle:headers()
    local method = headers:get(":method")
    local path = headers:get(":path")
    local log_msg = string.format("Request - Method: %s, Path: %s", method, path)
    print(log_msg)
end

-- 处理响应的函数，输出响应相关日志
function envoy_on_response(response_handle)
    local headers = response_handle:headers()
    local status = headers:get(":status")
    local log_msg = string.format("Response - Status: %s", status)
    print(log_msg)
end
