local redis = require 'redis';
local client = redis.connect('127.0.0.1', 6379);

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
