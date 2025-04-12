-- local cjson = require "cjson"
local feature = {} -- 特征存储容器

-- 请求阶段收集基础特征
function envoy_on_request(request_handle)
    request_handle:logErr("hello world, tanjunchen")
    -- 获取网络层特征
    local stream_info = request_handle:streamInfo()
    feature["srcip"] = stream_info.downstreamRemoteAddress
    feature["dstip"] = stream_info.downstreamLocalAddress()
    feature["proto"] = stream_info.protocol()

    -- 获取传输层特征
    local dynamic_metadata = stream_info:dynamicMetadata()
    local tcp_info = dynamic_metadata:get("envoy.transport_socket.tcp")
    if tcp_info then
        feature["sport"] = tcp_info.source_port
        feature["dsport"] = tcp_info.destination_port
        feature["sttl"] = tcp_info.ttl
        feature["dttl"] = tcp_info.ttl
    end

    -- 记录请求开始时间
    feature["stime"] = os.date("!%Y-%m-%dT%H:%M:%S.000Z")
    request_handle:logError("feature: " .. (feature))
end

-- 响应阶段收集统计特征
function envoy_on_response(response_handle)
    -- 获取流量统计
    local stream_info = response_handle:streamInfo()
    feature["dur"] = stream_info.requestComplete():seconds()

    -- 获取字节数特征
    local bytes_info = stream_info:downstreamAddressBytes()
    feature["sbytes"] = bytes_info.sent
    feature["dbytes"] = bytes_info.received

    -- 获取数据包统计
    feature["spkts"] = stream_info.downstreamSslConnection():numPacketsSent()
    feature["dpkts"] = stream_info.downstreamSslConnection():numPacketsReceived()

    -- 计算时间间隔特征
    feature["ltime"] = os.date("!%Y-%m-%dT%H:%M:%S.000Z")
    feature["sintpkt"] = stream_info.downstreamSslConnection():lastPacketSentDuration()
    feature["dintpkt"] = stream_info.downstreamSslConnection():lastPacketReceivedDuration()

    -- 生成复合特征
    feature["is_sm_ips_ports"] = (feature.srcip == feature.dstip) and
        (feature.sport == feature.dsport) and 1 or 0

    -- 发送特征数据到预处理服务
    --local headers = {
    -- [":method"] = "POST",
    --[":path"] = "/collect",
    --["content-type"] = "application/json"
    --}
    --response_handle:httpCall(
    -- "preprocessing_service",
    -- headers,
    --  cjson.encode(feature),
    --  500,
    --  true
    --)
end
