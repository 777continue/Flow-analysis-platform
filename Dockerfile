# 基于官方 Envoy 镜像
FROM envoyproxy/envoy:v1.27.0

USER root

# 安装 LuaRocks 和编译工具
RUN apt-get update && apt-get install -y \
    luarocks \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 lua-resty-redis 库
RUN luarocks install redis-lua

# 设置工作目录
WORKDIR /etc/envoy