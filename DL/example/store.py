import redis
import pandas as pd

# 连接 Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# 假设已经有预测结果
predictions = pd.DataFrame({"src_ip": X_test["src_ip"], "dst_ip": X_test["dst_ip"], "prediction": y_pred})

# 将结果存储到 Redis
for index, row in predictions.iterrows():
    key = f"{row['src_ip']}_{row['dst_ip']}"
    r.set(key, row["prediction"])