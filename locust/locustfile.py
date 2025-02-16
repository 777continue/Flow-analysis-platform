from locust import HttpUser, task, between
import random
class MyUser(HttpUser):
    wait_time = between(1, 5)
    @task(8)  # 80% 的概率发送正常请求
    def normal_request(self):
        self.client.get("/")
    @task(2)  # 20% 的概率发送恶意请求
    def malicious_request(self):
        # 这里可以模拟恶意请求，例如发送异常参数
        self.client.get("/?param=malicious_value")