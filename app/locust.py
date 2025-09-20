import json
from locust import HttpUser, task, between
from locust.exception import RescheduleTask
import random

class LLMOpsUser(HttpUser):
    wait_time = between(1,3) 


    def on_start(self):
        response = self.client.get("/health")
        if response.status_code != 200:
            print(f"health check failed : {response.status_code}")

    @task(1)
    def generate_short_text(self):
        payload = {
            "prompt": random.choice([
                "Summarize machine learning in one sentence",
                "What is artificial intelligence",
                "Explain neural netowrs breifly",
                "What are transformers in AI?"
            ]),
            "max_tokens": random.randint(20, 50),
            "temperature": 0.1
        }

        with self.client.post("/generate", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "text" in result and result["text"]:
                        response.success()
                    else:
                        response.failure("Empty or invalid response")
                except json.JSONDecodeError as ex:
                    response.failure("Invalid json")
            else:
                response.failure(f"HTTP {response.status_code}")


    @task(2)
    def generate_medium_text(self):
        payload = {
            "prompt": random.choice([
                "Write a detailed explanation of howw LLM work.",
                "Compare supervised and unsupervised learning in detail",
                "Explain the transformer architecture step by step.",
                "Describe the process of training a neural network."
            ]),
            "max_tokens": random.randint(100, 200),
            "temperature": 0.3
        }

        with self.client.post("/generate", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "text" in result and result["text"]:
                        response.success()
                    else:
                        response.failure("Empty or invalid response")
                except json.JSONDecodeError as ex:
                    response.failure("Invalid json")
            else:
                response.failure(f"HTTP {response.status_code}")


from locust import events

@events.request.add_listener
def my_req_handler(reuest_type, name , response_time, response_length, response, context, exception, start_time, **kwargs):
    if exception:
        print(f"Request failed: {name} - {exception}")
    elif response and hasattr(response, 'status_code') and response.status_code != 200:
        print(f"http error")


@events.request.add_listener
def track_tokens(name, response, **kwargs):
    if response and hasattr(response, 'json'):
        try:
             data = response.json()
             if 'tokens_used' in data and data['tokens_used'] > 0 :
                 print(f"token: {data['tokens_used']}")
        except:
            pass

