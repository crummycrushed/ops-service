import time
from typing import Dict
from fastapi import HTTPException
from app.metrics import RATE_LIMIT_EXCEEDED


user_requests: Dict[str, list] = {}


RATE_LIMTITS = {
    "alice" : 20,
    "bob": 1,
    "premium": 50,
    "default" : 5
}

def get_user_limit(user: str) -> int:
    return RATE_LIMTITS.get(user, RATE_LIMTITS["default"])


def check_rate_limit(user: str) -> bool:
    current_time = time.time()
    limit = get_user_limit(user)

    if user not in user_requests:
        user_requests[user] = []

    cutoff_time = current_time - 60
    user_requests[user] = [
        req_time for req_time in user_requests[user]
        if req_time > cutoff_time
    ]

    if len(user_requests[user]) >= limit:
        RATE_LIMIT_EXCEEDED.labels(user=user).inc()
        return False
    
    user_requests[user].append(current_time)
    return True

def enforce_rate_limit(user: str):
    if not check_rate_limit(user):
        current_requests = len(user_requests.get(user, []))
        limit = get_user_limit(user)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded : {current_requests}/{limit} requests per minute"
        )