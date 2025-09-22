import json
import logging
import re
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest
import requests
import time
import tiktoken
import os
import jwt

from app.governance import enforce_rate_limit
from app.metrics import (
    CONTENT_FILTER, COST_BLOCKED_REQUEST, GUARDRAIL_VIOLATION, SAFETY_VIOLATION, record_requst, ACTIVE_REQUESTS, MODEL_INFO, TOKENS_PER_SECOND
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="LLMOps LLM service",
    version="2.0.0"
)

OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"

OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
MODEL_NAME = "tinyllama"
BACKEND = "ollama"


JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_SECONDS = int(os.getenv("EXP", str(3600 * 4)))

try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logger.warning(f"Could not load tokenizer: {e}")
    tokenizer = None

security = HTTPBearer()

def create_jwt_for_user(username: str, role: str = "user") -> str:
    now = int(time.time())
    payload = {
        "sub": username, # subject, idetiied the user
        "role": role,  # what permission use has
        "iat": now, # current timestamp when this token was issue s
        "exp": now + JWT_EXPIRY_SECONDS
    }

    token = jwt.encode(payload, JWT_SECRET, JWT_ALGORITHM)

    #pyjwt return str in v2+ , while it return bytes
    if isinstance(token, bytes):
        token = token.decode()
    return token

def decode_jwt_token(token: str) -> Dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, JWT_ALGORITHM)
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.INnvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

MODEL_INFO.info(
    {
        'model_name' : MODEL_NAME,
        'backend': BACKEND,
        'version': '2.0.0'
    }
)


# Safety and control filter

class ContentFilter:

    def __init__(self):
        self.banned_keywords = [
            'hack', 'virus', 'malware', 'phising', 'exploit',
            'bomb', 'weapon', 'drug', 'violence'
        ]
        
        self.injection_patterns = [
            re.compile(r"ignore\s+(all|previous)\s+intructions", re.I),
            re.compile(r"you\s+are\s+now", re.I)
        ]

        self.pii_patterns = {
            'ssn': re.compile(r'\b^\d{3}-\d{2}-\d{4}$\b'),
            'credit_card': re.compile(r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})$'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        }

    def check_content_safety(self, text: str) -> Dict:
        violations = []
        severity = "none"

        text_lower = text.lower()


            #checked banned keywords
        for keyword in self.banned_keywords:
            if keyword in text_lower:
                violations.append(f"banned_keywords: {keyword}")
                severity = "high"

        for pattern in self.injection_patterns:
            if pattern.search(text):
                violations.append("prompt_injection_detected")
        # check for pii
        for pii_type, pattern in self.pii_patterns.items():
            if pattern.search(text):
                violations.append(f"pii_detected: {pii_type}")
                if severity == "none":
                    severity = "medium"

        return {
            "safe": len(violations) == 0,
            "violations": violations,
            "severity": severity
        }


    def sanitize_output(self, text: str) -> str:
        text = re.sub(self.pii_patterns['email'], 'EMAIL ********', text)
        return text


content_filter = ContentFilter()



class Guardrails:
    def __init__(self):
        self.max_prompt_length = 400
        self.max_output_tokens = 300


    def validate_Request(self, payload: Dict, user: str) -> Dict:
        violations = []
        prompt = payload.get("prompt", "")
        max_tokens = payload.get("max_tokens", 50)

        if len(prompt) > self.max_prompt_length:
            violations.append(f"prompt too longs : {len(prompt) > {self.max_prompt_length}}")


        if max_tokens > self.max_output_tokens:
            violations.append(f"token_limit_exceeded: {int(max_tokens)} > {self.max_output_tokens}")

        if user == "restricted_user" and max_tokens > 100:
            violations.append(f"user_token_limit: restricted_user max 100 tokens")

        return {
            "valid": len(violations) == 0,
            "violations": violations
        }


guardrails = Guardrails()

COST_PER_INPUT_TOKEN = 0.001
COST_PER_OUTPUT_TOKEN = 0.004

class CostController:
    
    def __init__(self):
        self.daily_limits = {
            "alice" : 10.0,
            "bob" : 100.0,
            "premium": 100.0,
            "default": 0.5
        }

        self.daily_spending: Dict[str, float] = {}
        self.last_rest = time.time()

    def reset_daily_if_needed(self):
        current_time = time.time()
        if current_time - self.last_rest > 86400:
            self.daily_spending = {}
            self.last_rest = current_time

    def calculate_cost(self, input_tokens: int, output_token: int) -> float:
        input_cost = input_tokens * COST_PER_INPUT_TOKEN
        output_cost = output_token * COST_PER_OUTPUT_TOKEN
        return input_cost + output_cost
    
    
    def check_cost_limit(self, user: str, estimated_cost: float) -> Dict:
        self.reset_daily_if_needed()

        current_spending = self.daily_spending.get(user, 0.0)
        daily_limit = self.daily_limits.get(user, self.daily_limits["default"])

        if current_spending + estimated_cost > daily_limit:
            return {
                "allowed": False,
                "reason": f"Daily limit exceeded ${current_spending + estimated_cost:.4f} > ${daily_limit}",
                "current_pending": current_spending,
                "daily_limit": daily_limit
            }
        return{"allowed": True}
    
    def record_spending(self, user: str, cost: float):
        self.reset_daily_if_needed()
        self.daily_spending[user] = self.daily_spending.get(user, 0.0) + cost

costcontroller = CostController()

def _count_tokens(text: str) -> int:
    if not text or not tokenizer:
        return len(text.split())
    return len(tokenizer.encode(text))


@app.on_event("startup")
async def startup_event():
    logger.info("Starting LLMOps Ollama tiny llm service")
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        logger.info(f"Ollama heaklth check: {response.status_code}")
    except Exception as e:
        logger.error(f"Could not connect to Ollama: {e}")


@app.get("/")
def root():
    return {
        "service" : "Hello World"
    }

@app.get("/health")
def health():
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        healthy = response.status_code == 200
    except:
        healthy = False

    status = "healthy" if healthy else "degraded"

    return {
        "status" : status,
        "ollama_backend": healthy,
        "model": MODEL_NAME,
        "active_requests" : int(ACTIVE_REQUESTS._value._value),
        "timestamp" : time.time()
    }

@app.get("/metrics")
def get_metrics():
    return PlainTextResponse(
        generate_latest(),
        media_type="text/plain"
    )

@app.post("/generate")
async def generate_text(payload: dict):
    user = payload.get("user", "anyonymous")
    prompt = payload.get("prompt", "").strip()
    max_tokens = min(payload.get("max_tokens", 50), 200)
    temperature = max(0.0, min(2.0, payload.get("temperature", 0.7)))


    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt not available")
    
    #Step 1: Content safety check
    safety_result = content_filter.check_content_safety(prompt)
    if not safety_result["safe"]:
        SAFETY_VIOLATION.labels(
            violation_type = "content_filter",
            severity = safety_result["severity"]
        ).inc()

        CONTENT_FILTER.labels(user=user, filter_type="safety").inc()

        raise HTTPException(
            status_code=400,
            detail= f"content safety violations: {', '.join(safety_result['violations'])}"
        )
    
    # step 2: business guarrails:
    guardrails_result = guardrails.validate_Request(payload=payload, user=user)
    if not guardrails_result["valid"]:
        GUARDRAIL_VIOLATION.labels(user=user, violation_type = "business_rule").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Guardrail violations: {', '.join(guardrails_result['violations'])}"
        )
    # Enforce governance
    enforce_rate_limit(user)

    input_tokens = _count_tokens(prompt)
    estimated_output_tokens = min(max_tokens, 100)
    estimated_cost = costcontroller.calculate_cost(input_tokens, estimated_output_tokens)

    cost_check = costcontroller.check_cost_limit(user, estimated_cost)
    if not cost_check["allowed"]:
        COST_BLOCKED_REQUEST.labels(user=user).inc()
        raise HTTPException(
            status_code=429,
            detail=f"Cost limit exceeded: {cost_check['reason']}"
        )

    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        ollama_payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        logger.info(f"Processing Request for User {user} --> Ollama : {OLLAMA_URL}")

        response = requests.post(
            OLLAMA_URL,
            json=ollama_payload,
            timeout=60,
            headers={"Content-Type": "application/json"}
        )

        response.raise_for_status()

        #Robuse json parsing
        text = response.text.strip()
        try:
            result = response.json()
            logger.info(f"result: {result}")
        except ValueError as ex:
            logger.error(f"Value error occured : {str(ex)}")

        generated_text = None
        if "response" in result and isinstance(result["response"], str):
            generated_text = result["response"].strip()
        elif "completions" in result and len(result["completions"]) > 0:
            generated_text = result["completions"][0].get("text", "").strip()


        if not generated_text:
            raise HTTPException(
                status_code=500,
                detail="NO valid response"
            )
        
        # step 5:  Output safety check
        output_safety = content_filter.check_content_safety(generated_text)
        if not output_safety["safe"]:
            SAFETY_VIOLATION.labels(
                violiation_type = "output_filter",
                severity=output_safety["severity"]
            ).inc()
            generate_text = "[CONTENT FILTERED - SAFETY VIOLATION]"
        else:
            generate_text = content_filter.sanitize_output(generated_text)
        end_time = time.time()
        latency = end_time - start_time

        # count tokens
        input_tokens = _count_tokens(prompt)
        output_tokens = _count_tokens(generated_text)

        actual_cost = costcontroller.calculate_cost(input_tokens=input_tokens, output_token=output_tokens)

        total_tokens = input_tokens + output_tokens

        costcontroller.record_spending(user, actual_cost)

        record_requst(
            backend=BACKEND,
            user=user,
            model=MODEL_NAME,
            status="200",
            latency=latency,
            tokens_in=input_tokens,
            tokens_out=output_tokens
        )
        logger.info(
            f"Request completd: user={user}, latency={latency:.2f}, tokens={total_tokens}"
        )

        return {
            "backend": BACKEND,
            "model": MODEL_NAME,
            "user": user,
            "generated_text": generated_text,
            "metrics": {
                "latency_seconds": round(latency, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "max_tokens": max_tokens
            }
        }
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        record_requst(BACKEND, user, MODEL_NAME, "503", time.time() - start_time)
        raise HTTPException(
            status_code=503,
            detail=f"Model service unavailable : {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        record_requst(BACKEND, user, MODEL_NAME, "500", time.time() - start_time)
        raise HTTPException(
            status_code=5003,
            detail=f"internal server error: {str(e)}"
        )
    finally:
        ACTIVE_REQUESTS.dec()

if __name__ == "main":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)