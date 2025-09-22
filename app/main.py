import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest
import requests
import time
import tiktoken
import os

from app.governance import enforce_rate_limit
from app.metrics import (
    record_requst, ACTIVE_REQUESTS, MODEL_INFO, TOKENS_PER_SECOND
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="LLMOps LLM service",
    version="2.0.0"
)

OLLAMA_HOST = "ollama"
OLLAMA_PORT = "11434"

OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
MODEL_NAME = "tinyllama"
BACKEND = "ollama"


try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logger.warning(f"Could not load tokenizer: {e}")
    tokenizer = None


MODEL_INFO.info(
    {
        'model_name' : MODEL_NAME,
        'backend': BACKEND,
        'version': '2.0.0'
    }
)


def _count_tokens(text: str) -> int:
    if not text or not tokenizer:
        return len(text.split())
    return len(tokenizer.encode(text))


@app.on_event("startup")
async def startup_event():
    logger.info("Starting LLMOps Ollama tiny llm service")
    try:
        response = requests.get("http://ollama:11434/api/version", timeout=5)
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
        response = requests.get("http://ollama:11434/api/version", timeout=5)
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
    
    # Enforce governance
    enforce_rate_limit(user)

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
        
        end_time = time.time()
        latency = end_time - start_time

        # count tokens
        input_tokens = _count_tokens(prompt)
        output_tokens = _count_tokens(generated_text)

        total_tokens = input_tokens + output_tokens

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