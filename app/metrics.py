from prometheus_client import Counter, Histogram, Gauge, Info


REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["backend", "user", "status", "model"]
)


REQUEST_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "Request processing time",
    ["backend", "model"]
)

ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Current processing requests"
)


INPUT_TOKENS = Counter(
    "llm_input_tokens_total",
    "Total input tokens processed",
    ["backend", "user", "model"]
)


OUTPUT_TOKENS = Counter(
    "llm_output_tokens_total",
    "Total output tokesn processed",
    ["backend", "user", "model"]
)

TOKENS_PER_SECOND = Gauge(
    "llm_tokens_per_second",
    "Current token generation rate",
    ["backend", "model"]
)


MODEL_INFO = Info(
    "llm_model_info",
    "Information about the deployed model"
)


#basic governance metrics
RATE_LIMIT_EXCEEDED = Counter(
  "llm_rate_limit_exceeded_total",
  "Rate limit violations",
  ["user"]
)

USER_REQUEST_COUNT = Counter(
    "llm_user_requests_total",
    "Request per user",
    ["user"]
)


# Helper func
def record_requst(backend, user, model, status, latency, tokens_in=0, tokens_out=0):
    REQUEST_COUNT.labels(
        backend = backend,
        user = user,
        status= status,
        model = model
    ).inc()


    REQUEST_LATENCY.labels(backend=backend, model=model).observe(latency)

    if tokens_in > 0:
        INPUT_TOKENS.labels(backend=backend, user=user, model=model).inc(tokens_in)
    if tokens_out > 0:
        OUTPUT_TOKENS.labels(backend=backend, user=user, model=model).inc(tokens_out)

    total_tokens = tokens_in + tokens_out
    if latency>0 and total_tokens>0:
        TOKENS_PER_SECOND.labels(backend=backend, model=model).set(total_tokens / latency)