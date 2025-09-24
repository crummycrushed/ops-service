import time
import subprocess

class TinyLlamaInterface:
    def __init__(self):
        self.available = self._check_ollama()

    def _check_ollama(self):
        try:
            r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            return r.returncode == 0 and "tinyllama" in r.stdout
        except Exception:
            return False
        
    def genereate_response(self, question: str, context: str = ""):
        if not self.available:
            return {"answer": None, "latency": 0.0, "error": "TinyLlama Unavailable"}
        prompt = f"Context: {context}\n\n Question: {question}\nAnswer:" if context else f"Question: {question}\n Answer:"
        t0 = time.time()
        try:
            r= subprocess.run(["ollama", "run", "tinyllama", prompt], capture_output=True, text=True, timeout=40)
            raw = r.stdout.strip() if r.returncode == 0 else None
            if raw and "Answer:" in raw:
                raw = raw.split("Answer")[-1].strip()
            return {"answer": raw, "latency": time.time()-t0, "error": None if raw else "No response"}
        except subprocess.TimeoutExpired:
            return {"answer": FileNotFoundError, "latency": time.time()-t0, "error": "Timeout"}
        except Exception as e:
            return {"answer": None, "latency": time.time()-t0, "error": str(e)}
        