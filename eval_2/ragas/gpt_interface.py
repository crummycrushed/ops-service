import time
from openai import OpenAI

class GPTInterface:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        try:
            self.client = OpenAI()
            self.available = True
        except Exception as e:
            print(f"GPT error {str(e)}")

    def generate_response(self, question: str, context: str = ""):
        if not self.available:
            return {"answer": None, "latency": 0.0, "error": "GPT Unavailable"}
        prompt = f"Context: {context}\n\n Question: {question}\nAnswer:" if context else f"Question: {question}\n Answer:"
        t0 = time.time()

        try:
            print(f"promt: {prompt}")
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpfl assitant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            print(f"resp:{resp}")
            ans = resp.choices[0].message.content
            print(f"GPTans: {ans}")
            return {"answer": ans, "latency": time.time()-t0, "error": None}
        except Exception as e:
            print(f"gpterr: {str(e)}")
            return {"answer": None, "latency": time.time()-t0, "error": None}
