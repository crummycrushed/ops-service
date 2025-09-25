import numpy as np
from typing import List, Dict, Any
from ragas.dataset_schema import SingleTurnSample
from sentence_transformers import SentenceTransformer, util

class HallucinationDetector:

    def __init__(self):
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    def score(self, samples: List[SingleTurnSample], predictions: List[str]) -> Dict[str, Any]:
        halluc_scores, details = [], []

        for sample, pred in zip(samples, predictions):
            context_text = " ".join(sample.retrieved_contexts)

            # overlap
            ctx_tokens, pred_tokens = set(context_text.split()), set(pred.split())
            overlap = len(pred_tokens & ctx_tokens) / max(len(pred_tokens), 1)

            # semantic similarity
            ctx_emb = self.similarity_model.encode(context_text, convert_to_tensor=True)
            pred_emb = self.similarity_model.encode(pred, convert_to_tensor=True)

            sim = util.cos_sim(pred_emb, ctx_emb).item()

            halluc_score = 1 - (0.5 * overlap + 0.5 * sim)
            halluc_scores.append(halluc_score)


            details.append({
                "question": sample.user_input,
                "prediction": pred,
                "hallucination_score": halluc_score,
                "overlap": overlap,
                "semantic": sim
            })
        return {"avg_hallucination": float(np.mean(halluc_scores)), "details": details}
    
    def run(self, samples: List[SingleTurnSample], predictions: List[str]) -> Dict[str, Any]:
        return self.score(samples, predictions)