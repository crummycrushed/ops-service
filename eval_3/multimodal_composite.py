import time
from typing import List, Dict, Any
import numpy as np

from ragas.dataset_schema import SingleTurnSample
from hallucination_detector import HallucinationDetector
from factual_consistency import FactualConsistencyMetric
from tiny_llama_interface import TinyLlamaInterface


class AdvanacedCompositeMetric:

    def __init__(self, metrics_config: Dict[str, Dict]):
        self.config = metrics_config
        self.metrics = {
            "hallucination": HallucinationDetector(),
            "factual_consistency": FactualConsistencyMetric()
        }

    def score(self, samples: List[SingleTurnSample], predictions: List[str]) -> Dict[str, Any]:
        metrics_score = {}
        detailed_results = {}

        for metric_name, metrics in self.metrics.items():
            if metric_name in self.config:
                result = metrics.run(samples, predictions)
                main_score_key = list(result.keys())[0]
                metrics_score[metric_name] = result[main_score_key]
                detailed_results[metric_name] = result


        composite_score = 0
        weighted_scores = {}
        for metric_name, score in metrics_score.items():
            config = self.config[metric_name]
            weight = config.get('weight', 1.0)
            if config.get('inverse', False):
                normalized_score = 1 - score
            else:
                normalized_score = score
            weighted_scores = weight * normalized_score
            composite_score = composite_score  + weighted_scores


        # Grading and recommendation
        quality_grade = self._calculate_grade(composite_score)
        recommendations = self._generate_recommendations(metrics_score)

        return {
            "composite_score": composite_score,
            "quality_grade": quality_grade,
            "individual_scores": metrics_score,
            "weighted_scores": weighted_scores,
            "detailed_results" : detailed_results,
            "recommendations": recommendations
        }



    
    def _calculate_grade(self, score: float) -> str:
        if score >= 0.9: return "A+"
        elif score >= 0.8: return "A"
        elif score >= 0.6: return "B"
        elif score >= 0.5: return "C"
        return "F"
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        recs = []
        if scores.get('hallucination', 0) > 0.4:
            recs.append("reduce hallucinations by better context")
        if scores.get("factual_consistency", 1) < 0.6:
            recs.append("Enhance factual accuracy and consistency")
        return recs
    



class MultiModelEvaluator:

    def __init__(self):
        self.models = {}
        self.composite_metrics = AdvanacedCompositeMetric(
            {
                'hallucination': {'weight': 0.4, 'threshold': 0.3, 'inverse': True}, 
                'factual_consistency': {'weight': 0.6, 'threshold': 0.3}, 
            }
        )

    def add_model(self, name: str, model_fuction):
        self.models[name] = model_fuction

    
    def compare_models(self, test_Samples: List[SingleTurnSample]) -> Dict[str, Any]:
        results = {}

        for model_name, model_func in self.models.items():
            predictions = []
            for sample in test_Samples:
                try:
                    pred = model_func(sample.user_input, sample.retrieved_contexts)
                    predictions.append(pred)
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
                    predictions.append("Error")

            evaluation_results = self.composite_metrics.score(test_Samples, predictions)
            results[model_name] = evaluation_results

        return {
            "individual_results": results,
            "rankings": self._rank_models(results)
        }
    
    def _rank_models(self, results: Dict) -> List[Dict]:
        ranking = []
        for mode_name, result in results.items():
            ranking.append(
                {
                    "model": mode_name,
                    "composite_score": result["composite_score"],
                    "grade": result['quality_grade']
                }
            )

        return sorted(ranking,
                       key=lambda x: x['composite_score'],
                         reverse=True)
    

def tiny_llama_model(question: str, context: List[str]) -> str:
    llama = TinyLlamaInterface()
    out = llama.genereate_response(question, " ".join(context))
    return out["answer"] or ""
    

if __name__ == "__main__":
    test_samples = [
        SingleTurnSample(
            user_input="What is photosynthesis?",
            retrieved_contexts=["Photosyntehsis is the process of conversion into chemical energy"],
            response="Photosynthesis is how plants use sunlight to produce energy.",
        ),
        SingleTurnSample(
            user_input="Explain the water cycle",
            retrieved_contexts=["The water cycle describes how water evaporates,"],
            response="Water evaporates, forms clouds, and rains back to the Earth"
        )
    ]

    evaluator = MultiModelEvaluator()
    evaluator.add_model("TinyLlama", tiny_llama_model)

    results = evaluator.compare_models(test_samples)

    print(results)