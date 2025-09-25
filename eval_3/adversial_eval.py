import time
import subprocess
import numpy as np
from typing import List

from ragas.dataset_schema import SingleTurnSample
from factual_consistency import FactualConsistencyMetric
from hallucination_detector import HallucinationDetector
from tiny_llama_interface import TinyLlamaInterface


class AdversarialEvaluator:
    def __init__(self, llm: TinyLlamaInterface):
        self.llm = llm
        self.hallu = HallucinationDetector()
        self.facts = FactualConsistencyMetric()


    def add_noise(self, sample: SingleTurnSample):
        return SingleTurnSample(
            user_input=sample.user_input + " actually really ",
            retrieved_contexts=sample.retrieved_contexts,
            response=sample.response,
            reference=sample.reference,
            metadata={"type": "noise"}
        )
    
    def add_contradiction(self, sample: SingleTurnSample):
        return SingleTurnSample(
            user_input=sample.user_input,
            retrieved_contexts=sample.retrieved_contexts + ["However, the opposite might be true."],
            response=sample.response,
            reference=sample.reference,
            metadata={"type": "contradiction"}
        )
    
    def generate_adversarials(self, samples: List[SingleTurnSample]):
        adv = []
        for s in samples:
            adv.append(self.add_noise(s))
            adv.append(self.add_contradiction(s))
        return adv
    
    def evaluate(self, samples: List[SingleTurnSample]):
        predictions = []
        for s in samples:
            out = self.llm.genereate_response(s.user_input, " ".join(s.retrieved_contexts))
            predictions.append(out["answer"] or "")
        
        return {
            "hallucination": self.hallu.run(samples, predictions),
            "factual": self.facts.run(samples, predictions)
        }
    
    def run(self, base_samples: List[SingleTurnSample]):
        adv_samples = self.generate_adversarials(base_samples)
        return self.evaluate(adv_samples)
    


samples = [
    SingleTurnSample(
        user_input="What is photosynthesis?",
        retrieved_contexts=["Photosyntehsis is the process by which plants convert light energy into chemical energy"],
        response="Photosynthesis is how plants use sunlight to produce energy.",
        reference="Photosynthesis is the process by which green plats and some other organism use sunligh to synthesize foods with the help of chlorophyll"
    )
]


llm = TinyLlamaInterface()


predictions = [llm.genereate_response(s.user_input, " ".join(s.retrieved_contexts))["answer"] or "" for s in samples]
halluc = HallucinationDetector().run(samples, predictions)
facts = FactualConsistencyMetric().run(samples, predictions)

print("Baseline hallucination: ", halluc)
print("baseline Factual consistency: ", facts)



adv_eval = AdversarialEvaluator(llm)
adv_results = adv_eval.run(samples)

print("Adversarial results: ", adv_results)

