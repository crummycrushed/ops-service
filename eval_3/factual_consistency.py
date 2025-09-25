import time
import numpy as np
from typing import List, Dict, Any
from ragas.dataset_schema import SingleTurnSample
from transformers import pipeline

# Check if the model prediction contradicts known context using NLI
# step 1:  Prepare hypotehsis and premis 
#.   premisa  --> all context text
#.  Hypotheis : model predictions
# step 2 : NLI classificaiton </s> special toke use to seperate sequence in RoBERT
# labels : ENTAILMENT L. prediction is supported by context score : 1.0
# Neutral : prediciont is neither supported not contradicted --> score 0.5
# Contradiction prediction contradicts context 

class FactualConsistencyMetric:

    def __init__(self):
        self.nli = pipeline("text-classification", model="roberta-large-mnli")

    def score(self, samples: List[SingleTurnSample], predictions: List[str]) -> Dict[str, Any]:
        results = []
        for sample, pred in zip(samples, predictions):
            premise = " ".join(sample.retrieved_contexts)
            hypothesis = pred
            res = self.nli(f"{premise} </s></s> {hypothesis}")
            label = res[0]["label"]


            if label == "CONTRADICTION":
               score = 0.0
            elif label == "NEUTRAL":
                score = 0.5
            else:
                score = 1.0
            results.append(score)


        return {"factual_consistency": float(np.mean(results))}
    

    def run(self, samples: List[SingleTurnSample], predictions: List[str]) -> Dict[str, Any]:
        return self.score(samples, predictions)



