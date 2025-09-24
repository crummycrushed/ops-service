import pandas as pd
from datasets import Dataset
from ragas.metrics import BleuScore
from ragas.metrics import RougeScore
from ragas import evaluate
from ragas import EvaluationDataset

class RagasEvalautor:
    def __init__(self):
        self.avaialble = False
        self.metrics = []

        try:
            self.metrics.append(BleuScore())
            self.metrics.append(RougeScore())


            self.evaluate_func = evaluate
            self.avaialble = True

            metric_names = [type(m).__name__ for m in self.metrics]
            print(f"RAGAS loaded with len {len(self.metrics)} metrics: {metric_names}")

        except Exception as e:
            print(f"Ragas initialisation error: {e}")

    def evaluate(self, questions, contexts, answers, ground_truths):
        if not self.avaialble:
            return pd.DataFrame({"evaluation_starus": ["unavailable"]})
        
        if not answers or not ground_truths:
            return pd.DataFrame({"evaluation_starus": ["no data"]})
        

        try:
            ds = Dataset.from_dict({
                "response": [str(a) for a in answers],
                "reference": [str(g) for g in ground_truths]
            })

        # convert convert ragas format and evaluate
            eval_ds = EvaluationDataset.from_hf_dataset(ds)
            result = self.evaluate_func(eval_ds, metrics=self.metrics)

            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                if not df.empty:
                    results = {}
                    for col in df.columns:
                        if df[col].dtype in ['float64', 'int64']:
                            results[col] = float(df[col].mean())

                    if results:
                        print(f"Evaluation completed with {len(results)} metrics")
                        return pd.DataFrame([results])

            fallback = {
                "bleu_score" : 0.0,
                "rouge_score": 0.08,
            }
            return pd.DataFrame([fallback])
        except Exception as e:
            return pd.DataFrame([{"bleu_score": 0.00}])




