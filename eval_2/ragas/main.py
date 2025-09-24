import os
import time
import pandas as pd
from dataset import RAGEvaluationDataset
from tiny_llama_interface import TinyLlamaInterface
from util import align_data
from ragas_evaluator import RagasEvalautor
from gpt_interface import GPTInterface


class LLMEvaluatorFramework:
    def __init__(self):
        self.tiny = TinyLlamaInterface()
        self.gpt = GPTInterface()
        self.dataset = RAGEvaluationDataset().data
        self.ragas = RagasEvalautor()


    def run_evaluation(self):
        question = self.dataset["question"]
        contexts = self.dataset["contexts"]
        ground_truths = self.dataset["ground_truth"]


        tiny_results= []
        gpt_Results = []

        for i, (q, c) in enumerate(zip(question, contexts)):
            ctx = c[0] if isinstance(c, (list, tuple)) and len(c) > 0 else (c if isinstance(c, str) else "")
            gpt_r = self.gpt.generate_response(q, ctx)
            tiny_r = self.tiny.genereate_response(q, ctx)

            tiny_results.append(tiny_r)
            gpt_Results.append(gpt_r)

        # Ragas evalaition

        tiny_answers = [r.get("answer") for r in tiny_results]
        gpt_answers = [r.get("answer") for r in gpt_Results]


        qs_t, cs_t, ans_t, gts_t , _ = align_data(question, contexts, tiny_answers, ground_truths)
        tiny_ragas = self.ragas.evaluate(qs_t, cs_t, ans_t, gts_t)


        qs_g, cs_g, ans_g, gts_g , _ = align_data(question, contexts, gpt_answers, ground_truths)
        gpt_ragas = self.ragas.evaluate(qs_g, cs_g, ans_g, gts_g)

        print("Ragas Evaluation Results")


        if not gpt_ragas.empty:
            for col in gpt_ragas.columns:
                value = gpt_ragas[col].iloc[0] if len(gpt_ragas) > 0 else 0
                try:
                    numeric_value = float(value)
                    print(f" {col}: {numeric_value}")
                except (ValueError, TypeError):
                     print(f" {col}: {value}")
        else:
            print(" No GPT ragas results available")

        if not tiny_ragas.empty:
            for col in tiny_ragas.columns:
                value = tiny_ragas[col].iloc[0] if len(tiny_ragas) > 0 else 0
                try:
                    numeric_value = float(value)
                    print(f" {col}: {numeric_value}")
                except (ValueError, TypeError):
                     print(f" {col}: {value}")
        else:
            print(" No tinyllama ragas results available")

        rows = []
        tiny_ragas_summary = {}
        gpt_ragas_summary = {}

        if not gpt_ragas.empty:
            for col in gpt_ragas.columns:
                gpt_ragas_summary[f"gpt_ragas_{col}"] = gpt_ragas[col].iloc[0]


        if not tiny_ragas.empty:
            for col in tiny_ragas.columns:
                tiny_ragas_summary[f"tiny_ragas_{col}"] = tiny_ragas[col].iloc[0]


        for i, q in enumerate(question):
            t = tiny_results[i]
            g = gpt_Results[i]

            row = {
                "question": q,
                "context": contexts[i][0] if isinstance(contexts[i], (list, tuple)) else contexts[i],
                "tiny_answer": t.get("answer"),
                "tiny_latency": t.get("latency"),
                "tiny_error": t.get("error"),
                "gpt_answer": g.get("answer"),
                "gpt_latency": g.get("latency"),
                "gpt_error": g.get("error")
            }
            row.update(gpt_ragas_summary)
            row.update(tiny_ragas_summary)
            rows.append(row)
        
        df_all =  pd.DataFrame(rows)
        out_csv  = os.path.join("results", "llm_evaluation.csv")
        os.makedirs("results", exist_ok=True)
        df_all.to_csv(out_csv, index=False)

        return {
            "results_df": df_all,
            "tiny_ragas": tiny_ragas,
            "gpt_ragas": gpt_ragas
        }


def main():
    framework  = LLMEvaluatorFramework()
    out = framework.run_evaluation()
    print("\nDone")


if __name__ == "__main__":
    main()

