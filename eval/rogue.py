import numpy as np
from rouge_score import rouge_scorer

def cacluate_rouge(reference, candidates):
    scorer  = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


    rouge_Results = {'rouge1':[], 'rouge2': [], 'rougeL': []}


    for ref, cand in zip(reference, candidates):
        scores = scorer.score(ref, cand)


        for metrics in rouge_Results.keys():
            rouge_Results[metrics].append(
                {
                    'precision': scores[metrics].precision,
                    'recall': scores[metrics].recall,
                    'fmeasure': scores[metrics].fmeasure

                }
            )

    avg_results = {}
    for metrics in rouge_Results.keys():
        avg_results[f'{metrics}_precision'] = np.mean([s['precision'] for s in rouge_Results[metrics]])
        avg_results[f'{metrics}_recall'] = np.mean([s['recall'] for s in rouge_Results[metrics]])
        avg_results[f'{metrics}_f1'] = np.mean([s['fmeasure'] for s in rouge_Results[metrics]])

    return avg_results

references = ["the cat sat on the mat", "Paris is the captial of France"]
candidates = ["the cat sits on the mat", "Paris is the France's capital city"]

rogue_results = cacluate_rouge(references, candidates)
print("Rogue Results:", rogue_results)



# ROGUE 1"

Many words overlap : "brown", "fox", "over", "the", "lazy", "dof"

precsion is high ---> most wors in cadidates are useful

reacall also high --> candidates covered much of references
F1 is close to 1.0  --> overlap is strong 


ROGUE 2

ref biagrams : "quick brown", "brown fox", "fox jumps"
canidate biagrams. "fast brown", "brown fox", "fox leap]"

precision and recall drops
f1 drops

response from model is not of high quality


p