from bert_score import score

def calculate(reference, candidates, model_type="microsoft/deberta-xlarge-mnli"):
    P, R, F1 = score(candidates, reference, model_type=model_type, verbose=False)

    return {
        'bertscore_prescion': P.mean().item(),
        'bertscore_recall': R.mean().item(),
        'bertscore_f1': F1.mean().item(),
        'bertscore_prescion_std': P.std().item(),
        'bertscore_recall_std': R.std().item(),
        'bertscore_f1_std': F1.std().item()
    }




references = ["the cat sat on the mat", "the quick brown fox jumps over the lazy dog"]
candidates = ["the cat sits on the mat", "a fast brown fox leaped over the lazy dog"]

bertscore_results = calculate(references, candidates)
print("BERTScore results:", bertscore_results)





f1 score = [0.8, 0.9. 0.7]

std devaition = sqrt((0.8-0.8)^2  + (0.9-0.8)^2 + (0.7-0.8)^2 /2)