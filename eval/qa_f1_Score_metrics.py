def qa_f1_score(reference_answer, predicted_answer):

    def normalise_answer(text):
        import re
        text = re.sub(r'\b(a|an|the)\b', '', text.lower())
        text = re.sub(r'\b[^\w\s]\b', '', text.lower())
        text = re.sub(r'\s+', ' ', text.lower())
        return text.strip()
    

    ref_tokesn = normalise_answer(reference_answer).split()
    pred_tokens = normalise_answer(predicted_answer).split()


    if not ref_tokesn and not pred_tokens:
        return 1.0
    if not ref_tokesn or not pred_tokens:
        return 0.0
    

    common_tokens = set(ref_tokesn) & set(pred_tokens)

    if not common_tokens:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokesn)

    f1 = 2 * precision * recall / (precision + recall)

    return f1

print(qa_f1_score("The capital of France", "berlin"))

