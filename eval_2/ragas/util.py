def align_data(questions, contexts, answers, ground_truths):
    valid_idx = [i for i, a in enumerate(answers) if a and isinstance(a, str) and len(a.strip()) > 0]
    qs = [questions[i] for i in valid_idx]
    cs = [contexts[i] for i in valid_idx]
    ans = [answers[i] for i in valid_idx]
    gts = [ground_truths[i] for i in valid_idx]
    return qs, cs, ans, gts, valid_idx