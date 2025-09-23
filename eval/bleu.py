from sacrebleu import BLEU

# n grams overlap
# [ "A", "is", "word", "x"]
# ["A", "word" , "is", "y"]
def calculate_bleu(references, candidates):
    bleu = BLEU()


    refs_formatted = [[ref] for ref in references]
    score = bleu.corpus_score(candidates, refs_formatted)
    return {
        'bleu_score': score.score,
        'bleu_1': score.precisions[0],
        'bleu_2': score.precisions[1],
        'bleu_3': score.precisions[2],
        'bleu_4': score.precisions[3],
        'brevity_penalty': score.bp
    }


references = ["the cat sat on the mat", "Paris is the captial of France"]
candidates = ["the cat sits on the mat", "Paris is the France's capital city"]

bleu_results = calculate_bleu(references, candidates)
print("BLEU Results:", bleu_results)