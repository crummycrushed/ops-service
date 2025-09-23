def fluency_score(text):


    sentences = text.split('.')
    words = text.split()


    if not words:
        return 0.0
    

    unique_words = set(words)
    repetition_penalty = len(unique_words) / len(words)

    avg_sent_length = len(words) / len(sentences) if sentences else 0
    length_score = min(1.0, avg_sent_length/15) if avg_sent_length <= 25 else 0.5


    #gramma check
    has_capitals = any(c.isupper() for c in text)
    has_punctuation = any(c in '.,!?;:' for c in text)

    grammar_Score = 0.5 * has_capitals + 0.5 * has_punctuation

    return (repetition_penalty + length_score + grammar_Score) / 3


text = "This is a well-wriiten sentence with proper grammar."
text2 = "this text has no capitals or puncutations and repeats words words words"


print(f"Fluency score 1: {fluency_score(text)}")
print(f"Fluency score 2: {fluency_score(text2)}")