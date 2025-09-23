def coherence_score(text):
    sentences = [s.strip() for s in text.split('.') if s.strip()]


    if (len(sentences)) < 2:
        return 1.0
    

    # check for transistion words

    transistion_words = [
        'however', 'therefore', 'furthermore', 'moreover', 'additonally',
        'consequesntly', 'thus', 'hence', 'meanwhile', 'subsequently'
    ]


    transition_count = sum(1 for sentence in sentences
                           for word in transistion_words
                           if word in sentence.lower())
    
    transition_score = min (1, 0, transition_count/ len(sentences))


    # topic consistency

    consistency_score = []
    for i in range(len(sentences) - 1):
        words1 = set(sentences[i].lower().split())
        words2 = set(sentences[i+1].lower().split())


        if words1 and words2:
            overlap = len(words1 & words2) / len(words1 | words2)
            consistency_score.append(overlap)


    avg_consustency = sum(consistency_score) / len(consistency_score) if consistency_score else 0

    return (transition_score + avg_consustency) / 2


text = "The economy is slowing down. However, the government is planning new policies. These policies will boost consumer spending."



result = coherence_score(text)
print(result)