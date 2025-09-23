def toxicity_score(text):
    toxic_words = [
        'hate', 'kill', 'murder', 'bomb', 'weapon'
    ]



    text_lower = text.lower()

    toxic_count = sum(1 for word in toxic_words if word in text_lower)

    total_words = len(text.split())

    if total_words == 0:
        return 0.0
    
    toxicity_ratio = toxic_count / total_words

    return min(1.0, toxicity_ratio)



text = "I hate waiting in lines."
print(toxicity_score(text))