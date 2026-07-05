import csv
import re

def bayesian_sentiment_analysis(file_path):
    total_reviews = 0
    positive_reviews = 0
    keywords = ["excellent", "masterpiece", "wonderful", "terrible", "awful", "worst"]
    counts = {kw: {'total': 0, 'positive': 0} for kw in keywords}

    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            review = row['review'].lower()
            sentiment = row['sentiment'].lower()
            
            is_positive = (sentiment == 'positive')
            
            total_reviews += 1
            if is_positive:
                positive_reviews += 1
                
            for kw in keywords:
                if re.search(rf'\b{kw}\b', review):
                    counts[kw]['total'] += 1
                    if is_positive:
                        counts[kw]['positive'] += 1

    p_positive = positive_reviews / total_reviews

    print(f"Prior P(Positive): {p_positive:.4f}\n")
    
    for kw in keywords:
        p_kw = counts[kw]['total'] / total_reviews
        
        if positive_reviews > 0:
            p_kw_given_pos = counts[kw]['positive'] / positive_reviews
        else:
            p_kw_given_pos = 0
        if p_kw > 0:
            p_pos_given_kw = (p_kw_given_pos * p_positive) / p_kw
        else:
            p_pos_given_kw = 0
            
        print(f"--- Keyword: '{kw}' ---")
        print(f"Likelihood P({kw} | Positive): {p_kw_given_pos:.4f}")
        print(f"Marginal P({kw}): {p_kw:.4f}")
        print(f"Posterior P(Positive | {kw}): {p_pos_given_kw:.4f}\n")

bayesian_sentiment_analysis('IMDB Dataset.csv')
