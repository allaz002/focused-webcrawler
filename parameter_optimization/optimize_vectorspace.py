import json
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from itertools import product

# Pfade
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINING_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training_data', 'vsm_training.json')

# Daten laden
with open(TRAINING_DATA, 'r', encoding='utf-8') as f:
    data = json.load(f)

topic_texts = [item['text'] for item in data if item['label'] == 'topic']
idf_texts = [item['text'] for item in data if item['label'] == 'idf']
all_texts = idf_texts + topic_texts

# Parameter
param_grid = {
    'ngram_range': [(1, 1), (1, 2)],
    'min_df': [1, 2],
    'max_df': [0.6, 0.7, 0.8, 0.9],
    'max_features': [5000, 10000, 20000]
}

def evaluate_params(params):
    """Parameter durch Trennschärfe bestimmen"""
    try:
        # Vokabular erstellen
        vocab = CountVectorizer(**params).fit(all_texts)
        if len(vocab.vocabulary_) < 100:
            return 0.0

        # TF-IDF mit diesem Vokabular
        vectorizer = TfidfVectorizer(vocabulary=vocab.vocabulary_, ngram_range=params['ngram_range'])
        vectorizer.fit(idf_texts)

        # Topic-Vektor
        topic_vecs = vectorizer.transform(topic_texts)
        topic_vector = normalize(np.asarray(topic_vecs.mean(axis=0)).reshape(1, -1))

        # Trennschärfe berechnen
        topic_sims = [cosine_similarity(vectorizer.transform([t]), topic_vector)[0, 0]
                      for t in topic_texts[:25]]
        idf_sims = [cosine_similarity(vectorizer.transform([t]), topic_vector)[0, 0]
                    for t in idf_texts[:25]]

        return np.mean(topic_sims) - np.mean(idf_sims)
    except:
        return 0.0

# Optimierung
results = []

for params in [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]:
    score = evaluate_params(params)
    results.append({**params, 'score': score})

# Ergebnisse
results.sort(key=lambda x: x['score'], reverse=True)
best = results[0]
print(f"ngram_range: {best['ngram_range']}")
print(f"min_df: {best['min_df']}")
print(f"max_df: {best['max_df']}")
print(f"max_features: {best['max_features']}")
print(f"separability_score: {best['score']:.4f}")