import json
import os
import sys
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Pfade
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINING_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training_data', 'nb_training.json')

# Daten laden
with open(TRAINING_DATA, 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
labels = [item['label'] for item in data]

# Pipeline und Parameter
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

param_grid = {
    'vectorizer__ngram_range': [(1,1), (1,2)],
    'vectorizer__min_df': [1, 2, 3],
    'vectorizer__max_df': [0.7, 0.8, 0.9],
    'vectorizer__max_features': [3000, 5000, 8000],
    'classifier__alpha': [0.5, 1.0, 2.0]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='f1_weighted',
    n_jobs=1,
    verbose=0
)

# Grid Search
grid_search.fit(texts, labels)

# Ergebnisse
best = grid_search.best_params_
print(f"ngram: {best['vectorizer__ngram_range']}")
print(f"min_df: {best['vectorizer__min_df']}")
print(f"max_df: {best['vectorizer__max_df']}")
print(f"max_features: {best['vectorizer__max_features']}")
print(f"alpha: {best['classifier__alpha']}")
print(f"f1_score: {grid_search.best_score_:.4f}")