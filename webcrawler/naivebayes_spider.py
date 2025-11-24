from .base_spider import BaseTopicalSpider
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
import numpy as np
from pathlib import Path

class NaiveBayesSpider(BaseTopicalSpider):
    """Naive-Bayes-Ansatz"""

    name = 'naivebayes_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Pfade aus Config lesen
        project_root = Path(__file__).resolve().parent.parent
        self.model_path = project_root / self.config['NAIVEBAYES']['MODEL_PATH']
        self.vectorizer_path = project_root / self.config['NAIVEBAYES']['VECTORIZER_PATH']
        self.training_data_path = project_root / self.config['NAIVEBAYES']['TRAINING_DATA_PATH']
        self.alpha = float(self.config['NAIVEBAYES']['ALPHA'])
        self.load_or_train_model()

    def select_training_labels(self, training_data):
        """Labels laden"""
        texts = []
        labels = []

        for sample in training_data:
            processed_text = self.preprocess_text(sample['text'])
            if processed_text:
                texts.append(processed_text)
                labels.append(sample['label'])

        return texts, labels

    def train_model(self, texts, labels):
        """Trainiert Klassifikator"""
        if not texts:
            raise ValueError("Keine gültigen Trainingsdaten vorhanden!")

        # CountVectorizer einstellen
        vectorizer_config = self.config['NAIVEBAYES']
        self.vectorizer = CountVectorizer(
            max_features=int(vectorizer_config['MAX_FEATURES']),
            ngram_range=(int(vectorizer_config['NGRAM_MIN']),
                         int(vectorizer_config['NGRAM_MAX'])),
            min_df=int(vectorizer_config['MIN_DF']),
            max_df=float(vectorizer_config['MAX_DF'])
        )

        # Vektorisiere und trainiere
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        self.classifier = MultinomialNB(alpha=self.alpha)
        self.classifier.fit(X, y)

        # Speichere Modell
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        n_relevant = sum(1 for l in labels if l == 1)
        n_irrelevant = len(labels) - n_relevant
        print(f"Trainiert mit {len(texts)} Beispielen ({n_relevant} relevant, {n_irrelevant} irrelevant)")

    def calculate_text_relevance(self, text):
        """Berechnet Relevanz für Texte"""
        if not text:
            return 0.0

        processed_text = self.preprocess_text(text)
        if not processed_text:
            return 0.0

        try:
            text_vector = self.vectorizer.transform([processed_text])
            probabilities = self.classifier.predict_proba(text_vector)[0]
            return float(probabilities[1] if len(probabilities) > 1 else 0.0)
        except Exception:
            return 0.0

    def parse(self, response):
        yield from super().parse(response)