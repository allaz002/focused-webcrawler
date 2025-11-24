from .base_spider import BaseTopicalSpider
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

class VectorSpaceSpider(BaseTopicalSpider):
    """Vektorraum-Ansatz"""

    name = 'vectorspace_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Pfade aus Config lesen
        project_root = Path(__file__).resolve().parent.parent
        self.model_path = project_root / self.config['VECTORSPACE']['MODEL_PATH']
        self.vectorizer_path = project_root / self.config['VECTORSPACE']['VECTORIZER_PATH']
        self.training_data_path = project_root / self.config['VECTORSPACE']['TRAINING_DATA_PATH']

        self.load_or_train_model()

        # VectorSpace Themenvektor initialisieren
        if hasattr(self, 'classifier'):
            self.topic_vector = self.classifier

        print("VectorSpace Spider mit TF-IDF initialisiert")

    def select_training_labels(self, training_data):
        """Themenvektor und IDF-Training"""
        topic_vector_texts = []
        idf_texts = []

        for sample in training_data:
            processed_text = self.preprocess_text(sample['text'])
            if processed_text:
                if sample['label'] == "idf":
                    idf_texts.append(processed_text)
                elif sample['label'] == "topic":
                    topic_vector_texts.append(processed_text)

        return (topic_vector_texts, idf_texts), None

    def train_model(self, texts_tuple, labels):
        """Trainiert TF-IDF Vectorizer und erstellt Themenvektor"""
        topic_vector_texts, idf_texts = texts_tuple

        print(f"Trainingsdaten: {len(topic_vector_texts)} Themenvektor, "
              f"{len(idf_texts)} IDF")

        # Vokabular aus beiden Datensammlungen
        vectorizer_config = self.config['VECTORSPACE']
        vocab_builder = CountVectorizer(
            max_features=int(vectorizer_config.get('MAX_FEATURES', 1000)),
            ngram_range=(int(vectorizer_config['NGRAM_MIN']),
                         int(vectorizer_config['NGRAM_MAX'])),
            min_df=int(vectorizer_config['MIN_DF']),
            max_df=float(vectorizer_config['MAX_DF'])
        )

        # Lernt Vokabular aus beiden Datensammlungen
        all_texts = idf_texts + topic_vector_texts
        vocab_builder.fit(all_texts)
        print(f"Vokabular erstellt: {len(vocab_builder.vocabulary_)} Terme")

        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            vocabulary=vocab_builder.vocabulary_,
            ngram_range=(int(vectorizer_config['NGRAM_MIN']),
                         int(vectorizer_config['NGRAM_MAX'])),
            norm='l2'
        )

        # IDF-Werte aus dem Wikipedia-Korpus
        self.vectorizer.fit(idf_texts)
        print(f"IDF-Werte aus {len(idf_texts)} diversen Dokumenten berechnet")

        # Themenvektor aus astronomischen Artikeln
        vectors = self.vectorizer.transform(topic_vector_texts)

        # Berechne Mittelwert und normalisiere
        topic_vec = np.asarray(vectors.mean(axis=0)).reshape(1, -1)
        self.topic_vector = normalize(topic_vec, norm='l2', axis=1)

        # Speichere als classifier
        self.classifier = self.topic_vector

        # Speichere Modell
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        print(f"Topic-Vektor aus {len(topic_vector_texts)} relevanten Dokumenten erstellt")

    def calculate_text_relevance(self, text):
        """Berechnet Relevanz für Texte"""
        if not text:
            return 0.0

        processed_text = self.preprocess_text(text)
        if not processed_text:
            return 0.0

        try:
            # Transformiere Text
            text_vector = self.vectorizer.transform([processed_text])
            if hasattr(text_vector, "nnz") and text_vector.nnz == 0:
                return 0.0

            # Cosinus-Ähnlichkeit berechnen
            similarity = float(cosine_similarity(text_vector, self.topic_vector)[0, 0])
            return max(0.0, similarity)

        except Exception:
            return 0.0

    def parse(self, response):
        yield from super().parse(response)