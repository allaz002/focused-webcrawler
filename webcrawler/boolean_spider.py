from .base_spider import BaseTopicalSpider


class BooleanSpider(BaseTopicalSpider):
    """Boolescher-Ansatz"""

    name = 'boolean_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lade Keywords und konvertiere zu Set
        keywords_raw = [kw.strip().lower() for kw in
                        self.config['BOOLEAN']['KEYWORDS'].split(',')]
        self.keywords = set(kw for kw in keywords_raw if kw and ' ' not in kw)

        # Schwellwerte für Textfelder
        self.anchor_min = int(self.config['BOOLEAN'].get('ANCHOR_MIN_KEYWORDS', '1'))
        self.title_min = int(self.config['BOOLEAN'].get('TITLE_MIN_KEYWORDS', '1'))
        self.heading_min = int(self.config['BOOLEAN'].get('HEADING_MIN_KEYWORDS', '1'))
        self.paragraph_min = int(self.config['BOOLEAN'].get('PARAGRAPH_MIN_KEYWORDS', '3'))

        print(f"Boolean Spider: {len(self.keywords)} Keywords geladen")

    def calculate_text_relevance(self, text):
        """Berechnet Relevanz für Texte"""
        if not text:
            return 0.0

        processed_text = self.preprocess_text(text)
        if not processed_text:
            return 0.0

        # Schlagwortsuche
        text_words = set(processed_text.split())
        found_keywords = text_words.intersection(self.keywords)

        # Binäre Bewertung
        return 1.0 if len(found_keywords) >= self.anchor_min else 0.0

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """Gewichtete Relevanz des Elterndokuments"""
        # Bewertung für Textfelder
        title_score = 0.0
        if title:
            processed = self.preprocess_text(title)
            if processed:
                words = set(processed.split())
                keywords_found = len(words.intersection(self.keywords))
                title_score = 1.0 if keywords_found >= self.title_min else 0.0

        heading_score = 0.0
        if headings:
            processed = self.preprocess_text(headings)
            if processed:
                words = set(processed.split())
                keywords_found = len(words.intersection(self.keywords))
                heading_score = 1.0 if keywords_found >= self.heading_min else 0.0

        paragraph_score = 0.0
        if paragraphs:
            processed = self.preprocess_text(paragraphs)
            if processed:
                words = set(processed.split())
                keywords_found = len(words.intersection(self.keywords))
                paragraph_score = 1.0 if keywords_found >= self.paragraph_min else 0.0

        # Gewichtete Kombination
        weighted_score = (
                self.title_weight * title_score +
                self.heading_weight * heading_score +
                self.paragraph_weight * paragraph_score
        )

        return min(1.0, weighted_score)

    def parse(self, response):
        yield from super().parse(response)