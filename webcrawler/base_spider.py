import scrapy
from scrapy.exceptions import CloseSpider
import heapq
from datetime import datetime, timedelta
import json
from urllib.parse import urlparse, unquote, urlunparse
from pathlib import Path
import configparser
import re
import nltk
from nltk.corpus import stopwords
import time
import pickle
import os
import psutil
import gc

class BaseTopicalSpider(scrapy.Spider):
    """Basisklasse für alle Strategien"""

    # Einfacher Duplikat-Eliminator, da eigener bereits vorhanden
    custom_settings = {
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.BaseDupeFilter',
        'REQUEST_FINGERPRINTER_IMPLEMENTATION': '2.7'
    }

    # Umwandlung in MiB
    MiB = 1024 * 1024

    @staticmethod
    def get_memory_mib(p: psutil.Process) -> float:
        """Ermittelt Speicherverbrauch in MiB, wobei USS > RSS"""
        try:
            info = p.memory_full_info()
            return (getattr(info, "uss", info.rss)) / BaseTopicalSpider.MiB
        except (psutil.Error, AttributeError, NotImplementedError):
            return p.memory_info().rss / BaseTopicalSpider.MiB

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Prozessinformationen
        self.process = psutil.Process()

        # Konfiguration laden
        project_root = Path(__file__).resolve().parent.parent
        self.config = configparser.ConfigParser()
        self.config.read(project_root / 'config_crawler.ini')

        # Verzeichnisse
        self.results_dir = project_root / self.config["PATHS"]["DATA_DIR"]
        self.backup_dir = project_root / self.config["PATHS"]["BACKUP_DIR"]

        # Crawler-Parameter
        self.batch_size = int(self.config['CRAWLER']['BATCH_SIZE'])
        self.max_pages = int(self.config['CRAWLER']['MAX_PAGES'])
        self.max_runtime = int(self.config['CRAWLER']['MAX_RUNTIME_MINUTES'])
        self.frontier_max_size = int(self.config['CRAWLER']['FRONTIER_MAX_SIZE'])

        # Domain und Namespace Filter
        self.allowed_domains = [d.strip() for d in self.config['CRAWLER']['ALLOWED_DOMAINS'].split(',')]
        self.ignored_namespaces = [ns.strip() for ns in self.config['CRAWLER']['IGNORED_NAMESPACES'].split(',')]

        # Gewichtungen
        self.link_weight = float(self.config['WEIGHTS']['ANCHOR_TEXT_WEIGHT'])
        self.parent_weight = float(self.config['WEIGHTS']['PARENT_DOCUMENT_WEIGHT'])
        self.title_weight = float(self.config['WEIGHTS']['TITLE_WEIGHT'])
        self.heading_weight = float(self.config['WEIGHTS']['HEADING_WEIGHT'])
        self.paragraph_weight = float(self.config['WEIGHTS']['PARAGRAPH_WEIGHT'])

        # NLTK Stoppwörter
        try:
            nltk.data.find('corpora/stopwords')
            self.stop_words = set(stopwords.words('german'))
        except LookupError:
            self.stop_words = set()

        # Seed URLs
        self.start_urls = [url.strip() for url in self.config['CRAWLER']['SEED_URLS'].split(',')]

        # Frontier Initialisierung
        self.frontier = []
        self.frontier_urls = set()
        self.visited_urls = set()

        # Tracking
        self.current_batch = 0
        self.visit_counter = 0
        self.final_report_done = False
        self.pending_requests = 0

        # Nur erfolgreich gecrawlte Seiten
        self.crawled_pages = []

        self.stats = {
            'total_crawled': 0,
            'start_time': datetime.now()
        }

        # Export vorbereiten
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.export_file = self.results_dir / f"{self.name}_{self.timestamp}.json"

    def normalize_url(self, url):
        """Normalisiert URL"""
        parsed = urlparse(url)

        # Hostname normalisieren
        hostname = parsed.netloc.lower()
        if ':' in hostname:
            hostname = hostname.split(':')[0]

        # Wikipedia Normalisierung
        if 'wikipedia.org' in hostname:
            hostname = re.sub(r'^(www\.|m\.|mobile\.)', '', hostname)

        # Schema normalisieren
        scheme = 'https'

        # Path beibehalten
        path = parsed.path.rstrip('/')
        if not path:
            path = '/'

        # Normalisierte URL ohne Query/Fragment
        normalized = urlunparse((
            scheme,
            hostname,
            path,
            '',  # params
            '',  # query
            ''   # fragment
        ))

        return normalized

    def is_valid_url(self, url):
        """Prüft ob URL gecrawlt werden soll"""
        parsed = urlparse(url)

        # Fragment oder Query direkt ablehnen
        if parsed.fragment or parsed.query:
            return False

        # Domain-Prüfung
        hostname = parsed.netloc.lower()
        if ':' in hostname:
            hostname = hostname.split(':')[0]

        domain_valid = False
        for domain in self.allowed_domains:
            if hostname == domain or hostname.endswith('.' + domain):
                domain_valid = True
                break

        if not domain_valid:
            return False

        # Namespace-Filter
        decoded_path = unquote(parsed.path)
        for namespace in self.ignored_namespaces:
            if f'/{namespace}' in decoded_path or f'/wiki/{namespace}' in decoded_path:
                return False

        return True

    def add_to_frontier(self, url, score):
        """Fügt URL zur Frontier hinzu"""
        # Erst validieren
        if not self.is_valid_url(url):
            return

        normalized = self.normalize_url(url)

        # Bereits besucht?
        if normalized in self.visited_urls:
            return

        # In Frontier einfügen wenn noch nicht drin
        if normalized not in self.frontier_urls:
            # Frontier-Größe prüfen
            if len(self.frontier) >= self.frontier_max_size:
                new_neg = -score
                # schlechtestes Element finden
                worst_idx, (worst_neg, worst_url) = max(
                    enumerate(self.frontier), key=lambda x: x[1][0]
                )
                if new_neg < worst_neg:
                    self.frontier_urls.discard(self.normalize_url(worst_url))
                    self.frontier[worst_idx] = (new_neg, url)
                    self.frontier_urls.add(normalized)
                    heapq.heapify(self.frontier)
            else:
                heapq.heappush(self.frontier, (-score, url))
                self.frontier_urls.add(normalized)

    def start_requests(self):
        """Initialisierung mit Seed-URLs"""
        for url in self.start_urls:
            self.add_to_frontier(url, 0.0)

        # Ersten Batch verarbeiten
        for request in self.process_batch():
            yield request

    def process_batch(self):
        """Verarbeitet nächsten Batch"""
        self.current_batch += 1

        # Beendigungskriterien prüfen
        if self.check_termination():
            if not self.final_report_done:
                self.final_report_done = True
                self.export_json()
            raise CloseSpider('Beendigungskriterium erreicht')

        # Batch aus Frontier holen
        batch = []
        for _ in range(min(self.batch_size, len(self.frontier))):
            if self.frontier:
                neg_score, url = heapq.heappop(self.frontier)
                normalized = self.normalize_url(url)

                # Aus Frontier entfernen
                self.frontier_urls.discard(normalized)
                batch.append((url, -neg_score, normalized))

        # Requests erstellen
        for url, score, normalized in batch:
            self.pending_requests += 1
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                errback=self.handle_error,
                meta={'frontier_score': score, 'normalized_url': normalized},
                dont_filter=True
            )

    def handle_error(self, failure):
        """Behandelt fehlgeschlagene Requests"""
        self.pending_requests -= 1

        # Nächster Batch, sobald alle Requests fertig
        if self.pending_requests == 0:
            for request in self.process_batch():
                yield request

    def parse(self, response):
        """Parser für gecrawlte Seiten"""
        self.pending_requests -= 1

        # Finale Duplikatsprüfung
        final_normalized = self.normalize_url(response.url)
        if final_normalized in self.visited_urls:
            if self.pending_requests == 0:
                for request in self.process_batch():
                    yield request
            return

        # Als besucht markieren
        self.visited_urls.add(final_normalized)
        self.visited_urls.add(response.meta['normalized_url'])

        # Bereinigte URL prüfen
        canonical = response.xpath('//link[@rel="canonical"]/@href').get()
        if canonical:
            canonical_norm = self.normalize_url(response.urljoin(canonical))
            if canonical_norm != final_normalized:
                if canonical_norm in self.visited_urls:
                    if self.pending_requests == 0:
                        for request in self.process_batch():
                            yield request
                    return
                self.visited_urls.add(canonical_norm)

        self.stats['total_crawled'] += 1
        self.visit_counter += 1

        # Parsing
        title = response.xpath('//title/text()').get('')
        headings = ' '.join(response.xpath(
            '//h1/text() | //h2/text() | //h3/text() | //h4/text() | //h5/text() | //h6/text()').getall())
        paragraphs = ' '.join(response.xpath('//p/text()').getall())

        # Relevanzwert holen
        frontier_score = response.meta.get('frontier_score', 0.0)

        # Messpunkt 1
        gc.collect()  # Garbage Collector
        memory_baseline_mib = self.get_memory_mib(self.process)  # Basisverbrauch
        time_start_ns = time.perf_counter_ns()  # Timer
        try:
            parent_relevance = self.calculate_parent_relevance(title, headings, paragraphs)
        finally:
            # Messpunkt 2
            eval_time_ns = time.perf_counter_ns() - time_start_ns  # Laufzeit (ns)

        # Abgeleitete Speichermetriken
        memory_after = self.get_memory_mib(self.process)  # Basis nach Bewertung
        uss_delta_mib = memory_after - memory_baseline_mib  # Nettoänderung (MiB)

        # Metriken für Analyse speichern
        self.crawled_pages.append({
            'url': response.url,
            'title': title[:100],
            'index': self.visit_counter,
            'score': frontier_score,
            'eval_time_ns': eval_time_ns,
            'memory_baseline_mib': memory_baseline_mib,
            'uss_delta_mib': uss_delta_mib,
        })

        # Links extrahieren und zur Frontier hinzufügen
        for link in response.xpath('//a[@href]'):
            href = link.xpath('@href').get()
            anchor_text = link.xpath('string()').get('').strip()
            url = response.urljoin(href)

            # Gesamte Relevanz berechnen
            anchor_score = self.calculate_text_relevance(anchor_text)
            link_relevance = self.link_weight * anchor_score + self.parent_weight * parent_relevance

            self.add_to_frontier(url, link_relevance)

        # Fortschritt ausgeben
        if self.stats['total_crawled'] % 50 == 0:
            self.print_progress()

        # Neuer Batch, wenn alle Requests fertig
        if self.pending_requests == 0:
            for request in self.process_batch():
                yield request

    def preprocess_text(self, text):
        """Textvorverarbeitung"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()

        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        else:
            tokens = [t for t in tokens if len(t) > 1]

        return ' '.join(tokens)

    def check_termination(self):
        """Prüft Beendigungskriterien"""
        runtime = datetime.now() - self.stats['start_time']

        if runtime > timedelta(minutes=self.max_runtime):
            return True
        if self.stats['total_crawled'] >= self.max_pages:
            return True
        if not self.frontier and self.pending_requests == 0:
            return True

        return False

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """Berechnet gewichtete Relevanz für Quelldokument"""
        title_score = self.calculate_text_relevance(title) if title else 0.0
        heading_score = self.calculate_text_relevance(headings) if headings else 0.0
        paragraph_score = self.calculate_text_relevance(paragraphs) if paragraphs else 0.0

        weighted_score = (
                self.title_weight * title_score +
                self.heading_weight * heading_score +
                self.paragraph_weight * paragraph_score
        )

        return min(1.0, weighted_score)

    def calculate_text_relevance(self, text):
        """Abstrakte Methode"""
        raise NotImplementedError("Subklassen müssen calculate_text_relevance implementieren")

    def print_progress(self):
        """Fortschritt"""
        print(f"[Batch {self.current_batch}] "
              f"Gecrawlt: {self.stats['total_crawled']} | "
              f"Frontier: {len(self.frontier)} URLs")

    def export_json(self):
        """JSON-Export"""
        runtime = datetime.now() - self.stats['start_time']

        # Nach Score sortieren
        sorted_pages = sorted(self.crawled_pages, key=lambda x: x['score'], reverse=True)

        # JSON exportieren
        export_data = {
            'summary': {
                'spider': self.name,
                'timestamp': self.timestamp,
                'total_execution_time_s': runtime.total_seconds(),
                'total_pages_visited': self.stats['total_crawled'],
                'batch_size_N': self.batch_size,
                'frontier_max_size': self.frontier_max_size,
                'pages': sorted_pages
            }
        }

        with open(self.export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    def load_or_train_model(self):
        """Lädt oder trainiert ML-Modell"""
        if hasattr(self, 'model_path') and hasattr(self, 'vectorizer_path'):
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                with open(self.model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print("Existierendes Modell geladen")
            else:
                with open(self.training_data_path, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
                texts, labels = self.select_training_labels(training_data)
                self.train_model(texts, labels)

    def select_training_labels(self, training_data):
        """Für ML-Ansätze"""
        raise NotImplementedError("ML-Ansätze müssen select_training_labels implementieren")

    def train_model(self, texts, labels):
        """Für ML-Ansätze"""
        raise NotImplementedError("ML-Ansätze müssen train_model implementieren")