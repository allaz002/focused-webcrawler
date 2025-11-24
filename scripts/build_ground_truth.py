import re
import json
import time
import requests
import configparser
from pathlib import Path
from urllib.parse import urlparse, unquote
from collections import deque


class WikiGroundTruthBuilder:

    def __init__(self, config_file='config_ground_truth.ini'):

        # Konfiguration laden
        self.config = configparser.ConfigParser(interpolation=None)
        self.config.optionxform = str
        config_path = Path(config_file)

        if not config_path.is_absolute():
            config_path = Path(__file__).parent / config_file

        self.config.read(str(config_path), encoding='utf-8')

        # Topic-Parameter
        self.seed_categories = self._get_list('TOPIC', 'SEED_CATEGORIES')
        self.max_depth = self.config.getint('TOPIC', 'MAX_CATEGORY_DEPTH')
        self.exclude_patterns = [
            re.compile(p, re.I)
            for p in self._get_list('TOPIC', 'CATEGORY_EXCLUDE_REGEX')
        ]

        # Filter-Parameter
        self.min_articles_per_category = self.config.getint('TOPIC', 'MIN_ARTICLES_PER_CATEGORY', fallback=5)
        self.max_articles_per_category = self.config.getint('TOPIC', 'MAX_ARTICLES_PER_CATEGORY', fallback=500)
        self.exclude_category_keywords = self._get_list('TOPIC', 'EXCLUDE_CATEGORY_KEYWORDS')
        self.core_articles_whitelist = self._get_list('TOPIC', 'CORE_ARTICLES_WHITELIST')

        # API-Parameter
        self.endpoint = self.config.get('API', 'ENDPOINT')
        self.max_rps = self.config.getfloat('API', 'MAX_REQUESTS_PER_SEC')

        # Verzeichnisse
        self.root = config_path.parent
        self.exports_dir = self.root / self.config.get('IO', 'EXPORTS_DIR')
        self.cache_path = self.root / self.config.get('IO', 'CACHE_PATH')
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Build-Optionen
        self.force_rebuild = self.config.getboolean('OPTIONS', 'FORCE_REBUILD_TOPIC', fallback=False)
        self.update_topic = self.config.getboolean('OPTIONS', 'UPDATE_TOPIC', fallback=False)
        self.clear_cache = self.config.getboolean('OPTIONS', 'CLEAR_CACHE', fallback=False)

        # Cache und API initialisieren
        self.cache = {} if self.clear_cache else self._load_cache()
        self.api = WikiAPI(self.endpoint, self.max_rps, self.cache)
        self.ground_truth_path = self.root / 'exports' / 'ground_truth.json'
        self.ground_truth_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_list(self, section, key):
        """Konfigurationsliste parsen"""
        raw = self.config.get(section, key, fallback='').strip()
        raw = raw.replace('\n', ',').replace(';', ',')
        items = []
        for item in raw.split(','):
            item = item.strip()
            if item:
                items.append(item)
        return items

    def _load_cache(self):
        """Cache laden"""
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding='utf-8'))
            except:
                pass
        return {'categorymembers': {}, 'redirects': {}}

    def _save_cache(self):
        """Cache sichern"""
        # Duplikate entfernen
        for key, value in list(self.cache.get('categorymembers', {}).items()):
            if isinstance(value, list):
                self.cache['categorymembers'][key] = sorted(set(value))

        # Identische Redirects entfernen
        redirects = self.cache.get('redirects', {})
        self.cache['redirects'] = {k: v for k, v in redirects.items() if k != v}

        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding='utf-8'
        )

    def _is_category_relevant(self, category_name, article_count):
        """Prüft ob Kategorie relevant ist"""
        if any(pattern.search(category_name) for pattern in self.exclude_patterns):
            return False

        if article_count < self.min_articles_per_category:
            return False
        if article_count > self.max_articles_per_category:
            return False

        if self.exclude_category_keywords:
            category_lower = category_name.lower()
            if any(kw.lower() in category_lower for kw in self.exclude_category_keywords):
                return False

        return True

    def build_ground_truth(self):
        """Erstellt Ground Truth"""
        topic_titles = set()
        queue = deque()
        visited = set()

        # Zuerst White-List Artikel hinzufügen
        if self.core_articles_whitelist:
            for article_ref in self.core_articles_whitelist:
                norm_title = normalize_wiki_title(article_ref)
                if norm_title:
                    topic_titles.add(norm_title)

        # Seed-Kategorien zur Queue hinzufügen
        for seed in self.seed_categories:
            seed = seed.strip()
            if seed and not any(p.search(seed) for p in self.exclude_patterns):
                queue.append((seed, 0))

        if not queue:
            return topic_titles

        # Kategorienbaum durchsuchen
        while queue:
            category, depth = queue.popleft()
            if category in visited:
                continue
            visited.add(category)

            # Artikel der Kategorie holen
            articles = self.api.get_category_articles(category)
            article_count = len(articles)

            is_relevant = True
            rejection_reason = None

            if any(p.search(category) for p in self.exclude_patterns):
                is_relevant = False
                rejection_reason = 'excluded_pattern'
            elif article_count < self.min_articles_per_category:
                is_relevant = False
                rejection_reason = 'too_few_articles'
            elif article_count > self.max_articles_per_category:
                is_relevant = False
                rejection_reason = 'too_many_articles'
            elif self.exclude_category_keywords:
                category_lower = category.lower()
                if any(kw.lower() in category_lower for kw in self.exclude_category_keywords):
                    is_relevant = False
                    rejection_reason = 'excluded_keyword'

            # Artikel hinzufügen wenn relevant
            if is_relevant:
                mapping = resolve_redirects_batch(self.api, articles)
                resolved = set(mapping.values())
                topic_titles.update(resolved)

            # Unterkategorien in die Warteschlange hinzufügen
            if depth < self.max_depth:
                for subcat in self.api.get_subcategories(category):
                    if not any(p.search(subcat) for p in self.exclude_patterns):
                        queue.append((subcat, depth + 1))

        return topic_titles

    def build_and_export_ground_truth(self):
        """Baut Ground Truth + exportiert als JSON"""
        if self.force_rebuild or self.update_topic or not self.ground_truth_path.exists():
            ground_truth = self.build_ground_truth()

            # Merge mit vorhandener GT, falls nötig
            if self.update_topic and self.ground_truth_path.exists():
                previous = set(json.loads(self.ground_truth_path.read_text(encoding='utf-8')))
                ground_truth = previous | ground_truth

            # JSON exportieren
            self.ground_truth_path.write_text(
                json.dumps(sorted(ground_truth), ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            self._save_cache()
        else:
            ground_truth = set(json.loads(self.ground_truth_path.read_text(encoding='utf-8')))

        return ground_truth


class WikiAPI:
    """Wikipedia API Wrapper"""

    def __init__(self, endpoint, max_rps, cache):
        self.endpoint = endpoint
        self.min_interval = 1.0 / max_rps if max_rps > 0 else 0
        self.last_request = 0.0
        self.cache = cache

        # Session einstellen
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BA-Evaluation/1.0 (University Research)'
        })

    def _throttle(self):
        """Übertragungsbegrenzung"""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()

    def _request(self, params):
        """API-Request"""
        for attempt in range(5):
            try:
                self._throttle()
                response = self.session.get(self.endpoint, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(0.5 * (attempt + 1))

    def get_category_articles(self, category):
        """Artikel einer Kategorie laden"""
        cache_key = f"ARTS::{category}"
        if cache_key in self.cache.get('categorymembers', {}):
            return self.cache['categorymembers'][cache_key]

        articles = []
        continuation = None

        # API-Abfrage
        while True:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Kategorie:{category}',
                'cmnamespace': '0',
                'cmlimit': '500',
                'format': 'json'
            }
            if continuation:
                params['cmcontinue'] = continuation

            data = self._request(params)

            # Artikel extrahieren
            for member in data.get('query', {}).get('categorymembers', []):
                if member.get('ns') == 0 and 'title' in member:
                    articles.append(member['title'])

            continuation = data.get('continue', {}).get('cmcontinue')
            if not continuation:
                break

        # Cache aktualisieren
        self.cache.setdefault('categorymembers', {})[cache_key] = articles
        return articles

    def get_subcategories(self, category):
        subcategories = []
        continuation = None

        # API-Abfrage
        while True:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Kategorie:{category}',
                'cmnamespace': '14',
                'cmlimit': '500',
                'format': 'json'
            }
            if continuation:
                params['cmcontinue'] = continuation

            data = self._request(params)

            # Kategorien extrahieren
            for member in data.get('query', {}).get('categorymembers', []):
                if member.get('ns') == 14 and 'title' in member:
                    subcategories.append(member['title'].split(':', 1)[1])

            continuation = data.get('continue', {}).get('cmcontinue')
            if not continuation:
                break

        return subcategories


def normalize_wiki_title(input_string):
    """Titel normalisieren"""
    if not input_string:
        return None

    # URL zu Titel konvertieren
    if input_string.startswith('http'):
        parsed = urlparse(input_string)
        if parsed.netloc.lower() != 'de.wikipedia.org':
            return None
        if not parsed.path.startswith('/wiki/'):
            return None
        title = unquote(parsed.path[6:])
    else:
        title = input_string

    # Titel normalisieren
    title = title.replace('_', ' ').strip()
    if not title:
        return None

    return title[0].upper() + title[1:]


def resolve_redirects_batch(api, titles):
    """Redirects auflösen"""
    titles = [t for t in titles if t]
    if not titles:
        return {}

    mapping = {}
    chunk_size = 50

    # Batches verarbeiten
    for i in range(0, len(titles), chunk_size):
        chunk = titles[i:i + chunk_size]

        data = api._request({
            'action': 'query',
            'titles': '|'.join(chunk),
            'redirects': '1',
            'format': 'json'
        })

        # Normale Titel
        pages = data.get('query', {}).get('pages', {})
        for page_data in pages.values():
            title = page_data.get('title')
            if title:
                mapping.setdefault(title, title)

        for redirect in data.get('query', {}).get('redirects', []):
            mapping[redirect['from']] = redirect['to']

        for title in chunk:
            mapping.setdefault(title, title)

    # Cache aktualisieren
    api.cache.setdefault('redirects', {}).update(
        {k: v for k, v in mapping.items() if k != v}
    )

    return mapping


if __name__ == '__main__':
    builder = WikiGroundTruthBuilder('../config_ground_truth.ini')
    builder.build_and_export_ground_truth()