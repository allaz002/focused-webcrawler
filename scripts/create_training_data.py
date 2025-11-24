import requests
from bs4 import BeautifulSoup
import configparser
import os
import re
import json
from pathlib import Path
from datetime import datetime
import textwrap

class TrainingDataGenerator:
    """Erstellt Trainingsdaten aus Wikipedia-Artikeln"""

    def __init__(self, config_file='config_training.ini'):

        # Konfiguration laden
        project_root = Path(__file__).resolve().parent.parent
        self.config = configparser.ConfigParser(interpolation=None)
        self.config.optionxform = str
        self.config.read(project_root / config_file, encoding='utf-8')

        # Ausgabepfade
        self.output_nb = project_root / self.config['PATHS']['OUTPUT_NB']
        self.output_vsm = project_root / self.config['PATHS']['OUTPUT_VSM']
        self.backup_dir = project_root / self.config['PATHS']['BACKUP_DIR']

        # Request-Parameter
        self.user_agent = self.config['SETTINGS']['USER_AGENT']
        self.timeout = int(self.config['SETTINGS']['TIMEOUT'])
        self.min_text_length = int(self.config['SETTINGS']['MIN_TEXT_LENGTH'])

        # Naive Bayes Artikel
        nb_relevant_raw = self.config['NB_URLS']['RELEVANT_URLS'].split(',')
        nb_irrelevant_raw = self.config['NB_URLS']['IRRELEVANT_URLS'].split(',')
        self.nb_relevant_urls = sorted(list(set([url.strip() for url in nb_relevant_raw if url.strip()])))
        self.nb_irrelevant_urls = sorted(list(set([url.strip() for url in nb_irrelevant_raw if url.strip()])))

        # Vektorraummodell Artikel
        vsm_topic_raw = self.config['VSM_URLS']['TOPIC_VECTOR_URLS'].split(',')
        vsm_idf_raw = self.config['VSM_URLS']['IDF_URLS'].split(',')
        self.vsm_topic_urls = sorted(list(set([url.strip() for url in vsm_topic_raw if url.strip()])))
        self.vsm_idf_urls = sorted(list(set([url.strip() for url in vsm_idf_raw if url.strip()])))

        # Verzeichnisse erstellen
        Path(os.path.dirname(self.output_nb)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.output_vsm)).mkdir(parents=True, exist_ok=True)
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)

        # Request-Einstellungen
        self.headers = {'User-Agent': self.user_agent}
        self.url_cache = {}

    def extract_content(self, url):
        """Extrahiert Textinhalt aus Artikel"""
        if url in self.url_cache:
            return self.url_cache[url]

        try:
            # HTTP-Request
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Scripts und Styles entfernen
            for script in soup(['script', 'style']):
                script.decompose()

            # Titel extrahieren
            title = soup.find('title')
            title_text = title.text if title else ''

            # Überschriften sammeln
            headings = []
            for i in range(1, 7):
                for heading in soup.find_all(f'h{i}'):
                    headings.append(heading.get_text(strip=True))
            headings_text = ' '.join(headings)

            # Paragraphen sammeln
            paragraphs = []
            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 20:
                    paragraphs.append(text)
            paragraphs_text = ' '.join(paragraphs)

            # Anker-Texte sammeln
            anchors = []
            for a in soup.find_all('a'):
                anchor_text = a.get_text(strip=True)
                if anchor_text and len(anchor_text) > 2:
                    anchors.append(anchor_text)
            anchors_text = ' '.join(anchors[:50])

            # Text kombinieren
            combined_text = f"{title_text} {headings_text} {paragraphs_text} {anchors_text}"

            # Leerzeichen normalisieren
            combined_text = re.sub(r'\s+', ' ', combined_text)
            combined_text = combined_text.strip()

            # Cache aktualisieren
            self.url_cache[url] = combined_text
            return combined_text

        except Exception as e:

            self.url_cache[url] = None
            return None

    def backup_existing_files(self):
        """Erstellt Backups für existierende Dateien"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Naive-Bayes-Backup
        if os.path.exists(self.output_nb):
            backup_file = os.path.join(self.backup_dir, f"nb_backup_{timestamp}.json")
            os.rename(self.output_nb, backup_file)

        # Vektorraummodell-Backup
        if os.path.exists(self.output_vsm):
            backup_file = os.path.join(self.backup_dir, f"vsm_backup_{timestamp}.json")
            os.rename(self.output_vsm, backup_file)

    def process_naive_bayes_data(self):
        """Verarbeitet Naive Bayes Trainingsdaten"""
        nb_data = []

        # Irrelevante URLs verarbeiten
        for i, url in enumerate(self.nb_irrelevant_urls, 1):
            content = self.extract_content(url)

            if content and len(content) >= self.min_text_length:
                nb_data.append({
                    "label": 0,
                    "text": content
                })

        # Relevante URLs verarbeiten
        for i, url in enumerate(self.nb_relevant_urls, 1):
            content = self.extract_content(url)

            if content and len(content) >= self.min_text_length:
                nb_data.append({
                    "label": 1,
                    "text": content
                })

        # JSON exportieren
        with open(self.output_nb, 'w', encoding='utf-8') as f:
            json.dump(nb_data, f, ensure_ascii=False, indent=2)


    def process_vector_space_data(self):
        """Verarbeitet Vector Space Model Daten"""
        vsm_data = []

        # IDF URLs verarbeiten
        for i, url in enumerate(self.vsm_idf_urls, 1):
            content = self.extract_content(url)

            if content and len(content) >= self.min_text_length:
                vsm_data.append({
                    "label": "idf",
                    "text": content
                })

        # Topic-Vector URLs verarbeiten
        for i, url in enumerate(self.vsm_topic_urls, 1):
            content = self.extract_content(url)

            if content and len(content) >= self.min_text_length:
                vsm_data.append({
                    "label": "topic",
                    "text": content
                })

        # JSON exportieren
        with open(self.output_vsm, 'w', encoding='utf-8') as f:
            json.dump(vsm_data, f, ensure_ascii=False, indent=2)

def main():
    generator = TrainingDataGenerator()
    generator.backup_existing_files()
    generator.process_naive_bayes_data()
    generator.process_vector_space_data()

if __name__ == '__main__':
    main()