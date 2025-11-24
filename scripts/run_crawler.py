"""
Hauptprogramm zum Starten der Crawler
"""

from pathlib import Path
import sys
import os

# Projektverzeichnis finden
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from webcrawler.boolean_spider import BooleanSpider
from webcrawler.vectorspace_spider import VectorSpaceSpider
from webcrawler.naivebayes_spider import NaiveBayesSpider

def print_usage():
    """Verwendung"""
    print("""Verwendung: python run_crawler.py [boolean|vectorspace|naivebayes]""")

def main():
    """Hauptfunktion"""
    
    # Argumente prüfen
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
        
    crawler = sys.argv[1].lower()
    
    # Crawler mappen
    crawler_map = {
        'boolean': BooleanSpider,
        'vectorspace': VectorSpaceSpider,
        'naivebayes': NaiveBayesSpider
    }
    
    if crawler not in crawler_map:
        print(f"Fehler: Unbekanntes Modell '{crawler}'")
        print_usage()
        sys.exit(1)
        
    # Konfigurationsdatei existiert?
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config_crawler.ini')

    if not os.path.exists(config_path):
        print("Fehler: config_crawler.ini nicht gefunden!")
        sys.exit(1)
        
    # Scrapy-Prozess initialisieren
    settings = get_project_settings()
    process = CrawlerProcess(settings)

    # Crawler starten
    spider_class = crawler_map[crawler]
    print(f"\nStarte {crawler.upper()} Crawler...\n")
    
    process.crawl(spider_class)
    process.start()
    
    print(f"\nBeende {crawler.upper()} Crawler...\n")


if __name__ == '__main__':
    main()