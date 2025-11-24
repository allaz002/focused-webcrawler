# Analyse und Vergleich von Ansätzen zum themenbasierten Crawling im Web

## Entwicklungsumgebung
- Windows 11
- PyCharm
- Python 3.13.7

## Installation
1. In das Hauptverzeichnis wechseln:
```bash
cd "C:Users\...\bachelorarbeit_alexander_lazarev_projekt\pythonProject"
```

2. Virtuelle Umgebung erstellen:
```bash
python -m venv .venv
```

3. Virtuelle Umgebung aktivieren:
```bash
.\.venv\Scripts\Activate
```

4. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

## Skripte Ausführen

 **Crawler starten**
```bash
cd "C:Users\...\bachelorarbeit_alexander_lazarev_projekt\pythonProject\scripts"
python ./run_crawler.py [boolean|vectorspace|naivebayes]
```

**Trainingsdaten erstellen**
```bash
cd "C:Users\...\bachelorarbeit_alexander_lazarev_projekt\pythonProject\scripts"
python ./create_training_data.py
```

**Ground-Truth erstellen**
```bash
cd "C:Users\...\bachelorarbeit_alexander_lazarev_projekt\pythonProject\scripts"
python ./build_ground_truth.py
```
