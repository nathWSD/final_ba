# 🚀 RAG-Strategie-Evaluierungsrahmen
*(Retrieval-Augmented Generation Performance Evaluation )*

## Inhaltsverzeichnis
*   [Über das Projekt](#-über-das-projekt)
*   [Merkmale](#-merkmale)
*   [Architektur & Ablauf](#-architektur--ablauf)
*   [Technologien](#-technologien)
*   [Erste Schritte](#-erste-schritte)
    *   [Voraussetzungen](#voraussetzungen)
    *   [Installation](#installation)
    *   [Umgebungsvariablen](#umgebungsvariablen)
    *   [Datenbank- und Services-Setup](#datenbank--und-services-setup)
    *   [Experimente durchfuehren](#experimente-durchfuehren) 
*   [Ergebnisse & Analyse](#ergebnisse--analyse)
*   [Danksagungen](#danksagungen)
---

## 💡 Über das Projekt

Dieses Projekt ist ein umfassender Rahmen zur systematischen Evaluierung und zum Vergleich verschiedener Retrieval-Augmented Generation (RAG)-Strategien. Ziel ist es, die Leistung unterschiedlicher Abrufmethoden – sowohl **vektorbasiert** als auch **graphbasiert** – bei der Beantwortung von Fragen unter variierenden Schwierigkeitsgraden zu bewerten.

Das Framework automatisiert den gesamten Prozess von der Datenaufnahme, über den Kontextabruf und die LLM-gestützte Antwortgenerierung, bis hin zur detaillierten Metrikenanalyse. Die Ergebnisse liefern fundierte Einblicke in die Stärken und Schwächen jeder Strategie und helfen bei der Auswahl der optimalen RAG-Architektur für spezifische Anwendungsfälle.

## ✨ Merkmale

*   **Vielseitige Retrieval-Strategien:** Evaluation von 9 verschiedenen Suchverfahren, darunter:
    *   **Vektorbasiert:** Dichte Suche (`vector_dense_search`), Spärliche Suche (`vector_sparse_search`), Hybride Suche (`vector_hybrid_search`).
    *   **Graphbasiert:** Klassische Globale Suche (`graph_classical_global_search`), Klassische Lokale Suche (`graph_classical_local_search`), Klassische Drift-Suche (`graph_classical_drift_search`), LLM Globale Suche (`graph_llm_global_search`), LLM Lokale Suche (`graph_llm_local_search`), LLM Drift-Suche (`graph_llm_drift_search`).
*   **Automatisierte Datenaufnahme:** Ingestion von Dokumenten in Qdrant (für Vektor-Embeddings) und Neo4j (für Graph-Strukturen), mit Unterstützung für dichte, spärliche und hybride Einbettungen.
*   **LLM-gestützte Antwortgenerierung:** Nutzung von Large Language Models (Google Gemini) zur Beantwortung von Fragen basierend auf dem abgerufenen Kontext unter Einhaltung strikter Prompt-Anweisungen zur Kontexterdung.
*   **Umfassende Metrikevaluation:** Bewertung der generierten Antworten und Retrieval-Leistung mittels vielfältiger Metriken, darunter: Präzision, Recall, Relevanz, ROUGE-1, Kosinus-Ähnlichkeit, Zeitverbrauch sowie Eingabe- und Ausgabetoken-Anzahl.
*   **Schwierigkeitsgrad-Analyse:** Bewertung der Strategien über einen dreistufigen Datensatz von 54 Frage-Antwort-Paaren (Level 1, Level 2, Level 3), mit 18 Paaren pro Level.
*   **Strukturierte Ergebnisspeicherung:** Detaillierte Speicherung aller Evaluierungsergebnisse in JSON-Dateien **und in einer MySQL-Datenbank** für einfache Analyse und Visualisierung.
*   **Monitoring & Visualisierung:** Integration mit Grafana zur Überwachung und Visualisierung der Metrikverläufe und Ergebnisse.

## 📐 Architektur & Ablauf

Das Projekt durchläuft für jeden Testfall (Frage-Antwort-Paar) und jede Retrieval-Strategie einen definierten Ablauf:

1.  **Kontextabruf:** Die Frage wird an jede der neun Retrieval-Strategien gesendet, um den relevantesten Kontext aus den Datenbanken (Qdrant für Vektoren, Neo4j für Graphen) abzurufen.
2.  **Antwortgenerierung:** Der abgerufene Kontext und die ursprüngliche Frage werden an ein Large Language Model (LLM) übergeben, um eine kontextbasierte Antwort zu generieren. Hierbei wird ein speziell definierter Prompt verwendet, der das LLM anleitet, Antworten strikt basierend auf dem bereitgestellten Kontext zu formulieren.
3.  **Evaluierung:** Die generierte Antwort, der abgerufene Kontext, die Originalfrage und die erwartete Antwort werden einem Evaluator zugeführt, der die Performance anhand der umfassend definierten Metriken bewertet.
4.  **Ergebnisspeicherung:** Die detaillierten Metrikwerte und alle relevanten Informationen werden in strukturierten JSON-Dateien sowie in einer MySQL-Datenbank gespeichert, um eine umfassende Analyse und Weiterverarbeitung zu ermöglichen.

Für die **Vektor-Strategien** werden Textelemente als hochdimensionale Einbettungen in Qdrant gespeichert, wobei die Suche über Ähnlichkeitsmessungen (Kosinus-Ähnlichkeit) der Vektoren erfolgt. Für die **Graph-Strategien** werden Wissensgraphen in Neo4j erstellt, und der Kontext wird durch Traversierung oder spezielle Graphenalgorithmen ermittelt.

## 🛠️ Technologien

*   **Programmiersprache:** Python (3.11+)
*   **Containerisierung:** [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)
*   **Vektordatenbank:** [Qdrant](https://qdrant.tech/)
*   **Graphdatenbank:** [Neo4j](https://neo4j.com/)
*   **Relationale Datenbank:** [MySQL](https://www.mysql.com/)
*   **Monitoring & Visualisierung:** [Grafana](https://grafana.com/)
*   **Einbettungsmodelle:**
    *   [FastEmbed](https://qdrant.github.io/fastembed/latest/) (`TextEmbedding` für dichte, `SparseTextEmbedding` für spärliche Einbettungen)
    *   [SentenceTransformers](https://www.sbert.net/) (für dichte Modelle)
*   **LLM-Integration:** [LangChain](https://www.langchain.com/) (für RAG-Kette, Prompt-Management)
*   **Large Language Model:** Google Gemini API ( `gemini-1.5-pro`)

## 🚀 Erste Schritte

Um das Projekt lokal einzurichten und auszuführen, sollten die folgenden Komponenten installiert werden.

### Voraussetzungen
*   [Python 3.11+](https://www.python.org/downloads/)
*   [Docker Desktop](https://www.docker.com/products/docker-desktop/) (oder Docker Engine und Docker Compose)
*   Internetzugang (für das Herunterladen von Embedding-Modellen und den Zugriff auf die Google Gemini API)

### Installation

1.  **Repository klonen:**
    ```bash
    git clone https://github.com/nathWSD/final_ba.git
    cd final_ba
    ```
2.  **Virtuelle Umgebung erstellen und aktivieren:**
    ```bash
    python -m venv venv
    # Auf Windows:
    .\venv\Scripts\activate
    # Auf macOS/Linux:
    source venv/bin/activate
    ```
3.  **Abhängigkeiten installieren:**
    ```bash
    pip install poetry
    poetry install
    ```
### Umgebungsvariablen

Eine .env-Datei sollte im Stammverzeichnis des Projekts erstellt und mit den erforderlichen Umgebungsvariablen gefüllt werden. Die Platzhalter müssen durch die tatsächlichen Werte ersetzt werden.

**Beispiel `.env`:**

```dotenv
# Qdrant Konfiguration
QDRANT_URL=http://localhost:6333
QDRANT_TIMEOUT=20
QDRANT_PORT=6333 
QDRANT_CONTAINER_NAME=ba_qdrant_vector_db 

# Neo4j Konfiguration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password 

# MySQL Konfiguration
MYSQL_PORT=3306 
MYSQL_ROOT_PASSWORD=your_mysql_root_password 
MYSQL_DATABASE=rag_evaluation_db 
MYSQL_USER=rag_user 
MYSQL_PASSWORD=your_mysql_user_password 

# Grafana Konfiguration
GRAFANA_HOST_PORT=3000
GRAFANA_IMAGE_NAME=grafana/grafana-oss:latest 
GRAFANA_CONTAINER_NAME=grafana_service 
GRAFANA_ADMIN_USER=admin 
GRAFANA_ADMIN_PASSWORD=admin_password 

# Google API Key für Gemini
GOOGLE_API_KEY=your_gemini_api_key

# Embedding Modell Schlüssel (Modellnamen, die von FastEmbed/SentenceTransformers geladen werden)
DENSE_MODEL_KEY=BAAI/bge-small-en-v1.5 
SPARSE_MODEL_KEY=Qdrant/all-MiniLM-L6-v2 

# Qdrant Collection Namen 
DENSE_COLLECTION_NAME=dense_collection
SPARSE_COLLECTION_NAME=sparse_collection
HYBRID_COLLECTION_NAME=hybrid_collection
```

## Datenbank- und Services-Setup
Mit der bereitgestellten docker-compose.yml-Datei können alle benötigten Datenbanken (Neo4j, MySQL, Qdrant) sowie der Grafana-Monitoring-Dienst mit einem einzigen Befehl gestartet werden.
```bash
docker-compose up -d
```
Um alle gestarteten Container und ihre Netzwerke zu stoppen und zu entfernen (und die Volumendaten zu erhalten):
 ```bash
docker-compose down
```
## Experimente durchfuehren
Im main.py-Skript kann eine Liste von URLs angegeben werden, die gecrawlt werden. Diese dienen als Datenquelle für die gesamte Pipeline. Um die gesamte RAG-Evaluierungspipeline auszuführen: Ingestion, Kontextabruf, LLM-basierte Antwortgenerierung und Metrikenberechnung für alle definierten Retrieval-Strategien und Frage-Antwort-Paare.
 ```bash
cd final_ba
python main.py
```
## Ergebnisse & Analyse
Für interaktive Dashboards und zur Visualisierung des Metrikverlaufs können die in Grafana vorkonfigurierten Dashboards genutzt werden. Der Zugriff erfolgt über http://localhost:3000, wobei die Anmeldung mit den in der .env-Datei festgelegten Admin-Anmeldedaten durchgeführt wird.

## Danksagungen
Ein besonderer Dank gilt:
Meinen beiden Prüfern für ihre wertvolle Betreuung und Unterstützung während dieses Projekts.
Meinen Kollegen für ihre Unterstützung.
Google Gemini, für die Bereitstellung leistungsstarker Large Language Models, die einen integralen Bestandteil dieser Arbeit bilden.
Den Entwicklern und Communities der weiteren verwendeten Open-Source-Bibliotheken und Technologien, insbesondere Qdrant, Neo4j, MySQL, Grafana, FastEmbed, SentenceTransformers und LangChain.
