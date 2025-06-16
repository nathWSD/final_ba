# 🚀 RAG-Strategie-Evaluierungsrahmen
*(Retrieval-Augmented Generation Performance Evaluation Framework)*

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/YOUR_USERNAME/YOUR_REPO_NAME/main/Build%20and%20Test?style=flat-square) <!-- Ersetzen Sie 'Build and Test' mit Ihrem tatsächlichen Workflow-Namen, falls abweichend -->
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)
![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=flat-square)
<!-- Fügen Sie hier weitere Badges hinzu, falls zutreffend (z.B. Code-Coverage, letzte Commit-Zeit) -->

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
    *   [Datenaufnahme (Ingestion)](#datenaufnahme-ingestion)
    *   [Experimente durchführen](#experimente-durchführen)
*   [Ergebnisse & Analyse](#-ergebnisse--analyse)
*   [Projektstruktur](#-projektstruktur)
*   [Lizenz](#-lizenz)
*   [Kontakt](#-kontakt)
*   [Danksagungen](#-danksagungen)

---

## 💡 Über das Projekt

Dieses Projekt ist ein umfassender Rahmen zur systematischen Evaluierung und zum Vergleich verschiedener Retrieval-Augmented Generation (RAG)-Strategien. Ziel ist es, die Leistung unterschiedlicher Abrufmethoden – sowohl **vektorbasiert** als auch **graphbasiert** – bei der Beantwortung von Fragen unter variierenden Schwierigkeitsgraden zu bewerten.

Das Framework automatisiert den gesamten Prozess von der Datenaufnahme, über den Kontextabruf und die LLM-gestützte Antwortgenerierung, bis hin zur detaillierten Metrikenanalyse. Die Ergebnisse liefern fundierte Einblicke in die Stärken und Schwächen jeder Strategie und helfen bei der Auswahl der optimalen RAG-Architektur für spezifische Anwendungsfälle.

## ✨ Merkmale

*   **Vielseitige Retrieval-Strategien:** Evaluation von 9 verschiedenen Suchverfahren, darunter:
    *   **Vektorbasiert:** Dichte Suche (`vector_dense_search`), Spärliche Suche (`vector_sparse_search`), Hybride Suche (`vector_hybrid_search`).
    *   **Graphbasiert:** Klassische Globale Suche (`graph_classical_global_search`), Klassische Lokale Suche (`graph_classical_local_search`), Klassische Drift-Suche (`graph_classical_drift_search`), LLM Globale Suche (`graph_llm_global_search`), LLM Lokale Suche (`graph_llm_local_search`), LLM Drift-Suche (`graph_llm_drift_search`).
*   **Automatisierte Datenaufnahme:** Ingestion von Dokumenten in Qdrant (für Vektor-Embeddings) und Neo4j (für Graph-Strukturen), mit Unterstützung für dichte, spärliche und hybride Einbettungen.
*   **LLM-gestützte Antwortgenerierung:** Nutzung von Large Language Models (z.B. Google Gemini) zur Beantwortung von Fragen basierend auf dem abgerufenen Kontext unter Einhaltung strikter Prompt-Anweisungen zur Kontexterdung.
*   **Umfassende Metrikevaluation:** Bewertung der generierten Antworten und Retrieval-Leistung mittels vielfältiger Metriken, darunter: Präzision, Recall, Relevanz, ROUGE-1, Kosinus-Ähnlichkeit, Zeitverbrauch sowie Eingabe- und Ausgabetoken-Anzahl.
*   **Schwierigkeitsgrad-Analyse:** Bewertung der Strategien über einen dreistufigen Datensatz von 54 Frage-Antwort-Paaren (Level 1, Level 2, Level 3), mit 18 Paaren pro Level.
*   **Strukturierte Ergebnisspeicherung:** Detaillierte Speicherung aller Evaluierungsergebnisse in JSON-Dateien **und in einer MySQL-Datenbank** für einfache Analyse und Visualisierung.
*   **Monitoring & Visualisierung:** Integration mit Grafana zur Echtzeit-Überwachung und Visualisierung der Metrikverläufe und Ergebnisse.
*   **Modulares Design:** Einfache Erweiterung um neue Retrieval-Strategien, LLMs oder Metriken.

## 📐 Architektur & Ablauf

Das Projekt durchläuft für jeden Testfall (Frage-Antwort-Paar) und jede Retrieval-Strategie einen definierten Ablauf, der in der Projektdokumentation schematisch dargestellt ist:

1.  **Kontextabruf:** Die Frage wird an jede der neun Retrieval-Strategien gesendet, um den relevantesten Kontext aus den Datenbanken (Qdrant für Vektoren, Neo4j für Graphen) abzurufen.
2.  **Antwortgenerierung:** Der abgerufene Kontext und die ursprüngliche Frage werden an ein Large Language Model (LLM) übergeben, um eine kontextbasierte Antwort zu generieren. Hierbei wird ein speziell definierter Prompt (wie in der Projektdokumentation beschrieben) verwendet, der das LLM anleitet, Antworten strikt basierend auf dem bereitgestellten Kontext zu formulieren.
3.  **Evaluierung:** Die generierte Antwort, der abgerufene Kontext, die Originalfrage und die erwartete Antwort werden einem Evaluator zugeführt, der die Performance anhand der umfassend definierten Metriken bewertet.
4.  **Ergebnisspeicherung:** Die detaillierten Metrikwerte und alle relevanten Informationen werden in strukturierten JSON-Dateien sowie in einer MySQL-Datenbank gespeichert, um eine umfassende Analyse und Weiterverarbeitung zu ermöglichen.

Für die **Vektor-Strategien** werden Textelemente als hochdimensionale Einbettungen in Qdrant gespeichert, wobei die Suche über Ähnlichkeitsmessungen (z.B. Kosinus-Ähnlichkeit) der Vektoren erfolgt. Für die **Graph-Strategien** werden Wissensgraphen in Neo4j erstellt, und der Kontext wird durch Traversierung oder spezielle Graphenalgorithmen ermittelt.

## 🛠️ Technologien

*   **Programmiersprache:** Python (3.11+)
*   **Containerisierung:** [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)
*   **Vektordatenbank:** [Qdrant](https://qdrant.tech/)
*   **Graphdatenbank:** [Neo4j](https://neo4j.com/)
*   **Relationale Datenbank:** [MySQL](https://www.mysql.com/)
*   **Monitoring & Visualisierung:** [Grafana](https://grafana.com/)
*   **Einbettungsmodelle:**
    *   [FastEmbed](https://qdrant.github.io/fastembed/latest/) (`TextEmbedding` für dichte, `SparseTextEmbedding` für spärliche Einbettungen)
    *   [SentenceTransformers](https://www.sbert.net/) (für bestimmte dichte Modelle)
*   **LLM-Integration:** [LangChain](https://www.langchain.com/) (für RAG-Kette, Prompt-Management)
*   **Large Language Model:** Google Gemini API (z.B. `gemini-1.5-pro`)
*   **Weitere Bibliotheken:** `numpy`, `pandas`, `tqdm` (für Fortschrittsanzeigen), `python-dotenv` (für Umgebungsvariablen), `loguru` (für Logging), `mysql-connector-python`, `neo4j-driver`.

## 🚀 Erste Schritte

Befolgen Sie diese Anweisungen, um das Projekt lokal einzurichten und auszuführen.

### Voraussetzungen
Stellen Sie sicher, dass Sie Folgendes installiert haben:
*   [Python 3.9+](https://www.python.org/downloads/)
*   [Docker Desktop](https://www.docker.com/products/docker-desktop/) (oder Docker Engine und Docker Compose)
*   Internetzugang (für das Herunterladen von Embedding-Modellen und den Zugriff auf die Google Gemini API)

### Installation

1.  **Repository klonen:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
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
    pip install -r requirements.txt
    ```
    *(Stellen Sie sicher, dass eine `requirements.txt`-Datei in Ihrem Stammverzeichnis vorhanden ist, die alle Abhängigkeiten auflistet. Sie können diese mit `pip freeze > requirements.txt` generieren, nachdem Sie alle benötigten Pakete installiert haben.)*

### Umgebungsvariablen

Erstellen Sie eine `.env`-Datei im Stammverzeichnis Ihres Projekts und füllen Sie diese mit den erforderlichen Umgebungsvariablen aus. **Ersetzen Sie die Platzhalter durch Ihre tatsächlichen Werte.**

**Beispiel `.env`:**

```dotenv
# Qdrant Konfiguration
QDRANT_URL=http://localhost:6333
QDRANT_TIMEOUT=20
QDRANT_PORT=6333 # Standard-Port, kann bei Bedarf geändert werden
QDRANT_CONTAINER_NAME=ba_qdrant_vector_db # Standard-Containername

# Neo4j Konfiguration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password # ERSETZEN SIE DIESES PASSWORT

# MySQL Konfiguration
MYSQL_PORT=3306 # Standard-Port, kann bei Bedarf geändert werden
MYSQL_ROOT_PASSWORD=your_mysql_root_password # ERSETZEN SIE DIESES PASSWORT
MYSQL_DATABASE=rag_evaluation_db # ERSETZEN SIE DIESEN DATENBANKNAMEN
MYSQL_USER=rag_user # ERSETZEN SIE DIESEN BENUTZERNAMEN
MYSQL_PASSWORD=your_mysql_user_password # ERSETZEN SIE DIESES PASSWORT

# Grafana Konfiguration
GRAFANA_HOST_PORT=3000 # Host-Port für Grafana UI
GRAFANA_IMAGE_NAME=grafana/grafana-oss:latest # Standard-Grafana Image
GRAFANA_CONTAINER_NAME=grafana_service # Standard-Containername
GRAFANA_ADMIN_USER=admin # Standard-Benutzer für Grafana Admin
GRAFANA_ADMIN_PASSWORD=admin_password # ERSETZEN SIE DIESES PASSWORT

# Google API Key für Gemini
GOOGLE_API_KEY=your_gemini_api_key # ERSETZEN SIE DIESEN API KEY

# Embedding Modell Schlüssel (Modellnamen, die von FastEmbed/SentenceTransformers geladen werden)
DENSE_MODEL_KEY=BAAI/bge-small-en-v1.5 # Beispiel: Ersetzen Sie dies durch Ihr tatsächlich verwendetes dichtes Modell
SPARSE_MODEL_KEY=Qdrant/all-MiniLM-L6-v2 # Beispiel: Ersetzen Sie dies durch Ihr tatsächlich verwendetes spärliches Modell

# Qdrant Collection Namen (Optional: Falls Sie die Standardnamen ändern möchten)
DENSE_COLLECTION_NAME=dense_collection
SPARSE_COLLECTION_NAME=sparse_collection
HYBRID_COLLECTION_NAME=hybrid_collection
