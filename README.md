💡 Über das Projekt
Dieses Projekt ist ein umfassender Rahmen zur systematischen Evaluierung und zum Vergleich verschiedener Retrieval-Augmented Generation (RAG)-Strategien. Ziel ist es, die Leistung unterschiedlicher Abrufmethoden – sowohl vektorbasiert als auch graphbasiert – bei der Beantwortung von Fragen unter variierenden Schwierigkeitsgraden zu bewerten.
Das Framework automatisiert den gesamten Prozess von der Datenaufnahme, über den Kontextabruf und die LLM-gestützte Antwortgenerierung, bis hin zur detaillierten Metrikenanalyse. Die Ergebnisse liefern Einblicke in die Stärken und Schwächen jeder Strategie und helfen bei der Auswahl der optimalen RAG-Architektur für spezifische Anwendungsfälle.
✨ Merkmale
Vielseitige Retrieval-Strategien: Evaluation von 9 verschiedenen Suchverfahren, darunter:
Vektorbasiert: Dichte Suche, Spärliche Suche, Hybride Suche.
Graphbasiert: Klassische Globale Suche, Klassische Lokale Suche, Klassische Drift-Suche, LLM Globale Suche, LLM Lokale Suche, LLM Drift-Suche.
Automatisierte Datenaufnahme: Ingestion von Dokumenten in Qdrant (für Vektor-Embeddings) und Neo4j (für Graph-Strukturen).
LLM-gestützte Antwortgenerierung: Nutzung von Large Language Models (z.B. Google Gemini) zur Beantwortung von Fragen basierend auf dem abgerufenen Kontext.
Umfassende Metrikevaluation: Bewertung der generierten Antworten und Retrieval-Leistung mittels vielfältiger Metriken:
Präzision (Precision)
Recall
Relevanz (Relevancy)
ROUGE-1
Kosinus-Ähnlichkeit (Cosine Similarity)
Zeitverbrauch (Time Taken)
Token-Anzahl (Input & Output Tokens)
Schwierigkeitsgrad-Analyse: Bewertung der Strategien über drei verschiedene Schwierigkeitsgrade von Frage-Antwort-Paaren (Level 1, Level 2, Level 3).
Strukturierte Ergebnisspeicherung: Speicherung aller Evaluierungsergebnisse in JSON-Dateien für einfache Analyse und Visualisierung.
Modulares Design: Einfache Erweiterung um neue Retrieval-Strategien, LLMs oder Metriken.
📐 Architektur & Strategien
Das Projekt durchläuft für jeden Testfall (Frage-Antwort-Paar) und jede Retrieval-Strategie einen definierten Ablauf:
Kontextabruf: Die Frage wird an die jeweilige Retrieval-Strategie (vektorbasiert oder graphbasiert) gesendet, um den relevantesten Kontext aus den Datenbanken (Qdrant, Neo4j) abzurufen.
Antwortgenerierung: Der abgerufene Kontext und die ursprüngliche Frage werden an ein Large Language Model (LLM) übergeben, um eine kontextbasierte Antwort zu generieren. Dabei werden strikte Prompts verwendet, um eine hohe Kontexterdung zu gewährleisten.
Evaluierung: Die generierte Antwort, der abgerufene Kontext, die Originalfrage und die erwartete Antwort werden einem Evaluator zugeführt, der die Performance anhand der definierten Metriken bewertet.
Ergebnisspeicherung: Die detaillierten Metrikwerte und alle relevanten Informationen werden in strukturierten JSON-Dateien gespeichert.
Für die Vektor-Strategien werden Textelemente als hochdimensionale Einbettungen in Qdrant gespeichert. Die Suche erfolgt über Ähnlichkeitsmessungen (Kosinus-Ähnlichkeit) der Vektoren.
Für die Graph-Strategien werden Wissensgraphen in Neo4j erstellt, wobei der Kontext durch Traversierung oder spezielle Graphenalgorithmen ermittelt wird.
🛠️ Technologien
Programmiersprache: Python (3.9+)
Vektordatenbank: Qdrant
Graphdatenbank: Neo4j
Einbettungsmodelle:
FastEmbed (TextEmbedding, SparseTextEmbedding)
SentenceTransformers
LLM-Integration: LangChain (für RAG-Kette, Prompts)
Large Language Model: Google Gemini API (z.B. gemini-1.5-pro)
Weitere Bibliotheken: numpy, pandas, tqdm (für Fortschrittsanzeigen), python-dotenv (für Umgebungsvariablen).
🚀 Erste Schritte
Befolgen Sie diese Anweisungen, um das Projekt lokal einzurichten und auszuführen.
Voraussetzungen
Stellen Sie sicher, dass Sie Folgendes installiert haben:
Python 3.9+
Docker Desktop (für Qdrant und Neo4j Datenbanken)
Internetzugang (für das Herunterladen von Embedding-Modellen und den Zugriff auf die Google Gemini API)
Installation
