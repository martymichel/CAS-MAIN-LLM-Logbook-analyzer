---
output:
  pdf_document: default
  html_document: default
---
# 📋 Logbuch Analyzer

Eine intelligente Anwendung zur semantischen Suche und KI-gestützten Analyse von handschriftlichen Logbuch-Einträgen.

## 🚀 Features

- **🧠 Intelligente Ergebnisoptimierung**: Automatische Bestimmung der optimalen Anzahl Suchergebnisse
- **🤖 Integrierte KI-Analyse**: Vollautomatische LLM-Integration für detaillierte Auswertungen
- **⚡ Auto-Initialisierung**: Embedding-Modell und Ollama-Status werden beim Start automatisch geprüft
- **📊 Live System-Monitoring**: Terminal-Ausgaben und System-Status direkt in der App
- **🔍 Semantische Suche**: Findet relevante Einträge auch bei ungenauen Suchbegriffen
- **📈 Relevanz-basierte Anzeige**: Farbkodierte Ergebnisse nach Relevanz-Score
- **📊 Flexible CSV-Unterstützung**: Komma, Semikolon, Leerzeichen als Trennzeichen
- **🔒 100% Lokal**: Keine Cloud-Services, Ihre Daten bleiben bei Ihnen
- **🎯 Logbuch-optimiert**: Speziell für Struktur: Datum, Zeit, Lot-Nr., Subsystem, Ereignis & Maßnahme, Visum

---

## 📦 Installation

### Schritt 1: Repository klonen/downloaden

```bash
# Arbeitsverzeichnis erstellen
mkdir logbuch-analyzer
cd logbuch-analyzer

# Dateien herunterladen:
# - logbook_analyzer.py
# - start_app.py  
# - requirements.txt
```

### Schritt 2: Python Virtual Environment

```bash
# Virtual Environment erstellen
python -m venv logbook_env

# Aktivieren
# Windows:
logbook_env\Scripts\activate

# Linux/Mac:
source logbook_env/bin/activate
```

### Schritt 3: Python Dependencies installieren

```bash
# Requirements installieren
pip install -r requirements.txt

# Bei Problemen: pip upgrade
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🦙 Ollama Installation & Setup

### Schritt 1: Ollama herunterladen

**Website:** https://ollama.ai

**Direkte Downloads:**
- **Windows**: https://ollama.ai/download/windows
- **Mac**: https://ollama.ai/download/mac  
- **Linux**: 
  ```bash
  curl -fsSL https://ollama.ai/install.sh | sh
  ```

### Schritt 2: Ollama installieren

- **Windows**: `.exe` Datei ausführen und Installationsanweisungen folgen
- **Mac**: `.dmg` Datei öffnen und in Applications ziehen
- **Linux**: Automatische Installation über Curl-Befehl

### Schritt 3: Ollama Server starten

```bash
# Ollama Server starten (Terminal/CMD offen lassen)
ollama serve
```

**✅ Erfolg**: Sie sollten sehen: `Ollama server starting...`

### Schritt 4: Modell herunterladen

```bash
# Neues Terminal/CMD-Fenster öffnen
# Empfohlenes Modell (Deutsch-optimiert):
ollama pull llama3.1:8b

# Alternative kleinere Modelle (schneller, weniger RAM):
ollama pull llama3.2:3b
ollama pull mistral:7b

# Verfügbare Modelle anzeigen:
ollama list
```

**⏱️ Dauer**: 5-15 Minuten je nach Internetverbindung

### Schritt 5: Ollama testen

```bash
# Test-Chat
ollama run llama3.1:8b "Hallo, kannst du mir auf Deutsch antworten?"

# Erwartete Antwort: Deutsche Begrüßung
# Mit Ctrl+C beenden
```

---

## 🏃‍♂️ App starten

### Methode 1: Optimierter Starter (empfohlen)

```bash
# Stelle sicher, dass Virtual Environment aktiv ist
# Arbeitsverzeichnis: logbuch-analyzer/

python start_app.py
```

### Methode 2: Direkter Start

```bash
streamlit run logbook_analyzer.py --server.fileWatcherType none --logger.level error
```

### Methode 3: Einfacher Start

```bash
streamlit run logbook_analyzer.py
```

**🌐 App öffnet sich automatisch im Browser:** http://localhost:8501

---

## 📋 Erste Schritte

### 1. **Automatische Initialisierung**
- App startet sich selbst und lädt alle Komponenten
- System-Status wird in der Sidebar angezeigt
- Ollama-Verbindung wird automatisch getestet

### 2. **CSV-Datei hochladen**
- Hauptbereich → "CSV-Datei hochladen"
- Unterstützte Formate: `.csv` mit beliebigen Trennzeichen
- Automatische Erkennung von Encoding und Separatoren

**Erwartete Spalten:**
- Datum
- Zeit  
- Lot-Nr.
- Subsystem
- Ereignis & Maßnahme
- Visum

### 3. **Daten verarbeiten**
- "🔄 Daten verarbeiten" klicken
- Automatische Embedding-Erstellung
- System bereit für intelligente Suche

### 4. **Intelligente Suche nutzen**
- Natürlichsprachliche Anfragen stellen
- **Automatische Ergebnisoptimierung**: System bestimmt beste Anzahl Ergebnisse
- **Integrierte KI-Analyse**: Detaillierte Auswertung wird automatisch erstellt
- Beispiel-Queries:
  ```
  Probleme mit Lot 12345
  Temperaturprobleme in der Heizung
  Alle Einträge von System XY letzte Woche
  Wann gab es Ausfälle im Ventilsystem?
  ```

### 5. **System-Monitoring**
- Sidebar zeigt Live-Status aller Komponenten
- Terminal-Logs werden direkt in der App angezeigt
- Automatische Ollama-Statusprüfung

---

## 🔧 Fehlerbehebung

### ❌ "Ollama Server nicht erreichbar"

**Lösung:**
```bash
# 1. Prüfen ob Ollama läuft
curl http://localhost:11434/api/tags

# 2. Falls Fehler, Ollama neu starten
ollama serve

# 3. Modell prüfen
ollama list
```

### ❌ "CSV konnte nicht gelesen werden"

**Lösung:**
- Prüfen Sie Trennzeichen (Komma, Semikolon)
- Encoding ändern (UTF-8 → Latin1)
- Spaltenheader überprüfen

### ❌ "Embedding-Modell Fehler"

**Lösung:**
```bash
# Internet-Verbindung prüfen
pip install --upgrade sentence-transformers

# Bei RAM-Problemen kleineres Modell verwenden
# Modell-Name in Code ändern: 'all-MiniLM-L6-v2'
```

### ❌ Python/Pip Probleme

**Lösung:**
```bash
# Python Version prüfen (>=3.8)
python --version

# Pip upgraden
pip install --upgrade pip

# Virtual Environment neu erstellen
deactivate
rm -rf logbook_env
python -m venv logbook_env
```

---

## 💡 Tipps & Tricks

### **Ohne LLM verwenden**
- Sidebar → "LLM-Analyse verwenden" deaktivieren
- Semantische Suche funktioniert auch ohne Ollama

### **Performance optimieren**
- Kleineres Modell: `ollama pull llama3.2:3b`
- Weniger Suchergebnisse: Slider auf 5-10 setzen
- Starke Hardware: `llama3.1:70b` für beste Qualität

### **CSV-Format optimieren**
```csv
Datum,Zeit,Lot-Nr.,Subsystem,Ereignis & Massnahme,Visum
2024-01-15,14:30,LOT-001,Pumpe,Druckabfall erkannt - Ventil gereinigt,MK
2024-01-15,15:45,LOT-002,Heizung,Temperatur zu niedrig - Thermostat justiert,AB
```

### **Beispiel-Searches**
```
Natürlichsprachlich:
- "Zeige mir alle Probleme mit Lot 12345"
- "Wann gab es Temperaturprobleme?"
- "Alle Einträge von Benutzer MK"

Stichwort-basiert:
- "Druckabfall Pumpe"
- "Heizung Ausfall"
- "Qualität Problem"
```

---

## 📊 Systemanforderungen

### **Minimum:**
- Python 3.8+
- 4 GB RAM
- 2 GB freier Speicher
- Internet (initial für Downloads)

### **Empfohlen:**
- Python 3.10+
- 8 GB RAM  
- 5 GB freier Speicher
- GPU (optional, für bessere Performance)

---

## 🆘 Support

### **Häufige Probleme:**
1. **Ollama Status prüfen**: http://localhost:11434
2. **Logs ansehen**: Terminal-Ausgabe beachten
3. **Virtual Environment**: Immer aktiviert lassen
4. **Firewall**: Port 11434 freigeben

### **Bei weiteren Problemen:**
- Detaillierte Fehlermeldung kopieren
- Python Version und OS angeben
- Screenshots des Fehlers

---

## 📁 Dateistruktur

```
logbuch-analyzer/
├── logbook_analyzer.py    # Hauptanwendung
├── start_app.py          # Optimierter Starter
├── requirements.txt      # Python Dependencies
├── README.md            # Diese Anleitung
└── logbook_env/         # Virtual Environment (nach Setup)
```

---

## ⚖️ Lizenz & Datenschutz

- **100% lokal**: Keine Daten verlassen Ihren Computer
- **Open Source**: Frei verwendbar und anpassbar
- **Keine Telemetrie**: Keine Datensammlung

---

**🎉 Viel Erfolg mit Ihrem Logbuch Analyzer!**