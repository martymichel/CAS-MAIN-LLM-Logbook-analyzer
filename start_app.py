#!/usr/bin/env python3
"""
Verbessertes Startscript für Logbook Analyzer Pro
Optimiert für maximale Kompatibilität und Performance
"""

import os
import sys
import warnings
import subprocess
import platform

# Umfassende Warning-Unterdrückung
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Optimierte Umgebungsvariablen für bessere Performance
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_SHARE_MEMORY_FILE_SYSTEM'] = '0'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow Warnings unterdrücken
os.environ['PYTHONWARNINGS'] = 'ignore'

def check_dependencies():
    """Prüft kritische Dependencies"""
    critical_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'chardet'
    ]
    
    missing = []
    for package in critical_packages:
        try:
            __import__(package.replace('_', '-'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Fehlende Pakete: {', '.join(missing)}")
        print("💡 Installieren Sie mit: pip install -r requirements.txt")
        return False
    
    return True

def check_ollama_recommendation():
    """Gibt Ollama-Empfehlungen aus"""
    print("\n🦙 OLLAMA SETUP-EMPFEHLUNGEN:")
    print("1. Installieren: https://ollama.ai/download")
    print("2. Starten: ollama serve")
    print("3. Beste Qualität: ollama pull qwen2.5:14b")
    print("4. Alternative: ollama pull llama3.1:8b")
    print("5. Kompakt: ollama pull mistral:7b")

def get_system_info():
    """Zeigt System-Informationen"""
    print(f"\n💻 SYSTEM INFO:")
    print(f"- Python: {sys.version.split()[0]}")
    print(f"- Plattform: {platform.system()} {platform.release()}")
    print(f"- Architektur: {platform.machine()}")

def main():
    """Startet die verbesserte Streamlit App"""
    
    print("🚀 LOGBOOK ANALYZER PRO")
    print("=" * 50)
    
    # System-Info
    get_system_info()
    
    # Dependency-Check
    print("\n📦 Prüfe Dependencies...")
    if not check_dependencies():
        return
    
    print("✅ Alle Dependencies verfügbar")
    
    # Ollama-Empfehlungen
    check_ollama_recommendation()
    
    try:
        print("\n🌐 Starte Streamlit App...")
        
        # Optimierte Streamlit-Parameter für bessere Performance
        cmd = [
            sys.executable, "-m", "streamlit", "run", "logbook_analyzer.py",
            "--server.headless", "true",
            "--server.port", "8501",
            "--server.fileWatcherType", "none",  # Reduziert CPU-Last
            "--logger.level", "error",           # Minimale Logs
            "--server.enableCORS", "false",      # Sicherheit
            "--server.enableXsrfProtection", "false",  # Lokale Nutzung
            "--server.maxUploadSize", "200",     # 200MB Upload-Limit
            "--theme.primaryColor", "#1f77b4",   # Ansprechende Farben
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f0f2f6"
        ]
        
        print("🎯 App läuft auf: http://localhost:8501")
        print("📊 Features:")
        print("  - E5-Large Embedding-Modell (höchste Präzision)")
        print("  - Automatische CSV-Erkennung")
        print("  - Intelligente KI-Analyse")
        print("  - Enter = Suche absenden")
        print("🛑 Stoppen mit Ctrl+C\n")
        
        # App starten
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n✅ App erfolgreich gestoppt")
        
    except ImportError as e:
        print(f"\n❌ Import-Fehler: {e}")
        print("💡 Lösung:")
        print("   pip install -r requirements.txt")
        print("   pip install --upgrade sentence-transformers")
        
    except FileNotFoundError:
        print("\n❌ logbook_analyzer.py nicht gefunden!")
        print("💡 Stellen Sie sicher, dass alle Dateien im gleichen Ordner sind")
        
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        print("💡 Versuchen Sie:")
        print("   1. Python-Version prüfen (>= 3.8)")
        print("   2. Dependencies neu installieren")
        print("   3. Virtuelles Environment nutzen")

if __name__ == "__main__":
    main()