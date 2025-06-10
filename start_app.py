#!/usr/bin/env python3
"""
Startscript für Logbook Analyzer
Umgeht bekannte Torch/Streamlit Kompatibilitätsprobleme
"""

import os
import sys
import warnings

# Warnings unterdrücken
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Torch spezifische Umgebungsvariablen setzen
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_SHARE_MEMORY_FILE_SYSTEM'] = '0'

def main():
    """Startet die Streamlit App mit optimierten Einstellungen"""
    
    print("🚀 Starte Logbook Analyzer...")
    print("📋 Lade Dependencies...")
    
    try:
        # Imports in richtiger Reihenfolge
        import streamlit as st
        import subprocess
        
        # Streamlit App starten
        cmd = [
            sys.executable, "-m", "streamlit", "run", "logbook_analyzer.py",
            "--server.headless", "true",
            "--server.port", "8501",
            "--server.fileWatcherType", "none",  # Reduziert Watcher-Probleme
            "--logger.level", "error"  # Reduziert Log-Spam
        ]
        
        print("🌐 Öffne Browser: http://localhost:8501")
        print("⚠️  Ignoriere Torch-Warnings - sie beeinträchtigen nicht die Funktionalität")
        print("🛑 Stoppen mit Ctrl+C\n")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n✅ App gestoppt")
    except ImportError as e:
        print(f"❌ Import Fehler: {e}")
        print("💡 Bitte installieren Sie die Requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")

if __name__ == "__main__":
    main()