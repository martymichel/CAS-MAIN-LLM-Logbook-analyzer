import streamlit as st
import pandas as pd
import numpy as np
import faiss
import requests
import json
import re
from datetime import datetime
import logging
from typing import List, Tuple, Optional
import pickle
import os
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import time
import chardet

# Torch Import Probleme vermeiden
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Custom Logger für Streamlit
class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
    
    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_entry}")
        # Nur die letzten 50 Logs behalten
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]

# Sentence Transformers Import nach anderen Imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Logging Setup mit Streamlit Integration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Streamlit Log Handler
if 'log_handler' not in st.session_state:
    st.session_state.log_handler = StreamlitLogHandler()
    st.session_state.log_handler.setFormatter(
        logging.Formatter('%(levelname)s - %(message)s')
    )
    logger.addHandler(st.session_state.log_handler)

class LogbookAnalyzer:
    def __init__(self):
        self.model = None
        self.index = None
        self.df = None
        self.embeddings = None
        # Optimiertes Modell: Balance zwischen Geschwindigkeit und Qualität
        self.model_name = 'intfloat/multilingual-e5-small'  # Schnell und trotzdem präzise
        self.ollama_status = "unbekannt"
        self.ollama_model = "deepseek-r1:latest"
        
    def auto_initialize(self):
        """Automatische Initialisierung beim App-Start"""
        try:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                st.error("❌ sentence-transformers nicht installiert. Bitte installieren Sie es mit: pip install sentence-transformers")
                logger.error("sentence-transformers nicht verfügbar")
                return False
            
            # Embedding-Modell laden
            with st.spinner('🤖 Lade Embedding-Modell (multilingual-e5-small)...'):
                success = self.load_embedding_model()
                if success:
                    st.success("✅ Embedding-Modell geladen")
                    logger.info("Embedding-Modell erfolgreich geladen")
                else:
                    st.error("❌ Embedding-Modell konnte nicht geladen werden")
                    st.info("💡 Versuchen Sie: pip install --upgrade sentence-transformers")
                    return False
            
            # Ollama Status prüfen - OHNE Spinner hier, da check_ollama_status() eigene Updates macht
            st.info("🦙 Prüfe Ollama-Verbindung...")
            ollama_available = self.check_ollama_status()
            
            if not ollama_available:
                st.error("❌ Ollama ist nicht verfügbar. Diese Anwendung benötigt eine funktionierende Ollama-Verbindung.")
                st.info("💡 Verwenden Sie die Auto-Reparatur in der Sidebar oder starten Sie Ollama manuell")
                return False
            else:
                st.success("✅ Ollama erfolgreich verbunden")
                
            return success
            
        except Exception as e:
            logger.error(f"Fehler bei Auto-Initialisierung: {e}")
            st.error(f"Fehler bei der Initialisierung: {e}")
            return False
        
    def load_embedding_model(self):
        """Lädt das Embedding-Modell mit Fallback-Strategien"""
        try:
            logger.info(f"Versuche Modell zu laden: {self.model_name}")
            
            # Hauptversuch mit E5-Small
            self.model = SentenceTransformer(
                self.model_name,
                device='cpu'
            )
            self.model.max_seq_length = 256
            logger.info(f"Embedding-Modell erfolgreich geladen: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden von {self.model_name}: {e}")
            
            # Fallback 1: Versuche andere E5-Varianten
            fallback_models = [
                'intfloat/multilingual-e5-base',
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'sentence-transformers/all-MiniLM-L6-v2'
            ]
            
            for fallback_model in fallback_models:
                try:
                    logger.info(f"Fallback-Versuch mit: {fallback_model}")
                    self.model = SentenceTransformer(fallback_model, device='cpu')
                    self.model_name = fallback_model
                    logger.info(f"Fallback erfolgreich: {fallback_model}")
                    st.warning(f"⚠️ Fallback-Modell verwendet: {fallback_model}")
                    return True
                except Exception as fallback_error:
                    logger.warning(f"Fallback {fallback_model} fehlgeschlagen: {fallback_error}")
                    continue
            
            # Alle Modelle fehlgeschlagen
            logger.error("Alle Embedding-Modelle fehlgeschlagen")
            return False
            
        except ImportError as import_error:
            logger.error(f"Import-Fehler bei sentence-transformers: {import_error}")
            st.error("❌ sentence-transformers nicht richtig installiert")
            st.code("pip install --upgrade sentence-transformers torch")
            return False
    
    def kill_existing_ollama_processes(self):
        """Beendet alle laufenden Ollama-Prozesse robust"""
        try:
            import subprocess
            import psutil
            import signal
            
            logger.info("Suche nach laufenden Ollama-Prozessen...")
            
            # Finde alle Ollama-Prozesse
            ollama_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    if proc_info['name'] and 'ollama' in proc_info['name'].lower():
                        ollama_processes.append(proc)
                        logger.info(f"Gefunden: Ollama-Prozess PID {proc_info['pid']}")
                    elif proc_info['cmdline']:
                        cmdline_str = ' '.join(proc_info['cmdline']).lower()
                        if 'ollama' in cmdline_str:
                            ollama_processes.append(proc)
                            logger.info(f"Gefunden: Ollama in Kommandozeile PID {proc_info['pid']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            if not ollama_processes:
                logger.info("Keine laufenden Ollama-Prozesse gefunden")
                return True
            
            # Versuche graceful shutdown
            logger.info(f"Beende {len(ollama_processes)} Ollama-Prozesse...")
            for proc in ollama_processes:
                try:
                    logger.info(f"Beende Prozess PID {proc.pid} (graceful)")
                    proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Warte auf graceful shutdown
            time.sleep(3)
            
            # Prüfe ob Prozesse noch laufen und force kill wenn nötig
            for proc in ollama_processes:
                try:
                    if proc.is_running():
                        logger.warning(f"Force kill Prozess PID {proc.pid}")
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Finale Wartezeit
            time.sleep(2)
            
            # Verifiziere dass alle Prozesse beendet sind
            remaining = []
            for proc in ollama_processes:
                try:
                    if proc.is_running():
                        remaining.append(proc.pid)
                except psutil.NoSuchProcess:
                    continue
            
            if remaining:
                logger.error(f"Konnte Prozesse nicht beenden: {remaining}")
                return False
            else:
                logger.info("Alle Ollama-Prozesse erfolgreich beendet")
                return True
                
        except ImportError:
            logger.error("psutil nicht verfügbar - installieren Sie: pip install psutil")
            return False
        except Exception as e:
            logger.error(f"Fehler beim Beenden der Ollama-Prozesse: {e}")
            return False

    def start_ollama_server(self):
        """Startet Ollama-Server robust mit Retry-Logik und Feedback"""
        try:
            import subprocess
            
            logger.info("Starte Ollama-Server...")
            
            # Starte Ollama im Hintergrund
            if os.name == 'nt':  # Windows
                process = subprocess.Popen(
                    ['ollama', 'serve'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:  # Unix/Linux/Mac
                process = subprocess.Popen(
                    ['ollama', 'serve'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
            
            logger.info(f"Ollama-Server gestartet mit PID {process.pid}")
            
            # Warte bis Server bereit ist (mit Feedback)
            max_wait_time = 20  # Reduziert auf 20 Sekunden
            wait_interval = 1   # Jede Sekunde prüfen
            
            for attempt in range(max_wait_time):
                # Prüfe ob Server erreichbar ist
                try:
                    response = requests.get("http://localhost:11434/", timeout=2)
                    if response.status_code == 200:
                        logger.info(f"Ollama-Server bereit nach {attempt + 1} Sekunden")
                        return True
                except requests.exceptions.RequestException:
                    # Zeige Progress nur alle 3 Sekunden um UI nicht zu spammen
                    if attempt % 3 == 0:
                        logger.debug(f"Server noch nicht bereit (Versuch {attempt + 1})")
                    time.sleep(wait_interval)
                    continue
            
            logger.error("Ollama-Server nicht rechtzeitig bereit geworden")
            return False
            
        except FileNotFoundError:
            logger.error("Ollama-Executable nicht gefunden - ist Ollama installiert?")
            st.error("❌ Ollama nicht gefunden - ist es installiert?")
            return False
        except Exception as e:
            logger.error(f"Fehler beim Starten des Ollama-Servers: {e}")
            st.error(f"❌ Server-Start Fehler: {e}")
            return False

    def check_ollama_status(self):
        """Robuste Ollama-Status-Prüfung mit automatischer Reparatur"""
        try:
            # Schritt 1: Prüfe grundsätzliche Erreichbarkeit
            test_urls = [
                "http://localhost:11434/api/tags",
                "http://127.0.0.1:11434/api/tags"
            ]
            
            server_reachable = False
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        server_reachable = True
                        models_data = response.json()
                        available_models = [model['name'] for model in models_data.get('models', [])]
                        logger.info(f"Ollama erreichbar mit {len(available_models)} Modellen")
                        break
                except requests.exceptions.RequestException as e:
                    logger.debug(f"URL {url} nicht erreichbar: {e}")
                    continue
            
            # Schritt 2: Wenn Server nicht erreichbar, versuche Reparatur
            if not server_reachable:
                logger.warning("Ollama-Server nicht erreichbar - starte Reparaturprozess")
                
                # 2a: Beende möglicherweise hängende Prozesse
                st.info("🔧 Bereinige hängende Ollama-Prozesse...")
                if self.kill_existing_ollama_processes():
                    st.success("✅ Ollama-Prozesse bereinigt")
                else:
                    st.warning("⚠️ Prozessbereinigung teilweise fehlgeschlagen")
                
                # 2b: Warte kurz
                time.sleep(3)
                
                # 2c: Starte Server neu
                st.info("🚀 Starte Ollama-Server neu...")
                if self.start_ollama_server():
                    st.success("✅ Ollama-Server gestartet")
                    server_reachable = True
                else:
                    st.error("❌ Ollama-Server-Start fehlgeschlagen")
                    self.ollama_status = "❌ Server konnte nicht gestartet werden"
                    return False
            
            # Schritt 3: Prüfe verfügbare Modelle
            if server_reachable:
                try:
                    response = requests.get("http://localhost:11434/api/tags", timeout=10)
                    models_data = response.json()
                    available_models = [model['name'] for model in models_data.get('models', [])]
                    
                    # Prüfe auf empfohlene Modelle
                    recommended_models = [
                        "deepseek-r1:latest",
                        "llama3.2:latest",
                        "llama3:latest",
                        "qwen2:latest"
                    ]
                    
                    best_available = None
                    for rec_model in recommended_models:
                        if rec_model in available_models:
                            best_available = rec_model
                            break
                    
                    if not best_available and available_models:
                        # Verwende das erste verfügbare Modell als Fallback
                        best_available = available_models[0]
                        logger.warning(f"Kein empfohlenes Modell gefunden, verwende: {best_available}")
                    
                    if best_available:
                        self.ollama_model = best_available
                        
                        # Schritt 4: Teste Modell-Funktionalität mit erweiterten Versuchen
                        st.info(f"🧪 Teste Modell {best_available}...")
                        test_success = False
                        
                        for attempt in range(3):  # 3 Versuche
                            try:
                                test_success = self.test_ollama_model()
                                if test_success:
                                    break
                                else:
                                    logger.warning(f"Modell-Test Versuch {attempt + 1} fehlgeschlagen")
                                    time.sleep(2)  # Kurze Pause zwischen Versuchen
                            except Exception as e:
                                logger.warning(f"Modell-Test Versuch {attempt + 1} Fehler: {e}")
                                time.sleep(2)
                        
                        if test_success:
                            self.ollama_status = f"✅ Aktiv mit {best_available}"
                            st.success(f"🦙 Ollama: {self.ollama_status}")
                            logger.info(f"Ollama vollständig funktionsfähig mit {best_available}")
                            return True
                        else:
                            self.ollama_status = f"⚠️ Modell {best_available} antwortet nicht korrekt"
                            st.error(f"🦙 Ollama: {self.ollama_status}")
                            st.info("💡 Versuchen Sie: `ollama pull deepseek-r1:latest`")
                            return False
                    else:
                        self.ollama_status = "❌ Keine Modelle verfügbar"
                        st.error(f"🦙 Ollama: {self.ollama_status}")
                        st.info("💡 Installieren Sie ein Modell: `ollama pull deepseek-r1:latest`")
                        return False
                        
                except Exception as e:
                    logger.error(f"Fehler beim Abrufen der Modelle: {e}")
                    self.ollama_status = f"❌ API-Fehler: {str(e)[:50]}"
                    st.error(f"🦙 Ollama: {self.ollama_status}")
                    return False
            
            return False
            
        except Exception as e:
            self.ollama_status = f"❌ Kritischer Fehler: {str(e)[:50]}"
            st.error(f"🦙 Ollama: {self.ollama_status}")
            logger.error(f"Kritischer Fehler bei Ollama-Status-Check: {e}")
            return False
    
    def test_ollama_model(self) -> bool:
        """Testet das Ollama-Modell mit robusten Versuchen"""
        try:
            # Mehrere einfache Test-Prompts für bessere Zuverlässigkeit
            test_prompts = [
                "Antworte nur mit 'OK'",
                "Sage nur 'TEST'",
                "Gib nur 'JA' aus"
            ]
            
            for i, test_prompt in enumerate(test_prompts, 1):
                try:
                    logger.info(f"Modell-Test {i}/{len(test_prompts)}: '{test_prompt}'")
                    
                    response = self.query_ollama(
                        test_prompt, 
                        "", 
                        timeout=20  # Längerer Timeout für Tests
                    )
                    
                    if response:
                        response_clean = response.strip().upper()
                        expected_responses = ['OK', 'TEST', 'JA']
                        
                        # Prüfe ob eine erwartete Antwort enthalten ist
                        for expected in expected_responses:
                            if expected in response_clean:
                                logger.info(f"Modell-Test erfolgreich: '{response[:50]}'")
                                return True
                        
                        # Auch akzeptieren wenn Antwort vernünftig ist (nicht leer/Error)
                        if len(response) > 1 and "error" not in response.lower():
                            logger.info(f"Modell antwortet vernünftig: '{response[:50]}'")
                            return True
                    
                    logger.warning(f"Test {i} unzureichende Antwort: '{response[:100] if response else 'Keine Antwort'}'")
                    
                except Exception as test_error:
                    logger.warning(f"Test {i} Fehler: {test_error}")
                    continue
            
            logger.error("Alle Modell-Tests fehlgeschlagen")
            return False
            
        except Exception as e:
            logger.error(f"Kritischer Fehler beim Modell-Test: {e}")
            return False

    def detect_encoding_and_separator(self, uploaded_file) -> Tuple[Optional[str], Optional[str]]:
        """Robuste Encoding- und Separator-Erkennung"""
        try:
            # Datei-Inhalt als Bytes lesen
            uploaded_file.seek(0)
            raw_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            
            # Encoding automatisch erkennen mit chardet
            detected = chardet.detect(raw_bytes)
            primary_encoding = detected.get('encoding', 'utf-8') if detected else 'utf-8'
            confidence = detected.get('confidence', 0) if detected else 0
            
            logger.info(f"Erkanntes Encoding: {primary_encoding} (Konfidenz: {confidence:.2f})")
            
            # Fallback-Encodings definieren
            encodings_to_try = [primary_encoding, 'utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
            # Duplikate entfernen, Reihenfolge beibehalten
            encodings_to_try = list(dict.fromkeys(encodings_to_try))
            
            # Separatoren mit Priorität
            separators = [',', ';', '\t', '|', ' ']
            
            best_result = None
            max_columns = 0
            
            for encoding in encodings_to_try:
                try:
                    # Text dekodieren
                    text_content = raw_bytes.decode(encoding)
                    
                    # Erste Zeilen analysieren für Separator-Erkennung
                    lines = text_content.split('\n')[:10]  # Mehr Zeilen für bessere Analyse
                    
                    for separator in separators:
                        # Teste jeden Separator
                        try:
                            from io import StringIO
                            test_io = StringIO(text_content)
                            test_df = pd.read_csv(
                                test_io, 
                                sep=separator, 
                                encoding=None,  # Bereits dekodiert
                                on_bad_lines='skip',
                                nrows=50  # Teste nur erste 50 Zeilen
                            )
                            
                            # Bewerte Qualität der Erkennung
                            num_cols = len(test_df.columns)
                            num_rows = len(test_df)
                            
                            # Qualitätskriterien
                            if (num_cols >= 3 and  # Mindestens 3 Spalten
                                num_rows >= 1 and  # Mindestens 1 Datenzeile
                                num_cols > max_columns):  # Mehr Spalten = besser
                                
                                max_columns = num_cols
                                best_result = (encoding, separator, test_df)
                                logger.info(f"Bessere Konfiguration gefunden: {encoding}/{separator} ({num_cols} Spalten)")
                        
                        except Exception as sep_error:
                            logger.debug(f"Separator {separator} mit {encoding} fehlgeschlagen: {sep_error}")
                            continue
                
                except UnicodeDecodeError as enc_error:
                    logger.debug(f"Encoding {encoding} fehlgeschlagen: {enc_error}")
                    continue
            
            if best_result:
                encoding, separator, df = best_result
                logger.info(f"Finale Auswahl: {encoding}/{separator} mit {len(df.columns)} Spalten")
                return encoding, separator
            else:
                logger.error("Keine gültige Encoding/Separator-Kombination gefunden")
                return None, None
                
        except Exception as e:
            logger.error(f"Kritischer Fehler bei Encoding/Separator-Erkennung: {e}")
            return None, None

    def load_csv_robust(self, uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
        """Vollständig überarbeitete robuste CSV-Ladung"""
        try:
            # Automatische Erkennung
            encoding, separator = self.detect_encoding_and_separator(uploaded_file)
            
            if not encoding or not separator:
                logger.error("Automatische Erkennung fehlgeschlagen")
                return None, None, None
            
            # Finale CSV-Ladung mit erkannten Parametern
            uploaded_file.seek(0)
            
            df = pd.read_csv(
                uploaded_file,
                sep=separator,
                encoding=encoding,
                on_bad_lines='skip',
                skipinitialspace=True,  # Entferne führende Leerzeichen
                dtype=str  # Alles als String laden für konsistente Verarbeitung
            )
            
            # Validierung
            if len(df.columns) < 2:
                logger.error(f"Zu wenige Spalten erkannt: {len(df.columns)}")
                return None, None, None
            
            if len(df) == 0:
                logger.error("Keine Datenzeilen gefunden")
                return None, None, None
            
            logger.info(f"CSV erfolgreich geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
            logger.info(f"Spalten: {list(df.columns)}")
            
            return df, encoding, separator
            
        except Exception as e:
            logger.error(f"Kritischer Fehler beim CSV-Laden: {e}")
            return None, None, None
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verbesserte Datenbereinigung und -validierung"""
        try:
            # Spalten-Mapping mit erweiterten Varianten
            expected_columns = ['Datum', 'Zeit', 'Lot-Nr.', 'Subsystem', 'Ereignis & Massnahme', 'Visum']
            
            df_columns = [col.strip() for col in df.columns.tolist()]  # Leerzeichen entfernen
            column_mapping = {}
            
            # Erweiterte Varianten für bessere Erkennung
            column_variants = {
                'Datum': ['datum', 'date', 'time-stamp', 'zeitstempel', 'tag', 'day'],
                'Zeit': ['zeit', 'time', 'uhrzeit', 'hour', 'minute'],
                'Lot-Nr.': ['lot', 'lotnummer', 'lot-nr', 'lot-nummer', 'charge', 'batch', 'serie'],
                'Subsystem': ['subsystem', 'system', 'bereich', 'modul', 'komponente', 'teil'],
                'Ereignis & Massnahme': ['ereignis', 'eintrag', 'massnahme', 'entry', 'beschreibung', 'inhalt', 'problem', 'bemerkung', 'kommentar'],
                'Visum': ['visum', 'signature', 'unterschrift', 'benutzer', 'user', 'name', 'person']
            }
            
            # Intelligente Spaltenzuordnung
            for expected in expected_columns:
                matched = False
                
                # Exakte Übereinstimmung (Case-insensitive)
                for col in df_columns:
                    if expected.lower() == col.lower():
                        column_mapping[expected] = col
                        matched = True
                        break
                
                if not matched:
                    # Fuzzy Matching mit Varianten
                    variants = column_variants.get(expected, [expected.lower()])
                    
                    for col in df_columns:
                        col_clean = col.lower().replace('-', '').replace('_', '').replace(' ', '')
                        
                        for variant in variants:
                            variant_clean = variant.replace('-', '').replace('_', '').replace(' ', '')
                            
                            if variant_clean in col_clean or col_clean in variant_clean:
                                column_mapping[expected] = col
                                matched = True
                                break
                        
                        if matched:
                            break
                
                # Falls immer noch nicht gefunden, verwende beste Ähnlichkeit
                if not matched:
                    best_match = None
                    best_score = 0
                    
                    for col in df_columns:
                        # Einfache Ähnlichkeitsberechnung
                        common_chars = set(expected.lower()) & set(col.lower())
                        score = len(common_chars) / max(len(expected), len(col))
                        
                        if score > best_score and score > 0.3:  # Mindest-Ähnlichkeit
                            best_score = score
                            best_match = col
                    
                    if best_match:
                        column_mapping[expected] = best_match
                        logger.info(f"Ähnlichkeits-Zuordnung: '{expected}' -> '{best_match}' (Score: {best_score:.2f})")
            
            # Prüfe fehlende Spalten
            missing_cols = [col for col in expected_columns if col not in column_mapping]
            if missing_cols:
                logger.warning(f"Fehlende Spalten werden mit Platzhaltern gefüllt: {missing_cols}")
                for col in missing_cols:
                    # Erstelle leere Spalte
                    df[col] = 'N/A'
                    column_mapping[col] = col
            
            # Erstelle neuen DataFrame mit korrekter Struktur
            processed_df = pd.DataFrame()
            for expected, actual in column_mapping.items():
                if actual in df.columns:
                    processed_df[expected] = df[actual].astype(str).fillna('').str.strip()
                else:
                    processed_df[expected] = 'N/A'
            
            # Verbesserte Datum/Zeit-Verarbeitung
            if 'Datum' in processed_df.columns:
                try:
                    # Versuche verschiedene Datumsformate
                    date_formats = [
                        '%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%m/%d/%Y',
                        '%Y-%m-%d %H:%M:%S', '%d.%m.%Y %H:%M:%S'
                    ]
                    
                    parsed_dates = None
                    for fmt in date_formats:
                        try:
                            parsed_dates = pd.to_datetime(processed_df['Datum'], format=fmt, errors='coerce')
                            if parsed_dates.notna().sum() > len(parsed_dates) * 0.8:  # 80% erfolgreich
                                break
                        except:
                            continue
                    
                    if parsed_dates is None:
                        # Fallback: automatische Erkennung
                        parsed_dates = pd.to_datetime(processed_df['Datum'], errors='coerce')
                    
                    processed_df['Datum_Parsed'] = parsed_dates
                    
                except Exception as e:
                    logger.warning(f"Datum-Parsing fehlgeschlagen: {e}")
            
            # Kombiniere Datum und Zeit falls vorhanden
            if 'Datum' in processed_df.columns and 'Zeit' in processed_df.columns:
                try:
                    datetime_combined = pd.to_datetime(
                        processed_df['Datum'].astype(str) + ' ' + processed_df['Zeit'].astype(str),
                        errors='coerce'
                    )
                    processed_df['DateTime_Combined'] = datetime_combined
                except Exception as e:
                    logger.warning(f"Datum/Zeit-Kombination fehlgeschlagen: {e}")
            
            # Bereinige leere Ereignis-Einträge (Hauptinhalt)
            if 'Ereignis & Massnahme' in processed_df.columns:
                before_count = len(processed_df)
                processed_df = processed_df[
                    (processed_df['Ereignis & Massnahme'].str.strip() != '') & 
                    (processed_df['Ereignis & Massnahme'] != 'N/A')
                ]
                after_count = len(processed_df)
                
                if before_count != after_count:
                    logger.info(f"Leere Ereignis-Einträge entfernt: {before_count - after_count}")
            
            # Finale Validierung
            if len(processed_df) == 0:
                logger.error("Keine gültigen Datenzeilen nach Bereinigung")
                return None
            
            logger.info(f"Datenverarbeitung abgeschlossen: {len(processed_df)} Einträge")
            logger.info(f"Finale Spalten: {list(processed_df.columns)}")
            
            return processed_df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Kritischer Fehler bei der Datenverarbeitung: {e}")
            return None
    
    def create_embeddings(self, df: pd.DataFrame) -> bool:
        """Optimierte Embedding-Erstellung für Geschwindigkeit und Qualität"""
        try:
            with st.spinner(''):
                # Kompakte aber effektive Textkombination
                texts = []
                for _, row in df.iterrows():
                    # Fokus auf wichtigste Informationen für Geschwindigkeit
                    parts = []
                    
                    # Nur die essentiellen Felder für schnelle Verarbeitung
                    if 'Lot-Nr.' in row and row['Lot-Nr.'] != 'N/A':
                        parts.append(f"Lot: {row['Lot-Nr.']}")
                    
                    if 'Subsystem' in row and row['Subsystem'] != 'N/A':
                        parts.append(f"System: {row['Subsystem']}")
                    
                    # Hauptinhalt (wichtigster Teil)
                    if 'Ereignis & Massnahme' in row and row['Ereignis & Massnahme'] != 'N/A':
                        ereignis = str(row['Ereignis & Massnahme'])[:200]  # Begrenzt für Geschwindigkeit
                        parts.append(f"Ereignis: {ereignis}")
                    
                    # Zeitinfo nur wenn verfügbar (optional für Geschwindigkeit)
                    if 'Datum' in row and pd.notna(row['Datum']) and row['Datum'] != 'N/A':
                        parts.append(f"Datum: {row['Datum']}")
                    
                    # Kompakte Kombination
                    combined_text = " | ".join(parts)
                    texts.append(combined_text)
                
                # Optimierte Embedding-Parameter für Geschwindigkeit
                self.embeddings = self.model.encode(
                    texts,
                    batch_size=32,  # Größere Batches für bessere Performance
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    # Keine speziellen Prompts für Geschwindigkeit
                )
                
                # Standard FAISS-Konfiguration
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(self.embeddings.astype('float32'))
                
                self.df = df
                
                logger.info(f"Schnelle Embeddings erstellt:")
                logger.info(f"  - Anzahl: {len(texts)}")
                logger.info(f"  - Dimension: {dimension}")
                logger.info(f"  - Modell: {self.model_name}")
                
                return True
                
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Embeddings: {e}")
            st.error(f"Fehler beim Erstellen der Embeddings: {e}")
            return False

    def semantic_search(self, query: str) -> Tuple[List[int], List[float], int]:
        """Semantische Suche OHNE Begrenzung der Ergebnisse"""
        try:
            if self.index is None or self.model is None:
                logger.error("Index oder Modell nicht verfügbar")
                return [], [], 0
            
            # Query-Optimierung für E5-Modell
            optimized_query = f"query: {query}"
            
            # Erstelle Query-Embedding mit verbesserter Konfiguration
            query_embedding = self.model.encode(
                [optimized_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Suche ALLE verfügbaren Einträge
            available_count = self.index.ntotal
            
            # Durchführung der Suche für ALLE Einträge
            scores, indices = self.index.search(query_embedding.astype('float32'), available_count)
            
            # Konvertiere zu Listen für sichere Verarbeitung
            scores_list = [float(score) for score in scores[0]]
            indices_list = [int(idx) for idx in indices[0]]
            
            logger.info(f"Semantische Suche abgeschlossen: {len(indices_list)} Einträge gefunden")
            
            return indices_list, scores_list, len(indices_list)
            
        except Exception as e:
            logger.error(f"Fehler bei der semantischen Suche: {e}")
            return [], [], 0
    
    def query_ollama(self, prompt: str, context: str, model: str = None, timeout: int = 120) -> str:
        """Robuste Ollama-Anfrage - MUSS erfolgreich sein"""
        if model is None:
            model = self.ollama_model
            
        try:
            possible_urls = [
                "http://localhost:11434/api/generate",
                "http://127.0.0.1:11434/api/generate"
            ]
            
            # Erweiterte und optimierte System-Prompts
            system_prompt = """Du bist ein hochspezialisierter Experte für die Analyse von industriellen Logbuch-Einträgen und Qualitätsmanagement.

DEINE AUFGABE:
- Analysiere die bereitgestellten Logbuch-Daten präzise und strukturiert
- Beantworte die Benutzerfrage vollständig und detailliert
- Verwende ausschließlich Informationen aus den bereitgestellten Daten
- Strukturiere deine Antwort professionell mit klaren Überschriften

ANTWORT-FORMAT:
1. **Direkte Antwort** auf die Frage (2-3 Sätze)
2. **Relevante Einträge** (nummeriert mit wichtigsten Details)
3. **Zusammenfassung** der Erkenntnisse
4. **Empfehlungen** (falls angebracht)

WICHTIGE REGELN:
- Zitiere spezifische Eintragsnummern bei Verweisen
- Bei unklaren Anfragen stelle gezielte Rückfragen
- Hebe kritische Probleme oder Muster hervor
- Verwende präzise deutsche Fachterminologie
- Falls keine relevanten Daten: erkläre was fehlt

Antworte präzise, strukturiert und professionell auf Deutsch."""
            
            # Optimierter Prompt
            full_prompt = f"""{system_prompt}

=== KONTEXT-DATEN ===
{context[:6000]}

=== BENUTZER-ANFRAGE ===
{prompt}

=== IHRE ANTWORT ==="""
            
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_ctx": 8192,
                    "num_predict": 2048,
                    "repeat_penalty": 1.1,
                    "stop": ["=== BENUTZER-ANFRAGE ===", "=== KONTEXT-DATEN ==="]
                }
            }
            
            # Versuche verschiedene URLs
            for url in possible_urls:
                try:
                    logger.info(f"Sende Anfrage an Ollama: {url}")
                    response = requests.post(url, json=payload, timeout=timeout)
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get('response', '').strip()
                        
                        # Erweiterte Qualitätsprüfung
                        if len(answer) > 10 and "error" not in answer.lower():
                            logger.info("Ollama-Antwort erfolgreich erhalten")
                            return answer
                        else:
                            logger.warning(f"Qualitätsprüfung fehlgeschlagen: {answer[:100]}")
                            raise Exception(f"Unzureichende Antwortqualität von Ollama")
                    else:
                        logger.error(f"HTTP {response.status_code}: {response.text}")
                        raise Exception(f"Ollama HTTP Error: {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    logger.error(f"Timeout bei {url} nach {timeout}s")
                    raise Exception(f"Ollama Timeout nach {timeout}s")
                except requests.exceptions.ConnectionError:
                    logger.error(f"Verbindungsfehler bei {url}")
                    raise Exception("Ollama Verbindungsfehler")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request-Fehler bei {url}: {e}")
                    raise Exception(f"Ollama Request-Fehler: {e}")
            
            # Alle URLs fehlgeschlagen
            raise Exception("Alle Ollama-URLs fehlgeschlagen")
                
        except Exception as e:
            logger.error(f"Kritischer Fehler bei Ollama-Anfrage: {e}")
            # KEIN Fallback - Fehler weiterreichen
            raise Exception(f"Ollama-Anfrage fehlgeschlagen: {e}")

    def extract_highlighted_entries(self, llm_response: str) -> List[int]:
        """Erweiterte Extraktion von hervorgehobenen Einträgen"""
        highlighted = []
        
        # Erweiterte Muster für verschiedene Referenzstile
        patterns = [
            r'Eintrag\s*(\d+)',      # "Eintrag 3"
            r'Nr\.\s*(\d+)',         # "Nr. 5"
            r'#(\d+)',               # "#7"
            r'(\d+)\.',              # "3."
            r'siehe\s*(\d+)',        # "siehe 2"
            r'Punkt\s*(\d+)',        # "Punkt 4"
            r'Position\s*(\d+)',     # "Position 1"
            r'Entry\s*(\d+)',        # "Entry 2"
            r'Zeile\s*(\d+)',        # "Zeile 5"
            r'Nummer\s*(\d+)',       # "Nummer 3"
            r'\((\d+)\)',            # "(4)"
            r'Pos\.\s*(\d+)',        # "Pos. 6"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, llm_response, re.IGNORECASE)
            for match in matches:
                try:
                    entry_num = int(match)
                    # Plausibilitätsprüfung
                    if 1 <= entry_num <= 1000 and entry_num not in highlighted:
                        highlighted.append(entry_num)
                except ValueError:
                    continue
        
        # Sortiere und protokolliere
        highlighted = sorted(highlighted)
        logger.info(f"LLM hervorgehobene Einträge: {highlighted}")
        
        return highlighted
    
    def get_top_relevant_entries(self, scores: List[float], threshold: float = 0.65) -> List[int]:
        """Verbesserte Ermittlung der relevantesten Einträge"""
        if not scores:
            return []
        
        top_entries = []
        
        # Adaptive Schwellenwerte basierend auf Score-Verteilung
        scores_array = np.array(scores)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        
        # Dynamischer Schwellenwert
        adaptive_threshold = max(threshold, mean_score + 0.5 * std_score)
        
        for i, score in enumerate(scores, 1):
            if score >= adaptive_threshold:
                top_entries.append(i)
        
        # Fallback: Mindestens Top 20% oder 5 Einträge
        if len(top_entries) < 5:
            num_top = max(5, int(len(scores) * 0.2))
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            top_entries = [i + 1 for i in sorted_indices[:num_top]]
        
        logger.info(f"Top relevante Einträge (adaptiver Schwellenwert {adaptive_threshold:.3f}): {top_entries}")
        return top_entries

    def analyze_query(self, query: str) -> dict:
        """Verbesserte Hauptfunktion für intelligente Analyse"""
        if self.df is None or self.index is None:
            return {"error": "Keine Daten geladen"}
        
        try:
            logger.info(f"Starte Analyse für: '{query}'")
            
            # Semantische Suche OHNE Begrenzung
            indices, scores, result_count = self.semantic_search(query)
            
            if not indices or result_count == 0:
                return {"error": "Keine Ergebnisse gefunden. Versuchen Sie andere Suchbegriffe."}
            
            logger.info(f"Gefunden: {result_count} Einträge")
            
            # Extrahiere ALLE relevanten Einträge
            relevant_entries = self.df.iloc[indices].copy()
            relevant_entries['similarity_score'] = scores
            
            # Verbesserte Kontext-Aufbereitung für LLM
            context_parts = []
            for i, (_, row) in enumerate(relevant_entries.iterrows(), 1):
                # Strukturierter Kontext mit allen verfügbaren Informationen
                entry_text = f"""Eintrag {i}:
- Datum: {row.get('Datum', 'N/A')}
- Zeit: {row.get('Zeit', 'N/A')}  
- Lot-Nr.: {row.get('Lot-Nr.', 'N/A')}
- Subsystem: {row.get('Subsystem', 'N/A')}
- Ereignis & Maßnahme: {row.get('Ereignis & Massnahme', 'N/A')}
- Visum: {row.get('Visum', 'N/A')}
- Relevanz-Score: {scores[i-1]:.3f}
"""
                context_parts.append(entry_text)
            
            context = "\n".join(context_parts)
            
            # Ergebnis-Dictionary
            result = {
                "relevant_entries": relevant_entries,
                "context": context,
                "query": query,
                "result_count": result_count,
                "total_available": len(self.df),
                "average_relevance": np.mean(scores) if scores else 0,
                "max_relevance": max(scores) if scores else 0
            }
            
            # LLM-Analyse - MUSS erfolgreich sein
            try:
                logger.info("Starte LLM-Analyse...")
                # DIREKT zur Ollama-Anfrage - kein vorheriger Test
                llm_response = self.query_ollama(query, context)
                result["llm_analysis"] = llm_response
                
                # Intelligente Hervorhebung kombinieren
                highlighted_by_llm = self.extract_highlighted_entries(llm_response)
                top_relevant = self.get_top_relevant_entries(scores)
                
                # Kombiniere und dedupliziere Hervorhebungen
                all_highlighted = list(set(highlighted_by_llm + top_relevant))
                all_highlighted.sort()
                result["highlighted_entries"] = all_highlighted
                
            except Exception as llm_error:
                logger.error(f"LLM-Analyse fehlgeschlagen: {llm_error}")
                return {"error": f"LLM-Analyse fehlgeschlagen: {llm_error}. Bitte prüfen Sie die Ollama-Verbindung mit dem manuellen Test-Button."}
            
            # Zusätzliche Metadaten für bessere UX
            result["search_quality"] = "Hoch" if np.mean(scores) > 0.6 else "Mittel" if np.mean(scores) > 0.4 else "Niedrig"
            
            return result
            
        except Exception as e:
            logger.error(f"Kritischer Fehler bei der Analyse: {e}")
            return {"error": f"Fehler bei der Analyse: {str(e)}"}

def main():
    st.set_page_config(
        page_title="Logbuch Analyzer Pro", 
        page_icon="📋", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📋 Logbuch Analyzer Pro")
    st.markdown("**🚀 KI-gestützte Suche und Analyse von Logbuch-Einträgen**")
    
    # Session State initialisieren
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = LogbookAnalyzer()
        st.session_state.initialized = False
        st.session_state.data_processed = False
    
    analyzer = st.session_state.analyzer
    
    # Auto-Initialisierung beim ersten Start
    if not st.session_state.initialized:
        st.info("🚀 Initialisiere Anwendung...")
        
        # Debug: Prüfe sentence-transformers Import
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.error("❌ sentence-transformers nicht verfügbar")
            st.code("pip install sentence-transformers")
            st.stop()
        else:
            try:
                import sentence_transformers
                st.success(f"✅ sentence-transformers verfügbar (Version: {sentence_transformers.__version__})")
            except Exception as e:
                st.error(f"❌ sentence-transformers Fehler: {e}")
                st.stop()
        
        success = analyzer.auto_initialize()
        st.session_state.initialized = success
        
        if not success:
            st.error("❌ Initialisierung fehlgeschlagen - App kann nicht gestartet werden")
            st.info("💡 Stellen Sie sicher, dass Ollama läuft und das empfohlene Modell verfügbar ist")
            st.stop()
        else:
            st.success("✅ Initialisierung erfolgreich")
        
        st.rerun()
    
    # Prüfe ob alle Komponenten verfügbar sind
    if not st.session_state.initialized:
        st.error("❌ Anwendung nicht korrekt initialisiert")
        st.stop()
    
    # Sidebar für System-Info
    with st.sidebar:
        st.header("🔧 System Status Pro")
        
        # Embedding-Modell Status mit Debug-Info
        if analyzer.model is not None:
            st.success("✅ Embedding-Modell: Aktiv")
            st.caption(f"Modell: {analyzer.model_name}")
        else:
            st.error("❌ Embedding-Modell: Nicht geladen")
            if st.button("🔄 Modell neu laden"):
                with st.spinner("Lade Modell..."):
                    success = analyzer.load_embedding_model()
                    if success:
                        st.success("✅ Modell erfolgreich geladen!")
                        st.rerun()
                    else:
                        st.error("❌ Modell-Loading fehlgeschlagen")
        
        # Ollama Status mit automatischer Reparatur
        st.markdown(f"🦙 **Ollama:** {analyzer.ollama_status}")
        
        # Erweiterte Ollama-Kontrollen
        if "❌" in analyzer.ollama_status or "⚠️" in analyzer.ollama_status:
            st.error("🚨 Ollama-Problem erkannt!")
            
            col_repair1, col_repair2 = st.columns(2)
            
            with col_repair1:
                if st.button("🔧 Auto-Reparatur"):
                    with st.spinner("🔧 Repariere Ollama-Verbindung..."):
                        # WICHTIG: Rufe check_ollama_status() direkt auf, NICHT die alte test-basierte Version
                        success = analyzer.check_ollama_status()
                        if success:
                            st.success("✅ Ollama repariert!")
                            st.rerun()
                        else:
                            st.error("❌ Reparatur fehlgeschlagen")
            
            with col_repair2:
                if st.button("⚡ Prozesse beenden"):
                    with st.spinner("⚡ Beende Ollama-Prozesse..."):
                        if analyzer.kill_existing_ollama_processes():
                            st.success("✅ Prozesse beendet")
                            time.sleep(2)
                            # Versuche automatisch neu zu starten
                            if analyzer.start_ollama_server():
                                st.success("🚀 Server neu gestartet")
                                time.sleep(3)
                                # WICHTIG: Nur Health-Check, kein Test
                                analyzer.check_ollama_status()
                                st.rerun()
                        else:
                            st.error("❌ Konnte nicht alle Prozesse beenden")
            
            st.markdown("💡 **Manuelle Schritte (falls Auto-Reparatur fehlschlägt):**")
            st.code("""
# Windows Task Manager oder:
taskkill /F /IM ollama.exe

# Dann neu starten:
ollama serve

# Modell installieren falls nötig:
ollama pull deepseek-r1:latest
            """)
            
            # Zusätzliche Diagnose-Informationen
            with st.expander("🔍 Diagnose-Informationen"):
                try:
                    import psutil
                    st.write("**Laufende Ollama-Prozesse:**")
                    ollama_procs = []
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            proc_info = proc.info
                            if (proc_info['name'] and 'ollama' in proc_info['name'].lower()) or \
                               (proc_info['cmdline'] and any('ollama' in cmd.lower() for cmd in proc_info['cmdline'])):
                                ollama_procs.append(f"PID {proc_info['pid']}: {proc_info['name']}")
                        except:
                            continue
                    
                    if ollama_procs:
                        for proc in ollama_procs:
                            st.text(proc)
                    else:
                        st.text("Keine Ollama-Prozesse gefunden")
                        
                except ImportError:
                    st.warning("psutil nicht verfügbar - installieren Sie: pip install psutil")
                except Exception as e:
                    st.error(f"Diagnose-Fehler: {e}")
                
                # Port-Check
                st.write("**Port 11434 Status:**")
                try:
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex(('127.0.0.1', 11434))
                    sock.close()
                    
                    if result == 0:
                        st.success("Port 11434 ist belegt (gut)")
                    else:
                        st.error("Port 11434 ist frei (Ollama läuft nicht)")
                except Exception as e:
                    st.error(f"Port-Check fehlgeschlagen: {e}")
        
        else:
            st.success("✅ Ollama funktioniert korrekt")
            col_test1, col_test2 = st.columns(2)
            
            with col_test1:
                if st.button("🧪 Modell-Test"):
                    with st.spinner("Teste Ollama-Modell..."):
                        test_result = analyzer.test_ollama_model_on_demand()
                        if test_result:
                            st.success("✅ Modell-Test erfolgreich")
                        else:
                            st.error("❌ Modell-Test fehlgeschlagen")
                            st.info("💡 Modell funktioniert möglicherweise trotzdem - Health-Check war erfolgreich")
            
            with col_test2:
                if st.button("📋 Modell-Info"):
                    models = analyzer.get_ollama_models_fast()
                    if models:
                        st.info(f"Verfügbare Modelle: {', '.join(models[:3])}")
                        st.info(f"Aktives Modell: {analyzer.ollama_model}")
                    else:
                        st.warning("Keine Modell-Info verfügbar")
        
        # Daten Status
        if analyzer.df is not None:
            st.success(f"📊 Daten: {len(analyzer.df)} Einträge verarbeitet")
            if st.session_state.data_processed:
                st.success("✅ Embeddings: Erstellt und bereit")
                if st.button("🗑️ Daten zurücksetzen"):
                    analyzer.df = None
                    analyzer.index = None
                    analyzer.embeddings = None
                    st.session_state.data_processed = False
                    st.success("Daten zurückgesetzt")
                    st.rerun()
            else:
                st.warning("⏳ Embeddings werden erstellt...")
        else:
            st.info("📄 Keine Daten geladen")
        
        st.markdown("---")
        
        # Performance Metriken
        st.subheader("📊 Performance Metriken")
        if analyzer.df is not None:
            st.metric("Verarbeitete Einträge", len(analyzer.df))
            if hasattr(analyzer, 'embeddings') and analyzer.embeddings is not None:
                st.metric("Embedding Dimension", analyzer.embeddings.shape[1])
        
        # System Logs
        st.subheader("📝 System Logs")
        if hasattr(st.session_state, 'log_handler'):
            logs = st.session_state.log_handler.logs[-8:]
            if logs:
                for log in logs:
                    if "ERROR" in log:
                        st.error(log)
                    elif "WARNING" in log:
                        st.warning(log)
                    else:
                        st.text(log)
            else:
                st.text("Keine Logs verfügbar")
        
        # Debug-Informationen
        with st.expander("🔧 Debug Info"):
            st.write("**Python Pfad:**", sys.executable)
            st.write("**Arbeitsverzeichnis:**", os.getcwd())
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    import sentence_transformers
                    st.write("**SentenceTransformers:**", sentence_transformers.__version__)
                except:
                    st.write("**SentenceTransformers:** ❌ Fehler beim Import")
            else:
                st.write("**SentenceTransformers:** ❌ Nicht verfügbar")
            
            try:
                import torch
                st.write("**PyTorch:**", torch.__version__)
            except:
                st.write("**PyTorch:** ❌ Nicht verfügbar")
        
        # Refresh und Hilfe
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 System Refresh"):
                with st.spinner("Aktualisiere System-Status..."):
                    # WICHTIG: Nur Health-Check, kein Modell-Test
                    analyzer.check_ollama_status()
                    st.rerun()
        with col2:
            if st.button("❓ Hilfe"):
                st.info("💡 Bei Ollama-Problemen: Auto-Reparatur verwenden oder Prozesse manuell beenden")

    # Prüfe kritische Abhängigkeiten vor Hauptbereich
    if not st.session_state.initialized:
        st.error("❌ Anwendung nicht korrekt initialisiert")
        st.info("🔧 Verwenden Sie die Auto-Reparatur in der Sidebar")
        st.stop()
    
    # Zusätzliche Dependency-Checks
    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False
        st.warning("⚠️ psutil nicht verfügbar - einige Auto-Reparatur-Funktionen sind eingeschränkt")
        st.info("💡 Installieren Sie psutil für erweiterte Prozessverwaltung: `pip install psutil`")
    
    # Hauptbereich
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📤 Daten Upload")
        
        uploaded_file = st.file_uploader(
            "CSV-Datei hochladen",
            type=['csv'],
            help="Automatische Erkennung von Encoding und Trennzeichen. Erwartete Spalten: Datum, Zeit, Lot-Nr., Subsystem, Ereignis & Massnahme, Visum"
        )
        
        if uploaded_file is not None:
            # Verhindere mehrfache Verarbeitung derselben Datei
            file_hash = str(hash(uploaded_file.getvalue()))
            
            if 'last_file_hash' not in st.session_state:
                st.session_state.last_file_hash = None
            
            # Nur verarbeiten wenn neue Datei oder Daten nicht verarbeitet
            if (st.session_state.last_file_hash != file_hash or 
                not st.session_state.data_processed):
                
                st.session_state.last_file_hash = file_hash
                
                # Automatische Verarbeitung ohne Button
                with st.spinner("🔄 Analysiere und lade CSV-Datei..."):
                    df, encoding, separator = analyzer.load_csv_robust(uploaded_file)
                    
                    if df is not None:
                        st.success(f"✅ {len(df)} Zeilen geladen")
                        st.info(f"📝 Erkannt: {encoding} / Trennzeichen: '{separator}'")
                        
                        # Datenvorschau
                        with st.expander("📊 Datenvorschau", expanded=False):
                            st.dataframe(df.head(10))
                            st.caption(f"Spalten: {', '.join(df.columns)}")
                        
                        # Automatische Datenverarbeitung
                        with st.spinner("🔄 Verarbeite Daten..."):
                            processed_df = analyzer.preprocess_data(df)
                            
                            if processed_df is not None:
                                st.success(f"✅ {len(processed_df)} Einträge verarbeitet")
                                
                                # Automatische Embedding-Erstellung
                                if analyzer.model is not None:
                                    with st.spinner("🧠 Erstelle Embeddings..."):
                                        if analyzer.create_embeddings(processed_df):
                                            st.success("🎯 Bereit für intelligente Suche!")
                                            st.session_state.data_processed = True
                                        else:
                                            st.error("❌ Embedding-Erstellung fehlgeschlagen")
                                            st.session_state.data_processed = False
                                else:
                                    st.error("❌ Embedding-Modell nicht verfügbar")
                                    st.session_state.data_processed = False
                            else:
                                st.error("❌ Datenverarbeitung fehlgeschlagen")
                                st.session_state.data_processed = False
                    else:
                        st.error("❌ CSV-Datei konnte nicht gelesen werden")
                        st.info("💡 Tipp: Unterstützte Formate: UTF-8, Latin1, CP1252 mit Komma, Semikolon oder Tab als Trennzeichen")
                        st.session_state.data_processed = False
            else:
                # Datei bereits verarbeitet
                st.info("📄 Diese Datei wurde bereits verarbeitet")
                if analyzer.df is not None:
                    st.success(f"✅ {len(analyzer.df)} Logbucheinträge bereit für Suche")
                    
                    # Option zum Neuverarbeiten
                    if st.button("🔄 Datei neu verarbeiten"):
                        st.session_state.last_file_hash = None
                        st.session_state.data_processed = False
                        st.rerun()
    
    with col2:
        st.header("🔍 Intelligente Suche & Analyse")
        
        if analyzer.df is not None and st.session_state.data_processed:
            # Erfolgreiche Initialisierung anzeigen
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📊 Einträge", len(analyzer.df))
            with col_b:
                st.metric("KI-Modell", "E5-Small")
            
            st.markdown("---")
            
            # Sucheingabe
            query = st.text_input(
                "🔎 Frage oder Suchanfrage an die Daten:",
                placeholder="z.B. 'Alle Probleme in Lot 12345' oder 'Wann gab es Qualitätsfehler?'",
                help="Alle Einträge werden durchsucht und von der KI analysiert"
            )
            
            # Automatische Suche bei Eingabe
            if query and query.strip():
                with st.spinner("🧠 Führe vollständige KI-Analyse durch..."):
                    results = analyzer.analyze_query(query.strip())
                    
                    if "error" in results:
                        st.error(f"❌ {results['error']}")
                        
                        # Hilfe bei Ollama-Problemen
                        if "Ollama" in results["error"]:
                            st.info("💡 Prüfen Sie:")
                            st.code("ollama serve")
                            st.code("ollama list")  # Zeige verfügbare Modelle
                            
                    else:
                        # Erfolgs-Metriken
                        col_x, col_y, col_z = st.columns(3)
                        with col_x:
                            st.metric("Analysierte Einträge", results["result_count"])
                        with col_y:
                            st.metric("Durchschn. Relevanz", f"{results.get('average_relevance', 0):.3f}")
                        with col_z:
                            st.metric("Max. Relevanz", f"{results.get('max_relevance', 0):.3f}")
                        
                        # LLM-Antwort prominent anzeigen
                        if "llm_analysis" in results and results["llm_analysis"]:
                            st.subheader("🤖 KI-Analyse (alle Einträge)")
                            st.markdown(results["llm_analysis"])
                            st.markdown("---")
                        
                        # Suchergebnisse mit verbesserter Darstellung
                        st.subheader("📋 Alle durchsuchten Einträge")
                        
                        relevant_df = results["relevant_entries"]
                        highlighted = results.get("highlighted_entries", [])
                        
                        # Zeige alle Einträge, nicht nur die ersten
                        for i, (_, row) in enumerate(relevant_df.iterrows(), 1):
                            score = row['similarity_score']
                            
                            # Relevanz-Indikatoren
                            if score >= 0.75:
                                score_indicator = "🟢 Sehr hoch"
                            elif score >= 0.60:
                                score_indicator = "🟡 Hoch"
                            elif score >= 0.45:
                                score_indicator = "🟠 Mittel"
                            else:
                                score_indicator = "🔴 Niedrig"
                            
                            # Hervorhebung für wichtige Einträge
                            is_highlighted = i in highlighted
                            title_prefix = "⭐ " if is_highlighted else ""
                            
                            # Nur die ersten 20 automatisch aufklappen, Rest bei Bedarf
                            auto_expand = is_highlighted or (i <= 20 and score >= 0.4)
                            
                            with st.expander(
                                f"{title_prefix}Eintrag {i} - {score_indicator} ({score:.3f})",
                                expanded=auto_expand
                            ):
                                # Strukturierte Darstellung
                                info_col, detail_col = st.columns([1, 2])
                                
                                with info_col:
                                    st.markdown("**📅 Zeitinformation:**")
                                    st.write(f"Datum: {row.get('Datum', 'N/A')}")
                                    st.write(f"Zeit: {row.get('Zeit', 'N/A')}")
                                    
                                    st.markdown("**🏭 Identifikation:**")
                                    st.write(f"Lot-Nr.: {row.get('Lot-Nr.', 'N/A')}")
                                    st.write(f"Subsystem: {row.get('Subsystem', 'N/A')}")
                                    st.write(f"Visum: {row.get('Visum', 'N/A')}")
                                    
                                    st.markdown("**📊 Metriken:**")
                                    st.write(f"Relevanz: {score_indicator}")
                                    st.write(f"Score: {score:.3f}")
                                
                                with detail_col:
                                    st.markdown("**📝 Ereignis & Maßnahme:**")
                                    ereignis = row.get('Ereignis & Massnahme', 'N/A')
                                    st.markdown(f"*{ereignis}*")
                                    
                                    # Zusätzliche Kontext-Informationen
                                    if is_highlighted:
                                        st.success("⭐ Von KI als besonders relevant markiert")
                        
                        # Download-Optionen
                        col_down1, col_down2 = st.columns(2)
                        with col_down1:
                            csv_data = relevant_df.drop('similarity_score', axis=1).to_csv(index=False)
                            st.download_button(
                                label="📥 Alle Ergebnisse als CSV",
                                data=csv_data,
                                file_name=f"logbuch_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col_down2:
                            # Vollständiger Report
                            report = f"""# Logbuch Analyse Report

## Suchanfrage
{query}

## Zusammenfassung
- Analysierte Einträge: {results['result_count']}
- Durchschnittliche Relevanz: {results.get('average_relevance', 0):.3f}
- Max. Relevanz: {results.get('max_relevance', 0):.3f}

## KI-Analyse
{results.get('llm_analysis', 'Nicht verfügbar')}

## Detaillierte Ergebnisse
{results['context']}
"""
                            st.download_button(
                                label="📄 Vollständiger Report",
                                data=report,
                                file_name=f"logbuch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
        
        elif analyzer.df is not None:
            st.info("⏳ Daten werden verarbeitet...")
        else:
            st.info("👆 Bitte laden Sie zuerst eine CSV-Datei hoch")
            
            # Hilfreiche Beispiele anzeigen
            with st.expander("💡 Beispiel-Suchanfragen", expanded=False):
                st.markdown("""
                **Problemanalyse:**
                - "Alle Fehler der letzten Woche"
                - "Probleme in Subsystem Pumpe"
                
                **Lot-spezifische Suche:**
                - "Alle Einträge für Lot 12345"
                - "Qualitätsprobleme in Charge ABC"
                
                **Zeitbasierte Analyse:**
                - "Was passierte am 15.03.2024?"
                - "Wartungsarbeiten im März"
                
                **Systemanalyse:**
                - "Alle Ereignisse im Kühlsystem"
                - "Übersicht Produktionslinie 2"
                """)

if __name__ == "__main__":
    main()