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
except ImportError:
    st.error("sentence-transformers nicht installiert. Bitte installieren Sie es mit: pip install sentence-transformers")
    st.stop()

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
        self.ollama_model = "llama3.1:8b"  # Größeres Modell für bessere Qualität
        
    def auto_initialize(self):
        """Automatische Initialisierung beim App-Start"""
        try:
            # Embedding-Modell laden
            with st.spinner('🤖 Lade Embedding-Modell (multilingual-e5-small)...'):
                success = self.load_embedding_model()
                if success:
                    st.success("✅ Embedding-Modell geladen")
                    logger.info("Embedding-Modell erfolgreich geladen")
                else:
                    st.error("❌ Embedding-Modell konnte nicht geladen werden")
                    # Zeige Fehlerbehebung
                    st.info("💡 Versuchen Sie: pip install --upgrade sentence-transformers")
                    return False
            
            # Ollama Status prüfen
            with st.spinner('🦙 Prüfe Ollama-Verbindung...'):
                self.check_ollama_status()
                
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
    
    def check_ollama_status(self):
        """Prüft Ollama-Status und empfiehlt bessere Modelle"""
        try:
            test_urls = [
                "http://localhost:11434/api/tags",
                "http://127.0.0.1:11434/api/tags"
            ]
            
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        models_data = response.json()
                        available_models = [model['name'] for model in models_data.get('models', [])]
                        
                        # Prüfe auf bessere Modelle
                        recommended_models = [
                            "qwen2.5:14b",      # Sehr gut für Analyse
                            "llama3.1:8b",      # Standard
                            "mistral:7b",       # Alternative
                            "qwen2.5:7b"        # Kompakt aber gut
                        ]
                        
                        best_available = None
                        for rec_model in recommended_models:
                            if rec_model in available_models:
                                best_available = rec_model
                                break
                        
                        if best_available:
                            self.ollama_model = best_available
                            test_success = self.test_ollama_model()
                            if test_success:
                                self.ollama_status = f"✅ Aktiv mit {best_available}"
                                st.success(f"🦙 Ollama: {self.ollama_status}")
                                logger.info(f"Ollama funktioniert mit {best_available}")
                            else:
                                self.ollama_status = "⚠️ Verbunden, aber Modell antwortet nicht"
                                st.warning(f"🦙 Ollama: {self.ollama_status}")
                        else:
                            self.ollama_status = f"⚠️ Kein empfohlenes Modell verfügbar"
                            st.warning(f"🦙 Ollama: {self.ollama_status}")
                            if available_models:
                                st.info(f"Verfügbare Modelle: {', '.join(available_models[:3])}")
                                st.info("💡 Empfohlen: `ollama pull qwen2.5:14b` für beste Qualität")
                        return
                        
                except requests.exceptions.RequestException:
                    continue
            
            self.ollama_status = "❌ Server nicht erreichbar"
            st.error(f"🦙 Ollama: {self.ollama_status}")
            st.info("💡 Starten Sie Ollama mit: `ollama serve`")
            
        except Exception as e:
            self.ollama_status = f"❌ Fehler: {str(e)[:50]}"
            st.error(f"🦙 Ollama: {self.ollama_status}")
            logger.error(f"Ollama-Status-Check fehlgeschlagen: {e}")
    
    def test_ollama_model(self) -> bool:
        """Testet das Ollama-Modell mit einer einfachen Anfrage"""
        try:
            test_prompt = "Antworte nur mit 'OK'"
            response = self.query_ollama(test_prompt, "", timeout=10)
            
            if response and "OK" in response and "❌" not in response:
                return True
            return False
            
        except Exception as e:
            logger.error(f"Ollama-Modell-Test fehlgeschlagen: {e}")
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
    
    def determine_optimal_results(self, scores: List[float], query: str) -> int:
        """Verbesserte automatische Bestimmung der optimalen Ergebnisanzahl"""
        if not scores or len(scores) == 0:
            return 0
        
        scores = np.array(scores)
        
        # Angepasste Schwellenwerte für das verbesserte E5-large Modell
        excellent_threshold = 0.75    # Exzellente Übereinstimmung
        high_relevance_threshold = 0.65  # Hohe Relevanz
        medium_relevance_threshold = 0.50  # Mittlere Relevanz
        min_relevance_threshold = 0.35   # Minimale Relevanz
        
        # Zähle Ergebnisse nach Qualitätsstufen
        excellent = int(np.sum(scores >= excellent_threshold))
        high_relevant = int(np.sum(scores >= high_relevance_threshold))
        medium_relevant = int(np.sum(scores >= medium_relevance_threshold))
        min_relevant = int(np.sum(scores >= min_relevance_threshold))
        
        # Intelligente adaptive Logik
        if excellent >= 3:
            # Mehrere exzellente Treffer: zeige alle + einige gute
            optimal_count = min(excellent + min(3, high_relevant - excellent), 20)
        elif high_relevant >= 2:
            # Einige hochrelevante: zeige alle relevanten
            optimal_count = min(high_relevant + min(4, medium_relevant - high_relevant), 25)
        elif medium_relevant >= 1:
            # Mittlere Relevanz: zeige alle über mittlerem Schwellenwert
            optimal_count = min(medium_relevant + min(2, min_relevant - medium_relevant), 15)
        elif min_relevant >= 1:
            # Minimale Relevanz: zeige wenige für Kontext
            optimal_count = min(min_relevant, 10)
        else:
            # Keine relevanten: zeige Top 3 für Feedback
            optimal_count = min(3, len(scores))
        
        # Query-spezifische Anpassungen mit verbesserter Erkennung
        query_lower = query.lower()
        
        # Spezifische Suchtypen
        if any(keyword in query_lower for keyword in ['alle', 'list', 'zeige', 'übersicht', 'vollständig']):
            optimal_count = min(optimal_count * 2, 40)
        
        # Zeitbasierte Suchen
        if any(keyword in query_lower for keyword in ['wann', 'zeit', 'datum', 'letzte', 'heute', 'gestern', 'monat']):
            optimal_count = min(optimal_count + 3, 30)
        
        # Problem-/Fehlersuchen (oft wichtiger)
        if any(keyword in query_lower for keyword in ['problem', 'fehler', 'defekt', 'ausfall', 'störung']):
            optimal_count = min(optimal_count + 2, 25)
        
        # Lot-/Batch-spezifische Suchen
        if any(keyword in query_lower for keyword in ['lot', 'charge', 'batch', 'serie']):
            optimal_count = min(optimal_count + 5, 35)
        
        # Qualitätskontrolle: nicht zu wenige, nicht zu viele
        optimal_count = max(2, min(optimal_count, 40))
        
        logger.info(f"Optimierte Ergebnisanzahl: {optimal_count}")
        logger.info(f"Qualitäts-Verteilung - Exzellent: {excellent}, Hoch: {high_relevant}, Mittel: {medium_relevant}, Min: {min_relevant}")
        
        return optimal_count

    def semantic_search(self, query: str, max_results: int = 60) -> Tuple[List[int], List[float], int]:
        """Verbesserte semantische Suche mit Query-Optimierung"""
        try:
            if self.index is None or self.model is None:
                logger.error("Index oder Modell nicht verfügbar")
                return [], [], 0
            
            # Query-Optimierung für E5-Modell
            # E5 funktioniert besser mit "query:" Prefix für Suchanfragen
            optimized_query = f"query: {query}"
            
            # Erstelle Query-Embedding mit verbesserter Konfiguration
            query_embedding = self.model.encode(
                [optimized_query],
                convert_to_numpy=True,
                normalize_embeddings=True  # Konsistent mit Index-Erstellung
            )
            
            # Begrenze Suche auf verfügbare Daten
            available_count = self.index.ntotal
            actual_max = min(max_results, available_count)
            
            # Durchführung der Suche
            scores, indices = self.index.search(query_embedding.astype('float32'), actual_max)
            
            # Konvertiere zu Listen für sichere Verarbeitung
            scores_list = [float(score) for score in scores[0]]
            indices_list = [int(idx) for idx in indices[0]]
            
            # Bestimme optimale Anzahl mit verbesserter Logik
            optimal_count = self.determine_optimal_results(scores_list, query)
            optimal_count = min(optimal_count, len(scores_list))
            
            return indices_list[:optimal_count], scores_list[:optimal_count], optimal_count
            
        except Exception as e:
            logger.error(f"Fehler bei der semantischen Suche: {e}")
            return [], [], 0
    
    def query_ollama(self, prompt: str, context: str, model: str = None, timeout: int = 120) -> str:
        """Robuste Ollama-Anfrage mit erweiterten Fallback-Mechanismen"""
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
{context[:4000]}  # Begrenzt für bessere Performance

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
                    "num_ctx": 6144,  # Reduziert für bessere Performance
                    "num_predict": 1024,  # Kürzere Antworten
                    "repeat_penalty": 1.1,
                    "stop": ["=== BENUTZER-ANFRAGE ===", "=== KONTEXT-DATEN ==="]
                }
            }
            
            # Versuche verschiedene URLs mit erweiterten Timeouts
            for url in possible_urls:
                try:
                    logger.info(f"Sende Anfrage an Ollama: {url}")
                    response = requests.post(url, json=payload, timeout=timeout)
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get('response', '').strip()
                        
                        # Erweiterte Qualitätsprüfung
                        if (len(answer) > 10 and 
                            "❌" not in answer and
                            "error" not in answer.lower()):
                            logger.info("Ollama-Antwort erfolgreich erhalten")
                            return answer
                        else:
                            logger.warning(f"Qualitätsprüfung fehlgeschlagen: {answer[:100]}")
                            continue
                    else:
                        logger.warning(f"HTTP {response.status_code}: {response.text}")
                        continue
                        
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout bei {url} nach {timeout}s")
                    continue
                except requests.exceptions.ConnectionError:
                    logger.warning(f"Verbindungsfehler bei {url}")
                    continue
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request-Fehler bei {url}: {e}")
                    continue
            
            # Alle URLs fehlgeschlagen - Fallback
            logger.warning("Alle Ollama-URLs fehlgeschlagen - verwende Fallback")
            return self.fallback_analysis(prompt, context)
                
        except Exception as e:
            logger.error(f"Kritischer Fehler bei Ollama-Anfrage: {e}")
            return self.fallback_analysis(prompt, context)
    
    def fallback_analysis(self, prompt: str, context: str) -> str:
        """Verbesserte Fallback-Analyse ohne LLM"""
        lines = context.split('\n')
        entry_count = len([line for line in lines if line.startswith('Eintrag')])
        
        # Einfache Keyword-Analyse
        prompt_lower = prompt.lower()
        context_lower = context.lower()
        
        analysis_points = []
        
        # Problem-Erkennung
        if any(word in prompt_lower for word in ['problem', 'fehler', 'defekt', 'ausfall']):
            problem_indicators = ['fehler', 'problem', 'defekt', 'ausfall', 'störung']
            problem_count = sum(context_lower.count(word) for word in problem_indicators)
            analysis_points.append(f"🔍 **Problem-Analyse:** {problem_count} potentielle Probleme erkannt")
        
        # Zeit-Analyse
        if any(word in prompt_lower for word in ['wann', 'zeit', 'datum']):
            analysis_points.append("📅 **Zeitanalyse:** Chronologische Darstellung der Einträge")
        
        # System-Analyse
        if 'subsystem' in prompt_lower or 'system' in prompt_lower:
            analysis_points.append("⚙️ **System-Analyse:** Subsystem-spezifische Auswertung")
        
        return f"""📊 **Automatische Basis-Analyse** (Ollama nicht verfügbar)

**🎯 Direkte Antwort:**
Es wurden {entry_count} relevante Einträge zu Ihrer Anfrage gefunden.

**📋 Analyseergebnisse:**
{chr(10).join(analysis_points) if analysis_points else '- Allgemeine Logbuch-Einträge gefunden'}

**📄 Gefundene Einträge:**
{context[:800]}{'...' if len(context) > 800 else ''}

**💡 Empfehlung:**
Für detaillierte KI-Analyse starten Sie Ollama mit: `ollama serve`
Empfohlenes Modell für beste Qualität: `ollama pull qwen2.5:14b`

**🔧 Hinweis:** Diese Basis-Analyse basiert auf einfachen Mustern. Für semantische Analyse und intelligente Erkenntnisse ist Ollama erforderlich."""

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
                    if 1 <= entry_num <= 100 and entry_num not in highlighted:
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
        
        # Fallback: Mindestens Top 20% oder 3 Einträge
        if len(top_entries) < 3:
            num_top = max(3, int(len(scores) * 0.2))
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            top_entries = [i + 1 for i in sorted_indices[:num_top]]
        
        logger.info(f"Top relevante Einträge (adaptiver Schwellenwert {adaptive_threshold:.3f}): {top_entries}")
        return top_entries

    def analyze_query(self, query: str) -> dict:
        """Verbesserte Hauptfunktion für intelligente Analyse"""
        if self.df is None or self.index is None:
            return {"error": "Keine Daten geladen"}
        
        try:
            logger.info(f"Starte verbesserte Analyse für: '{query}'")
            
            # Erweiterte semantische Suche
            indices, scores, result_count = self.semantic_search(query)
            
            if not indices or result_count == 0:
                return {"error": "Keine relevanten Ergebnisse gefunden. Versuchen Sie andere Suchbegriffe."}
            
            logger.info(f"Gefunden: {result_count} hochrelevante Einträge")
            
            # Extrahiere und bereichere relevante Einträge
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
                "average_relevance": np.mean(scores),
                "max_relevance": max(scores) if scores else 0
            }
            
            # LLM-Analyse mit verbessertem Prompt
            logger.info("Starte erweiterte LLM-Analyse...")
            llm_response = self.query_ollama(query, context)
            result["llm_analysis"] = llm_response
            
            # Intelligente Hervorhebung kombinieren
            highlighted_by_llm = self.extract_highlighted_entries(llm_response)
            top_relevant = self.get_top_relevant_entries(scores)
            
            # Kombiniere und dedupliziere Hervorhebungen
            all_highlighted = list(set(highlighted_by_llm + top_relevant))
            all_highlighted.sort()
            result["highlighted_entries"] = all_highlighted
            
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
        try:
            import sentence_transformers
            st.success(f"✅ sentence-transformers verfügbar (Version: {sentence_transformers.__version__})")
        except ImportError as e:
            st.error(f"❌ sentence-transformers Import fehlgeschlagen: {e}")
            st.code("pip install sentence-transformers")
            st.stop()
        
        success = analyzer.auto_initialize()
        st.session_state.initialized = True
        
        if not success:
            st.warning("⚠️ Initialisierung teilweise fehlgeschlagen - App läuft eingeschränkt")
        
        st.rerun()
    
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
                        st.info("Prüfen Sie die Logs unten für Details")
        
        # Ollama Status mit manueller Verbindung
        st.markdown(f"🦙 **Ollama:** {analyzer.ollama_status}")
        if "❌" in analyzer.ollama_status:
            st.markdown("💡 **Empfehlung:**")
            st.code("ollama serve")
            st.code("ollama pull qwen2.5:14b")
            
            # Manueller Verbindungsversuch
            if st.button("🔄 Ollama neu verbinden"):
                with st.spinner("Verbinde mit Ollama..."):
                    analyzer.check_ollama_status()
                    st.rerun()
        elif "⚠️" in analyzer.ollama_status:
            if st.button("🔄 Ollama neu testen"):
                with st.spinner("Teste Ollama-Verbindung..."):
                    analyzer.check_ollama_status()
                    st.rerun()
        
        # Daten Status mit Zurücksetzen-Option
        if analyzer.df is not None:
            st.success(f"📊 Daten: {len(analyzer.df)} Einträge verarbeitet")
            if st.session_state.data_processed:
                st.success("✅ Embeddings: Erstellt und bereit")
                # Reset-Option bei Problemen
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
        
        # Erweiterte System-Informationen
        st.subheader("📊 Performance Metriken")
        if analyzer.df is not None:
            st.metric("Verarbeitete Einträge", len(analyzer.df))
            if hasattr(analyzer, 'embeddings') and analyzer.embeddings is not None:
                st.metric("Embedding Dimension", analyzer.embeddings.shape[1])
        
        # System Logs mit mehr Details
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
            try:
                import sentence_transformers
                st.write("**SentenceTransformers:**", sentence_transformers.__version__)
            except:
                st.write("**SentenceTransformers:** ❌ Nicht verfügbar")
            
            try:
                import torch
                st.write("**PyTorch:**", torch.__version__)
            except:
                st.write("**PyTorch:** ❌ Nicht verfügbar")
        
        # Refresh und Hilfe
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh"):
                analyzer.check_ollama_status()
                st.rerun()
        with col2:
            if st.button("❓ Hilfe"):
                st.info("Tipps: Verwenden Sie spezifische Begriffe für bessere Ergebnisse")
    
    # Hauptbereich - Verbesserte Layout
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
                    # Robuste CSV-Ladung
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
                                    with st.spinner("🧠 Erstelle optimierte Embeddings..."):
                                        if analyzer.create_embeddings(processed_df):
                                            st.success("🎯 Bereit für schnelle intelligente Suche!")
                                            st.session_state.data_processed = True
                                            # KEIN automatisches Rerun mehr!
                                            # st.rerun() entfernt
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
            col_a, col_b, col_c = st.columns(2)
            with col_a:
                st.metric("📊 Einträge", len(analyzer.df))
            with col_b:
                st.metric("KI-Modell", "E5-Small (schnell)")
            
            st.markdown("---")
            
            # Verbesserte Sucheingabe - Enter zum Absenden
            query = st.text_input(
                "🔎 Frage oder Suchanfrage an die Daten:",
                placeholder="z.B. 'Alle Probleme in Lot 12345' oder 'Wann gab es Qualitätsfehler?'",
                help="Drücken Sie Enter zum Absenden. Für neue Zeile verwenden Sie Shift+Enter"
            )
            
            # Automatische Suche bei Enter (query change)
            if query and query.strip():
                with st.spinner("🧠 Führe intelligente Analyse durch..."):
                    results = analyzer.analyze_query(query.strip())
                    
                    if "error" in results:
                        st.error(f"❌ {results['error']}")
                    else:
                        # Erfolgs-Metriken
                        col_x, col_y, col_z = st.columns(3)
                        with col_x:
                            st.metric("Gefunden", results["result_count"])
                        with col_y:
                            st.metric("Durchschn. Relevanz", f"{results.get('average_relevance', 0):.3f}")
                        with col_z:
                            st.metric("Qualität", results.get('search_quality', 'Unbekannt'))
                        
                        # LLM-Antwort prominent anzeigen
                        if "llm_analysis" in results and results["llm_analysis"]:
                            st.subheader("🤖 KI-Analyse")
                            st.markdown(results["llm_analysis"])
                            st.markdown("---")
                        
                        # Suchergebnisse mit verbesserter Darstellung
                        st.subheader("📋 Detaillierte Ergebnisse")
                        
                        relevant_df = results["relevant_entries"]
                        highlighted = results.get("highlighted_entries", [])
                        
                        for i, (_, row) in enumerate(relevant_df.iterrows(), 1):
                            score = row['similarity_score']
                            
                            # Verbesserte Relevanz-Indikatoren
                            if score >= 0.75:
                                score_indicator = "🟢 Sehr hoch"
                                score_color = "green"
                            elif score >= 0.60:
                                score_indicator = "🟡 Hoch"
                                score_color = "orange"
                            elif score >= 0.45:
                                score_indicator = "🟠 Mittel"
                                score_color = "orange"
                            else:
                                score_indicator = "🔴 Niedrig"
                                score_color = "red"
                            
                            # Hervorhebung für wichtige Einträge
                            is_highlighted = i in highlighted
                            title_prefix = "⭐ " if is_highlighted else ""
                            
                            with st.expander(
                                f"{title_prefix}Eintrag {i} - {score_indicator} ({score:.3f})",
                                expanded=is_highlighted or i <= 3  # Erste 3 automatisch aufgeklappt
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
                                label="📥 Ergebnisse als CSV",
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
- Gefundene Einträge: {results['result_count']}
- Durchschnittliche Relevanz: {results.get('average_relevance', 0):.3f}
- Suchqualität: {results.get('search_quality', 'Unbekannt')}

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