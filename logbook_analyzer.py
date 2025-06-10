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
import threading
import time

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
        self.model_name = 'intfloat/multilingual-e5-small'
        self.ollama_status = "unbekannt"
        self.ollama_model = "llama3.1:8b"
        
    def auto_initialize(self):
        """Automatische Initialisierung beim App-Start"""
        try:
            # Embedding-Modell laden
            with st.spinner('🤖 Lade Embedding-Modell...'):
                success = self.load_embedding_model()
                if success:
                    st.success("✅ Embedding-Modell geladen")
                    logger.info("Embedding-Modell erfolgreich geladen")
                else:
                    st.error("❌ Embedding-Modell konnte nicht geladen werden")
                    return False
            
            # Ollama Status prüfen
            with st.spinner('🦙 Prüfe Ollama-Verbindung...'):
                self.check_ollama_status()
                
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei Auto-Initialisierung: {e}")
            st.error(f"Fehler bei der Initialisierung: {e}")
            return False
        
    def load_embedding_model(self):
        """Lädt das Embedding-Modell"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding-Modell {self.model_name} geladen")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Laden des Embedding-Modells: {e}")
            return False
    
    def check_ollama_status(self):
        """Prüft Ollama-Status und verfügbare Modelle"""
        try:
            # Teste verschiedene Endpunkte
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
                        
                        if self.ollama_model in available_models:
                            # Test-Query senden
                            test_success = self.test_ollama_model()
                            if test_success:
                                self.ollama_status = "✅ Aktiv und funktionsfähig"
                                st.success(f"🦙 Ollama: {self.ollama_status}")
                                logger.info(f"Ollama funktioniert mit Modell {self.ollama_model}")
                            else:
                                self.ollama_status = "⚠️ Verbunden, aber Modell antwortet nicht"
                                st.warning(f"🦙 Ollama: {self.ollama_status}")
                        else:
                            self.ollama_status = f"⚠️ Modell '{self.ollama_model}' nicht verfügbar"
                            st.warning(f"🦙 Ollama: {self.ollama_status}")
                            if available_models:
                                st.info(f"Verfügbare Modelle: {', '.join(available_models[:3])}")
                        return
                        
                except requests.exceptions.RequestException:
                    continue
            
            # Alle URLs fehlgeschlagen
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
            
            # Prüfe ob Response sinnvoll ist
            if response and "OK" in response and "❌" not in response:
                return True
            return False
            
        except Exception as e:
            logger.error(f"Ollama-Modell-Test fehlgeschlagen: {e}")
            return False
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bereinigt und validiert die CSV-Daten"""
        try:
            # Spalten validieren und normalisieren - Ihre Logbuch-Struktur
            expected_columns = ['Datum', 'Zeit', 'Lot-Nr.', 'Subsystem', 'Ereignis & Massnahme', 'Visum']
            
            # Flexible Spaltenerkennung
            df_columns = df.columns.tolist()
            column_mapping = {}
            
            # Mapping-Dictionary für verschiedene Varianten
            column_variants = {
                'Datum': ['datum', 'date', 'time-stamp', 'zeitstempel'],
                'Zeit': ['zeit', 'time', 'uhrzeit'],
                'Lot-Nr.': ['lot', 'lotnummer', 'lot-nr', 'lot-nummer', 'charge'],
                'Subsystem': ['subsystem', 'system', 'bereich', 'modul'],
                'Ereignis & Massnahme': ['ereignis', 'eintrag', 'massnahme', 'entry', 'beschreibung', 'inhalt'],
                'Visum': ['visum', 'signature', 'unterschrift', 'benutzer', 'user']
            }
            
            for expected in expected_columns:
                # Exakte Übereinstimmung
                if expected in df_columns:
                    column_mapping[expected] = expected
                else:
                    # Ähnliche Spalten finden
                    found = False
                    for col in df_columns:
                        col_lower = col.lower().strip()
                        # Prüfe Varianten für diese Spalte
                        variants = column_variants.get(expected, [expected.lower()])
                        if any(variant in col_lower for variant in variants):
                            column_mapping[expected] = col
                            found = True
                            break
                    
                    if not found:
                        # Fallback: beste Übereinstimmung finden
                        for col in df_columns:
                            if expected.lower().replace('-', '').replace(' ', '') in col.lower().replace('-', '').replace(' ', ''):
                                column_mapping[expected] = col
                                break
            
            # Fehlende Spalten prüfen
            missing_cols = [col for col in expected_columns if col not in column_mapping]
            if missing_cols:
                st.warning(f"Fehlende Spalten werden mit Platzhaltern gefüllt: {missing_cols}")
                for col in missing_cols:
                    df[col] = 'N/A'
                    column_mapping[col] = col
            
            # DataFrame mit korrekten Spaltennamen erstellen
            processed_df = pd.DataFrame()
            for expected, actual in column_mapping.items():
                processed_df[expected] = df[actual]
            
            # Datum und Zeit kombinieren falls separate Spalten
            if 'Datum' in processed_df.columns and 'Zeit' in processed_df.columns:
                try:
                    # Kombiniere Datum und Zeit
                    datetime_combined = pd.to_datetime(
                        processed_df['Datum'].astype(str) + ' ' + processed_df['Zeit'].astype(str),
                        errors='coerce'
                    )
                    processed_df['DateTime_Combined'] = datetime_combined
                except Exception as e:
                    logger.warning(f"Datum/Zeit-Kombination fehlgeschlagen: {e}")
            elif 'Datum' in processed_df.columns:
                try:
                    processed_df['Datum'] = pd.to_datetime(
                        processed_df['Datum'], 
                        errors='coerce'
                    )
                except Exception as e:
                    logger.warning(f"Datum-Parsing fehlgeschlagen: {e}")
            
            # Leere Einträge behandeln
            for col in expected_columns:
                if col in processed_df.columns:
                    processed_df[col] = processed_df[col].fillna('').astype(str)
            
            # Leere Ereignis-Einträge entfernen
            if 'Ereignis & Massnahme' in processed_df.columns:
                processed_df = processed_df[processed_df['Ereignis & Massnahme'].str.strip() != '']
            
            logger.info(f"Daten verarbeitet: {len(processed_df)} Einträge")
            return processed_df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Fehler bei der Datenverarbeitung: {e}")
            st.error(f"Fehler bei der Datenverarbeitung: {e}")
            return None
    
    def create_embeddings(self, df: pd.DataFrame) -> bool:
        """Erstellt Embeddings für alle Einträge"""
        try:
            with st.spinner('Erstelle Embeddings...'):
                # Kombiniere relevante Felder für bessere Suche
                texts = []
                for _, row in df.iterrows():
                    # Verwende Ihre Logbuch-Struktur
                    parts = []
                    if 'Datum' in row and pd.notna(row['Datum']):
                        parts.append(f"Datum: {row['Datum']}")
                    if 'Zeit' in row and pd.notna(row['Zeit']):
                        parts.append(f"Zeit: {row['Zeit']}")
                    if 'Lot-Nr.' in row and row['Lot-Nr.']:
                        parts.append(f"Lot: {row['Lot-Nr.']}")
                    if 'Subsystem' in row and row['Subsystem']:
                        parts.append(f"System: {row['Subsystem']}")
                    if 'Ereignis & Massnahme' in row and row['Ereignis & Massnahme']:
                        parts.append(f"Ereignis: {row['Ereignis & Massnahme']}")
                    if 'Visum' in row and row['Visum']:
                        parts.append(f"Visum: {row['Visum']}")
                    
                    combined_text = " ".join(parts)
                    texts.append(combined_text)
                
                # Embeddings erstellen
                self.embeddings = self.model.encode(
                    texts, 
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                
                # FAISS Index erstellen
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # Inner Product für Cosine Similarity
                
                # Normalisiere Embeddings für Cosine Similarity
                faiss.normalize_L2(self.embeddings)
                self.index.add(self.embeddings)
                
                self.df = df
                logger.info(f"Embeddings erstellt: {len(texts)} Texte, Dimension: {dimension}")
                return True
                
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Embeddings: {e}")
            st.error(f"Fehler beim Erstellen der Embeddings: {e}")
            return False
    
    def determine_optimal_results(self, scores: List[float], query: str) -> int:
        """Bestimmt automatisch die optimale Anzahl von Ergebnissen basierend auf Relevanz"""
        if not scores or len(scores) == 0:
            return 0
        
        scores = np.array(scores)
        
        # Methode 1: Signifikante Ähnlichkeitsschwelle
        high_relevance_threshold = 0.7
        medium_relevance_threshold = 0.5
        
        high_relevant = int(np.sum(scores >= high_relevance_threshold))
        medium_relevant = int(np.sum(scores >= medium_relevance_threshold))
        
        # Methode 2: Relativer Abfall in der Ähnlichkeit
        if len(scores) > 1:
            score_diffs = np.diff(scores)
            # Finde große Sprünge in der Ähnlichkeit
            if len(score_diffs) > 0:
                mean_diff = float(np.mean(score_diffs))
                std_diff = float(np.std(score_diffs))
                significant_drop = mean_diff - 1.5 * std_diff  # Negativer Wert für Abfall
                
                drop_indices = np.where(score_diffs < significant_drop)[0]
                if len(drop_indices) > 0:
                    natural_cutoff = int(drop_indices[0]) + 1
                else:
                    natural_cutoff = len(scores)
            else:
                natural_cutoff = len(scores)
        else:
            natural_cutoff = len(scores)
        
        # Methode 3: Query-Komplexität berücksichtigen
        query_words = len(query.split())
        if query_words <= 2:
            # Einfache Queries: weniger Ergebnisse
            complexity_factor = 0.7
        elif query_words <= 5:
            # Mittlere Komplexität
            complexity_factor = 1.0
        else:
            # Komplexe Queries: mehr Ergebnisse
            complexity_factor = 1.3
        
        # Kombiniere alle Methoden
        if high_relevant >= 3:
            # Viele hochrelevante Ergebnisse
            optimal_count = min(high_relevant + 2, 15)
        elif medium_relevant >= 5:
            # Einige mittelrelevante Ergebnisse
            optimal_count = min(medium_relevant, 12)
        else:
            # Wenige relevante Ergebnisse, nutze natürlichen Cutoff
            optimal_count = min(natural_cutoff, 8)
        
        # Anpassung basierend auf Query-Komplexität
        optimal_count = int(optimal_count * complexity_factor)
        
        # Grenzen einhalten
        optimal_count = max(3, min(optimal_count, 20))
        
        logger.info(f"Automatische Ergebnisanzahl: {optimal_count} (von {len(scores)} verfügbar)")
        return optimal_count
    def semantic_search(self, query: str, max_results: int = 50) -> Tuple[List[int], List[float], int]:
        """Führt semantische Suche durch und bestimmt optimale Ergebnisanzahl"""
        try:
            # Validierung
            if self.index is None or self.model is None:
                logger.error("Index oder Modell nicht verfügbar")
                return [], [], 0
            
            # Query Embedding
            query_embedding = self.model.encode([f"Suche: {query}"], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Begrenze max_results auf verfügbare Daten
            available_count = self.index.ntotal
            actual_max = min(max_results, available_count)
            
            # Suche mit mehr Ergebnissen für Analyse
            scores, indices = self.index.search(query_embedding, actual_max)
            
            # Konvertiere zu Python-Listen für sicherere Verarbeitung
            scores_list = [float(score) for score in scores[0]]
            indices_list = [int(idx) for idx in indices[0]]
            
            # Bestimme optimale Anzahl
            optimal_count = self.determine_optimal_results(scores_list, query)
            
            # Stelle sicher, dass optimal_count nicht größer als verfügbare Ergebnisse ist
            optimal_count = min(optimal_count, len(scores_list))
            
            return indices_list[:optimal_count], scores_list[:optimal_count], optimal_count
            
        except Exception as e:
            logger.error(f"Fehler bei der semantischen Suche: {e}")
            return [], [], 0
    
    def query_ollama(self, prompt: str, context: str, model: str = None, timeout: int = 60) -> str:
        """Sendet Anfrage an lokales Ollama LLM"""
        if model is None:
            model = self.ollama_model
            
        try:
            # Prüfe verschiedene Ollama Endpunkte
            possible_urls = [
                "http://localhost:11434/api/generate",
                "http://127.0.0.1:11434/api/generate"
            ]
            
            system_prompt = """Du bist ein Experte für die Analyse von Logbuch-Einträgen. 
            Analysiere die gegebenen Daten sorgfältig und beantworte die Frage präzise und strukturiert. 
            Verwende nur Informationen aus den bereitgestellten Daten.
            Antworte auf Deutsch und strukturiere deine Antwort klar."""
            
            full_prompt = f"{system_prompt}\n\nKontext-Daten:\n{context}\n\nFrage: {prompt}\n\nAntwort:"
            
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_ctx": 4096
                }
            }
            
            # Versuche verschiedene URLs
            for url in possible_urls:
                try:
                    response = requests.post(url, json=payload, timeout=timeout)
                    
                    if response.status_code == 200:
                        result = response.json()
                        return result.get('response', 'Keine Antwort erhalten')
                    else:
                        continue
                        
                except requests.exceptions.RequestException:
                    continue
            
            # Alle URLs fehlgeschlagen - verwende Fallback-Analyse
            return self.fallback_analysis(prompt, context)
                
        except Exception as e:
            logger.error(f"Fehler bei Ollama-Anfrage: {e}")
            return self.fallback_analysis(prompt, context)
    
    def fallback_analysis(self, prompt: str, context: str) -> str:
        """Fallback-Analyse ohne LLM"""
        lines = context.split('\n')
        entry_count = len([line for line in lines if line.startswith('Eintrag')])
        
        return f"""📊 **Automatische Analyse** (Ollama nicht verfügbar)

**Gefundene Einträge:** {entry_count}

**Zusammenfassung der Suchergebnisse:**
{context[:500]}{'...' if len(context) > 500 else ''}

💡 **Hinweis:** Für detaillierte KI-Analyse starten Sie Ollama mit: `ollama serve`
"""
    
    def analyze_query(self, query: str) -> dict:
        """Hauptfunktion für die intelligente Analyse"""
        if self.df is None or self.index is None:
            return {"error": "Keine Daten geladen"}
        
        try:
            logger.info(f"Starte Analyse für Query: '{query}'")
            
            # Intelligente semantische Suche
            indices, scores, result_count = self.semantic_search(query)
            
            if not indices or result_count == 0:
                return {"error": "Keine relevanten Ergebnisse gefunden"}
            
            logger.info(f"Gefunden: {result_count} relevante Einträge")
            
            # Relevante Einträge extrahieren
            relevant_entries = self.df.iloc[indices].copy()
            relevant_entries['similarity_score'] = scores
            
            # Kontext für LLM vorbereiten
            context_parts = []
            for i, (_, row) in enumerate(relevant_entries.iterrows(), 1):
                context_parts.append(
                    f"Eintrag {i}:\n"
                    f"- Datum: {row.get('Datum', 'N/A')}\n"
                    f"- Zeit: {row.get('Zeit', 'N/A')}\n"
                    f"- Lot-Nr.: {row.get('Lot-Nr.', 'N/A')}\n"
                    f"- Subsystem: {row.get('Subsystem', 'N/A')}\n"
                    f"- Ereignis & Massnahme: {row.get('Ereignis & Massnahme', 'N/A')}\n"
                    f"- Visum: {row.get('Visum', 'N/A')}\n"
                    f"- Relevanz: {scores[i-1]:.3f}\n"
                )
            
            context = "\n".join(context_parts)
            
            result = {
                "relevant_entries": relevant_entries,
                "context": context,
                "query": query,
                "result_count": result_count,
                "total_available": len(self.df)
            }
            
            # Automatische LLM-Analyse
            logger.info("Starte LLM-Analyse...")
            llm_response = self.query_ollama(query, context)
            result["llm_analysis"] = llm_response
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei der Analyse: {e}")
            return {"error": f"Fehler bei der Analyse: {str(e)}"}

def main():
    st.set_page_config(
        page_title="Logbuch Analyzer", 
        page_icon="📋", 
        layout="wide"
    )
    
    st.title("📋 Logbuch Analyzer")
    st.markdown("**Intelligente KI-gestützte Suche und Analyse von Logbuch-Einträgen**")
    
    # Session State initialisieren
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = LogbookAnalyzer()
        st.session_state.initialized = False
    
    analyzer = st.session_state.analyzer
    
    # Auto-Initialisierung beim ersten Start
    if not st.session_state.initialized:
        st.info("🚀 Initialisiere Anwendung...")
        success = analyzer.auto_initialize()
        st.session_state.initialized = True
        if success:
            st.rerun()  # Refresh UI nach Initialisierung
    
    # Sidebar für System-Info und Logs
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Embedding-Modell Status
        if analyzer.model is not None:
            st.success("✅ Embedding-Modell: Aktiv")
        else:
            st.error("❌ Embedding-Modell: Nicht geladen")
        
        # Ollama Status
        st.write(f"🦙 **Ollama:** {analyzer.ollama_status}")
        
        # Daten Status
        if analyzer.df is not None:
            st.success(f"📊 Daten: {len(analyzer.df)} Einträge")
        else:
            st.info("📊 Daten: Nicht geladen")
        
        st.markdown("---")
        
        # System Logs
        st.subheader("📝 System Logs")
        if hasattr(st.session_state, 'log_handler'):
            logs = st.session_state.log_handler.logs[-10:]  # Nur die letzten 10
            if logs:
                log_container = st.container()
                with log_container:
                    for log in logs:
                        st.text(log)
            else:
                st.text("Keine Logs verfügbar")
        
        # Refresh-Button
        if st.button("🔄 System neu prüfen"):
            analyzer.check_ollama_status()
            st.rerun()
        
        st.markdown("---")
        st.markdown("**💡 Tipps:**")
        st.markdown("- Ollama Status: http://localhost:11434")
        st.markdown("- Automatische Ergebnisoptimierung")
        st.markdown("- KI-Analyse immer aktiv")
    
    # Hauptbereich
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📤 Daten Upload")
        
        uploaded_file = st.file_uploader(
            "CSV-Datei hochladen",
            type=['csv'],
            help="Erwartete Spalten: Datum, Zeit, Lot-Nr., Subsystem, Ereignis & Massnahme, Visum"
        )
        
        if uploaded_file is not None:
            try:
                # CSV mit verschiedenen Trennzeichen versuchen
                separators = [',', ';', ' ', '\t']
                df = None
                successful_separator = None
                
                for sep in separators:
                    try:
                        uploaded_file.seek(0)  # Zurück zum Anfang
                        test_df = pd.read_csv(uploaded_file, sep=sep, encoding='utf-8')
                        
                        # Prüfe ob DataFrame sinnvoll ist (mehr als 1 Spalte)
                        if len(test_df.columns) > 1:
                            df = test_df
                            successful_separator = sep
                            break
                    except:
                        continue
                
                # Falls UTF-8 fehlschlägt, versuche andere Encodings
                if df is None:
                    encodings = ['latin1', 'cp1252', 'iso-8859-1']
                    for encoding in encodings:
                        for sep in separators:
                            try:
                                uploaded_file.seek(0)
                                test_df = pd.read_csv(uploaded_file, sep=sep, encoding=encoding)
                                if len(test_df.columns) > 1:
                                    df = test_df
                                    successful_separator = sep
                                    st.info(f"Datei mit Encoding '{encoding}' und Trennzeichen '{sep}' gelesen")
                                    break
                            except:
                                continue
                        if df is not None:
                            break
                
                if df is not None:
                    st.success(f"✅ {len(df)} Zeilen geladen (Trennzeichen: '{successful_separator}')")
                
                # Datenvorschau
                with st.expander("📊 Datenvorschau"):
                    st.dataframe(df.head())
                    st.write(f"Spalten: {list(df.columns)}")
                
                # Daten verarbeiten
                if st.button("🔄 Daten verarbeiten"):
                    processed_df = analyzer.preprocess_data(df)
                    if processed_df is not None:
                        st.success(f"✅ {len(processed_df)} Einträge verarbeitet")
                        
                        # Embeddings erstellen
                        if analyzer.model is not None:
                            if analyzer.create_embeddings(processed_df):
                                st.success("✅ Embeddings erstellt - Bereit für intelligente Suche!")
                                logger.info("System bereit für Analysen")
                        else:
                            st.warning("⚠️ Embedding-Modell nicht verfügbar")
                            
            except Exception as e:
                st.error(f"Fehler beim Laden der CSV: {e}")
                st.info("💡 Tipp: Prüfen Sie Trennzeichen (Komma, Semikolon) und Encoding (UTF-8, Latin1)")
    
    with col2:
        st.header("🔍 Intelligente Suche & KI-Analyse")
        
        if analyzer.df is not None:
            st.success(f"📊 {len(analyzer.df)} Einträge bereit für Analyse")
            
            # Suchbereich
            query = st.text_area(
                "Ihre Frage/Suchanfrage:",
                placeholder="z.B. 'Alle Probleme mit Lot 12345' oder 'Wann gab es Temperaturprobleme in der Heizung?'",
                height=100
            )
            
            if st.button("🧠 Intelligente Analyse starten", type="primary") and query:
                with st.spinner("🔍 Suche läuft... (Ergebnisanzahl wird automatisch optimiert)"):
                    results = analyzer.analyze_query(query)
                    
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        # Ergebnis-Header mit Statistiken
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("📊 Gefundene Einträge", results["result_count"])
                        with col_stat2:
                            st.metric("📈 Gesamt verfügbar", results["total_available"])
                        with col_stat3:
                            if len(results["relevant_entries"]) > 0:
                                avg_score = float(np.mean(results["relevant_entries"]["similarity_score"]))
                                st.metric("🎯 Ø Relevanz", f"{avg_score:.2f}")
                            else:
                                st.metric("🎯 Ø Relevanz", "N/A")
                        
                        st.markdown("---")
                        
                        # KI-Analyse (immer aktiv)
                        st.subheader("🤖 KI-Analyse")
                        if "llm_analysis" in results:
                            st.write(results["llm_analysis"])
                        else:
                            st.error("KI-Analyse nicht verfügbar")
                        
                        st.markdown("---")
                        
                        # Detaillierte Suchergebnisse
                        st.subheader("📋 Detaillierte Ergebnisse")
                        st.caption(f"Automatisch optimiert: {results['result_count']} relevanteste Einträge angezeigt")
                        
                        relevant_df = results["relevant_entries"]
                        
                        for i, (_, row) in enumerate(relevant_df.iterrows(), 1):
                            # Relevanz-basierte Farbe
                            score = float(row['similarity_score'])
                            if score >= 0.7:
                                score_color = "🟢"  # Hoch relevant
                            elif score >= 0.5:
                                score_color = "🟡"  # Mittel relevant
                            else:
                                score_color = "🟠"  # Niedrig relevant
                            
                            with st.expander(f"{score_color} Eintrag {i} (Relevanz: {score:.3f})"):
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.write(f"**Datum:** {row.get('Datum', 'N/A')}")
                                    st.write(f"**Zeit:** {row.get('Zeit', 'N/A')}")
                                with col_b:
                                    st.write(f"**Lot-Nr.:** {row.get('Lot-Nr.', 'N/A')}")
                                    st.write(f"**Subsystem:** {row.get('Subsystem', 'N/A')}")
                                with col_c:
                                    st.write(f"**Visum:** {row.get('Visum', 'N/A')}")
                                    st.write(f"**Relevanz:** {score_color} {score:.3f}")
                                
                                st.write(f"**Ereignis & Maßnahme:** {row.get('Ereignis & Massnahme', 'N/A')}")
                        
                        # Download-Option
                        csv = relevant_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Ergebnisse als CSV herunterladen",
                            data=csv,
                            file_name=f"suchergebnisse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        
        else:
            st.info("👆 Bitte laden Sie zuerst eine CSV-Datei hoch und verarbeiten Sie die Daten")
            
            # Beispiel-CSV anzeigen
            with st.expander("📄 Beispiel CSV-Format"):
                example_data = {
                    'Datum': ['2024-01-15', '2024-01-15', '2024-01-16'],
                    'Zeit': ['14:30', '15:45', '09:15'],
                    'Lot-Nr.': ['LOT-001', 'LOT-002', 'LOT-003'],
                    'Subsystem': ['Pumpe', 'Heizung', 'Ventil'],
                    'Ereignis & Massnahme': [
                        'Druckabfall erkannt - Ventil gereinigt',
                        'Temperatur zu niedrig - Thermostat justiert',
                        'Ungewöhnliche Geräusche - Wartung durchgeführt'
                    ],
                    'Visum': ['MK', 'AB', 'CD']
                }
                example_df = pd.DataFrame(example_data)
                st.dataframe(example_df)

if __name__ == "__main__":
    main()