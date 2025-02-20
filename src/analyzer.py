# written by 0xSolanaceae

import re
import logging
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from typing import Dict
from utils import safe_parse_timestamp, calculate_entropy, load_parsing_patterns


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class LogAnalyzer:
    """Advanced Machine Learning-powered log analysis engine"""

    MAX_SAMPLE_LINES = 1000
    CHUNK_SIZE = 10000
    
    def __init__(self):
        logging.info("Initializing LogAnalyzer")
        self.dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
        self.LOG_TYPES = [os.path.splitext(f)[0] for f in os.listdir(self.dataset_dir) if f.endswith(".log")]
        self.log_type_model = self._init_log_type_model()
        self.analysis_pipeline = self._build_analysis_pipeline()
        self._compiled_patterns = load_parsing_patterns()

    def _init_log_type_model(self) -> OneVsRestClassifier:
        logging.info("Initializing log type model from training files")
        # Build training_files dynamically from dataset_dir and LOG_TYPES
        training_files = {
            log_type: os.path.join(self.dataset_dir, f"{log_type}.log")
            for log_type in self.LOG_TYPES
        }
        training_data = {}
        for log_type, file_path in training_files.items():
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    training_data[log_type] = [line.strip() for line in f if line.strip()]
            else:
                logging.warning("Training file for %s not found at %s", log_type, file_path)
                training_data[log_type] = []
        
        texts, labels = [], []
        for log_type, examples in training_data.items():
            texts.extend(examples)
            labels.extend([log_type] * len(examples))
        
        logging.info("Training log type model with %d examples", len(texts))
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), max_features=2000),
            OneVsRestClassifier(LogisticRegression(max_iter=2000, n_jobs=-1, class_weight='balanced'))
        ).fit(texts, labels)

    def parse_logs(self, log_file: str) -> pd.DataFrame:
        logging.info("Starting log parsing for file: %s", log_file)
        entries = []
        format_counts = {p: 0 for p in self.LOG_TYPES}
        format_counts['unknown'] = 0

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.read().splitlines()

        # Process in chunks
        for i in range(0, len(lines), self.CHUNK_SIZE):
            chunk = lines[i:i+self.CHUNK_SIZE]
            for line in chunk:
                line = line.strip()
                parsed = False

                # Try precompiled patterns first
                for pattern, fields in self._compiled_patterns:
                    if match := pattern.match(line):
                        logging.info("Matched pattern for line: %s", line)
                        entry = dict(zip(fields, match.groups()))
                        entry['timestamp'] = safe_parse_timestamp(entry.get('timestamp', ''))
                        entries.append(entry)
                        parsed = True
                        break

                # Fallback to ML-based parsing
                if not parsed:
                    proba = self.log_type_model.predict_proba([line])[0]
                    detected = self.LOG_TYPES[np.argmax(proba)]
                    format_counts[detected] += 1
                    logging.info("Fallback parsing for line: %s as type: %s", line, detected)
                    entries.append(self._parse_fallback(line, detected))

        logging.info("Parsed %d log entries", len(entries))
        logging.info("Format distribution: %s", format_counts)
        return pd.DataFrame(entries).fillna({
            'host': 'unknown',
            'level': 'UNKNOWN',
            'message': ''
        }).pipe(self._postprocess_df)

    def _parse_fallback(self, line: str, detected_type: str) -> Dict:
        logging.info("Using fallback parse for type: %s and line: %s", detected_type, line)
        type_patterns = {
            'apache': (r'^(\S+) (\S+) (\S+) \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\d+)$', 
                    ['ip', 'client_id', 'user_id', 'timestamp', 'method', 'url', 'protocol', 'status', 'size']),
            'syslog': (r'^(\w{3}\s+\d{1,2} \d{2}:\d{2}:\d{2}) (\S+) (\S+)\[(\d+)\]: (.*)$', 
                    ['timestamp', 'host', 'app', 'pid', 'message'])
        }

        if detected_type in type_patterns:
            pattern, fields = type_patterns[detected_type]
            if match := re.match(pattern, line):
                entry = dict(zip(fields, match.groups()))
                # For Apache logs, set 'message' using other fields if missing.
                if detected_type == 'apache' and 'message' not in entry:
                    entry['message'] = f"{entry.get('method', '')} {entry.get('url', '')} {entry.get('protocol', '')}"
                entry['timestamp'] = safe_parse_timestamp(entry.get('timestamp', ''))
                return entry

        logging.info("Default fallback parsing for line: %s", line)
        # Default fallback for unknown formats
        return {
            'timestamp': safe_parse_timestamp(line[:30]),
            'host': detected_type,
            'level': 'UNKNOWN',
            'message': line
        }

    def _postprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df['warning_count'] = df['message'].str.count(r'(?i)warning')
        df['http_method'] = df['message'].str.extract(r'^(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\b')
        df['msg_length'] = df['message'].str.len()
        df['error_count'] = df['message'].str.count(r'(?i)error|exception|fail|timeout')
        df['http_status'] = df['message'].str.extract(r'\b(\d{3})\b').astype(float)
        
        df['timestamp_delta'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
        df['status_5xx'] = df['http_status'].between(500, 599).astype(int)
        df['unique_terms'] = df['message'].apply(lambda x: len(set(x.split())))
        df['entropy'] = df['message'].apply(calculate_entropy)
        df['suspect_ua'] = df['message'].str.contains(r'(?:curl|wget|nikto|sqlmap)', case=False)
        
        return df

    def _build_analysis_pipeline(self):
        logging.info("Building analysis pipeline")
        numeric_features = ['msg_length', 'error_count', 'warning_count', 'http_status']
        categorical_features = ['http_method', 'level']

        pipeline = make_pipeline(
            ColumnTransformer([
                ('numeric', make_pipeline(
                    SimpleImputer(strategy='median'),
                    StandardScaler()
                ), numeric_features),
                ('categorical', make_pipeline(
                    SimpleImputer(strategy='constant', fill_value='missing'),
                    OneHotEncoder(handle_unknown='ignore')
                ), categorical_features),
                ('text', TfidfVectorizer(max_features=1000), 'message')
            ]),
            IsolationForest(
                contamination=0.05, 
                random_state=42, 
                n_jobs=-1,
                verbose=0
            )
        )
        logging.info("Analysis pipeline built")
        return pipeline

    def analyze(self, log_file: str) -> Dict:
        logging.info("Starting full analysis on file: %s", log_file)
        try:
            return self.anomaly_detection(log_file)
        except Exception as e:
            logging.error("Analysis failed: %s", str(e))
            raise

    def anomaly_detection(self, log_file):
        # Stage 1: Log type detection
        type_confidences = self.detect_log_types(log_file)
        logging.info("Log type confidences: %s", type_confidences)

        # Stage 2: Memory-efficient parsing
        logs = self.parse_logs(log_file)
        if logs.empty:
            raise ValueError("No parsable log entries found")
        logging.info("Log parsing completed. Records: %d", len(logs))

        # Stage 3: Anomaly detection
        logging.info("Fitting analysis pipeline for anomaly detection")
        self.analysis_pipeline.fit(logs)
        logs['anomaly_score'] = self.analysis_pipeline.decision_function(logs)
        logs['anomaly'] = self.analysis_pipeline.predict(logs)
        logging.info("Anomaly detection complete. Found anomalies: %d", len(logs[logs['anomaly'] == -1]))

        return {
            'type_confidences': type_confidences,
            'anomalies': logs[logs['anomaly'] == -1],
            'stats': self._compute_statistics(logs)
        }

    def _compute_statistics(self, logs: pd.DataFrame) -> Dict:
        logging.info("Computing statistics from logs")
        stats = {
            'total_entries': len(logs),
            'error_rate': logs['error_count'].sum() / len(logs),
            'common_hosts': logs['host'].value_counts().head(3).to_dict(),
            'anomaly_distribution': logs['anomaly'].value_counts().to_dict(),
            'http_status_distribution': logs['http_status'].value_counts().to_dict()
        }
        
        # Temporal analysis
        if not logs['timestamp'].isna().all():
            time_bins = pd.cut(logs['timestamp'], bins=10)
            stats['temporal_distribution'] = time_bins.value_counts().sort_index().to_dict()
        logging.info("Statistics computed: %s", stats)
        return stats
    
    def detect_log_types(self, log_file: str) -> Dict:
        logging.info("Detecting log types for file: %s", log_file)
        try:
            return self.dataframe_generation(log_file)
        except Exception as e:
            logging.error("Detection failed: %s", str(e))
            return {}

    def dataframe_generation(self, log_file):
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            sample = [
                line.strip()
                for line, _ in zip(f, range(self.MAX_SAMPLE_LINES))
                if line.strip()
            ]
        if not sample:
            logging.warning("Empty log file detected")
            return {}

        probas = self.log_type_model.predict_proba(sample)
        # Build a DataFrame using the classes learned by the model
        classes = self.log_type_model.classes_
        df_probas = pd.DataFrame(probas, columns=classes)
        # Reindex to ensure all types in self.LOG_TYPES are present, filling missing with 0
        df_probas = df_probas.reindex(columns=self.LOG_TYPES, fill_value=0)
        results = df_probas.mean().to_dict()
        logging.info("Log types detected: %s", results)
        return results