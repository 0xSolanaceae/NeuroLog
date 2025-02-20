# written by 0xSolanaceae

import re
import logging
import os
import pickle
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
        self.crf_model = self._load_crf_model()

    def _load_crf_model(self, model_path='crf_model.pkl'):
        """Load pre-trained CRF model from file"""
        try:
            with open(model_path, 'rb') as f:
                logging.info(f"Loaded CRF model from {model_path}")
                return pickle.load(f)
        except FileNotFoundError:
            logging.warning(f"CRF model not found at {model_path}. Using regex fallback only.")
            return None
        except Exception as e:
            logging.error(f"Error loading CRF model: {str(e)}")
            return None

    def _init_log_type_model(self) -> OneVsRestClassifier:
        logging.info("Initializing log type model from training files")
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

        for i in range(0, len(lines), self.CHUNK_SIZE):
            chunk = lines[i:i+self.CHUNK_SIZE]
            for line in chunk:
                line = line.strip()
                parsed = False

                for pattern, fields in self._compiled_patterns:
                    if match := pattern.match(line):
                        entry = dict(zip(fields, match.groups()))
                        entry['timestamp'] = safe_parse_timestamp(entry.get('timestamp', ''))
                        entries.append(entry)
                        parsed = True
                        break

                if not parsed:
                    proba = self.log_type_model.predict_proba([line])[0]
                    detected = self.LOG_TYPES[np.argmax(proba)]
                    format_counts[detected] += 1
                    entries.append(self._parse_fallback(line, detected))

        logging.info("Parsed %d log entries", len(entries))
        return pd.DataFrame(entries).fillna({
            'host': 'unknown',
            'level': 'UNKNOWN',
            'message': ''
        }).pipe(self._postprocess_df)

    def _parse_fallback(self, line: str, detected_type: str) -> Dict:
        """Parse line using CRF model or basic fallback"""
        return self._crf_parse(line) if self.crf_model else self._basic_fallback(line)
    def _crf_parse(self, line: str) -> Dict:
        """Parse line using CRF model"""
        tokens = self._tokenize_with_spans(line)
        if not tokens:
            return self._basic_fallback(line)
        
        features = self._extract_features([t[0] for t in tokens])
        labels = self.crf_model.predict_single(features)
        
        entry = {}
        current_field = None
        current_values = []
        
        for token, label in zip([t[0] for t in tokens], labels):
            if label == 'O':
                if current_field:
                    entry[current_field] = ' '.join(current_values)
                    current_field = None
                    current_values = []
                continue
                
            if label != current_field:
                if current_field:
                    entry[current_field] = ' '.join(current_values)
                current_field = label
                current_values = [token]
            else:
                current_values.append(token)
        
        if current_field:
            entry[current_field] = ' '.join(current_values)
        
        entry.setdefault('timestamp', '')
        entry['timestamp'] = safe_parse_timestamp(entry['timestamp'])
        entry.setdefault('host', 'unknown')
        entry.setdefault('level', 'UNKNOWN')
        entry.setdefault('message', line)
        return entry

    @staticmethod
    def _tokenize_with_spans(line):
        """Tokenize line with position tracking"""
        tokens = []
        for match in re.finditer(r'(\S+|\".*?\")', line):
            start, end = match.start(), match.end()
            token = match.group()
            tokens.append((token, start, end))
        return tokens

    @staticmethod
    def _extract_features(tokens):
        """Generate CRF features for tokens"""
        features = []
        for i, token in enumerate(tokens):
            feat = {
                'word': token,
                'lower': token.lower(),
                'isupper': token.isupper(),
                'islower': token.islower(),
                'isdigit': token.isdigit(),
                'prefix3': token[:3],
                'suffix3': token[-3:],
                'position': i,
                'length': len(token),
            }
            if i > 0:
                feat['prev_word'] = tokens[i-1]
            if i < len(tokens)-1:
                feat['next_word'] = tokens[i+1]
            features.append(feat)
        return features

    def _basic_fallback(self, line: str) -> Dict:
        """Basic regex fallback for failed CRF parsing"""
        return {
            'timestamp': safe_parse_timestamp(line[:30]),
            'host': 'unknown',
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

        return make_pipeline(
            ColumnTransformer(
                [
                    (
                        'numeric',
                        make_pipeline(
                            SimpleImputer(strategy='median'), StandardScaler()
                        ),
                        numeric_features,
                    ),
                    (
                        'categorical',
                        make_pipeline(
                            SimpleImputer(
                                strategy='constant', fill_value='missing'
                            ),
                            OneHotEncoder(handle_unknown='ignore'),
                        ),
                        categorical_features,
                    ),
                    ('text', TfidfVectorizer(max_features=1000), 'message'),
                ]
            ),
            IsolationForest(
                contamination=0.05, random_state=42, n_jobs=-1, verbose=0
            ),
        )

    def analyze(self, log_file: str) -> Dict:
        logging.info("Starting full analysis on file: %s", log_file)
        try:
            return self.anomaly_detection(log_file)
        except Exception as e:
            logging.error("Analysis failed: %s", str(e))
            raise

    def anomaly_detection(self, log_file):
        type_confidences = self.detect_log_types(log_file)
        logging.info("Log type confidences: %s", type_confidences)

        logs = self.parse_logs(log_file)
        if logs.empty:
            raise ValueError("No parsable log entries found")
        
        logging.info("Fitting analysis pipeline")
        self.analysis_pipeline.fit(logs)
        logs['anomaly_score'] = self.analysis_pipeline.decision_function(logs)
        logs['anomaly'] = self.analysis_pipeline.predict(logs)
        
        return {
            'type_confidences': type_confidences,
            'anomalies': logs[logs['anomaly'] == -1],
            'stats': self._compute_statistics(logs)
        }

    def _compute_statistics(self, logs: pd.DataFrame) -> Dict:
        stats = {
            'total_entries': len(logs),
            'error_rate': logs['error_count'].sum() / len(logs),
            'common_hosts': logs['host'].value_counts().head(3).to_dict(),
            'anomaly_distribution': logs['anomaly'].value_counts().to_dict(),
            'http_status_distribution': logs['http_status'].value_counts().to_dict()
        }
        
        if not logs['timestamp'].isna().all():
            time_bins = pd.cut(logs['timestamp'], bins=10)
            stats['temporal_distribution'] = time_bins.value_counts().sort_index().to_dict()
        return stats
    
    def detect_log_types(self, log_file: str) -> Dict:
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
            return {}

        probas = self.log_type_model.predict_proba(sample)
        classes = self.log_type_model.classes_
        df_probas = pd.DataFrame(probas, columns=classes)
        df_probas = df_probas.reindex(columns=self.LOG_TYPES, fill_value=0)
        return df_probas.mean().to_dict()