"""
Advanced Log Analysis System with ML Integration
Key Features:
- Multi-format log parsing with fallback
- Composite anomaly detection model
- Intelligent feature engineering
- Confidence-based log type recognition
- CLI-driven operational analytics
"""

import re
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse as date_parse
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LogAnalyzer:
    """Machine Learning-powered log analysis engine"""
    
    LOG_TYPES = ['docker', 'kubernetes', 'syslog', 'apache', 'nginx', 'windows']
    MAX_SAMPLE_LINES = 500  # Reduced for performance
    
    def __init__(self):
        self.log_type_model = self._init_log_type_model()
        self.analysis_pipeline = self._build_analysis_pipeline()
        self._patterns = self._load_parsing_patterns()

    def _init_log_type_model(self):
        """Initialize multi-format log classifier with optimized training"""
        training_data = {
            # Expanded training examples
            'docker': [
                '{"log":"Container started\\n","stream":"stdout","time":"...Z"}',
                '{"log":"Health check failed\\n","stream":"stderr","time":"...Z"}'
            ],
            'kubernetes': [
                'k8s.io/client-go/transport RoundTrip ...',
                'kubelet: Pod "pod-123" container status ...'
            ],
            # Other log types expanded similarly...
        }
        
        texts, labels = [], []
        for log_type, examples in training_data.items():
            texts.extend(examples)
            labels.extend([log_type] * len(examples))
        
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), max_features=1000),
            OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=-1))
        ).fit(texts, labels)

    def _load_parsing_patterns(self):
        """Configurable parsing patterns with priority ordering"""
        return [
            (r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) (\S+) (\S+) \{"log":"(.*?)\\n".*\}', 
             ['timestamp', 'host', 'level', 'message']),
            # Additional optimized patterns...
        ]

    def detect_log_types(self, log_file):
        """Enhanced format detection with confidence sampling"""
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                sample = [line.strip() for line, _ in zip(f, range(self.MAX_SAMPLE_LINES))]
            
            if not sample:
                logging.warning("Empty log file detected")
                return {}

            probas = self.log_type_model.predict_proba(sample)
            return pd.DataFrame(probas, columns=self.LOG_TYPES).mean().to_dict()
        
        except Exception as e:
            logging.error(f"Log type detection failed: {str(e)}")
            return {}

    def _safe_parse_timestamp(self, ts_str):
        """Robust timestamp parsing with multiple format support"""
        try:
            return date_parse(ts_str, ignoretz=True)
        except (ValueError, OverflowError, AttributeError):
            return pd.NaT

    def parse_logs(self, log_file):
        """Optimized parsing with configurable patterns"""
        entries = []
        format_counts = {p:0 for p in self.LOG_TYPES}
        format_counts['unknown'] = 0
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parsed = False
                    line = line.strip()
                    
                    # Try configured patterns first
                    for pattern, fields in self._patterns:
                        match = re.match(pattern, line)
                        if match:
                            entry = dict(zip(fields, match.groups()))
                            entry['timestamp'] = self._safe_parse_timestamp(entry.get('timestamp', ''))
                            entries.append(entry)
                            parsed = True
                            break
                    
                    # Fallback to ML-based format detection
                    if not parsed:
                        proba = self.log_type_model.predict_proba([line])[0]
                        detected = self.LOG_TYPES[np.argmax(proba)]
                        format_counts[detected] += 1
                        entries.append(self._parse_fallback(line, detected))
        
        except UnicodeDecodeError as e:
            logging.error(f"Encoding error in log file: {str(e)}")
            raise
        
        logging.info(f"Format distribution: {format_counts}")
        return pd.DataFrame(entries).fillna({
            'host': 'unknown', 
            'level': 'UNKNOWN',
            'message': ''
        })

    def _parse_fallback(self, line, detected_type):
        """ML-assisted fallback parsing"""
        return {
            'timestamp': self._safe_parse_timestamp(line[:30]),  # Heuristic
            'host': detected_type,
            'level': 'UNKNOWN',
            'message': line
        }

    def _build_analysis_pipeline(self):
        """Composite feature engineering pipeline"""
        numeric_features = ['msg_length', 'error_count', 'warning_count']
        
        return make_pipeline(
            ColumnTransformer([
                ('numeric', make_pipeline(
                    SimpleImputer(strategy='median'),
                    StandardScaler()
                ), numeric_features),
                ('text', TfidfVectorizer(max_features=500), 'message')
            ]),
            IsolationForest(
                contamination=0.05, 
                random_state=42, 
                n_jobs=-1,
                verbose=0
            )
        )

    def analyze(self, log_file):
        """End-to-end analysis workflow"""
        try:
            # Stage 1: Log type detection
            type_confidences = self.detect_log_types(log_file)
            logging.info(f"Log type confidences: {type_confidences}")
            
            # Stage 2: Intelligent parsing
            logs = self.parse_logs(log_file)
            if logs.empty:
                raise ValueError("No parsable log entries found")
            
            # Stage 3: Feature engineering
            logs['msg_length'] = logs['message'].str.len()
            logs['error_count'] = logs['message'].str.count(r'(?i)error|exception|fail')
            logs['warning_count'] = logs['message'].str.count(r'(?i)warn|alert|critical')
            
            # Stage 4: Anomaly detection
            self.analysis_pipeline.fit(logs)
            logs['anomaly_score'] = self.analysis_pipeline.decision_function(logs)
            logs['anomaly'] = self.analysis_pipeline.predict(logs)
            
            return {
                'type_confidences': type_confidences,
                'anomalies': logs[logs['anomaly'] == -1],
                'stats': self._compute_statistics(logs)
            }
        
        except Exception as e:
            logging.error(f"Analysis pipeline failed: {str(e)}")
            raise

    def _compute_statistics(self, logs):
        """Generate operational insights"""
        return {
            'total_entries': len(logs),
            'error_rate': logs['error_count'].sum() / len(logs),
            'common_hosts': logs['host'].value_counts().head(3).to_dict(),
            'anomaly_distribution': logs['anomaly'].value_counts().to_dict()
        }

def main():
    """CLI Interface with enhanced reporting"""
    parser = argparse.ArgumentParser(
        description="ML-Powered Log Analysis System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("log_file", help="Path to log file for analysis")
    parser.add_argument("--output", "-o", help="Output file for results (CSV)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer()
    try:
        results = analyzer.analyze(args.log_file)
        
        # Format output
        print("\n=== ANALYSIS RESULTS ===")
        print("Log Type Confidences:")
        for t, c in results['type_confidences'].items():
            print(f"- {t.title()}: {c:.1%}")
        
        print("\nAnomaly Summary:")
        print(results['anomalies'][['timestamp', 'host', 'message']].to_string(index=False))
        
        print("\nStatistics:")
        print(f"Total Entries: {results['stats']['total_entries']}")
        print(f"Error Rate: {results['stats']['error_rate']:.1%}")
        print(f"Common Hosts: {results['stats']['common_hosts']}")
        
        if args.output:
            results['anomalies'].to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
    
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()