import re
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_log_file(log_file):
    logging.info(f"Parsing log file: {log_file}")
    log_entries = []
    with open(log_file, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z', line):
                match = re.match(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) \S+ \S+ \{"log":"(.*?)\\n","stream":"\S+","time":"\S+"\}$', line)
                if match:
                    timestamp, message = match.groups()
                    log_entries.append([timestamp, "docker/kube", "INFO", message])
            else:
                match = re.match(r'^(\S+ \d+ \d+:\d+:\d+) (\S+) (\S+): (.*)$', line)
                if match:
                    timestamp, host, level, message = match.groups()
                    log_entries.append([timestamp, host, level, message])
    log_df = pd.DataFrame(log_entries, columns=["timestamp", "host", "level", "message"])
    logging.info(f"Parsed {len(log_df)} log entries")
    return log_df

def preprocess_logs(log_df):
    logging.info("Preprocessing log data")
    log_df["level_encoded"] = log_df["level"].astype("category").cat.codes
    log_df["msg_length"] = log_df["message"].apply(len)
    logging.info("Preprocessing complete")
    return log_df[["level_encoded", "msg_length"]]

def detect_anomalies(log_file):
    logging.info("Starting anomaly detection")
    log_df = parse_log_file(log_file)
    if log_df.empty:
        logging.warning("No log entries found or invalid log format.")
        return
    
    feature_data = preprocess_logs(log_df)
    model = IsolationForest(contamination=0.05, random_state=42)
    log_df["anomaly"] = model.fit_predict(feature_data)
    
    anomalies = log_df[log_df["anomaly"] == -1]
    logging.info(f"Detected {len(anomalies)} anomalous log entries")
    print("\nAnomalous Log Entries:")
    print(anomalies[["timestamp", "host", "level", "message"]])

if __name__ == "__main__":
    log_file_path = "data/kube.log"
    detect_anomalies(log_file_path)