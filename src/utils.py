#!/usr/bin/env python3

import re
import logging
import pandas as pd
from dateutil.parser import parse as date_parse
from collections import Counter
import math

def safe_parse_timestamp(ts_str: str) -> pd.Timestamp:
    """
    Safely parse a timestamp string into a pandas Timestamp.
    Tries multiple common date formats if the initial parse fails.
    """
    logging.info("Parsing timestamp: %s", ts_str)
    try:
        return date_parse(ts_str, ignoretz=True)
    except (ValueError, OverflowError, AttributeError):
        # Try alternative common formats
        for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%d/%b/%Y:%H:%M:%S %z', '%b %d %H:%M:%S']:
            try:
                return pd.to_datetime(ts_str, format=fmt)
            except ValueError:
                continue
        return pd.NaT

def calculate_entropy(text: str) -> float:
    """
    Calculate the Shannon entropy of a given text.
    Returns a float representing the entropy.
    """
    counts = Counter(text)
    probs = [count / len(text) for count in counts.values()]
    return -sum(p * math.log(p) for p in probs if p > 0)

def load_parsing_patterns():
    """
    Return a list of tuples, each containing a compiled regex pattern and the corresponding
    list of field names. These patterns are used to parse different log formats.
    """
    patterns = [
        # Docker JSON format
        (r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) (\S+) (\S+) \{"log":"(.*?)\\n".*\}',
         ['timestamp', 'host', 'level', 'message']),
        
        # Syslog format
        (r'^(\w{3}\s+\d{1,2} \d{2}:\d{2}:\d{2}) (\S+) (\S+)\[(\d+)\]: (.*)$',
         ['timestamp', 'host', 'app', 'pid', 'message']),
        
        # Apache Common Log Format
        (r'^(\S+) (\S+) (\S+) \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\d+)$',
         ['ip', 'client_id', 'user_id', 'timestamp', 'method', 'url', 'protocol', 'status', 'message']),
        
        # Nginx access log
        (r'^(\S+) - (\S+) \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\d+) "(.*?)" "(.*?)"$',
         ['ip', 'remote_user', 'timestamp', 'method', 'url', 'protocol', 'status', 'size', 'referer', 'user_agent']),
        
        # Windows Event Log
        (r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) (\S+) (\S+): (\S+): (.*)$',
         ['timestamp', 'host', 'source', 'event_id', 'message']),
        
        # Kubernetes container log pattern
        (r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+(stdout|stderr)\s+(\w+)\s+(.*)$',
         ['timestamp', 'stream', 'level', 'message'])
    ]
    return [(re.compile(pattern), fields) for pattern, fields in patterns]
