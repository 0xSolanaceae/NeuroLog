#!/usr/bin/env python3

import re
import logging
import pandas as pd
from dateutil.parser import parse as date_parse
from collections import Counter
import math
import yaml

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
    with open('log_patterns.yaml', 'r') as file:
        data = yaml.safe_load(file)

    return [
        (re.compile(item['pattern']), item['fields'])
        for item in data['patterns']
    ]