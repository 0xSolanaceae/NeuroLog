import os
import tempfile
import pandas as pd
import pytest
from src.main import LogAnalyzer  # see [src/main.py](src/main.py)

def create_temp_log_file(lines):
    """Helper to create a temporary log file with the given lines."""
    fd, path = tempfile.mkstemp(suffix=".log", text=True)
    with os.fdopen(fd, 'w', encoding='utf-8') as tmp:
        for line in lines:
            tmp.write(line + "\n")
    return path

@pytest.fixture
def sample_log_file():
    """Creates a temporary log file with a mix of matching and non-matching entries."""
    lines = [
        # Should match Docker JSON format:
        '2021-10-10T12:00:00.000000Z host1 INFO {"log":"Test log entry\\nextra"}',
        # Should match Syslog format:
        'Oct 10 12:00:00 host2 app[123]: Test syslog entry',
        # Non-matching log, forcing fallback:
        'Random log entry that does not match any pattern'
    ]
    path = create_temp_log_file(lines)
    yield path
    os.remove(path)

def test_parse_logs(sample_log_file):
    analyzer = LogAnalyzer()
    df = analyzer.parse_logs(sample_log_file)
    assert isinstance(df, pd.DataFrame)
    # Verify expected columns after postprocessing
    for col in ['timestamp', 'message']:
        assert col in df.columns
    # Ensure at least one entry parsed using a precompiled pattern or fallback.
    assert len(df) > 0

def test_safe_parse_timestamp():
    analyzer = LogAnalyzer()
    # Valid ISO timestamp
    ts = analyzer._safe_parse_timestamp("2021-10-10T12:00:00.000000Z")
    assert pd.notna(ts)
    # Invalid timestamp returns pd.NaT
    ts_invalid = analyzer._safe_parse_timestamp("not a timestamp")
    assert pd.isna(ts_invalid)

def test_fallback_parsing():
    analyzer = LogAnalyzer()
    sample_line = "This is a log entry with no known pattern"
    # We force a fallback using a made-up detected type.
    result = analyzer._parse_fallback(sample_line, "unknown")
    assert result.get('message') == sample_line
    # When no proper fields can be parsed, host defaults to the detected type.
    assert result.get('host') == "unknown"
    # Timestamp should be attempted from the first portion of the line.
    assert 'timestamp' in result