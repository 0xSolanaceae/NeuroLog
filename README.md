# NeuroLog: ML-Powered Log Analysis

## Overview

NeuroLog is a log analysis tool that leverages machine learning (ML) and natural language processing (NLP) to intelligently parse, analyze, and detect anomalies in diverse log formats. Designed for scalability and robustness, NeuroLog is ideal for DevOps, security analysts, and system administrators who need to extract actionable insights from complex log data.

---

## Key Features

- **ML-Powered Log Type Detection**: Uses a multi-class classifier to identify log formats with confidence scores.
- **Anomaly Detection**: Employs Isolation Forest to detect outliers in log data.
- **Feature Engineering**: Extracts HTTP methods, status codes, error counts, and more.
- **Memory-Efficient Processing**: Handles large files with chunked processing.

---

## Installation

- Clone the repository:

```bash
git clone https://github.com/0xSolanaceae/NeuroLog.git
```

- Navigate to project directory:

```bash
cd NeuroLog
```

- Install dependencies with `poetry`:

```bash
poetry install
```

If you don't have `poetry` installed, install it [here](https://python-poetry.org/docs/#installation).

## Usage:

### 1. Activate the poetry shell:

```bash
poetry shell
```

### 2. CD into `src`:
```bash
cd src
```

### 3. Full Log Analysis
```bash
poetry run python neurolog.py analyze /path/to/logfile.log --output anomalies.csv --format csv
```
- Detects anomalies and generates insights.
- Supports CSV, JSON, and HTML outputs.

### 4. Log Format Detection
```bash
poetry run python neurolog.py detect /path/to/logfile.log
```
- Displays probabilities for supported log formats.

### 5. Statistical Reporting
```bash
poetry run python neurolog.py stats /path/to/logfile.log --output stats.json
```
- Generates a JSON file with detailed log statistics.

---

## Scientific Foundations

### Machine Learning Pipeline
- **Log Type Detection**: TF-IDF + Logistic Regression for multi-class classification.
- **Anomaly Detection**: Isolation Forest with numeric, categorical, and text features.

### Feature Engineering
- Numeric: Message length, error/warning counts, HTTP status codes.
- Categorical: HTTP methods, log levels.
- Text: TF-IDF vectorization of log messages.

---

## Future Work

- Real-time log analysis with streaming platforms.
- Deep learning models for advanced log understanding.
- Custom log format support and dashboard integration.

---

## Contributing

Contributions are welcome! Fork the repository, create a feature branch, and submit a pull request.

---

## License

GPLv3 License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use NeuroLog in your work, cite this repository:

```bibtex
@software{NeuroLog,
  author = {Your Name},
  title = {NeuroLog: AI-Powered Log Analysis System},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/0xSolanaceae/NeuroLog}}
}