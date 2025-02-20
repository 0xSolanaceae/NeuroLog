# train_crf.py
import os
import pickle
from analyzer import LogAnalyzer
from sklearn_crfsuite import CRF

def train_and_save_model(model_path='models/crf_model.pkl'):
    # Initialize analyzer to access dataset and patterns
    analyzer = LogAnalyzer()
    
    # Collect training data using analyzer's existing patterns
    X_train = []
    y_train = []
    
    for log_type in analyzer.LOG_TYPES:
        file_path = os.path.join(analyzer.dataset_dir, f"{log_type}.log")
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        for line in lines:
            for pattern, fields in analyzer._compiled_patterns:
                if match := pattern.match(line):
                    tokens = analyzer._tokenize_with_spans(line)
                    if not tokens:
                        continue
                    labels = ['O'] * len(tokens)
                    
                    for group_idx in range(len(fields)):
                        field = fields[group_idx]
                        group_start = match.start(group_idx + 1)
                        group_end = match.end(group_idx + 1)
                        
                        for i, (_, start, end) in enumerate(tokens):
                            if start >= group_start and end <= group_end:
                                labels[i] = field
                    
                    features = analyzer._extract_features([t[0] for t in tokens])
                    X_train.append(features)
                    y_train.append(labels)
                    break

    # Train and save model
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    if X_train and y_train:
        crf.fit(X_train, y_train)
        with open(model_path, 'wb') as f:
            pickle.dump(crf, f)
        print(f"Model saved to {model_path}")
    else:
        print("No training data found!")

if __name__ == '__main__':
    train_and_save_model()