import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Configure logging (clean)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Constants
REQUIRED_FEATURES = [
    'Final_Exam_Score', 'Quiz_Scores', 'Assignment_Completion_Rate',
    'Time_Spent_on_Videos', 'Forum_Participation', 'Quiz_Attempts',
    'Feedback_Score', 'Learning_Style', 'Course_Name', 'Education_Level'
]
NUMERIC_FEATURES = [
    'Final_Exam_Score', 'Quiz_Scores', 'Assignment_Completion_Rate',
    'Time_Spent_on_Videos', 'Forum_Participation', 'Quiz_Attempts', 'Feedback_Score'
]
CATEGORICAL_FEATURES = ['Learning_Style', 'Course_Name', 'Education_Level']
CATEGORY_MAPPING = {
    'High Risk': 0,
    'At Risk': 1,
    'Steady': 2,
    'Star Learners': 3
}
REVERSE_CATEGORY_MAPPING = {v: k for k, v in CATEGORY_MAPPING.items()}
LEARNING_STYLE_MAP = {
    'Visual': 1.0,
    'Auditory': 0.5,
    'Kinesthetic': 0.25,
    'Reading/Writing': 0.0
}

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded: {file_path} (shape: {df.shape})")
        return df
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)

def preprocess_data(df_raw, scaler=None, encoder=None, fit=True):
    df = df_raw.copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if fit:
        scaler = MinMaxScaler()
        df[NUMERIC_FEATURES] = scaler.fit_transform(df[NUMERIC_FEATURES])
    else:
        df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])
    df['Learning_Style_Mapped'] = df['Learning_Style'].map(LEARNING_STYLE_MAP).fillna(0.0)
    perf = df[['Final_Exam_Score', 'Quiz_Scores', 'Assignment_Completion_Rate']].mean(axis=1)
    eng = df[['Time_Spent_on_Videos', 'Forum_Participation']].mean(axis=1)
    cons = (df[['Quiz_Attempts', 'Feedback_Score']].mean(axis=1) * 2 + df['Learning_Style_Mapped']) / 3
    df['SLEI_Score'] = (0.4 * perf + 0.4 * eng + 0.2 * cons) * 100
    def assign(score):
        if score >= 80: return 'Star Learners'
        elif score >= 60: return 'Steady'
        elif score >= 40: return 'At Risk'
        else: return 'High Risk'
    df['SLEI_Category'] = df['SLEI_Score'].apply(assign)
    logging.info(f"SLEI Distribution:\n{df['SLEI_Category'].value_counts()}")
    if fit:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df[CATEGORICAL_FEATURES])
    else:
        encoded = encoder.transform(df[CATEGORICAL_FEATURES])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=df.index)
    df = pd.concat([df, encoded_df], axis=1)
    features = NUMERIC_FEATURES + list(encoded_df.columns)
    X = df[features]
    y = df['SLEI_Category'].map(CATEGORY_MAPPING)
    return df, X, y, scaler, encoder

def main():
    train_df_raw = load_data('csv_80_percent.csv')
    test_df_raw = load_data('csv_20_percent.csv')
    train_df, X_train, y_train, scaler, encoder = preprocess_data(train_df_raw, fit=True)
    test_df, X_test, y_test, _, _ = preprocess_data(test_df_raw, scaler, encoder, fit=False)
    print(f"Training Samples: {X_train.shape[0]}")

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_tr, y_tr)

    weights = compute_class_weight('balanced', classes=np.unique(y_res), y=y_res)
    class_weights = dict(zip(np.unique(y_res), weights))
    sample_weights = np.array([class_weights[label] for label in y_res])

    model = XGBClassifier(
        objective='multi:softmax',
        num_class=4,
        eval_metric='mlogloss',
        random_state=42
    )

    model.fit(
        X_res,
        y_res,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pred_val = model.predict(X_val)
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_pred_val, target_names=[REVERSE_CATEGORY_MAPPING[i] for i in range(4)], zero_division=0))

    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average='macro')
    recall = recall_score(y_test, y_pred_test, average='macro')

    print("\nTest Metrics:")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=[REVERSE_CATEGORY_MAPPING[i] for i in range(4)], zero_division=0))

    cm = confusion_matrix(y_test, y_pred_test)
    print("\nConfusion Matrix:")
    print(cm)

if __name__ == '__main__':
    main()
