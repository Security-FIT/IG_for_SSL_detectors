import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

# Add project root to sys.path to import utils
sys.path.append(os.getcwd())
from utils.metrics import calculate_EER, calculate_minDCF

def load_scores(filepath, detector_name):
    # The score files have 3 columns: id, score, label
    # Assuming space separated
    df = pd.read_csv(filepath, sep=' ', header=None, names=['utt_id', f'{detector_name}_score', 'label'])
    return df

def main():
    base_path = 'scores/wavlm-base-plus'
    aasist_path = os.path.join(base_path, 'aasist_best_scores.txt')
    camhfa_path = os.path.join(base_path, 'camhfa_best_scores.txt')
    sls_path = os.path.join(base_path, 'sls_best_scores.txt')

    print("Loading scores...")
    df_aasist = load_scores(aasist_path, 'aasist')
    df_camhfa = load_scores(camhfa_path, 'camhfa')
    df_sls = load_scores(sls_path, 'sls')

    print("Merging dataframes...")
    # Merge on id
    df = df_aasist.merge(df_camhfa[['utt_id', 'camhfa_score']], on='utt_id', how='inner')
    df = df.merge(df_sls[['utt_id', 'sls_score']], on='utt_id', how='inner')

    print(f"Total samples: {len(df)}")

    # Assumes labels are consistent across files which they should be
    
    # Features and Target
    X = df[['aasist_score', 'camhfa_score', 'sls_score']]
    y = df['label']

    print(f"Training Logistic Regression Fusion on {len(X)} samples...")
    # Train Logistic Regression
    clf = LogisticRegression() # Adjusted for better convergence
    clf.fit(X, y)

    # Evaluate
    print("Evaluating...")
    y_pred_proba = clf.predict_proba(X)[:, 1] # Probability of class 1 (spoof)
    
    eer = calculate_EER(y, y_pred_proba)
    mindcf = calculate_minDCF(y, y_pred_proba)

    print(f"EER: {eer*100:.2f}%")
    print(f"minDCF: {mindcf:.4f}")

    print("\n--- Individual Detectors ---")
    for name in ['aasist', 'camhfa', 'sls']:
        score_col = f'{name}_score'
        scores = df[score_col]
        
        eer_dev = calculate_EER(y, scores)
        mindcf_dev = calculate_minDCF(y, scores)
        print(f"{name.upper()} - EER: {eer_dev*100:.2f}%, minDCF: {mindcf_dev:.4f}")

    # print("\nModel Coefficients:")
    # for feature, coef in zip(['aasist', 'camhfa', 'sls'], clf.coef_[0]):
    #     print(f"{feature}: {coef:.4f}")
    # print(f"Intercept: {clf.intercept_[0]:.4f}")

    # # Save model
    # model_path = 'models/fusion_logistic_regression.pkl'
    # joblib.dump(clf, model_path)
    # print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()
