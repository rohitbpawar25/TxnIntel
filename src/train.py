import os
import json
import logging
import hashlib
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
import shap

from src.config import SEED
from src.db import load_transactions_from_db
from src.preprocess import clean_data, split_chronologically, build_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("train")

def hash_dataframe(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def train_model():
    logger.info("Starting model training pipeline...")
    
    # 1. Load Data
    logger.info("Loading data from database...")
    try:
        df = load_transactions_from_db()
    except Exception as e:
        logger.error(f"Failed to load data from database: {e}")
        # Fallback for demonstration if DB is not populated
        logger.info("Falling back to local CSV for training...")
        df = pd.read_csv("data/raw/creditcard.csv")

    # 2. Clean and Split Data
    logger.info("Cleaning and splitting data chronologically...")
    df_clean = clean_data(df)
    train_df, test_df = split_chronologically(df_clean, train_ratio=0.8)

    X_train = train_df.drop(columns=['Class'])
    y_train = train_df['Class']
    X_test = test_df.drop(columns=['Class'])
    y_test = test_df['Class']

    logger.info(f"Training set: {X_train.shape[0]} rows, {y_train.sum()} frauds")
    logger.info(f"Testing set: {X_test.shape[0]} rows, {y_test.sum()} frauds")

    # 3. Build & Fit Preprocessing Pipeline
    logger.info("Fitting preprocessing pipeline...")
    pipeline = build_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    pca_cols = [f"V{i}" for i in range(1, 29)]
    feature_cols = ['Time', 'Amount'] + pca_cols

    # 4. Train Model
    logger.info("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        class_weight='balanced', 
        random_state=SEED, 
        n_jobs=-1
    )
    rf_model.fit(X_train_processed, y_train)

    # 5. Evaluate and Threshold Tuning
    logger.info("Evaluating model...")
    rf_probs = rf_model.predict_proba(X_test_processed)[:, 1]
    
    pr_auc = average_precision_score(y_test, rf_probs)
    roc_auc = roc_auc_score(y_test, rf_probs)
    
    logger.info(f"PR-AUC: {pr_auc:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")

    # Optimal threshold selected via cost-based reasoning in notebook
    final_threshold = 0.25
    final_preds = (rf_probs >= final_threshold).astype(int)
    
    precision = precision_score(y_test, final_preds)
    recall = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)

    logger.info(f"Metrics @ threshold {final_threshold}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    # 6. SHAP Values Extraction
    logger.info("Extracting SHAP feature importance...")
    explainer = shap.TreeExplainer(rf_model)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_cols)
    shap_values_raw = explainer.shap_values(X_test_df)
    
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
    elif len(shap_values_raw.shape) == 3:
        shap_values = shap_values_raw[:, :, 1]
    else:
        shap_values = shap_values_raw

    # 7. Save Artifacts
    logger.info("Saving model binaries and metadata...")
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(pipeline, "models/preprocessing_pipeline.pkl")
    joblib.dump(rf_model, "models/fraud_model.pkl")
    np.save("models/shap_values_test.npy", shap_values)
    
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
    top_shap = mean_abs_shap.sort_values(ascending=False)
    top_shap.to_csv("models/shap_feature_importance.csv", header=["mean_abs_shap"])

    training_data_hash = hash_dataframe(train_df)
    model_version = f"rf_seed{SEED}_{pd.Timestamp.now().strftime('%Y%m%d')}"

    summary = {
        'seed': SEED,
        'model_version': model_version,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'split': 'chronological 80/20 by Time',
        'imbalance_handling': 'class_weight=balanced (LR and RF)',
        'chosen_threshold': final_threshold,
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'test_pr_auc_rf': float(pr_auc),
        'test_roc_auc_rf': float(roc_auc),
        'training_data_hash': training_data_hash,
    }
    
    with open("models/metadata.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    train_model()
