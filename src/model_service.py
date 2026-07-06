import os
import joblib
import pandas as pd
import numpy as np
import shap

# Feature order in raw dataframe
feature_names = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
# Preprocessed feature order (output of ColumnTransformer: ['Time', 'Amount', 'V1', ..., 'V28'])
preprocessed_feature_order = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

_pipeline = None
_model = None
_explainer = None

def load_model_binaries():
    global _pipeline, _model, _explainer
    pipeline_path = "models/preprocessing_pipeline.pkl"
    model_path = "models/fraud_model.pkl"
    
    if os.path.exists(pipeline_path) and os.path.exists(model_path):
        _pipeline = joblib.load(pipeline_path)
        _model = joblib.load(model_path)
        _explainer = shap.TreeExplainer(_model)
        return True
    return False

def score_tx(tx_dict: dict):
    if _pipeline is None or _model is None:
        raise ValueError("Model is not loaded.")
    df = pd.DataFrame([tx_dict], columns=feature_names)
    processed = _pipeline.transform(df)
    prob = float(_model.predict_proba(processed)[:, 1][0])
    
    if prob >= 0.75:
        risk = "HIGH"
    elif prob >= 0.25:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return prob, risk

def get_shap_contributions(tx_dict: dict, top_n=4):
    if _pipeline is None or _model is None or _explainer is None:
        raise ValueError("Explainer is not loaded.")
    df = pd.DataFrame([tx_dict], columns=feature_names)
    processed = _pipeline.transform(df)
    processed_df = pd.DataFrame(processed, columns=preprocessed_feature_order)
    
    shap_vals_raw = _explainer.shap_values(processed_df)
    
    if isinstance(shap_vals_raw, list):
        shap_vals = shap_vals_raw[1][0]
    elif len(shap_vals_raw.shape) == 3:
        shap_vals = shap_vals_raw[0, :, 1]
    else:
        shap_vals = shap_vals_raw[0]

    abs_shap = np.abs(shap_vals)
    top_indices = abs_shap.argsort()[::-1][:top_n]
    
    features_list = []
    for idx in top_indices:
        feat_name = preprocessed_feature_order[idx]
        val = tx_dict[feat_name]
        contribution = shap_vals[idx]
        features_list.append({
            "feature": feat_name,
            "value": float(val),
            "shap_value": float(contribution),
            "direction": "increases fraud risk" if contribution > 0 else "decreases fraud risk"
        })
    return features_list
