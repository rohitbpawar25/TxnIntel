import os
import joblib
import pandas as pd
from src.preprocess import clean_data, split_chronologically

def test_preprocessing():
    print("Running pipeline verification test...")
    # Load pipeline
    pipeline_path = "models/preprocessing_pipeline.pkl"
    assert os.path.exists(pipeline_path), "Missing preprocessing pipeline binary."
    pipeline = joblib.load(pipeline_path)
    print("Pipeline binary loaded successfully.")
    
    # Mock data to test transforms
    feature_names = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    mock_data = {f: [0.0] for f in feature_names}
    df = pd.DataFrame(mock_data)
    
    transformed = pipeline.transform(df)
    assert transformed.shape == (1, 30), "Transformed shape mismatch."
    print("Mock transaction transform successful. Shape: (1, 30)")
    print("Pipeline test: SUCCESS")

if __name__ == "__main__":
    test_preprocessing()
