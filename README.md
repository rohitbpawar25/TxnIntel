# txnintel: Credit Card Fraud Analytics & Serving Layer

This repository contains an end-to-end Machine Learning pipeline and API serving layer for evaluating credit card fraud transactions, generating natural language audit explanations, and containerized deployment.

---

## 1. Project Directory Layout

```text
txnintel/
│
├── README.md                 # Setup instructions, architecture, decisions
├── REPORT.md                 # Model comparison, threshold selection, improvements, curl commands
├── requirements.txt          # Python package requirements
├── .env.example              # Sample environment configuration template
├── .gitignore                # Git ignore paths
├── Dockerfile                # API & models container build file
├── docker-compose.yml        # PostgreSQL + API deployment composer
│
├── data/
│   └── raw/
│       └── creditcard.csv     # Raw ULB Kaggle transactions (ignored by git)
│
├── notebooks/
│   ├── eda.ipynb              # Part A: Exploratory Data Analysis & Ingestion checks
│   ├── modeling.ipynb         # Part B: Reproducible Model training & Threshold tuning
│   └── llm_explanations.ipynb # Part C: LLM explanation layer, prompts, and templates
│
├── models/
│   ├── fraud_model.pkl        # Persisted Random Forest model binary
│   ├── preprocessing_pipeline.pkl # Preprocessing RobustScaler + Imputer binary
│   ├── metadata.json          # Reproducibility log (data hash, model version, validation metrics)
│   └── shap_values_test.npy   # Pre-computed SHAP values for verification
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Environment config loader
│   ├── db.py                  # Database connection utilities
│   ├── preprocess.py          # Cleaning, chronological splitting, and transformation steps
│   ├── train.py               # Model training script
│   ├── model_service.py       # Model inference & dynamic SHAP extraction service
│   ├── llm_service.py         # Gemini API interface with retry & fallback
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py            # FastAPI main router & event hooks
│       └── schemas.py         # Pydantic input/response validation models
│
└── tests/
    ├── test_pipeline.py       # Pipeline scale & shape validator
    └── test_api.py            # FastAPI endpoint integration test suite
```

---

## 2. System Architecture

```text
       Raw Transaction (creditcard.csv)
                      │
                      ▼
             Database Ingestion (PostgreSQL)
                      │
                      ▼
           Chronological Train-Test Split (80/20)
                      │
                      ▼
          Model Inference (Random Forest)
                      │
                      ├──────────────────────────┐
                      ▼                          ▼
            Fraud Probability [0-1]         SHAP Feature Contributions
                      │                          │
                      └─────────────┬────────────┘
                                    │
                                    ▼
                          Grounded Input Payload
                                    │
                                    ▼
                         LLM Explainer (Gemini)
                                    │
                                    ▼
                         Strict JSON Validation
                                    │
                                    ├─── [Success] ──► Structured Explanations
                                    │
                                    └─── [Failure] ──► Retry Once ──► Fallback Template
```

---

## 3. Setup & Running Locally

### A. Environment Configuration
Create a `.env` file in the root workspace folder matching `.env.example`:
```env
DATABASE_URL=postgresql://postgres:your_postgres_password@localhost:5432/txnintel
POSTGRES_PASSWORD=your_postgres_password
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
```

### B. Ingest and Train
1. Ensure your local PostgreSQL instance is running with the credentials above.
2. The initial ingestion of the Kaggle `creditcard.csv` and exploratory parsing are documented inside `notebooks/eda.ipynb`.
3. Model fitting, evaluation metrics, and artifact generation can be executed directly using the standalone training script:
   ```bash
   PYTHONPATH=. python3 src/train.py
   ```

### C. Docker Compose Build & Deploy (Recommended)
You can deploy both the PostgreSQL database and the FastAPI endpoint inside Docker containers with a single command:
```bash
docker-compose up --build
```
The serving endpoint will launch on `http://localhost:8000`.

### D. Running API Locally (Manual)
If running manually without Docker, activate your local python environment and launch the FastAPI server using Uvicorn:
```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

---

## 4. Running Test Suites

Automated unit and integration tests are available under the `tests/` directory:

1. **Pipeline Transform & Shape Verification**:
   ```bash
   PYTHONPATH=. python3 tests/test_pipeline.py
   ```
2. **API Endpoint Integration Testing**:
   Starts a local uvicorn subprocess and verifies `/health`, `/model/info`, `/score` (validation logic), and `/explain` (Gemini API schema validation):
   ```bash
   PYTHONPATH=. python3 tests/test_api.py
   ```

---

## 5. Key Decisions & Trade-Offs

- **Chronological Split vs. K-Fold**: Using a random split introduces temporal leakage because fraud trends change over time. Transactions are sorted by `Time` and split chronologically (80% train, 20% test).
- **Deduplication Prior to Splitting**: 1,081 identical duplicate records were removed. Leaving identical rows in both folds results in target leakage, artificially boosting test metrics.
- **Random Forest Classifier**: Selected over XGBoost due to platform compilation constraints (macOS requires `libomp` which is absent on this machine). Random Forest handles non-linear relationships, outliers, and class weights natively.
- **Decision Threshold (0.25)**: Tuned to minimize expected business cost ($500 per False Negative, $25 per False Positive) yielding massive savings over the default 0.5 threshold.
