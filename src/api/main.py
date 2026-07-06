import os
import time
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from src.api.schemas import TransactionInput, ScoreResponse, ExplainResponse
from src.model_service import load_model_binaries, score_tx, get_shap_contributions
from src.llm_service import configure_llm, explain_transaction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_api")

metadata = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler (replaces deprecated on_event)."""
    global metadata
    logger.info("Initializing models and services...")

    # Load metadata
    metadata_path = "models/metadata.json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logger.info("Model metadata loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")

    # Configure services
    configure_llm()
    success = load_model_binaries()
    if success:
        logger.info("Model and SHAP binaries initialized successfully.")
    else:
        logger.error("Binary models failed to load. API running in degraded state.")

    yield  # Application runs here
    # (shutdown logic can go here if needed)

app = FastAPI(
    title="txnintel API",
    description="Serving layer for fraud scoring and natural language transaction audits.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    from src.model_service import _model, _pipeline
    is_loaded = (_model is not None and _pipeline is not None)
    return {
        "status": "healthy" if is_loaded else "degraded",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_loaded": is_loaded
    }

@app.get("/model/info")
def model_info():
    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model info not loaded on server."
        )
    return metadata

monitor_logger = logging.getLogger("fraud_monitoring")

@app.post("/score", response_model=ScoreResponse)
def score(payload: TransactionInput):
    try:
        prob, risk = score_tx(payload.model_dump())
        # Monitoring Hook: Log prediction details to track distribution drift
        monitor_logger.info(f"MONITOR_DRIFT | Amount: {payload.Amount:.2f} | Score: {prob:.4f} | Risk: {risk}")
        return ScoreResponse(fraud_probability=prob, risk_level=risk)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Score endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error."
        )

@app.post("/explain", response_model=ExplainResponse)
def explain(payload: TransactionInput):
    try:
        tx_dict = payload.model_dump()
        prob, risk = score_tx(tx_dict)
        # Monitoring Hook: Log prediction details to track distribution drift
        monitor_logger.info(f"MONITOR_DRIFT | Amount: {payload.Amount:.2f} | Score: {prob:.4f} | Risk: {risk}")
        top_features = get_shap_contributions(tx_dict)
        explanation = explain_transaction(tx_dict, prob, top_features)
        return ExplainResponse(fraud_probability=prob, explanation=explanation)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Explain endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error."
        )
