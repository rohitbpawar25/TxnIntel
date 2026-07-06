from pydantic import BaseModel, Field
from typing import List

class TransactionInput(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first transaction", ge=0)
    Amount: float = Field(..., description="Transaction amount in currency units", ge=0)
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float; V8: float
    V9: float; V10: float; V11: float; V12: float; V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float; V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 48592.0, "Amount": 125.0,
                "V1": -1.2, "V2": 0.5, "V3": 1.1, "V4": 0.9, "V5": -0.4, "V6": -0.6, "V7": 0.8, "V8": 0.1,
                "V9": -0.5, "V10": -1.8, "V11": 1.2, "V12": -2.1, "V13": 0.4, "V14": -3.5, "V15": 0.2, "V16": -1.4,
                "V17": -2.5, "V18": -0.9, "V19": 0.4, "V20": 0.1, "V21": 0.2, "V22": -0.1, "V23": -0.2, "V24": 0.1,
                "V25": 0.3, "V26": -0.1, "V27": 0.2, "V28": -0.05
            }
        }
    }

class ScoreResponse(BaseModel):
    fraud_probability: float = Field(..., description="Estimated probability of fraud")
    risk_level: str = Field(..., description="Categorized risk tier: HIGH | MEDIUM | LOW")

class ExplanationSchema(BaseModel):
    risk_level: str
    summary: str
    key_indicators: List[str]
    recommended_action: str

class ExplainResponse(BaseModel):
    fraud_probability: float
    explanation: ExplanationSchema
