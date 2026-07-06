import os
import json
import time
import logging
import google.generativeai as genai
from src.config import GEMINI_API_KEY

logger = logging.getLogger("fraud_api")

def configure_llm():
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        return True
    return False

def generate_fallback_report(tx_dict: dict, score: float, top_features: list):
    indicators = [f"{f['feature']} value of {f['value']:.2f} ({f['direction']})" for f in top_features]
    
    if score >= 0.75:
        risk = "HIGH"
    elif score >= 0.25:
        risk = "MEDIUM"
    else:
        risk = "LOW"
        
    return {
        "risk_level": risk,
        "summary": f"Transaction of ${tx_dict['Amount']:.2f} flagged at Time={tx_dict['Time']:.0f}s with fraud score of {score:.2%}. (Generated via rule-based fallback system)",
        "key_indicators": indicators[:3],
        "recommended_action": "Freeze card immediately and contact customer." if risk == "HIGH" else "Queue for standard review."
    }

def explain_transaction(tx_dict: dict, score: float, top_features: list, retries=1):
    if score < 0.25:
        return {
            "risk_level": "LOW",
            "summary": "This transaction exhibits typical legitimate characteristics and has been classified as low risk.",
            "key_indicators": [f"{f['feature']} value of {f['value']:.2f} ({f['direction']})" for f in top_features[:2]],
            "recommended_action": "No analyst action required. Approve transaction."
        }
        
    if not GEMINI_API_KEY:
        return generate_fallback_report(tx_dict, score, top_features)
        
    input_data = {
        "transaction_details": {"Time": tx_dict["Time"], "Amount": tx_dict["Amount"]},
        "fraud_score": score,
        "top_feature_contributions": top_features
    }
    
    system_prompt = '''You are an expert Credit Card Fraud Analyst AI. Explain why this transaction was flagged.
Input Data contains parameters, score, and top SHAP contributions.

CRITICAL GROUNDING RULES:
1. You must ONLY use numbers and values present in the Input Data. Do not invent other numeric values.
2. Output must strictly be a valid JSON object matching the schema below exactly. No other text.

TARGET JSON SCHEMA:
{
  "risk_level": "HIGH | MEDIUM | LOW",
  "summary": "2-3 sentence explanation in plain English",
  "key_indicators": [
    "description of feature 1 and how its value relates to risk",
    "description of feature 2 and how its value relates to risk"
  ],
  "recommended_action": "action plan for human analyst"
}'''
    
    prompt = f"Input Data:\n{json.dumps(input_data, indent=2)}\n\nResponse:"
    
    for attempt in range(retries + 1):
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                f"{system_prompt}\n\n{prompt}",
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text.strip())
        except Exception as e:
            logger.error(f"Gemini API attempt {attempt+1} failed: {e}")
            if attempt < retries:
                time.sleep(1)
            else:
                logger.warning("Gemini API calls failed. Returning template fallback report.")
                return generate_fallback_report(tx_dict, score, top_features)
