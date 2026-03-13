import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List

app = FastAPI(
    title  = "Bank Churn Prediction API",
    version= "1.0.0",
)


@app.on_event("startup")
def load_artefacts():
    global pipeline, metadata, threshold
    try:
        metadata  = joblib.load(os.path.join("models", "metadata.joblib"))
        pipeline  = joblib.load(os.path.join("models", "xgboost_pipeline.joblib"))
        threshold = metadata["optimal_thresholds"]["XGBoost"]
        print(f"Model loaded. Threshold: {threshold:.4f}")
    except Exception as e:
        raise RuntimeError(f"Could not load model: {e}")


def treat_outliers(df):
    df = df.copy()
    df = df[df['scoring'] <= 1.0]
    df['income'] = df['income'].clip(upper=200000)
    return df


def engineer_features(df):
    df = df.copy()
    df['debt_to_income']     = df['credit_sum']        / (df['income'] + 1)
    df['income_vs_region']   = df['income']             / (df['average_region_wage'] + 1)
    df['monthly_obligation'] = df['credit_sum']         / (df['credit_length'] + 1)
    df['repayment_burden']   = df['monthly_obligation'] / (df['income'] + 1)
    df['overdue_rate']       = df['overdue_count']      / (df['credit_count'] + 1)
    df['has_overdue']        = (df['overdue_count'] > 0).astype(int)
    df['score_per_age']      = df['scoring']            / (df['age'] + 1)
    return df


class CustomerInput(BaseModel):
    gender             : str   = Field(..., example="Male")
    age                : int   = Field(..., example=35)
    marital_status     : float = Field(..., example=3.0)
    job_position       : int   = Field(..., example=14)
    credit_sum         : float = Field(..., example=25000)
    credit_length      : int   = Field(..., example=12)
    scoring            : float = Field(..., example=0.46)
    education          : float = Field(..., example=4.0)
    tariff_id          : int   = Field(..., example=2)
    region             : float = Field(..., example=31.0)
    average_region_wage: float = Field(..., example=35000)
    income             : float = Field(..., example=40000)
    credit_count       : float = Field(..., example=2.0)
    overdue_count      : float = Field(..., example=0.0)

    @validator("gender")
    def gender_valid(cls, v):
        if v not in {"Male", "Female"}:
            raise ValueError("gender must be 'Male' or 'Female'")
        return v

    @validator("scoring")
    def scoring_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("scoring must be between 0 and 1")
        return v


class PredictionResponse(BaseModel):
    churn_probability  : float
    churn_prediction   : int
    risk_segment       : str
    threshold_used     : float
    expected_profit_rub: float
    model_name         : str


def get_risk_segment(prob: float) -> str:
    if prob < 0.20:  return "LOW"
    if prob < 0.40:  return "MEDIUM"
    if prob < 0.60:  return "HIGH"
    return "VERY HIGH"


def compute_expected_profit(prob: float) -> float:
    scenario = metadata["profit_scenarios"]["Realistic (Market CLV)"]
    return round(
        prob * scenario["SUCCESS_RATE"] * scenario["CLV"] - scenario["COST"], 2
    )


@app.get("/health")
def health_check():
    return {
        "status"   : "healthy",
        "model"    : "XGBoost",
        "threshold": threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    try:
        raw = pd.DataFrame([customer.dict()])
        raw = engineer_features(raw)
        X   = raw[metadata["feature_names"]]
        
        prob = float(pipeline.predict_proba(X)[0, 1])
        pred = int(prob >= threshold)
        
        return PredictionResponse(
            churn_probability   = round(prob, 4),
            churn_prediction    = pred,
            risk_segment        = get_risk_segment(prob),
            threshold_used      = round(threshold, 4),
            expected_profit_rub = compute_expected_profit(prob),
            model_name          = "XGBoost",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(customers: List[CustomerInput]):
    if len(customers) > 10000:
        raise HTTPException(status_code=400, detail="Batch size must not exceed 10,000.")
    
    try:
        raw = pd.DataFrame([c.dict() for c in customers])
        raw = engineer_features(raw)
        X   = raw[metadata["feature_names"]]
        
        probs = pipeline.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        results = []
        for prob, pred in zip(probs, preds):
            results.append(PredictionResponse(
                churn_probability   = round(float(prob), 4),
                churn_prediction    = int(pred),
                risk_segment        = get_risk_segment(float(prob)),
                threshold_used      = round(threshold, 4),
                expected_profit_rub = compute_expected_profit(float(prob)),
                model_name          = "XGBoost",
            ))
        
        return {
            "predictions"              : results,
            "n_customers"              : len(results),
            "n_churners"               : int(sum(preds)),
            "total_expected_profit_rub": round(sum(r.expected_profit_rub for r in results), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))







