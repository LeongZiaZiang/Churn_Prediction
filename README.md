# Bank Customer Churn Prediction — Production ML Project

A production-level machine learning project that predicts customer churn for a Russian bank. The project covers the full ML lifecycle — from exploratory analysis and feature engineering to model deployment via a REST API.

Dataset source: [Kaggle — Bank Churn Dataset](https://www.kaggle.com/datasets/ilya2raev/bank-churn-dataset)

---

## Project Structure

```
├── Churn_Prediction.ipynb             # Full analysis notebook
├── api.py                             # FastAPI serving endpoint
└── requirements.txt                   # Dependencies
```

---

## Methodology

### 1. Data Preprocessing
- Removed 1,151 duplicate rows
- Outlier treatment: dropped 3 rows with invalid scoring (>1.0), capped income at 200,000 RUB
- Missing values handled via median/mode imputation inside sklearn Pipeline (fit on train only)

### 2. Feature Engineering
7 domain-driven features derived from raw columns:

| Feature | Description |
|---|---|
| `debt_to_income` | Credit sum relative to monthly income |
| `income_vs_region` | Income relative to regional average wage |
| `monthly_obligation` | Estimated monthly repayment |
| `repayment_burden` | Monthly obligation as proportion of income |
| `overdue_rate` | Proportion of credits that are overdue |
| `has_overdue` | Binary flag — any overdue loan |
| `score_per_age` | Scoring normalised by age |

### 3. Preprocessing Pipeline
Full sklearn `ColumnTransformer` Pipeline:
- Numerical: median imputation → StandardScaler
- Categorical: mode imputation → OneHotEncoder
- Binary: mode imputation

### 4. Models
Five models trained, all as end-to-end sklearn Pipelines:
- Logistic Regression (`class_weight='balanced'`)
- Random Forest (`class_weight='balanced_subsample'`)
- LightGBM (`scale_pos_weight` for class imbalance)
- XGBoost (`scale_pos_weight` for class imbalance)
- Stacking Ensemble (RF + LightGBM + XGBoost → Logistic Regression meta-learner)

Threshold tuned per model using **Youden's J statistic** on validation set.

---

## Results

### Test Set Performance

| Model | AUC-ROC | KS | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| XGBoost | 0.76 | 0.40 | 0.40 | 0.75 | 0.32 | 0.44 |
| Stacking Ensemble | 0.76 | 0.40 | 0.40 | 0.78 | 0.30 | 0.44 |
| LightGBM | 0.76 | 0.39 | 0.40 | 0.73 | 0.32 | 0.44 |
| Random Forest | 0.74 | 0.37 | 0.36 | 0.66 | 0.32 | 0.43 |
| Logistic Regression | 0.72 | 0.32 | 0.34 | 0.66 | 0.30 | 0.41 |

> KS Statistic of 0.40 meets the banking industry threshold for a good model (KS > 0.40).

---

## Key Findings

### SHAP Explainability
Top churn drivers identified via LightGBM SHAP values:
- `credit_length` — strongest predictor. Short-term loan customers churn more, likely due to lower financial commitment
- `tariff_id_19` — being on tariff 19 significantly increases churn probability
- `scoring` — non-linear relationship with churn
- Engineered features `score_per_age`, `debt_to_income`, `monthly_obligation` all appear in top 20, validating the feature engineering approach
- Tariffs 20, 25, 28 are protective against churn — characteristics worth replicating in other products

### Lift Analysis (Stacking Ensemble)
- Top 10% highest-risk customers are **2.65× more likely to churn** than average
- Targeting top 40% of customers captures **72% of all churners**
- Retention team can recover three quarters of potential churn losses at less than half the cost of a blanket campaign

### Profit Optimisation
Two business scenarios evaluated:

| Scenario | CLV | Cost | Success Rate | Best Model | Max Profit | Optimal Target % |
|---|---|---|---|---|---|---|
| Conservative | RUB 1,000 | RUB 170 | 30% | XGBoost | RUB 387,293 | 28.7% |
| Realistic | RUB 15,000 | RUB 200 | 20% | Logistic Regression | RUB 39,334,137 | 99.8% |

> Model selection is a business decision, not just a technical one. When intervention costs are high relative to CLV, precision-focused XGBoost is optimal. When CLV far exceeds cost, a wide-net strategy with Logistic Regression maximises profit.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the notebook
Open `Churn_Prediction.ipynb` and run all cells. This generates all serialised pipelines in the `models/` folder.

### 3. Start the API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open Swagger UI
```
http://localhost:8000/docs
```

---

## API Usage

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch prediction (up to 10,000) |

### Example — Single Prediction

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "age": 42,
    "marital_status": 4.0,
    "job_position": 14,
    "credit_sum": 80000,
    "credit_length": 24,
    "scoring": 0.32,
    "education": 4.0,
    "tariff_id": 19,
    "region": 31.0,
    "average_region_wage": 35000,
    "income": 38000,
    "credit_count": 3.0,
    "overdue_count": 1.0
  }'
```

**Response:**
```json
{
  "churn_probability": 0.638,
  "churn_prediction": 1,
  "risk_segment": "VERY HIGH",
  "threshold_used": 0.4475,
  "expected_profit_rub": 1713.97,
  "model_name": "XGBoost"
}
```

---

## Recommendations

- **Deploy XGBoost** via the REST API for real-time scoring and monthly batch campaign targeting
- **Prioritise top 40%** highest-risk customers for retention campaigns to capture 72% of churners
- **Investigate tariff 19** — elevated churn rate suggests product or pricing issues requiring review
- **Study tariffs 20, 25, 28** — these show protective characteristics worth replicating
- **Monitor monthly** — retrain if KS statistic drops below 0.35
