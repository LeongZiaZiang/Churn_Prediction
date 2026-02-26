# Project: Customer Churn Prediction and Profit Optimization

## Overview

This project develops and evaluates machine learning models to predict customer churn and optimize retention campaign profitability using profit-driven threshold selection. Data source is obtained from [Kaggle]([https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn](https://www.kaggle.com/datasets/ilya2raev/bank-churn-dataset)).

## Models Implemented

- Logistic Regression 
- XGBoost Classifier 

## Model Performance
Classification Metrics
| Model |	Accuracy | Recall |	Precision |	F1 |	ROC-AUC |
| ----- | ----- | ----- | ----- | ----- | ----- | 
| Logistic Regression |	0.64 |	0.60 |	0.27 |	0.37 |	0.668 |
| XGBoost |	0.61 |	0.75 |	0.28 |	0.41 |	0.728 |

XGBoost demonstrates superior churn discrimination ability.

## Lift Analysis

Top 10% lift:

- XGBoost: 2.41
- Logistic Regression: 2.10

The XGBoost model identifies customers who are 2.41× more likely to churn compared to average customers, significantly improving targeting efficiency.

## Profit Optimization

Business assumptions:

- Customer Lifetime Value: 1000
- Retention cost: 170
- Retention success rate: 30%

Results:
| Model |	Optimal targeting % |	Probability cutoff |	Maximum profit |
| Logistic Regression |	22.7% |	0.606 |	143,022 |
| XGBoost |	38.0% |	0.566 |	291,117 |

XGBoost delivers 103% higher profit compared to Logistic Regression.

## Key Business Insights

- XGBoost significantly improves churn prediction accuracy and ranking ability
- Profit optimization shows that default threshold (0.5) is suboptimal
- Targeting top 38% high-risk customers maximizes retention profit
- Machine learning-driven targeting substantially improves marketing efficiency

## Business Recommendation

Deploy XGBoost model in production using probability cutoff of 0.566.

Expected outcome:

- Maximum retention profit
- Improved customer retention efficiency
- Optimized marketing resource allocation

## Project Components

- Data preprocessing
- Logistic Regression modeling
- XGBoost modeling
- ROC curve and AUC analysis
- Lift and gain analysis
- Profit curve optimization
- Optimal threshold selection
