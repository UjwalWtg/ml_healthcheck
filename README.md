# ml_healthcheck
This is ML HealthCheck project. This project provides data drift detection, bias auditing, overfitting checks, and generates diagnostic reports.


A Python library that automatically detects common ML model issues (data drift, bias, overfitting, etc.) and suggests fixes.

1. Solves a real pain point: Many ML projects fail in production due to undetected issues.
2. Beginner-friendly: Helps new practitioners debug models.
3. Production-ready: Useful for MLOps pipelines.

# Key Features

**Data Drift Detection**	Compares training vs. production data distributions.
**Bias Audit**	Measures fairness metrics (demographic parity, equalized odds).
**Overfitting** Check	Flags high train-test accuracy gaps or suspicious learning curves.
**Error Analysis**	Identifies problematic samples (e.g., misclassified hard examples).

# Example 

from ml_healthcheck import ModelDoctor

# Load your trained model and data
model = load_model()  
X_train, X_test, y_train, y_test = load_data()

# Run diagnostics
doctor = ModelDoctor(model, X_train, X_test, y_train, y_test)
report = doctor.generate_report()

# Print issues and fixes
print(report.summary())
""" 
[!] Found 2 Warnings:
1. DATA DRIFT: Feature "age" drifted (p=0.01). 
   → Suggestion: Retrain with recent data.
2. BIAS: Model favors group "male" (disparity=0.2). 
   → Suggestion: Use Fairlearn's GridSearch.
"""
