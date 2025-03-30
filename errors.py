#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import shap
from eli5 import show_weights

class ErrorAnalyzer:
    def __init__(self, model):
        self.model = model
        
    def analyze_errors(self, X, y_true, y_pred):
        """
        Analyze model errors and identify problematic samples.
        
        Args:
            X: Features
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            dict: Error analysis results
        """
        # Get misclassified samples
        wrong_mask = y_true != y_pred
        X_wrong = X[wrong_mask]
        y_true_wrong = y_true[wrong_mask]
        y_pred_wrong = y_pred[wrong_mask]
        
        # Feature importance for errors
        try:
            explainer = shap.Explainer(self.model, X)
            shap_values = explainer(X_wrong)
            
            # Get top features contributing to errors
            mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
            top_error_features = pd.Series(mean_abs_shap, index=X.columns
                ).sort_values(ascending=False).head(5).to_dict()
        except Exception as e:
            top_error_features = {'error': str(e)}
        
        # Get hard samples (high confidence wrong predictions)
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X_wrong)
            confidences = np.max(probas, axis=1)
            hard_samples = np.argsort(confidences)[-5:]  # Top 5 most confident errors
        else:
            hard_samples = []
            
        return {
            'error_rate': float(np.mean(wrong_mask)),
            'top_error_features': top_error_features,
            'hard_samples': {
                'indices': hard_samples.tolist(),
                'true_labels': y_true_wrong[hard_samples].tolist(),
                'pred_labels': y_pred_wrong[hard_samples].tolist()
            }
        }

