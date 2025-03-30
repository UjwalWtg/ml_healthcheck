#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DataDriftDetector:
    def __init__(self, threshold=0.05):
        """
        Initialize drift detector with significance threshold.
        
        Args:
            threshold (float): p-value threshold for statistical tests (default: 0.05)
        """
        self.threshold = threshold
        self.scaler = StandardScaler()
        
    def detect_drift(self, X_train, X_prod, features=None):
        """
        Detect drift between training and production data.
        
        Args:
            X_train (pd.DataFrame): Training data
            X_prod (pd.DataFrame): Production data
            features (list): Specific features to check (default: all)
            
        Returns:
            dict: Drift results for each feature
        """
        if features is None:
            features = X_train.columns
            
        results = {}
        
        for feature in features:
            # Kolmogorov-Smirnov test for continuous features
            if pd.api.types.is_numeric_dtype(X_train[feature]):
                stat, p = stats.ks_2samp(X_train[feature], X_prod[feature])
                results[feature] = {
                    'test': 'KS',
                    'statistic': stat,
                    'p_value': p,
                    'drift': p < self.threshold
                }
            # Chi-square test for categorical features
            else:
                train_counts = X_train[feature].value_counts()
                prod_counts = X_prod[feature].value_counts()
                all_cats = list(set(train_counts.index) | set(prod_counts.index))
                
                # Align counts
                train_aligned = [train_counts.get(cat, 0) for cat in all_cats]
                prod_aligned = [prod_counts.get(cat, 0) for cat in all_cats]
                
                stat, p = stats.chisquare(train_aligned, prod_aligned)
                results[feature] = {
                    'test': 'Chi-square',
                    'statistic': stat,
                    'p_value': p,
                    'drift': p < self.threshold
                }
        
        # PCA-based drift detection
        pca_results = self._detect_pca_drift(X_train, X_prod)
        results['_pca'] = pca_results
        
        return results
    
    def _detect_pca_drift(self, X_train, X_prod):
        """Detect drift in PCA space"""
        try:
            # Scale data
            X_train_scaled = self.scaler.fit_transform(X_train.select_dtypes(include=np.number))
            X_prod_scaled = self.scaler.transform(X_prod.select_dtypes(include=np.number))
            
            # Fit PCA
            pca = PCA(n_components=2)
            train_pca = pca.fit_transform(X_train_scaled)
            prod_pca = pca.transform(X_prod_scaled)
            
            # Test each component
            pca_results = {}
            for i in range(2):
                stat, p = stats.ks_2samp(train_pca[:, i], prod_pca[:, i])
                pca_results[f'PC{i+1}'] = {
                    'test': 'KS',
                    'statistic': stat,
                    'p_value': p,
                    'drift': p < self.threshold
                }
            
            return pca_results
        except Exception as e:
            return {'error': str(e)}


# In[ ]:




