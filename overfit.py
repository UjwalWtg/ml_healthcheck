#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

class OverfitDetector:
    def __init__(self, cv=5):
        self.cv = cv
        
    def check_overfitting(self, model, X, y):
        """
        Check for overfitting using learning curves.
        
        Args:
            model: sklearn-compatible model
            X: Features
            y: Labels
            
        Returns:
            dict: Training and validation scores across sample sizes
            matplotlib.figure: Learning curve plot
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=self.cv,
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring='accuracy'
        )
        
        # Calculate means and stds
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Create plot
        fig, ax = plt.subplots()
        ax.plot(train_sizes, train_mean, 'o-', label='Training score')
        ax.plot(train_sizes, val_mean, 'o-', label='Validation score')
        ax.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1)
        ax.fill_between(train_sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.1)
        ax.set_xlabel('Training examples')
        ax.set_ylabel('Accuracy')
        ax.legend()
        
        # Determine overfitting
        overfit_score = (train_mean[-1] - val_mean[-1]) / train_mean[-1]
        is_overfit = overfit_score > 0.1  # 10% gap threshold
        
        return {
            'train_sizes': train_sizes.tolist(),
            'train_scores': train_mean.tolist(),
            'val_scores': val_mean.tolist(),
            'overfit_score': float(overfit_score),
            'is_overfit': bool(is_overfit)
        }, fig

