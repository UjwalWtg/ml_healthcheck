#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness

class BiasAuditor:
    def __init__(self, sensitive_features):
        """
        Initialize bias auditor.
        
        Args:
            sensitive_features (list): Column names of sensitive attributes
        """
        self.sensitive_features = sensitive_features
        
    def audit_bias(self, df, preds, labels):
        """
        Audit model for bias across sensitive groups.
        
        Args:
            df (pd.DataFrame): Data containing sensitive features
            preds (array-like): Model predictions (binary)
            labels (array-like): Ground truth labels
            
        Returns:
            dict: Bias metrics for each sensitive group
        """
        results = {}
        
        for feature in self.sensitive_features:
            # Prepare data for Aequitas
            audit_df = df[[feature]].copy()
            audit_df['score'] = preds
            audit_df['label_value'] = labels
            
            # Compute group metrics
            g = Group()
            xtab, _ = g.get_crosstabs(audit_df)
            
            # Compute bias metrics
            b = Bias()
            bdf = b.get_disparity_predefined_groups(
                xtab,
                original_df=audit_df,
                ref_groups_dict={feature: df[feature].mode()[0]},
                alpha=0.05
            )
            
            # Compute fairness metrics
            f = Fairness()
            fdf = f.get_group_value_fairness(bdf)
            
            # Store results
            results[feature] = {
                'disparity': bdf[[
                    'attribute_name', 'attribute_value',
                    'ppr_disparity', 'pprev_disparity', 'fdr_disparity'
                ]].to_dict('records'),
                'fairness': fdf[[
                    'attribute_name', 'attribute_value',
                    'equalized_odds', 'statistical_parity'
                ]].to_dict('records')
            }
            
        return results

