#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
from typing import Dict, Any
from .drift import DataDriftDetector
from .bias import BiasAuditor
from .overfit import OverfitDetector
from .errors import ErrorAnalyzer

class ModelDoctor:
    def __init__(self, model, X_train, X_test, y_train, y_test,
                 sensitive_features=None):
        """
        Initialize Model Doctor.
        
        Args:
            model: Trained model to diagnose
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            sensitive_features: List of sensitive feature names for bias audit
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.sensitive_features = sensitive_features or []
        
        # Initialize components
        self.drift_detector = DataDriftDetector()
        self.bias_auditor = BiasAuditor(sensitive_features)
        self.overfit_detector = OverfitDetector()
        self.error_analyzer = ErrorAnalyzer(model)
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        report = {}
        
        # Get predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Run all checks
        report['data_drift'] = self._check_data_drift()
        report['bias'] = self._check_bias(y_test_pred)
        report['overfitting'] = self._check_overfitting()
        report['errors'] = self._analyze_errors(y_test_pred)
        report['summary'] = self._generate_summary(report)
        
        return report
    
    def _check_data_drift(self):
        """Check for data drift between train and test sets"""
        return self.drift_detector.detect_drift(self.X_train, self.X_test)
    
    def _check_bias(self, y_pred):
        """Check for bias in model predictions"""
        if not self.sensitive_features:
            return {'warning': 'No sensitive features provided'}
            
        # Combine features and predictions
        df = self.X_test.copy()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, columns=[f'feature_{i}' for i in range(df.shape[1])])
            
        return self.bias_auditor.audit_bias(
            df, y_pred, self.y_test
        )
    
    def _check_overfitting(self):
        """Check for overfitting using learning curves"""
        results, fig = self.overfit_detector.check_overfitting(
            self.model, self.X_train, self.y_train
        )
        plt.close(fig)  # Close plot to free memory
        return results
    
    def _analyze_errors(self, y_pred):
        """Analyze model errors"""
        return self.error_analyzer.analyze_errors(
            self.X_test, self.y_test, y_pred
        )
    
    def _generate_summary(self, full_report):
        """Generate human-readable summary"""
        summary = []
        
        # Data drift summary
        drift_features = [
            feat for feat, res in full_report['data_drift'].items() 
            if not feat.startswith('_') and res['drift']
        if drift_features:
            summary.append(
                f"Data drift detected in {len(drift_features)} features: {', '.join(drift_features)}"
            )
        
        # Bias summary
        for feat, res in full_report['bias'].items():
            if feat != 'warning':
                disparities = [r['ppr_disparity'] for r in res['disparity']]
                if any(abs(d) > 1.25 for d in disparities):
                    summary.append(
                        f"Potential bias found for sensitive attribute '{feat}'"
                    )
        
        # Overfitting summary
        if full_report['overfitting']['is_overfit']:
            gap = full_report['overfitting']['overfit_score'] * 100
            summary.append(
                f"Possible overfitting (train-test gap: {gap:.1f}%)"
            )
        
        # Error summary
        error_rate = full_report['errors']['error_rate'] * 100
        summary.append(
            f"Error rate: {error_rate:.1f}%"
        )
        
        return summary
    
    def save_report(self, filepath):
        """Save report to JSON file"""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

