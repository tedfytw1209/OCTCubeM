"""
Uncertainty Quantification Module

This module provides uncertainty quantification methods for model evaluation:
- Reject Option Classification with confidence thresholds
- Conformal Prediction for prediction sets with coverage guarantees
- Model calibration assessment

"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class UncertaintyQuantifier:
    """
    Uncertainty quantification methods for model predictions.
    """
    
    def __init__(self, num_classes: int = 2, alpha: float = 0.1):
        self.num_classes = num_classes
        self.alpha = alpha  # For conformal prediction
        self.target_coverage = 1 - alpha
        
    def reject_option_classification(self, 
                                   probabilities: np.ndarray, 
                                   true_labels: np.ndarray,
                                   strategy: str = 'accuracy_coverage') -> Dict:
        """
        Perform Reject Option Classification analysis.
        
        Args:
            probabilities: Predicted probabilities (n_samples, n_classes)
            true_labels: True class labels (n_samples,)
            strategy: Threshold selection strategy
            
        Returns:
            Dictionary with ROC results
        """
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        
        # Test range of thresholds
        thresholds = np.linspace(0.1, 0.99, 50)
        results = []
        
        for threshold in thresholds:
            accepted_mask = confidences >= threshold
            
            if np.sum(accepted_mask) == 0:
                continue
                
            accepted_preds = predictions[accepted_mask]
            accepted_labels = true_labels[accepted_mask]
            
            coverage = np.sum(accepted_mask) / len(true_labels)
            accuracy = accuracy_score(accepted_labels, accepted_preds)
            
            results.append({
                'threshold': threshold,
                'coverage': coverage,
                'accuracy': accuracy,
                'n_accepted': np.sum(accepted_mask),
                'n_rejected': np.sum(~accepted_mask)
            })
        
        # Select optimal threshold based on strategy
        if strategy == 'accuracy_coverage':
            scores = [r['accuracy'] * r['coverage'] for r in results]
            best_idx = np.argmax(scores)
        elif strategy == 'target_coverage':
            coverages = [r['coverage'] for r in results]
            best_idx = np.argmin(np.abs(np.array(coverages) - 0.8))
        else:
            scores = [r['accuracy'] for r in results if r['coverage'] >= 0.7]
            if scores:
                best_idx = np.argmax([r['accuracy'] for r in results if r['coverage'] >= 0.7])
            else:
                best_idx = np.argmax([r['accuracy'] for r in results])
        
        optimal_result = results[best_idx]
        optimal_result['rejection_rate'] = optimal_result['n_rejected'] / len(true_labels)
        
        return {
            'optimal_threshold': optimal_result['threshold'],
            'coverage': optimal_result['coverage'],
            'accuracy': optimal_result['accuracy'],
            'rejection_rate': optimal_result['rejection_rate'],
            'n_accepted': optimal_result['n_accepted'],
            'n_rejected': optimal_result['n_rejected'],
            'all_thresholds': results
        }
    
    def conformal_prediction(self, 
                           cal_probabilities: np.ndarray,
                           cal_labels: np.ndarray,
                           test_probabilities: np.ndarray,
                           test_labels: np.ndarray) -> Dict:
        """
        Perform Conformal Prediction analysis.
        
        Args:
            cal_probabilities: Calibration set probabilities
            cal_labels: Calibration set labels
            test_probabilities: Test set probabilities
            test_labels: Test set labels
            
        Returns:
            Dictionary with conformal prediction results
        """
        # Calculate conformity scores on calibration set
        n_cal = len(cal_labels)
        cal_scores = []
        
        for i in range(n_cal):
            true_class = cal_labels[i]
            true_class_prob = cal_probabilities[i, true_class]
            score = 1 - true_class_prob
            cal_scores.append(score)
        
        cal_scores = np.array(cal_scores)
        
        # Calculate quantile for desired coverage
        quantile_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        quantile_level = min(quantile_level, 1.0)
        quantile = np.quantile(cal_scores, quantile_level)
        
        # Generate prediction sets for test samples
        prediction_sets = []
        n_test = len(test_labels)
        
        for i in range(n_test):
            threshold = 1 - quantile
            included_classes = []
            
            for class_idx in range(self.num_classes):
                if test_probabilities[i, class_idx] >= threshold:
                    included_classes.append(class_idx)
            
            if len(included_classes) == 0:
                best_class = np.argmax(test_probabilities[i])
                included_classes = [best_class]
            
            prediction_sets.append(included_classes)
        
        # Evaluate coverage and efficiency
        covered = 0
        set_sizes = []
        
        for i, (true_label, pred_set) in enumerate(zip(test_labels, prediction_sets)):
            if true_label in pred_set:
                covered += 1
            set_sizes.append(len(pred_set))
        
        empirical_coverage = covered / len(test_labels)
        avg_set_size = np.mean(set_sizes)
        singleton_rate = np.mean([size == 1 for size in set_sizes])
        
        return {
            'target_coverage': self.target_coverage,
            'empirical_coverage': empirical_coverage,
            'coverage_gap': empirical_coverage - self.target_coverage,
            'average_set_size': avg_set_size,
            'singleton_rate': singleton_rate,
            'quantile_threshold': quantile,
            'prediction_sets': prediction_sets,
            'n_calibration': n_cal,
            'n_test': n_test
        }
    
    def calibration_assessment(
        self,
        probabilities: np.ndarray,
        true_labels: np.ndarray,
        n_bins: int = 20
    ) -> Dict:
        """
        Evaluate model calibration, including:
        - Overall (top-1) Expected Calibration Error (ECE)
        - Class-wise ECE for each individual class

        Args:
            probabilities: Predicted probabilities, shape (N, C)
            true_labels: Ground-truth labels, shape (N,)
            n_bins: Number of bins to compute calibration

        Returns:
            Dictionary containing calibration metrics.
        """

        # ---------------------------------------------------------
        # Overall / top-1 calibration
        # ---------------------------------------------------------
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        accuracies = (predictions == true_labels).astype(float)

        # Calibration curve (for plotting / diagnostics)
        frac_pos, mean_pred = calibration_curve(
            accuracies, confidences, n_bins=n_bins
        )

        # Generic ECE helper
        def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 20) -> float:
            """
            Compute Expected Calibration Error (ECE) for binary probs/labels.
            """
            bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece_val = 0.0
            for bl, bu in zip(bin_lowers, bin_uppers):
                in_bin = (probs > bl) & (probs <= bu)
                if not np.any(in_bin):
                    continue

                bin_accuracy = labels[in_bin].mean()
                bin_confidence = probs[in_bin].mean()
                prop_in_bin = in_bin.mean()

                ece_val += np.abs(bin_confidence - bin_accuracy) * prop_in_bin

            return float(ece_val)

        # Overall ECE
        overall_ece = _compute_ece(confidences, accuracies, n_bins=n_bins)

        # ---------------------------------------------------------
        # Class-wise ECE
        # ---------------------------------------------------------
        num_classes = probabilities.shape[1]
        classwise_ece = []

        for k in range(num_classes):
            class_probs = probabilities[:, k]
            class_labels = (true_labels == k).astype(float)

            # Avoid ECE issues if the model never outputs class k
            if np.all(class_probs == 0):
                class_ece = 0.0
            else:
                class_ece = _compute_ece(class_probs, class_labels, n_bins=n_bins)

            classwise_ece.append(class_ece)

        mean_classwise_ece = float(np.mean(classwise_ece)) if num_classes > 0 else 0.0

        # ---------------------------------------------------------
        # Return results
        # ---------------------------------------------------------
        return {
            # Overall calibration
            "ECE": overall_ece,
            "mean_confidence": float(np.mean(confidences)),
            "mean_accuracy": float(np.mean(accuracies)),
            "fraction_positives": frac_pos.tolist(),
            "mean_predicted_values": mean_pred.tolist(),
            "confidence_distribution": confidences.tolist(),
            # Class-wise calibration
            "classwise_ECE": classwise_ece,        # list of length C
            "mean_classwise_ECE": mean_classwise_ece,
        }
