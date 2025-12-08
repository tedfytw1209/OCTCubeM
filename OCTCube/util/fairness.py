import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from typing import Dict, Optional

#save auroc
def auc_ignore_unseen(y_onehot, y_probs):
    """
    Compute macro AUROC (OVR) ignoring classes that do not appear in GT.
    y_onehot: shape (N, C)
    y_probs:  shape (N, C)
    """

    # Convert one-hot to label index vector
    y_true = np.argmax(y_onehot, axis=1)

    # Find only classes that appear in GT
    present_classes = np.unique(y_true)

    # If only a single class appears → skip AUROC for missing classes
    if len(present_classes) == 1:
        c = present_classes[0]
        # Use only that column for score (binary AUC degenerate)
        # AUROC is undefined for one class → use accuracy instead or set 0.5 baseline
        return 0

    # Filter to present classes
    y_onehot_filtered = y_onehot[:, present_classes]
    y_probs_filtered = y_probs[:, present_classes]

    # Compute AUROC on remaining classes only
    return roc_auc_score(
        y_onehot_filtered,
        y_probs_filtered,
        multi_class="ovr",
        average="macro"
    )

# fairness metric
def fariness_score(protected_ground_truth, privileged_ground_truth, protected_pred, privileged_pred, protected_probs=None, privileged_probs=None,protected_gt_onehot=None,privileged_gt_onehot=None):
    ## confusion matrix
    if len(set(protected_ground_truth)) == 1 or len(set(protected_pred)) == 1:
        protected_cm = confusion_matrix(protected_ground_truth, protected_pred, labels=[0, 1])  
    else:
        protected_cm = confusion_matrix(protected_ground_truth, protected_pred)

    if len(set(privileged_ground_truth)) == 1 or len(set(privileged_pred)) == 1:
        privileged_cm = confusion_matrix(privileged_ground_truth, privileged_pred, labels=[0, 1])  
    else:
        privileged_cm = confusion_matrix(privileged_ground_truth, privileged_pred)

    
    # basic model accuracy metrics
    ## protected
    protected_TP = protected_cm[1, 1]
    protected_FP = protected_cm[0, 1]
    protected_TN = protected_cm[0, 0]
    protected_FN = protected_cm[1, 0]
    
    ## privileged
    privileged_TP = privileged_cm[1, 1]
    privileged_FP = privileged_cm[0, 1]
    privileged_TN = privileged_cm[0, 0]
    privileged_FN = privileged_cm[1, 0]
    
    # No.1 Predictive parity: PPV (0 if division by zero)
    protected_PPV = protected_TP / (protected_TP + protected_FP) if (protected_TP + protected_FP) != 0 else 0
    privileged_PPV = privileged_TP / (privileged_TP + privileged_FP) if (privileged_TP + privileged_FP) != 0 else 0

    # No.2 False positive error rate balance: FPR (1 if division by zero)
    protected_FPR = protected_FP / (protected_FP + protected_TN) if (protected_FP + protected_TN) != 0 else 1
    privileged_FPR = privileged_FP / (privileged_FP + privileged_TN) if (privileged_FP + privileged_TN) != 0 else 1

    # No.3 Equalized odds: equal TPR and FPR (0 if division by zero)
    protected_TPR = protected_TP / (protected_TP + protected_FN) if (protected_TP + protected_FN) != 0 else 0
    privileged_TPR = privileged_TP / (privileged_TP + privileged_FN) if (privileged_TP + privileged_FN) != 0 else 0

    # No.4 Conditional use accuracy equality: equal PPV, NPV (0 if division by zero)
    protected_NPV = protected_TN / (protected_TN + protected_FN) if (protected_TN + protected_FN) != 0 else 0
    privileged_NPV = privileged_TN / (privileged_TN + privileged_FN) if (privileged_TN + privileged_FN) != 0 else 0

    # No.5 Treatment equality: FN/FP (0 if division by zero)
    protected_te = protected_FN / protected_FP if protected_FP != 0 else 0
    privileged_te = privileged_FN / privileged_FP if privileged_FP != 0 else 0

    # No.6 False negative error rate balance: FNR = FN/(FN + TP) (1 if division by zero)
    protected_FNR = protected_FN / (protected_FN + protected_TP) if (protected_FN + protected_TP) != 0 else 1
    privileged_FNR = privileged_FN / (privileged_FN + privileged_TP) if (privileged_FN + privileged_TP) != 0 else 1

    # No.7 Overall accuracy equality (0 if calculation fails)
    protected_ACC = (
        accuracy_score(protected_ground_truth, protected_pred)
        if protected_ground_truth is not None
        and protected_pred is not None
        and (
            (isinstance(protected_ground_truth, (pd.DataFrame, pd.Series)) and not protected_ground_truth.empty)
            or (isinstance(protected_ground_truth, np.ndarray) and protected_ground_truth.size > 0)
        )
        and (
            (isinstance(protected_pred, (pd.DataFrame, pd.Series)) and not protected_pred.empty)
            or (isinstance(protected_pred, np.ndarray) and protected_pred.size > 0)
        )
        else 0
    )

    privileged_ACC = (
        accuracy_score(privileged_ground_truth, privileged_pred)
        if privileged_ground_truth is not None
        and privileged_pred is not None
        and (
            (isinstance(privileged_ground_truth, (pd.DataFrame, pd.Series)) and not privileged_ground_truth.empty)
            or (isinstance(privileged_ground_truth, np.ndarray) and privileged_ground_truth.size > 0)
        )
        and (
            (isinstance(privileged_pred, (pd.DataFrame, pd.Series)) and not privileged_pred.empty)
            or (isinstance(privileged_pred, np.ndarray) and privileged_pred.size > 0)
        )
        else 0
    )

    # No.8 AUROC (Area Under ROC Curve) - requires probability scores
    protected_AUROC = 0
    privileged_AUROC = 0
    
    try:
        protected_AUROC = auc_ignore_unseen(
                protected_gt_onehot, 
                protected_probs, 
            )
    except Exception as e:
        protected_AUROC = 0
    
    try:
            privileged_AUROC = auc_ignore_unseen(
                    privileged_gt_onehot, 
                    privileged_probs,
                )
    except Exception as e:
        privileged_AUROC = 0

    return (round(protected_PPV, 4), round(privileged_PPV, 4), 
            round(protected_FPR, 4), round(privileged_FPR, 4), 
            round(protected_TPR, 4), round(privileged_TPR, 4), 
            round(protected_NPV, 4), round(privileged_NPV, 4), 
            round(protected_te, 4), round(privileged_te, 4), 
            round(protected_FNR, 4), round(privileged_FNR, 4),
            round(protected_ACC, 4), round(privileged_ACC, 4),
            round(protected_AUROC, 4), round(privileged_AUROC, 4))


class FairnessAnalyzerWithCI:
    """
    Enhanced fairness analyzer with confidence intervals and uncertainty integration.
    """
    
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def compute_fairness_metrics_with_ci(self,
                                       protected_gt: np.ndarray,
                                       privileged_gt: np.ndarray, 
                                       protected_pred: np.ndarray,
                                       privileged_pred: np.ndarray,
                                       protected_probs: np.ndarray = None,
                                       privileged_probs: np.ndarray = None,
                                       protected_gt_onehot: np.ndarray = None,
                                       privileged_gt_onehot: np.ndarray = None) -> Dict:
        """
        Compute fairness metrics with confidence intervals using bootstrap sampling.
        
        Args:
            protected_gt: Protected group ground truth labels
            privileged_gt: Privileged group ground truth labels
            protected_pred: Protected group predictions
            privileged_pred: Privileged group predictions
            
        Returns:
            Dictionary with fairness metrics and confidence intervals
        """
        
        def compute_single_fairness_metrics(prot_gt, priv_gt, prot_pred, priv_pred, prot_probs=None, priv_probs=None, prot_gt_onehot=None, priv_gt_onehot=None):
            """Compute fairness metrics for a single bootstrap sample."""
            
            # Use existing fairness function
            metrics = fariness_score(prot_gt, priv_gt, prot_pred, priv_pred, prot_probs, priv_probs, prot_gt_onehot, priv_gt_onehot)
            
            return {
                'protected_PPV': metrics[0],
                'privileged_PPV': metrics[1], 
                'protected_FPR': metrics[2],
                'privileged_FPR': metrics[3],
                'protected_TPR': metrics[4],
                'privileged_TPR': metrics[5],
                'protected_NPV': metrics[6],
                'privileged_NPV': metrics[7],
                'protected_TE': metrics[8],
                'privileged_TE': metrics[9],
                'protected_FNR': metrics[10],
                'privileged_FNR': metrics[11],
                'protected_ACC': metrics[12],
                'privileged_ACC': metrics[13],
                'protected_AUROC': metrics[14],
                'privileged_AUROC': metrics[15]
            }
        
        # Compute original metrics
        original_metrics = compute_single_fairness_metrics(
            protected_gt, privileged_gt, protected_pred, privileged_pred, protected_probs, privileged_probs, protected_gt_onehot, privileged_gt_onehot
        )
        
        # Bootstrap sampling for confidence intervals
        bootstrap_results = {key: [] for key in original_metrics.keys()}
        
        # Add fairness difference metrics
        fairness_differences = {
            'PPV_diff': [],
            'FPR_diff': [], 
            'TPR_diff': [],
            'NPV_diff': [],
            'TE_diff': [],
            'FNR_diff': [],
            'ACC_diff': [],
            'AUROC_diff': []
        }
        
        np.random.seed(42)  # For reproducibility
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample indices
            n_protected = len(protected_gt)
            n_privileged = len(privileged_gt)
            
            prot_indices = np.random.choice(n_protected, n_protected, replace=True)
            priv_indices = np.random.choice(n_privileged, n_privileged, replace=True)
            
            # Bootstrap samples
            boot_prot_gt = protected_gt[prot_indices]
            boot_prot_pred = protected_pred[prot_indices]
            boot_priv_gt = privileged_gt[priv_indices]
            boot_priv_pred = privileged_pred[priv_indices]
            
            # Bootstrap probability samples if available
            boot_prot_probs = protected_probs[prot_indices] if protected_probs is not None else None
            boot_priv_probs = privileged_probs[priv_indices] if privileged_probs is not None else None
            
            boot_prot_gt_onehot = protected_gt_onehot[prot_indices] if protected_gt_onehot is not None else None
            boot_priv_gt_onehot = privileged_gt_onehot[priv_indices] if privileged_gt_onehot is not None else None
            
            # Compute metrics for bootstrap sample
            boot_metrics = compute_single_fairness_metrics(
                boot_prot_gt, boot_priv_gt, boot_prot_pred, boot_priv_pred, boot_prot_probs, boot_priv_probs, boot_prot_gt_onehot, boot_priv_gt_onehot
            )
            
            for key, value in boot_metrics.items():
                bootstrap_results[key].append(value)
            
            # Compute fairness differences
            fairness_differences['PPV_diff'].append(
                boot_metrics['privileged_PPV'] - boot_metrics['protected_PPV']
            )
            fairness_differences['FPR_diff'].append(
                boot_metrics['privileged_FPR'] - boot_metrics['protected_FPR']
            )
            fairness_differences['TPR_diff'].append(
                boot_metrics['privileged_TPR'] - boot_metrics['protected_TPR']
            )
            fairness_differences['NPV_diff'].append(
                boot_metrics['privileged_NPV'] - boot_metrics['protected_NPV']
            )
            fairness_differences['TE_diff'].append(
                boot_metrics['privileged_TE'] - boot_metrics['protected_TE']
            )
            fairness_differences['FNR_diff'].append(
                boot_metrics['privileged_FNR'] - boot_metrics['protected_FNR']
            )
            fairness_differences['ACC_diff'].append(
                boot_metrics['privileged_ACC'] - boot_metrics['protected_ACC']
            )
            fairness_differences['AUROC_diff'].append(
                boot_metrics['privileged_AUROC'] - boot_metrics['protected_AUROC']
            )
        
        # Compute confidence intervals
        def compute_ci(values):
            if len(values) == 0:
                return (0, 0)
            lower_percentile = (self.alpha / 2) * 100
            upper_percentile = (1 - self.alpha / 2) * 100
            return (
                np.percentile(values, lower_percentile),
                np.percentile(values, upper_percentile)
            )
        
        # Prepare results
        results = {
            'original_metrics': original_metrics,
            'confidence_intervals': {},
            'fairness_differences': {},
            'fairness_differences_ci': {},
            'bootstrap_samples': len(bootstrap_results[list(bootstrap_results.keys())[0]]),
            'confidence_level': self.confidence_level
        }
        
        # Add confidence intervals for each metric
        for key, values in bootstrap_results.items():
            if len(values) > 0:
                results['confidence_intervals'][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'ci_lower': compute_ci(values)[0],
                    'ci_upper': compute_ci(values)[1]
                }
        
        # Add fairness differences with confidence intervals
        for key, values in fairness_differences.items():
            if len(values) > 0:
                original_diff = (
                    original_metrics[f'privileged_{key.split("_")[0]}'] - 
                    original_metrics[f'protected_{key.split("_")[0]}']
                )
                
                results['fairness_differences'][key] = {
                    'original': original_diff,
                    'absolute': abs(original_diff),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'ci_lower': compute_ci(values)[0],
                    'ci_upper': compute_ci(values)[1],
                    'is_significant': not (compute_ci(values)[0] <= 0 <= compute_ci(values)[1])
                }
        
        return results
    
    def plot_fairness_with_ci(self, fairness_results: Dict, save_path: Optional[str] = None):
        """
        Plot fairness metrics with confidence intervals.
        
        Args:
            fairness_results: Results from compute_fairness_metrics_with_ci
            save_path: Path to save the plot
        """
        
        # Extract metrics for plotting
        metrics_to_plot = ['PPV', 'FPR', 'TPR', 'NPV', 'ACC', 'AUROC']
        protected_values = []
        privileged_values = []
        protected_cis = []
        privileged_cis = []
        
        for metric in metrics_to_plot:
            prot_key = f'protected_{metric}'
            priv_key = f'privileged_{metric}'
            
            if prot_key in fairness_results['confidence_intervals']:
                protected_values.append(fairness_results['original_metrics'][prot_key])
                privileged_values.append(fairness_results['original_metrics'][priv_key])
                
                prot_ci = fairness_results['confidence_intervals'][prot_key]
                priv_ci = fairness_results['confidence_intervals'][priv_key]
                
                protected_cis.append([
                    fairness_results['original_metrics'][prot_key] - prot_ci['ci_lower'],
                    prot_ci['ci_upper'] - fairness_results['original_metrics'][prot_key]
                ])
                privileged_cis.append([
                    fairness_results['original_metrics'][priv_key] - priv_ci['ci_lower'],
                    priv_ci['ci_upper'] - fairness_results['original_metrics'][priv_key]
                ])
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Fairness metrics with error bars
        x_pos = np.arange(len(metrics_to_plot))
        width = 0.35
        
        ax1.bar(x_pos - width/2, protected_values, width, 
                yerr=np.array(protected_cis).T, label='Protected Group', 
                alpha=0.8, capsize=5)
        ax1.bar(x_pos + width/2, privileged_values, width,
                yerr=np.array(privileged_cis).T, label='Privileged Group',
                alpha=0.8, capsize=5)
        
        ax1.set_xlabel('Fairness Metrics')
        ax1.set_ylabel('Metric Value')
        ax1.set_title(f'Fairness Metrics with {fairness_results["confidence_level"]*100}% Confidence Intervals')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(metrics_to_plot)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Fairness differences with significance
        diff_metrics = []
        diff_values = []
        diff_cis = []
        colors = []
        
        for key, diff_data in fairness_results['fairness_differences'].items():
            metric_name = key.replace('_diff', '')
            diff_metrics.append(metric_name)
            diff_values.append(diff_data['original'])
            
            ci_lower = diff_data['ci_lower']
            ci_upper = diff_data['ci_upper']
            
            diff_cis.append([
                diff_data['original'] - ci_lower,
                ci_upper - diff_data['original']
            ])
            
            # Color based on significance
            colors.append('red' if diff_data['is_significant'] else 'blue')
        
        bars = ax2.bar(range(len(diff_metrics)), diff_values, 
                      yerr=np.array(diff_cis).T, capsize=5, color=colors, alpha=0.7)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Fairness Metrics')
        ax2.set_ylabel('Privileged - Protected (Difference)')
        ax2.set_title('Fairness Differences with Significance Testing')
        ax2.set_xticks(range(len(diff_metrics)))
        ax2.set_xticklabels(diff_metrics)
        ax2.grid(True, alpha=0.3)
        
        # Add legend for significance
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Significant Difference'),
            Patch(facecolor='blue', alpha=0.7, label='Non-significant Difference')
        ]
        ax2.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def bootstrap_ci(data: np.ndarray, statistic_func, n_bootstrap: int = 1000, confidence_level: float = 0.95):
    """
    Compute confidence interval for a statistic using bootstrap sampling.
    
    Args:
        data: Input data
        statistic_func: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for CI
        
    Returns:
        Tuple of (lower_bound, upper_bound, mean_estimate)
    """
    np.random.seed(42)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence_level
    
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    mean_estimate = np.mean(bootstrap_stats)
    
    return ci_lower, ci_upper, mean_estimate