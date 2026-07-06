'''
Evaluation Visualization Suite for SAM 3 Semantic Segmentation.

Generates two publication-ready visualizations:
1. A multi-class chart comparing Path, Obstacles, and Background on the same axes.
2. An overall 'Micro-Averaged' chart evaluating the model holistically.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from dotenv import load_dotenv
import os
from typing import Dict

def calculate_ece(y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 10) -> float:
    """Calculates the Expected Calibration Error (ECE) for model predictions."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        
        if i == n_bins - 1:
            in_bin = (y_scores >= bin_lower) & (y_scores <= bin_upper)
        else:
            in_bin = (y_scores >= bin_lower) & (y_scores < bin_upper)
        
        if not np.any(in_bin):
            continue
            
        bin_accuracy = np.mean(y_true[in_bin])
        bin_confidence = np.mean(y_scores[in_bin])
        bin_weight = np.sum(in_bin) / total_samples
        
        ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
    return ece

def find_probability_columns(df: pd.DataFrame) -> Dict[int, str]:
    """
    Dynamically identifies column names for each class ID by inspecting the DataFrame.
    Safely resolves variations like 'prob_class_0' or 'prob_path'.
    """
    columns = df.columns.tolist()
    mapping = {}
    
    # Check for literal class ID strings first
    for cid, pattern in [(0, "0"), (1, "1"), (2, "2")]:
        for col in columns:
            if pattern in col and "prob" in col.lower():
                mapping[cid] = col
                break
                
    # If not found numerically, fall back to semantic keyword matching
    semantic_patterns = {0: ["path", "road"], 1: ["obstacle", "obs"], 2: ["background", "bg"]}
    for cid, keywords in semantic_patterns.items():
        if cid not in mapping:
            for col in columns:
                if any(kw in col.lower() for kw in keywords) and "prob" in col.lower():
                    mapping[cid] = col
                    break
                    
    # Ultimate fallback assertion if columns are completely obscured
    for cid in [0, 1, 2]:
        if cid not in mapping:
            raise KeyError(
                f"Could not automatically detect probability column for class {cid}. "
                f"Available columns: {columns}"
            )
            
    return mapping

def plot_multiclass_metrics(df: pd.DataFrame, col_map: Dict[int, str], save_path: str) -> None:
    """Plots PR and Calibration curves for all classes on a single shared figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    classes = {0: 'Path', 1: 'Obstacles', 2: 'Background'}
    colors = {0: 'mediumseagreen', 1: 'crimson', 2: 'royalblue'}

    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

    for class_id, class_name in classes.items():
        # Binarize the labels for the specific class context
        y_true = (df['true_class'].values == class_id).astype(int)
        y_scores = df[col_map[class_id]].values

        # 1. Precision-Recall Curve Calculations
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        ar = np.mean(recall)
        
        metrics_label = f'{class_name} (AP: {ap:.3f} | AR: {ar:.3f})'
        ax1.plot(recall, precision, color=colors[class_id], linewidth=2, label=metrics_label)

        # 2. Calibration Curve & Brier Score Calculations
        ece = calculate_ece(y_true, y_scores, n_bins=10)
        brier = brier_score_loss(y_true, y_scores)
        prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10, strategy='uniform')
        
        calib_label = f'{class_name} (ECE: {ece:.3f} | Brier: {brier:.3f})'
        ax2.plot(prob_pred, prob_true, marker='o', color=colors[class_id], linewidth=2, label=calib_label)

    # Formatting ax1 (Precision-Recall Axes)
    ax1.set_title('Precision-Recall Curve by Class', fontsize=14)
    ax1.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax1.set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower left', fontsize=11)

    # Formatting ax2 (Reliability Axes)
    ax2.set_title('Reliability Diagram by Class', fontsize=14)
    ax2.set_xlabel('Mean Predicted Confidence', fontsize=12)
    ax2.set_ylabel('Actual Fraction of True Positives', fontsize=12)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=11)

    fig.suptitle('SAM 3 "Unfrozen" Semantic Segmentation (Epoch 24): Multi-Class Performance over Validation Set', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Multi-class plot saved to: {save_path}")

def plot_overall_metrics(df: pd.DataFrame, col_map: Dict[int, str], save_path: str) -> None:
    """Pools all classes together for a Micro-Averaged overall performance evaluation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pool all predictions into a single flat array (Micro-Averaging approach)
    y_true_all = []
    y_scores_all = []
    
    for class_id in [0, 1, 2]:
        y_true_all.extend((df['true_class'].values == class_id).astype(int))
        y_scores_all.extend(df[col_map[class_id]].values)
        
    y_true = np.array(y_true_all)
    y_scores = np.array(y_scores_all)

    # 1. Micro-Averaged Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    ar = np.mean(recall)
    
    ax1.plot(recall, precision, color='indigo', linewidth=2, label=f'Overall (AP: {ap:.3f} | AR: {ar:.3f})')
    ax1.set_title('Overall Precision-Recall Curve (Micro-Averaged)', fontsize=14)
    ax1.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax1.set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower left', fontsize=12)

    # 2. Micro-Averaged Calibration Curve
    ece = calculate_ece(y_true, y_scores, n_bins=10)
    brier = brier_score_loss(y_true, y_scores)
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10, strategy='uniform')

    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    ax2.plot(prob_pred, prob_true, marker='o', color='indigo', linewidth=2, 
             label=f'Overall Calibration (ECE: {ece:.3f} | Brier: {brier:.3f})')
    
    ax2.set_title('Overall Reliability Diagram', fontsize=14)
    ax2.set_xlabel('Mean Predicted Confidence', fontsize=12)
    ax2.set_ylabel('Actual Fraction of True Positives', fontsize=12)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=12)

    fig.suptitle('SAM 3 "Unfrozen" Semantic Segmentation (Epoch 24): Multi-Class Performance over Validation Set', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Overall metrics plot saved to: {save_path}")

if __name__ == "__main__":
    load_dotenv()
    ROOT_PATH = os.getenv("ROOT_PATH")

    if ROOT_PATH is None:
        raise ValueError("ROOT_PATH environment variable is not set. Please check your .env file.")
    
    DATA_RELATIVE_PATH = os.getenv("DATA_RELATIVE_PATH")

    if DATA_RELATIVE_PATH is None:
        raise ValueError("DATA_RELATIVE_PATH environment variable is not set. Please check your .env file.")
    
    CSV_FILE = os.path.join(ROOT_PATH, DATA_RELATIVE_PATH, "segmentation_metrics_val.csv")
    MULTICLASS_OUTPUT = os.path.join(ROOT_PATH, DATA_RELATIVE_PATH, "sam3_4_metrics_by_class.png")
    OVERALL_OUTPUT = os.path.join(ROOT_PATH, DATA_RELATIVE_PATH, "sam3_u24_metrics_overall.png")

    print(f"Reading evaluation metrics from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)

    # Dynamically match whatever naming convention is stored in measure.py output
    col_map = find_probability_columns(df)

    plot_multiclass_metrics(df, col_map, MULTICLASS_OUTPUT)
    plot_overall_metrics(df, col_map, OVERALL_OUTPUT)