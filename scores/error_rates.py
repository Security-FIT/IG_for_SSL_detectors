#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve

def load_scores(score_file):
    """
    Load scores and labels from a file.
    Format expected: file_name score label
    """
    scores = []
    labels = []
    with open(score_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Format: filename score label
            if len(parts) >= 3:
                try:
                    score = float(parts[1])
                    label = int(parts[2])
                    scores.append(score)
                    labels.append(label)
                except ValueError:
                    continue
    return np.array(labels), np.array(scores)

def plot_rates_vs_threshold(labels, scores, save_path=None, title=None):
    """
    Plot FMR and FNMR vs Threshold.
    """
    fpr, fnr, thresholds = det_curve(labels, scores, pos_label=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, fpr, label='FMR (False Match Rate)', color='red')
    plt.plot(thresholds, fnr, label='FNMR (False Non-Match Rate)', color='blue')
    
    # Find EER for annotation
    # EER is where FMR == FNMR
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer_threshold = thresholds[idx]
    eer_val = (fpr[idx] + fnr[idx]) / 2
    
    plt.axvline(x=eer_threshold, color='green', linestyle='--', label=f'EER Threshold $\\approx$ {eer_threshold:.4f}')
    plt.scatter([eer_threshold], [eer_val], color='green', zorder=5)
    plt.annotate(f'EER: {eer_val:.2%}', (eer_threshold, eer_val), xytext=(10, 10), textcoords='offset points')
    
    # Add thresholds at FMR 0.1% and 1%
    op_points = [0.01, 0.001]
    op_labels = ['1%', '0.1%']
    op_colors = ['orange', 'purple']

    for target, label, color in zip(op_points, op_labels, op_colors):
        # We want the threshold where FMR is closest to target but <= target (closest below)
        # Assuming we want the most generous threshold (highest FMR <= target) to minimize FNMR.
        valid_indices = np.where(fpr <= target)[0]
        if len(valid_indices) > 0:
            # Find the index among valid_indices that has the largest FPR
            # This corresponds to the operating point closest to the target limit
            best_local_idx = np.argmax(fpr[valid_indices])
            best_idx = valid_indices[best_local_idx]
            
            op_threshold = thresholds[best_idx]
            op_actual_fpr = fpr[best_idx]
            op_fnmr = fnr[best_idx]
            
            plt.axvline(x=op_threshold, color=color, linestyle=':', alpha=0.8, 
                        label=f'FMR {label} Thresh $\\approx$ {op_threshold:.4f}')
            
            # Plot markers
            plt.scatter([op_threshold], [op_actual_fpr], color=color, marker='o', zorder=5, s=30)
            plt.scatter([op_threshold], [op_fnmr], color=color, marker='x', zorder=5, s=30)
            
            # Annotate
            plt.annotate(f'FNMR @ {label} FMR: {op_fnmr:.2%}', (op_threshold, op_fnmr),
                         xytext=(10, 10 if op_fnmr < 0.5 else -20), 
                         textcoords='offset points', color=color, fontsize=9,
                         arrowprops=dict(arrowstyle="->", color=color, alpha=0.5))

    # Add thresholds for FNMR 1% and 0.1%
    fnmr_targets = [0.01, 0.001]
    fnmr_labels = ['1%', '0.1%']
    fnmr_colors = ['brown', 'teal']
    
    for fnmr_target, fnmr_label, fnmr_color in zip(fnmr_targets, fnmr_labels, fnmr_colors):
        # We want max FNMR <= target (closest from below)
        valid_indices_fnmr = np.where(fnr <= fnmr_target)[0]
        if len(valid_indices_fnmr) > 0:
            best_local_idx_fnmr = np.argmax(fnr[valid_indices_fnmr])
            best_idx_fnmr = valid_indices_fnmr[best_local_idx_fnmr]
            
            op_threshold_fnmr = thresholds[best_idx_fnmr]
            op_actual_fnmr = fnr[best_idx_fnmr]
            op_fpr_at_fnmr = fpr[best_idx_fnmr]
            
            plt.axvline(x=op_threshold_fnmr, color=fnmr_color, linestyle='-.', alpha=0.8,
                        label=f'FNMR {fnmr_label} Thresh $\\approx$ {op_threshold_fnmr:.4f}')
            
            plt.scatter([op_threshold_fnmr], [op_actual_fnmr], color=fnmr_color, marker='x', zorder=5, s=30)
            plt.scatter([op_threshold_fnmr], [op_fpr_at_fnmr], color=fnmr_color, marker='o', zorder=5, s=30)
            
            plt.annotate(f'FMR @ {fnmr_label} FNMR: {op_fpr_at_fnmr:.2%}', (op_threshold_fnmr, op_fpr_at_fnmr),
                            xytext=(-100, 10 if op_fpr_at_fnmr < 0.5 else -20), 
                            textcoords='offset points', color=fnmr_color, fontsize=9,
                            arrowprops=dict(arrowstyle="->", color=fnmr_color, alpha=0.5))

    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title(title if title else 'FMR and FNMR vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_det_curve(labels, scores, save_path=None, title=None):
    """
    Draw DET curve (FNMR vs FMR)
    """
    fpr, fnr, _ = det_curve(labels, scores, pos_label=1)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, fnr, linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('False Non-Match Rate (FNMR)')
    plt.title(title if title else 'DET Curve')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Plot diagonal (EER line)
    # EER is where plot intersects with y=x
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def evaluate_modifications(baseline_file, modified_file, save_threshold_path=None):
    """
    Calculate the baseline EER, save the threshold, and evaluate FAR, FRR of the modifications on the saved threshold.
    """
    base_labels, base_scores = load_scores(baseline_file)
    mod_labels, mod_scores = load_scores(modified_file)
    
    fpr, fnr, thresholds = det_curve(base_labels, base_scores, pos_label=1)
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer_threshold = thresholds[idx]
    base_eer = (fpr[idx] + fnr[idx]) / 2
    base_far = fpr[idx]
    base_frr = fnr[idx]
    
    print(f"Baseline EER: {base_eer:.2%} at threshold: {eer_threshold:.4f}")
    print(f"Baseline FAR: {base_far:.2%}")
    print(f"Baseline FRR: {base_frr:.2%}")
    
    if save_threshold_path:
        with open(save_threshold_path, 'w') as f:
            f.write(str(eer_threshold))
        print(f"Threshold saved to {save_threshold_path}")
    
    # Evaluate modified scores at the baseline EER threshold
    # Assuming higher score -> positive class (1)
    predictions = (mod_scores >= eer_threshold).astype(int)
    
    tp = np.sum((mod_labels == 1) & (predictions == 1))
    fn = np.sum((mod_labels == 1) & (predictions == 0))
    tn = np.sum((mod_labels == 0) & (predictions == 0))
    fp = np.sum((mod_labels == 0) & (predictions == 1))
    
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    print(f"Modified FAR: {far:.2%}")
    print(f"Modified FRR: {frr:.2%}")
    
    return base_eer, eer_threshold, far, frr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw FMR and FNMR curves from scores.")
    parser.add_argument("score_file", type=str, help="Path to the score file (format: filename score label)")
    parser.add_argument("--output", "-o", type=str, default="curve.png", help="Output filename for the plot")
    parser.add_argument("--mode", type=str, choices=["threshold", "det", "eval_mod"], default="threshold", help="Type of plot: 'threshold' (default), 'det', or 'eval_mod'")
    parser.add_argument("--baseline_file", type=str, help="Path to the baseline score file (required for eval_mod mode)")
    parser.add_argument("--save_threshold", type=str, help="Path to save the calculated baseline threshold")
    parser.add_argument("--title", type=str, help="Title of the plot")

    args = parser.parse_args()
    
    if not args.title:
        args.title = f"{args.mode.capitalize()} Curve for {args.score_file}"

    if args.mode == "eval_mod":
        if not args.baseline_file:
            print("Error: --baseline_file is required for eval_mod mode.")
            sys.exit(1)
        evaluate_modifications(args.baseline_file, args.score_file, args.save_threshold)
        sys.exit(0)

    print(f"Loading scores from {args.score_file}...")
    try:
        labels, scores = load_scores(args.score_file)
    except FileNotFoundError:
        print(f"Error: File {args.score_file} not found.")
        sys.exit(1)

    print(f"Loaded {len(scores)} scores.")
    
    if len(scores) == 0:
        print("No scores loaded. Check file format.")
        sys.exit(1)

    if args.mode == "threshold":
        plot_rates_vs_threshold(labels, scores, save_path=args.output, title=args.title)
    elif args.mode == "det":
        plot_det_curve(labels, scores, save_path=args.output, title=args.title)
