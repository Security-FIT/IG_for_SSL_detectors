#!/usr/bin/env python3
import sys
import argparse
import numpy as np
from sklearn.metrics import det_curve
from tqdm import tqdm

def load_scores(score_file):
    """
    Load scores and labels from a file.
    Returns: dict {filename: {'score': float, 'label': int}}
    """
    data = {}
    with open(score_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Format: filename score label
            if len(parts) >= 3:
                try:
                    filename = parts[0]
                    score = float(parts[1])
                    label = int(parts[2])
                    data[filename] = {'score': score, 'label': label}
                except ValueError:
                    continue
    return data

def get_thresholds(labels, scores):
    """
    Calculate T_high (FMR approx 0.1%) and T_low (FNMR approx 0.1%)
    """
    fpr, fnr, thresholds = det_curve(labels, scores, pos_label=1)
    
    # T_high: Threshold for FMR ~= 0.1% (0.001)
    # We want max FPR <= 0.001
    target_fmr = 0.001
    valid_indices_fpr = np.where(fpr <= target_fmr)[0]
    if len(valid_indices_fpr) > 0:
        best_idx_fpr = valid_indices_fpr[np.argmax(fpr[valid_indices_fpr])]
        t_high = thresholds[best_idx_fpr]
    else:
        # If we can't reach 0.1%, take the strictest possible (max threshold)
        t_high = np.max(thresholds)
        
    # T_low: Threshold for FNMR ~= 0.1% (0.001)
    # We want max FNMR <= 0.001
    target_fnmr = 0.001
    valid_indices_fnr = np.where(fnr <= target_fnmr)[0]
    if len(valid_indices_fnr) > 0:
        best_idx_fnr = valid_indices_fnr[np.argmax(fnr[valid_indices_fnr])]
        t_low = thresholds[best_idx_fnr]
    else:
        # If we can't reach 0.1%, take the loosest possible (min threshold)
        t_low = np.min(thresholds)
        
    return t_high, t_low

def categorize_samples(data_dict, t_high, t_low):
    """
    Categorize samples into CR, CW, Mid.
    CR: Confident Right
    CW: Confident Wrong
    Mid: Middle Ground
    
    Returns: dict {category: set(filenames)}
    """
    categories = {
        'CR': set(),
        'CW': set(),
        'Mid': set()
    }
    
    for filename, info in data_dict.items():
        score = info['score']
        label = info['label']
        
        # Determine confidence
        is_confident_bonafide = score > t_high
        is_confident_spoof = score < t_low
        is_middle = not (is_confident_bonafide or is_confident_spoof)
        
        if is_middle:
            categories['Mid'].add(filename)
            continue
            
        if label == 1: # Bonafide
            if is_confident_bonafide:
                categories['CR'].add(filename)
            elif is_confident_spoof: # Confidently classified as spoof (WRONG)
                categories['CW'].add(filename)
            # If it fell into middle, already handled
            
        elif label == 0: # Spoof
            if is_confident_spoof:
                categories['CR'].add(filename)
            elif is_confident_bonafide: # Confidently classified as bonafide (WRONG)
                categories['CW'].add(filename)
                
    return categories

def main():
    parser = argparse.ArgumentParser(description="Select consensus subsets of recordings.")
    parser.add_argument("--aasist", type=str, required=True, help="Path to AASIST scores")
    parser.add_argument("--camhfa", type=str, required=True, help="Path to CAM++ scores")
    parser.add_argument("--sls", type=str, required=True, help="Path to SLS scores")
    parser.add_argument("--output_dir", type=str, default="outputs/subsets", help="Directory to save output lists")
    
    args = parser.parse_args()
    
    # 1. Load Data
    print("Loading scores...")
    data_aasist = load_scores(args.aasist)
    data_camhfa = load_scores(args.camhfa)
    data_sls = load_scores(args.sls)
    
    # Ensure detailed overlap of filenames
    all_files = set(data_aasist.keys()) & set(data_camhfa.keys()) & set(data_sls.keys())
    print(f"Total common files: {len(all_files)}")
    
    # Process each detector
    detectors = [
        ('AASIST', data_aasist),
        ('CAMHFA', data_camhfa),
        ('SLS', data_sls)
    ]
    
    detector_cats = {}
    
    for name, data in detectors:
        print(f"\nProcessing {name}...")
        
        # Prepare arrays for det_curve
        scores_arr = []
        labels_arr = []
        for info in data.values():
            scores_arr.append(info['score'])
            labels_arr.append(info['label'])
            
        t_high, t_low = get_thresholds(np.array(labels_arr), np.array(scores_arr))
        print(f"  Thresholds: T_high (FMR~0.1%)={t_high:.4f}, T_low (FNMR~0.1%)={t_low:.4f}")
        
        cats = categorize_samples(data, t_high, t_low)
        detector_cats[name] = cats
        print(f"  Counts: CR={len(cats['CR'])}, CW={len(cats['CW'])}, Mid={len(cats['Mid'])}")

    # 2. Consensus
    print("\nCalculating consensus...")
    
    consensus_cr = detector_cats['AASIST']['CR'] & detector_cats['CAMHFA']['CR'] & detector_cats['SLS']['CR']
    consensus_cw = detector_cats['AASIST']['CW'] & detector_cats['CAMHFA']['CW'] & detector_cats['SLS']['CW']
    consensus_mid = detector_cats['AASIST']['Mid'] & detector_cats['CAMHFA']['Mid'] & detector_cats['SLS']['Mid']
    
    # Start intersection with common files to be safe
    consensus_cr = consensus_cr & all_files
    consensus_cw = consensus_cw & all_files
    consensus_mid = consensus_mid & all_files
    
    print(f"Consensus Confident Right: {len(consensus_cr)}")
    print(f"Consensus Confident Wrong: {len(consensus_cw)}")
    print(f"Consensus Middle Ground: {len(consensus_mid)}")
    
    # ---------------------------------------------------------
    # More detailed statistics about the consensus groups
    # ---------------------------------------------------------
    print("\n--- Detailed Consensus Statistics ---")
    
    def print_group_stats(group_name, file_set):
        if not file_set:
            print(f"{group_name}: Empty")
            return
            
        n_bonafide = 0
        n_spoof = 0
        
        for fid in file_set:
            label = data_aasist[fid]['label']
            if label == 1:
                n_bonafide += 1
            else:
                n_spoof += 1
        
        total = len(file_set)
        print(f"{group_name} Total: {total}")
        print(f"  - Bonafide: {n_bonafide} ({n_bonafide/total:.1%})")
        print(f"  - Spoof:    {n_spoof} ({n_spoof/total:.1%})")

    print_group_stats("Consensus Confident Right", consensus_cr)
    print_group_stats("Consensus Confident Wrong", consensus_cw)
    print_group_stats("Consensus Middle Ground", consensus_mid)
    
    # 3. Output
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    def save_list(filename, file_set):
        path = os.path.join(args.output_dir, filename)
        with open(path, 'w') as f:
            for fid in sorted(list(file_set)):
                f.write(f"{fid}\n")
        print(f"Saved {path}")

    save_list("consensus_confident_right.txt", consensus_cr)
    save_list("consensus_confident_wrong.txt", consensus_cw)
    save_list("consensus_middle_ground.txt", consensus_mid)

if __name__ == "__main__":
    main()
