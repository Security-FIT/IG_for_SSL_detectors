
import pandas as pd
import numpy as np
import itertools
import sys

def calculate_eer_numpy(target_scores, nontarget_scores):
    """
    Calculates EER using numpy.
    Auto-detects whether higher scores indicate target or non-target.
    Returns the lower EER of the two possibilities.
    """
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        return np.nan
        
    target_scores = np.array(target_scores)
    nontarget_scores = np.array(nontarget_scores)
    
    def _get_eer(t_scores, n_scores):
        all_scores = np.concatenate([t_scores, n_scores])
        all_labels = np.concatenate([np.ones(len(t_scores)), np.zeros(len(n_scores))])
        
        # Sort
        indices = np.argsort(all_scores)
        sorted_labels = all_labels[indices]
        
        # Cumulative sums
        cum_target = np.cumsum(sorted_labels)
        cum_nontarget = np.cumsum(1 - sorted_labels)
        
        n_targets = cum_target[-1]
        n_nontargets = cum_nontarget[-1]
        
        # At index i (threshold just above value[i]):
        # Classified Negative: 0..i
        # FN = targets in 0..i = cum_target[i]
        # TN = nontargets in 0..i = cum_nontarget[i]
        
        # FNR = FN / Total_Targets
        fnr = cum_target / n_targets
        
        # FPR = FP / Total_NonTargets
        # FP = Total_NonTargets - TN
        fpr = (n_nontargets - cum_nontarget) / n_nontargets
        
        # We need to prepend the state before the first element (threshold -inf)
        # where everything is Positive.
        # FN=0, FNR=0
        # TN=0, FP=Total_NonTarget, FPR=1
        
        fnr = np.concatenate([[0], fnr])
        fpr = np.concatenate([[1], fpr])
        
        # Find intersection
        diffs = np.abs(fpr - fnr)
        min_idx = np.argmin(diffs)
        
        return (fpr[min_idx] + fnr[min_idx]) / 2.0

    # Try both polarities
    eer_1 = _get_eer(target_scores, nontarget_scores)
    eer_2 = _get_eer(-target_scores, -nontarget_scores)
    
    return min(eer_1, eer_2)

def main():
    # 1. Load Data
    try:
        details_df = pd.read_csv("outputs/final_selection_132_details.csv")
        artefacts_df = pd.read_csv("outputs/ig_recordings_artefacts.csv")
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        sys.exit(1)

    artefacts_df.rename(columns={'name': 'FileID'}, inplace=True)
    merged_df = pd.merge(details_df, artefacts_df, on='FileID')

    metric_cols = [
        "duration_s",
        "speech_length_s",
        "leading_silence_s",
        "trailing_silence_s",
        "voiced_frames",
        "unvoiced_frames",
        "voiced_unvoiced_ratio",
        "rms_first_100ms",
        "rmse_mean",
        "rmse_amplitude",
    ]
    
    # Ensure numeric
    for col in metric_cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
        
    # remove rows with NaN in metrics
    merged_df.dropna(subset=metric_cols, inplace=True)

    # 2. Global Bonafide vs Spoof
    print("Computing Global EER (Bonafide vs Spoof)...")
    global_results = []
    
    bonafide_scores = merged_df[merged_df['Label'] == 'bonafide']
    spoof_scores = merged_df[merged_df['Label'] == 'spoof']
    
    for metric in metric_cols:
        eer = calculate_eer_numpy(bonafide_scores[metric], spoof_scores[metric])
        global_results.append({'Metric': metric, 'EER': eer})
        
    global_df = pd.DataFrame(global_results)
    print("\nGlobal EER Results:")
    print(global_df.to_string(index=False))
    global_df.to_csv("outputs/artefact_eer_global.csv", index=False)

    # 3. Pairwise SelectionReason
    print("\nComputing Pairwise EER between Selection Reasons...")
    reasons = sorted(merged_df['SelectionReason'].unique())
    pairwise_results = []
    
    for r1, r2 in itertools.combinations(reasons, 2):
        scores_1 = merged_df[merged_df['SelectionReason'] == r1]
        scores_2 = merged_df[merged_df['SelectionReason'] == r2]
        
        # Skip if too few samples
        if len(scores_1) < 2 or len(scores_2) < 2:
            continue
            
        for metric in metric_cols:
            eer = calculate_eer_numpy(scores_1[metric], scores_2[metric])
            pairwise_results.append({
                'Group_1': r1, 
                'Group_2': r2, 
                'Metric': metric, 
                'EER': eer
            })
            
    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_df.to_csv("outputs/artefact_eer_pairwise.csv", index=False)
    print(f"Calculated {len(pairwise_df)} pairwise comparisons. Saved to outputs/artefact_eer_pairwise.csv")

    # 4. Highlight interesting pairs (Bonafide vs Spoof subgroups)
    # Filter for pairs where one is Bonafide and one is Spoof
    print("\n--- Selected EERs: Bonafide Subgroups vs Spoof Subgroups ---")
    
    # Simple heuristic: Check if 'Bonafide' is in the name
    def is_bonafide(name): return 'Bonafide' in name
    def is_spoof(name): return 'Spoof' in name
    
    # Filter dataframe
    subset = pairwise_df[pairwise_df.apply(lambda row: 
        (is_bonafide(row['Group_1']) and is_spoof(row['Group_2'])) or 
        (is_bonafide(row['Group_2']) and is_spoof(row['Group_1'])), axis=1)]
        
    # Sort by EER ascending (best separators)
    subset_sorted = subset.sort_values('EER')
    
    pd.set_option('display.max_rows', 20)
    print(subset_sorted.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
