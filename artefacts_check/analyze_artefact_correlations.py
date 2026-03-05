import pandas as pd
import numpy as np
import os

def main():
    # Paths
    corr_path = "outputs/ig_artefact_correlations.csv"
    details_path = "outputs/final_selection_132_details.csv"
    
    if not os.path.exists(corr_path) or not os.path.exists(details_path):
        print(f"Error: Files not found. Check {corr_path} and {details_path}")
        return

    # Load data
    df_corr = pd.read_csv(corr_path)
    df_details = pd.read_csv(details_path)
    
    # Check loaded data
    print(f"Loaded {len(df_corr)} correlation rows.")
    print(f"Loaded {len(df_details)} detail rows.")

    # Merge
    # df_corr has 'file_id', df_details has 'FileID'
    merged = pd.merge(df_corr, df_details, left_on='file_id', right_on='FileID', how='inner')
    
    print(f"Merged dataset size: {len(merged)}")
    
    # Define metrics to analyze
    metrics = ['pearson_rmse', 'pearson_voiced', 'spearman_rmse', 'spearman_voiced']
    
    # Grouping 1: Per Model and Label (Bonafide vs Spoof)
    print("\n" + "="*80)
    print("ANALYSIS 1: Per Model and Label (Bonafide vs Spoof)")
    print("="*80)
    g1 = merged.groupby(['model', 'Label'])[metrics].agg(['mean', 'std', 'count'])
    print(g1)
    
    # Grouping 2: Per Model and Selection Reason (Detailed)
    print("\n" + "="*80)
    print("ANALYSIS 2: Per Model and Detailed Selection Reason")
    print("="*80)
    g2 = merged.groupby(['model', 'SelectionReason'])[metrics].agg(['mean', 'count'])
    print(g2)
    
    # Grouping 3: Per Label only (Average across models? Or just aggregating all data points)
    # Aggregating all data points ignores that models might behave differently, but good for overview
    print("\n" + "="*80)
    print("ANALYSIS 3: Global Per Label")
    print("="*80)
    g3 = merged.groupby(['Label'])[metrics].agg(['mean', 'std', 'count'])
    print(g3)

    # Flatten columns for CSV output
    g1.columns = ['_'.join(col).strip() for col in g1.columns.values]
    g2.columns = ['_'.join(col).strip() for col in g2.columns.values]

    # Save summary to files
    g1.to_csv("outputs/artefact_correlation_summary_by_label.csv")
    g2.to_csv("outputs/artefact_correlation_summary_by_reason.csv")
    print("\nSaved summaries to outputs/artefact_correlation_summary_by_label.csv and outputs/artefact_correlation_summary_by_reason.csv")

if __name__ == "__main__":
    main()
