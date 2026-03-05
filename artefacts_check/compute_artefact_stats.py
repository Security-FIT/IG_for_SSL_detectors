
import pandas as pd
import sys

# Load details
try:
    details_df = pd.read_csv("outputs/final_selection_132_details.csv")
except Exception as e:
    print(f"Error loading final_selection_132_details.csv: {e}")
    sys.exit(1)

# Load artefacts
try:
    artefacts_df = pd.read_csv("outputs/ig_recordings_artefacts.csv")
except Exception as e:
    print(f"Error loading ig_recordings_artefacts.csv: {e}")
    sys.exit(1)

# Merge
# Rename name column in artefacts to FileID to match
artefacts_df.rename(columns={'name': 'FileID'}, inplace=True)

merged_df = pd.merge(details_df, artefacts_df, on='FileID')

if merged_df.empty:
    print("Error: Merged dataframe is empty. Check FileIDs.")
    sys.exit(1)

# Metric columns
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

# Ensure metric columns are numeric
for col in metric_cols:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# Group by SelectionReason
stats = merged_df.groupby('SelectionReason')[metric_cols].agg(['mean', 'std'])
# Flatten MultiIndex columns
stats.columns = [f"{c[0]}_{c[1]}" for c in stats.columns]
stats = stats.reset_index()

# Also group by Label
label_stats = merged_df.groupby('Label')[metric_cols].agg(['mean', 'std'])
# Flatten MultiIndex columns
label_stats.columns = [f"{c[0]}_{c[1]}" for c in label_stats.columns]
label_stats = label_stats.reset_index()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

print("Stats per SelectionReason category:")
print(stats.to_string(index=False))
print("-" * 80)
print("\nStats per Label:")
print(label_stats.to_string(index=False))

# Save to file
stats.to_csv("outputs/artefact_stats_by_reason.csv", index=False)
label_stats.to_csv("outputs/artefact_stats_by_label.csv", index=False)
print("\nStats saved to outputs/artefact_stats_by_reason.csv and outputs/artefact_stats_by_label.csv")
