"""
PAN subgroup peak-difference analysis
--------------------------------------
Compare YPAN / CPAN / SPAN each vs Non-cancer reference (NOR, DIA, HBP, H.D., YNOR)
using plot_group_peak_difference.
"""

from pathlib import Path
import pandas as pd

from src.sers.visualization.analysis import plot_group_peak_difference

# --- Config ---
SPECTRA_PATH = Path("results/processed_spectra.csv")
OUTPUT_DIR = Path("results/pan_subgroup_peak_diff")

TARGET_GROUPS = ["YPAN", "CPAN", "SPAN"]
REFERENCE_GROUPS = ["NOR", "DIA", "HBP", "H.D.", "YNOR"]

TOP_K = 15

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SPECTRA_PATH)
    print(f"Loaded {len(df)} rows, groups: {df['group'].value_counts().to_dict()}")

    # Filter to only relevant groups
    relevant = TARGET_GROUPS + REFERENCE_GROUPS
    df_filtered = df[df["group"].isin(relevant)].copy()
    print(f"Filtered to {len(df_filtered)} rows ({df_filtered['group'].nunique()} groups)")

    for target in TARGET_GROUPS:
        n_target = (df_filtered["group"] == target).sum()
        n_ref = df_filtered["group"].isin(REFERENCE_GROUPS).sum()
        print(f"\n--- {target} (n={n_target}) vs Non-cancer (n={n_ref}) ---")

        out_path = OUTPUT_DIR / f"{target}_vs_noncancer_peak_diff.png"
        kwargs = {}
        if target == "YPAN":
            kwargs["target_color"] = "#7E57C2"  # purple for YPAN in this comparison
        result_df = plot_group_peak_difference(
            spectra_df=df_filtered,
            output_path=out_path,
            target_group=target,
            reference_groups=REFERENCE_GROUPS,
            group_col="group",
            feature_prefix="x_",
            top_k=TOP_K,
            title=f"{target} vs Non-cancer Peak Difference",
            **kwargs,
        )
        print(f"  Saved: {out_path}")
        print(f"  Top-{TOP_K} peaks:")
        print(result_df.to_string(index=False))

        # Save CSV too
        csv_path = OUTPUT_DIR / f"{target}_vs_noncancer_peaks.csv"
        result_df.to_csv(csv_path, index=False)

    print(f"\nAll outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
