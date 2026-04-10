"""
Medical 장비 데이터 평가 대시보드용 JSON 데이터 생성

Sections:
1. Wavenumber Calibration: Thermo vs Medical offset 분석
2. Background Removal: 장비 BG removal 품질 평가
3. Preprocessing Parameters: SG smoothing, baseline correction 비교
4. Data Overview: 폴더별 통계, 대표 스펙트럼
"""

import json
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

# ── paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_MEDICAL = DATA_ROOT / "raw_data_medical"
RAW_DATA = DATA_ROOT / "raw_data"
EQUIP_TEST = DATA_ROOT / "equipment_test_data"
OUTPUT = PROJECT_ROOT / "scripts" / "analysis" / "medical_eval_data.json"

FOLDER_TO_GROUP = {
    "1. Prostate cancer (100개)": "PRO",
    "2. Breast cancer (30개)": "BRE",
    "3. Ovarian cancer (70개)": "OVA",
    "4. Lung cancer (300개)": "LUN",
    "5. Normal (100개)": "NOR",
    "6. Diabetes (100개)": "DIA",
    "7. High blood pressure (100개)": "HBP",
    "8. High blood pressure + Diabetes (100개)": "H.D.",
    "9. Colorectal cancer (300개)": "CRC",
    "10-1. C-Pancreatic cancer (70개)": "CPAN",
    "10-2. S-Pancreatic cancer (72개)": "SPAN",
    "10-3. Y-Pancreatic cancer (30개)": "YPAN",
    "11. Bladdder Cancer (299개)": "BLC",
    "12. Y-Normal (29개)": "YNOR",
}


def read_spectrum(path, sep=None):
    """Read 2-col spectrum file. Returns (wn, intensity) arrays."""
    df = pd.read_csv(path, sep=sep, engine="python", header=None, usecols=[0, 1])
    df.columns = ["wn", "y"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna().sort_values("wn")
    return df.wn.values, df.y.values


def downsample(wn, y, max_points=500):
    """Downsample spectrum for JSON size reduction."""
    if len(wn) <= max_points:
        return wn.tolist(), y.tolist()
    idx = np.linspace(0, len(wn) - 1, max_points, dtype=int)
    return wn[idx].tolist(), y[idx].tolist()


def fingerprint_mask(wn, lo=400, hi=2200):
    return (wn >= lo) & (wn <= hi)


def noise_estimate(y_fp):
    """1st derivative std as noise proxy."""
    return float(np.std(np.diff(y_fp)))


# ═══════════════════════════════════════════════════════════════════════
# Section 1: Wavenumber Calibration
# ═══════════════════════════════════════════════════════════════════════
def section_calibration():
    print("[1/4] Wavenumber calibration analysis...")
    thermo_dir = EQUIP_TEST / "thermo" / "1. NOR"
    medical_dir = EQUIP_TEST / "medical_removed" / "1. NOR"

    # Collect offset data for all matching samples
    offsets = []
    overlay_spectra = []  # 5 representative pairs for overlay chart

    med_files = sorted(medical_dir.glob("NOR *_1.txt"))
    overlay_samples = med_files[:5]

    for mf in med_files:
        sample_name = mf.stem.replace(".txt", "") + ".CSV"
        tf = thermo_dir / sample_name
        if not tf.exists():
            # Try case variations
            candidates = list(thermo_dir.glob(mf.stem.split("_")[0] + "*.CSV"))
            if not candidates:
                continue
            tf = candidates[0]

        try:
            wn_m, y_m = read_spectrum(mf, sep="\t")
            wn_t, y_t = read_spectrum(tf)

            # Find strongest peak in 970-1060 range
            m_mask = (wn_m >= 970) & (wn_m <= 1060)
            t_mask = (wn_t >= 970) & (wn_t <= 1060)

            if m_mask.sum() == 0 or t_mask.sum() == 0:
                continue

            m_peak_wn = wn_m[m_mask][np.argmax(y_m[m_mask])]
            t_peak_wn = wn_t[t_mask][np.argmax(y_t[t_mask])]

            offsets.append({
                "sample": mf.stem,
                "medical_peak": round(float(m_peak_wn), 1),
                "thermo_peak": round(float(t_peak_wn), 1),
                "offset": round(float(m_peak_wn - t_peak_wn), 1),
            })

            # Overlay data for first 5 samples
            if mf in overlay_samples:
                fp_m = fingerprint_mask(wn_m)
                fp_t = fingerprint_mask(wn_t)
                wn_m_ds, y_m_ds = downsample(wn_m[fp_m], y_m[fp_m])
                wn_t_ds, y_t_ds = downsample(wn_t[fp_t], y_t[fp_t])

                # Normalize for visual comparison
                y_m_arr = np.array(y_m_ds)
                y_t_arr = np.array(y_t_ds)
                if y_m_arr.std() > 0:
                    y_m_norm = ((y_m_arr - y_m_arr.mean()) / y_m_arr.std()).tolist()
                else:
                    y_m_norm = y_m_ds
                if y_t_arr.std() > 0:
                    y_t_norm = ((y_t_arr - y_t_arr.mean()) / y_t_arr.std()).tolist()
                else:
                    y_t_norm = y_t_ds

                overlay_spectra.append({
                    "sample": mf.stem,
                    "medical": {"wn": wn_m_ds, "y": y_m_norm},
                    "thermo": {"wn": wn_t_ds, "y": y_t_norm},
                })
        except Exception as e:
            print(f"  Skip {mf.name}: {e}")

    # Statistics
    if offsets:
        offset_vals = [o["offset"] for o in offsets]
        stats = {
            "n_samples": len(offsets),
            "mean": round(np.mean(offset_vals), 1),
            "std": round(np.std(offset_vals), 1),
            "min": round(min(offset_vals), 1),
            "max": round(max(offset_vals), 1),
            "median": round(float(np.median(offset_vals)), 1),
        }
    else:
        stats = {}

    print(f"  {len(offsets)} samples analyzed, mean offset: {stats.get('mean', 'N/A')} cm⁻¹")
    return {
        "offsets": offsets,
        "stats": stats,
        "overlay_spectra": overlay_spectra,
    }


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Background Removal Evaluation
# ═══════════════════════════════════════════════════════════════════════
def section_bg_evaluation():
    print("[2/4] Background removal evaluation...")

    results = []
    overlay_data = []  # For BG overlay chart

    # Analyze multiple groups
    groups_to_check = [
        ("5. Normal (100개)", "NOR"),
        ("1. Prostate cancer (100개)", "PRO"),
        ("9. Colorectal cancer (300개)", "CRC"),
        ("4. Lung cancer (300개)", "LUN"),
    ]

    for folder_name, group in groups_to_check:
        folder = RAW_MEDICAL / folder_name
        bg_folder = folder / "Background"
        if not bg_folder.exists():
            continue

        # Get first 10 samples (rep 1 only)
        sample_files = sorted(
            [f for f in folder.glob("*.txt")
             if not f.name.endswith("_ave.txt")
             and "MultiData" not in f.name
             and "Zone" not in f.name
             and "_1.txt" in f.name]
        )[:10]

        for sf in sample_files:
            bg_file = bg_folder / sf.name
            if not bg_file.exists():
                continue

            try:
                wn, y_instr = read_spectrum(sf, sep="\t")
                _, y_bg = read_spectrum(bg_file, sep="\t")

                fp = fingerprint_mask(wn)
                y_sub = y_instr - y_bg  # Manual subtraction for comparison

                result = {
                    "group": group,
                    "sample": sf.stem,
                    # Instrument output (already BG-removed by instrument)
                    "instr_fp_mean": round(float(y_instr[fp].mean()), 1),
                    "instr_fp_std": round(float(y_instr[fp].std()), 1),
                    "instr_noise": round(noise_estimate(y_instr[fp]), 2),
                    # Background
                    "bg_fp_mean": round(float(y_bg[fp].mean()), 1),
                    "bg_fp_std": round(float(y_bg[fp].std()), 1),
                    "bg_noise": round(noise_estimate(y_bg[fp]), 2),
                    "bg_ratio_pct": round(float(y_bg[fp].mean() / (y_instr[fp].mean() + y_bg[fp].mean()) * 100), 1) if y_instr[fp].mean() + y_bg[fp].mean() > 0 else 0,
                    # Manual subtraction (instrument - bg)
                    "sub_fp_mean": round(float(y_sub[fp].mean()), 1),
                    "sub_noise": round(noise_estimate(y_sub[fp]), 2),
                    "sub_neg_pct": round(float((y_sub[fp] < 0).mean() * 100), 1),
                }
                results.append(result)

                # Overlay for first sample per group
                if sf == sample_files[0]:
                    wn_ds, y_instr_ds = downsample(wn[fp], y_instr[fp])
                    _, y_bg_ds = downsample(wn[fp], y_bg[fp])
                    _, y_sub_ds = downsample(wn[fp], y_sub[fp])

                    # Rolling minimum baseline on instrument output
                    baseline = pd.Series(y_instr[fp]).rolling(101, center=True, min_periods=1).min().values
                    _, baseline_ds = downsample(wn[fp], baseline)

                    overlay_data.append({
                        "group": group,
                        "sample": sf.stem,
                        "wn": wn_ds,
                        "instrument": y_instr_ds,
                        "background": y_bg_ds,
                        "manual_sub": y_sub_ds,
                        "baseline_101": baseline_ds,
                    })
            except Exception as e:
                print(f"  Skip {sf.name}: {e}")

    # Aggregate stats per group
    df = pd.DataFrame(results)
    group_stats = []
    if not df.empty:
        for grp, gdf in df.groupby("group"):
            group_stats.append({
                "group": grp,
                "n_samples": len(gdf),
                "instr_noise_mean": round(gdf.instr_noise.mean(), 2),
                "bg_noise_mean": round(gdf.bg_noise.mean(), 2),
                "bg_ratio_pct_mean": round(gdf.bg_ratio_pct.mean(), 1),
                "sub_noise_mean": round(gdf.sub_noise.mean(), 2),
                "sub_neg_pct_mean": round(gdf.sub_neg_pct.mean(), 1),
            })

    print(f"  {len(results)} samples across {len(groups_to_check)} groups")
    return {
        "samples": results,
        "group_stats": group_stats,
        "overlay_data": overlay_data,
    }


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Preprocessing Parameters
# ═══════════════════════════════════════════════════════════════════════
def section_preprocessing():
    print("[3/4] Preprocessing parameter comparison...")

    comparisons = []

    # Use NOR samples from both sources
    sources = [
        ("Medical", RAW_MEDICAL / "5. Normal (100개)", "\t"),
        ("Thermo (raw_data)", RAW_DATA / "5. Normal (100개)", None),
    ]

    for source_name, folder, sep in sources:
        if not folder.exists():
            continue
        pattern = "NOR *_1.txt" if sep == "\t" else "NOR *_1.*"
        files = sorted(folder.glob(pattern))
        # Exclude special files
        files = [f for f in files if "_ave" not in f.name and "MultiData" not in f.name and "Zone" not in f.name]
        files = files[:5]

        for sf in files:
            try:
                wn, y = read_spectrum(sf, sep=sep)
                fp = fingerprint_mask(wn)
                wn_fp = wn[fp]
                y_fp = y[fp]

                entry = {
                    "source": source_name,
                    "sample": sf.stem,
                    "raw_noise": round(noise_estimate(y_fp), 2),
                }

                # SG smoothing comparison
                for win in [7, 11, 21]:
                    y_sg = savgol_filter(y_fp, win, 3)
                    entry[f"sg{win}_noise"] = round(noise_estimate(y_sg), 2)

                # Baseline correction comparison
                for bw in [51, 101, 201]:
                    baseline = pd.Series(y_fp).rolling(bw, center=True, min_periods=1).min().values
                    corrected = y_fp - baseline
                    entry[f"bl{bw}_mean"] = round(float(corrected.mean()), 1)
                    entry[f"bl{bw}_max"] = round(float(corrected.max()), 1)

                comparisons.append(entry)
            except Exception as e:
                print(f"  Skip {sf.name}: {e}")

    # Representative spectra: full preprocessing pipeline comparison
    pipeline_overlay = []
    for source_name, folder, sep in sources:
        if not folder.exists():
            continue
        pattern = "NOR *_1.txt" if sep == "\t" else "NOR *_1.*"
        files = sorted(folder.glob(pattern))
        files = [f for f in files if "_ave" not in f.name and "MultiData" not in f.name and "Zone" not in f.name]
        if not files:
            continue

        sf = files[0]
        wn, y = read_spectrum(sf, sep=sep)
        fp = fingerprint_mask(wn)
        wn_fp = wn[fp]
        y_fp = y[fp]

        # Raw
        wn_ds, y_raw_ds = downsample(wn_fp, y_fp)

        # SG smoothed
        y_sg = savgol_filter(y_fp, 11, 3)
        _, y_sg_ds = downsample(wn_fp, y_sg)

        # Baseline corrected
        baseline = pd.Series(y_sg).rolling(101, center=True, min_periods=1).min().values
        y_bl = y_sg - baseline
        _, y_bl_ds = downsample(wn_fp, y_bl)

        # SNV normalized
        if np.std(y_bl) > 0:
            y_snv = (y_bl - np.mean(y_bl)) / np.std(y_bl)
        else:
            y_snv = y_bl
        _, y_snv_ds = downsample(wn_fp, y_snv)

        pipeline_overlay.append({
            "source": source_name,
            "sample": sf.stem,
            "wn": wn_ds,
            "raw": y_raw_ds,
            "smoothed": y_sg_ds,
            "baseline_corrected": y_bl_ds,
            "snv_normalized": y_snv_ds,
        })

    print(f"  {len(comparisons)} samples compared")
    return {
        "comparisons": comparisons,
        "pipeline_overlay": pipeline_overlay,
    }


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Data Overview
# ═══════════════════════════════════════════════════════════════════════
def section_overview():
    print("[4/4] Data overview...")

    folder_stats = []
    group_spectra = []  # Representative spectra per group

    for folder_name, group in FOLDER_TO_GROUP.items():
        folder = RAW_MEDICAL / folder_name
        if not folder.exists():
            print(f"  Folder not found: {folder_name}")
            continue

        # Count files (exclude special)
        all_files = [
            f for f in folder.glob("*.txt")
            if not f.name.endswith("_ave.txt")
            and "MultiData" not in f.name
            and "Zone" not in f.name
        ]

        # Count samples and replicates
        sample_ids = set()
        rep_counts = {}
        for f in all_files:
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2:
                sample_key = parts[0]
                sample_ids.add(sample_key)
                rep_counts[sample_key] = rep_counts.get(sample_key, 0) + 1

        has_bg = (folder / "Background").exists()
        has_ave = any(f.name.endswith("_ave.txt") for f in folder.glob("*.txt"))

        # Read first spectrum for range info
        first_file = all_files[0] if all_files else None
        wn_range = intensity_range = None
        if first_file:
            try:
                wn, y = read_spectrum(first_file, sep="\t")
                wn_range = [round(float(wn.min()), 1), round(float(wn.max()), 1)]
                intensity_range = [round(float(y.min()), 1), round(float(y.max()), 1)]
            except:
                pass

        reps = list(rep_counts.values())
        folder_stats.append({
            "folder": folder_name,
            "group": group,
            "n_samples": len(sample_ids),
            "n_spectra": len(all_files),
            "reps_per_sample": int(np.median(reps)) if reps else 0,
            "has_background": has_bg,
            "has_ave": has_ave,
            "wn_range": wn_range,
            "intensity_range": intensity_range,
        })

        # Representative spectrum (first sample, rep 1)
        if first_file:
            try:
                wn, y = read_spectrum(first_file, sep="\t")
                fp = fingerprint_mask(wn)
                wn_ds, y_ds = downsample(wn[fp], y[fp], max_points=300)
                group_spectra.append({
                    "group": group,
                    "sample": first_file.stem,
                    "wn": wn_ds,
                    "y": y_ds,
                })
            except:
                pass

    # Comparison with raw_data stats
    raw_data_stats = []
    raw_folder_map = {
        "5. Normal (100개)": "NOR",
        "1. Prostate cancer (100개)": "PRO",
        "9. Colorectal cancer (300개)": "CRC",
    }
    for folder_name, group in raw_folder_map.items():
        folder = RAW_DATA / folder_name
        if not folder.exists():
            continue
        files = list(folder.glob("*.CSV")) + list(folder.glob("*.csv"))
        files = [f for f in files if "_ave" not in f.name.lower()]
        if files:
            wn, y = read_spectrum(files[0])
            fp = fingerprint_mask(wn)
            raw_data_stats.append({
                "group": group,
                "n_files": len(files),
                "noise": round(noise_estimate(y[fp]), 2),
                "wn_range": [round(float(wn.min()), 1), round(float(wn.max()), 1)],
            })

    print(f"  {len(folder_stats)} folders analyzed")
    return {
        "folder_stats": folder_stats,
        "group_spectra": group_spectra,
        "raw_data_comparison": raw_data_stats,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Medical Data Evaluation - JSON Data Generation")
    print("=" * 60)

    data = {
        "calibration": section_calibration(),
        "bg_evaluation": section_bg_evaluation(),
        "preprocessing": section_preprocessing(),
        "overview": section_overview(),
        "metadata": {
            "generated_by": "prepare_medical_eval_data.py",
            "medical_dir": str(RAW_MEDICAL),
            "raw_data_dir": str(RAW_DATA),
            "equip_test_dir": str(EQUIP_TEST),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=None)

    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"\nOutput: {OUTPUT} ({size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
