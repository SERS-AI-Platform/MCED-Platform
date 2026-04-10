"""
Thermo vs Medical 피크 비교 시각화 데이터 생성

Sections:
1. 그룹별 mean spectrum overlay (SNV normalized)
2. 2nd derivative 피크 위치 비교
3. 피크 매칭 테이블
4. 1600 cm⁻¹ drop 분석
"""

import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_MEDICAL = DATA_ROOT / "raw_data_medical"
RAW_THERMO = DATA_ROOT / "raw_data"
OUTPUT = PROJECT_ROOT / "scripts" / "analysis" / "peak_comparison_data.json"

GROUPS = {
    "NOR": {"med_folder": "5. Normal (100개)", "thermo_folder": "5. Normal (100개)"},
    "PRO": {"med_folder": "1. Prostate cancer (100개)", "thermo_folder": "1. Prostate cancer (100개)"},
    "CRC": {"med_folder": "9. Colorectal cancer (300개)", "thermo_folder": "9. Colorectal cancer (300개)"},
    "LUN": {"med_folder": "4. Lung cancer (300개)", "thermo_folder": "4. Lung cancer (300개)"},
    "BRE": {"med_folder": "2. Breast cancer (30개)", "thermo_folder": "2. Breast cancer (30개)"},
    "OVA": {"med_folder": "3. Ovarian cancer (70개)", "thermo_folder": "3. Ovarian cancer (70개)"},
    "DIA": {"med_folder": "6. Diabetes (100개)", "thermo_folder": "6. Diabetes (100개)"},
    "HBP": {"med_folder": "7. High blood pressure (100개)", "thermo_folder": "7. High blood pressure (100개)"},
    "H.D.": {"med_folder": "8. High blood pressure + Diabetes (100개)", "thermo_folder": "8. High blood pressure + Diabetes (100개)"},
    "BLC": {"med_folder": "11. Bladdder Cancer (299개)", "thermo_folder": "11 BLC (299개)"},
}

GRID = np.linspace(402, 2198, 900)


def read_spec(path, sep=None):
    df = pd.read_csv(path, sep=sep, engine="python", header=None, usecols=[0, 1], names=["wn", "y"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna().sort_values("wn")
    return df.wn.values, df.y.values


def bl_correct(y, w=101):
    return y - pd.Series(y).rolling(w, center=True, min_periods=1).min().values


def snv(y):
    return (y - y.mean()) / y.std() if y.std() > 0 else y


def ds(arr, n=350):
    if len(arr) <= n:
        return arr.tolist()
    idx = np.linspace(0, len(arr) - 1, n, dtype=int)
    return arr[idx].tolist()


def load_group_spectra(group, info, n_samples=10):
    """Load up to n_samples (rep 1) for both Thermo and Medical."""
    results = {"thermo": [], "medical": []}

    # Medical: Background/ subdir (BG-removed)
    med_dir = RAW_MEDICAL / info["med_folder"] / "Background"
    if med_dir.exists():
        files = sorted([f for f in med_dir.glob("*.txt")
                        if "_ave" not in f.name and "Multi" not in f.name
                        and "Zone" not in f.name and "_1.txt" in f.name])[:n_samples]
        for f in files:
            try:
                wn, y = read_spec(f, sep="\t")
                fp = (wn >= 400) & (wn <= 2200)
                results["medical"].append(np.interp(GRID, wn[fp], y[fp]))
            except:
                pass

    # Thermo
    thermo_dir = RAW_THERMO / info["thermo_folder"]
    if thermo_dir.exists():
        patterns = ["*_1.CSV", "*_1.csv"]
        files = []
        for p in patterns:
            files.extend(sorted(thermo_dir.glob(p)))
        files = [f for f in files if "_ave" not in f.name.lower()][:n_samples]
        for f in files:
            try:
                wn, y = read_spec(f)
                fp = (wn >= 400) & (wn <= 2200)
                results["thermo"].append(np.interp(GRID, wn[fp], y[fp]))
            except:
                pass

    return results


def find_prominent_peaks(y_proc, grid, n_peaks=15, sg_win=15):
    d2 = savgol_filter(y_proc, sg_win, 3, deriv=2)
    peaks, props = find_peaks(-d2, prominence=0.0005)
    if len(peaks) == 0:
        return [], d2
    top = sorted(range(len(peaks)), key=lambda i: props["prominences"][i], reverse=True)[:n_peaks]
    peak_list = [{"wn": round(float(grid[peaks[i]]), 1),
                  "prom": round(float(props["prominences"][i]), 4)} for i in top]
    return sorted(peak_list, key=lambda p: p["wn"]), d2


def main():
    print("=" * 60)
    print("Peak Comparison Data Generation")
    print("=" * 60)

    data = {"groups": {}, "grid": ds(GRID), "summary": {}}

    all_matched = []

    for group, info in GROUPS.items():
        print(f"  [{group}] loading...")
        spectra = load_group_spectra(group, info)

        if not spectra["thermo"] or not spectra["medical"]:
            print(f"    Skip: thermo={len(spectra['thermo'])}, medical={len(spectra['medical'])}")
            continue

        # Mean spectra
        thermo_mean = np.mean(spectra["thermo"], axis=0)
        medical_mean = np.mean(spectra["medical"], axis=0)

        # Baseline correct + SNV
        thermo_proc = snv(bl_correct(thermo_mean))
        medical_proc = snv(bl_correct(medical_mean))

        # 2nd derivative peaks
        thermo_peaks, thermo_d2 = find_prominent_peaks(thermo_proc, GRID)
        medical_peaks, medical_d2 = find_prominent_peaks(medical_proc, GRID)

        # Peak matching (±20 cm⁻¹)
        matched = []
        t_wns = [p["wn"] for p in thermo_peaks]
        m_wns = [p["wn"] for p in medical_peaks]
        for tw in t_wns:
            if not m_wns:
                matched.append({"thermo": tw, "medical": None, "offset": None, "match": False})
                continue
            dists = [abs(tw - mw) for mw in m_wns]
            bi = int(np.argmin(dists))
            bd = dists[bi]
            matched.append({
                "thermo": tw, "medical": m_wns[bi],
                "offset": round(m_wns[bi] - tw, 1),
                "match": bd <= 20,
            })

        n_matched = sum(1 for m in matched if m["match"])
        all_matched.append({"group": group, "matched": n_matched, "total": len(t_wns)})

        # Correlation
        corr_snv = round(float(np.corrcoef(thermo_proc, medical_proc)[0, 1]), 4)
        corr_d2 = round(float(np.corrcoef(thermo_d2, medical_d2)[0, 1]), 4)

        # 1600 drop analysis
        m1 = (GRID >= 1400) & (GRID <= 1600)
        m2 = (GRID >= 1600) & (GRID <= 1800)
        m3 = (GRID >= 1800) & (GRID <= 2200)
        drop_thermo = round(float(thermo_mean[m2].mean() / thermo_mean[m1].mean()), 3) if thermo_mean[m1].mean() != 0 else 0
        drop_medical = round(float(medical_mean[m2].mean() / medical_mean[m1].mean()), 3) if medical_mean[m1].mean() != 0 else 0

        data["groups"][group] = {
            "thermo_mean": ds(thermo_proc),
            "medical_mean": ds(medical_proc),
            "thermo_d2": ds(thermo_d2),
            "medical_d2": ds(medical_d2),
            "thermo_peaks": thermo_peaks,
            "medical_peaks": medical_peaks,
            "matched": matched,
            "corr_snv": corr_snv,
            "corr_d2": corr_d2,
            "n_thermo": len(spectra["thermo"]),
            "n_medical": len(spectra["medical"]),
            "drop_1600": {"thermo": drop_thermo, "medical": drop_medical},
        }

        print(f"    T:{len(spectra['thermo'])} M:{len(spectra['medical'])} "
              f"peaks T:{len(thermo_peaks)} M:{len(medical_peaks)} "
              f"match:{n_matched}/{len(t_wns)} corr_d2:{corr_d2}")

    # Summary
    data["summary"] = {
        "matching": all_matched,
        "avg_match_rate": round(np.mean([m["matched"] / m["total"] for m in all_matched if m["total"] > 0]) * 100, 1),
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    size_kb = OUTPUT.stat().st_size / 1024
    print(f"\nOutput: {OUTPUT} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
