"""
Generate patient-level clinical reports for Pancreatic Cancer Screening.

MFDS Single-Cancer Diagnosis: Pancreatic Cancer vs Non-Cancer
Reads test_predictions.csv and generates one HTML report per patient.

Usage:
    python scripts/generate_pancreatic_reports.py
    python scripts/generate_pancreatic_reports.py --mode screening
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = PROJECT_ROOT / "results" / "pancreatic" / "experiment"
OUTPUT_DIR = EXPERIMENT_DIR / "clinical_reports"


def load_experiment():
    with open(EXPERIMENT_DIR / "experiment_log.json", encoding="utf-8") as f:
        return json.load(f)


def risk_level(prob, threshold):
    if prob >= 0.7:
        return "HIGH", "#dc2626", "#fef2f2"
    elif prob >= threshold:
        return "MODERATE", "#d97706", "#fffbeb"
    else:
        return "LOW", "#059669", "#f0fdf4"


def generate_report_html(row, experiment_log, mode="screening"):
    """Generate a single patient HTML report."""
    threshold = experiment_log["operating_modes"][mode]["threshold"]
    prob = row["probability"]
    is_positive = prob > threshold
    risk, risk_color, risk_bg = risk_level(prob, threshold)

    patient_id = row["patient_key"]
    group = row["group"]
    sample_id = row["sample_id"]
    age = row.get("age", "N/A")
    sex = row.get("sex", "N/A")
    bmi = row.get("bmi", "N/A")
    true_label = row["binary_label"]

    report_id = f"PAN-{datetime.now().strftime('%Y')}-{group}{sample_id:>04s}" if isinstance(sample_id, str) else f"PAN-{datetime.now().strftime('%Y')}-{group}{int(sample_id):04d}"
    report_date = datetime.now().strftime("%Y-%m-%d")

    # Model performance from experiment
    test_metrics = experiment_log["results"]["sers_only"]["test"]
    cv_auc = experiment_log["results"]["cv_auc"]
    mode_info = experiment_log["operating_modes"][mode]

    # Status text
    if is_positive:
        status_label = "PANCREATIC CANCER SIGNAL DETECTED"
        status_detail = "Abnormal SERS metabolite pattern consistent with pancreatic cancer. Further diagnostic workup recommended."
        status_class = "positive"
        status_icon = "!"
        rec_text = """
            <li><strong>Recommended:</strong> CA 19-9 tumor marker blood test</li>
            <li><strong>Recommended:</strong> Contrast-enhanced CT or MRI of the abdomen</li>
            <li><strong>Consider:</strong> Endoscopic ultrasound (EUS) if imaging is inconclusive</li>
            <li><strong>Refer to:</strong> Gastroenterology / Hepatobiliary-Pancreatic Surgery</li>
        """
    else:
        status_label = "NO PANCREATIC CANCER SIGNAL"
        status_detail = "SERS metabolite pattern within normal range. No evidence of pancreatic cancer detected."
        status_class = "negative"
        status_icon = "&#10003;"
        rec_text = """
            <li>No immediate follow-up required for pancreatic cancer</li>
            <li>Continue routine screening per clinical guidelines</li>
            <li>Repeat SERS screening in 12 months if risk factors present</li>
        """

    bmi_display = f"{bmi:.1f}" if isinstance(bmi, (int, float)) and not np.isnan(bmi) else "N/A"
    age_display = f"{int(age)}" if isinstance(age, (int, float)) and not np.isnan(age) else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AECD Platform &mdash; Pancreatic Cancer Screening Report &mdash; {patient_id}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

  :root {{
    --purple: #3b1f8e;
    --purple-light: #5a3db8;
    --purple-bg: rgba(59,31,142,0.04);
    --green: #059669;
    --green-bg: rgba(5,150,105,0.06);
    --red: #dc2626;
    --red-bg: rgba(220,38,38,0.04);
    --amber: #d97706;
    --amber-bg: rgba(217,119,6,0.05);
    --text: #1a1a2e;
    --text2: #4a5568;
    --text3: #8896ab;
    --border: #e4e8f0;
    --border-light: #eef1f6;
    --bg: #ffffff;
  }}

  @page {{ size: A4; margin: 16mm 14mm; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', -apple-system, sans-serif;
    color: var(--text);
    background: #f0f2f5;
    -webkit-font-smoothing: antialiased;
    line-height: 1.5;
  }}

  .report {{
    width: 210mm; min-height: 297mm;
    margin: 20px auto; background: var(--bg);
    box-shadow: 0 4px 40px rgba(0,0,0,0.08);
    padding: 0; position: relative;
  }}

  @media print {{
    body {{ background: #fff; }}
    .report {{ box-shadow: none; margin: 0; width: 100%; }}
  }}

  .report-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 24px 32px;
    border-bottom: 3px solid var(--purple);
    background: linear-gradient(135deg, rgba(59,31,142,0.02), rgba(59,31,142,0.06));
  }}
  .header-left {{ display: flex; align-items: center; gap: 16px; }}
  .brand {{ padding-left: 16px; border-left: 2px solid var(--border); }}
  .brand-name {{ font-size: 1.2rem; font-weight: 800; color: var(--purple); }}
  .brand-sub {{ font-size: 0.7rem; color: var(--text2); font-weight: 500; }}
  .header-right {{ text-align: right; }}
  .report-type {{ font-size: 0.75rem; font-weight: 700; color: var(--purple); text-transform: uppercase; letter-spacing: 1px; }}
  .report-id {{ font-size: 0.65rem; color: var(--text3); margin-top: 4px; }}

  .status-banner {{
    display: flex; align-items: center; gap: 20px;
    padding: 20px 32px; margin: 0;
  }}
  .status-banner.positive {{ background: var(--red-bg); border-left: 5px solid var(--red); }}
  .status-banner.negative {{ background: var(--green-bg); border-left: 5px solid var(--green); }}
  .status-icon {{
    width: 48px; height: 48px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; font-weight: 900; color: #fff; flex-shrink: 0;
  }}
  .positive .status-icon {{ background: var(--red); }}
  .negative .status-icon {{ background: var(--green); }}
  .status-label {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; color: var(--text3); }}
  .status-result {{ font-size: 1.15rem; font-weight: 900; margin: 2px 0; }}
  .positive .status-result {{ color: var(--red); }}
  .negative .status-result {{ color: var(--green); }}
  .status-detail {{ font-size: 0.72rem; color: var(--text2); }}
  .status-confidence {{ text-align: center; margin-left: auto; }}
  .conf-value {{ font-size: 1.6rem; font-weight: 900; }}
  .positive .conf-value {{ color: var(--red); }}
  .negative .conf-value {{ color: var(--green); }}
  .conf-label {{ font-size: 0.6rem; color: var(--text3); font-weight: 700; text-transform: uppercase; }}

  .report-body {{ padding: 24px 32px; }}

  .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
  .info-box {{ background: var(--border-light); border-radius: 10px; padding: 16px 20px; }}
  .info-box h4 {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: var(--purple); margin-bottom: 10px; font-weight: 800; }}
  .info-row {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid rgba(0,0,0,0.04); font-size: 0.78rem; }}
  .info-row .label {{ color: var(--text3); font-weight: 500; }}
  .info-row .value {{ color: var(--text); font-weight: 700; }}

  .section {{ margin-bottom: 24px; }}
  .section-title {{
    display: flex; align-items: center; gap: 10px;
    font-size: 0.85rem; font-weight: 800; color: var(--text);
    padding-bottom: 8px; border-bottom: 2px solid var(--purple);
    margin-bottom: 14px;
  }}
  .sec-num {{
    background: var(--purple); color: #fff; width: 24px; height: 24px;
    border-radius: 6px; display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 800;
  }}

  .risk-gauge {{
    display: flex; align-items: center; gap: 24px;
    padding: 20px; background: {risk_bg}; border-radius: 12px;
    border: 2px solid {risk_color}20; margin-bottom: 16px;
  }}
  .gauge-circle {{
    width: 100px; height: 100px; border-radius: 50%;
    border: 8px solid {risk_color}; display: flex; flex-direction: column;
    align-items: center; justify-content: center; flex-shrink: 0;
  }}
  .gauge-value {{ font-size: 1.4rem; font-weight: 900; color: {risk_color}; line-height: 1; }}
  .gauge-unit {{ font-size: 0.55rem; color: var(--text3); font-weight: 700; }}
  .gauge-detail h3 {{ font-size: 1rem; font-weight: 800; color: {risk_color}; margin-bottom: 4px; }}
  .gauge-detail p {{ font-size: 0.75rem; color: var(--text2); }}

  .threshold-bar {{
    position: relative; height: 32px; background: linear-gradient(90deg, var(--green), var(--amber), var(--red));
    border-radius: 6px; margin: 16px 0 8px;
  }}
  .threshold-marker {{
    position: absolute; top: -6px; width: 3px; height: 44px;
    background: var(--text); border-radius: 2px;
  }}
  .threshold-label {{
    position: absolute; top: 40px; font-size: 0.55rem; font-weight: 700;
    color: var(--text2); transform: translateX(-50%);
  }}
  .patient-marker {{
    position: absolute; top: -10px; width: 20px; height: 20px;
    background: var(--purple); border-radius: 50%; border: 3px solid #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }}
  .bar-labels {{ display: flex; justify-content: space-between; font-size: 0.6rem; color: var(--text3); font-weight: 600; margin-top: 22px; }}

  .perf-table {{ width: 100%; border-collapse: collapse; font-size: 0.72rem; margin-top: 12px; }}
  .perf-table th {{ background: var(--purple-bg); padding: 8px 12px; text-align: left; font-weight: 700; color: var(--purple); border-bottom: 2px solid var(--purple); }}
  .perf-table td {{ padding: 8px 12px; border-bottom: 1px solid var(--border-light); }}
  .perf-table tr.active {{ background: rgba(59,31,142,0.06); font-weight: 700; }}

  .rec-list {{ list-style: none; padding: 0; }}
  .rec-list li {{ padding: 8px 0 8px 24px; position: relative; font-size: 0.78rem; color: var(--text2); border-bottom: 1px solid var(--border-light); }}
  .rec-list li::before {{ content: "\\2022"; position: absolute; left: 8px; color: var(--purple); font-weight: 900; }}
  .rec-list li strong {{ color: var(--text); }}

  .report-footer {{
    padding: 20px 32px; border-top: 1px solid var(--border);
    background: var(--border-light);
  }}
  .disclaimer {{ font-size: 0.62rem; color: var(--text3); line-height: 1.7; margin-bottom: 12px; }}
  .disclaimer strong {{ color: var(--text2); }}
  .footer-row {{ display: flex; justify-content: space-between; align-items: center; }}
  .footer-meta {{ text-align: right; font-size: 0.6rem; color: var(--text3); line-height: 1.6; }}
</style>
</head>
<body>

<div class="report">

  <div class="report-header">
    <div class="header-left">
      <div class="brand" style="border-left:none;">
        <div class="brand-name">AECD Platform</div>
        <div class="brand-sub">Pancreatic Cancer Urine Screening Report</div>
      </div>
    </div>
    <div class="header-right">
      <div class="report-type">Single-Cancer Diagnostic Report</div>
      <div class="report-id">Report ID: {report_id} &nbsp;|&nbsp; MFDS Approved</div>
    </div>
  </div>

  <div class="status-banner {status_class}">
    <div class="status-icon">{status_icon}</div>
    <div class="status-text">
      <div class="status-label">Screening Result</div>
      <div class="status-result">{status_label}</div>
      <div class="status-detail">{status_detail}</div>
    </div>
    <div class="status-confidence">
      <div class="conf-value">{prob*100:.1f}%</div>
      <div class="conf-label">Cancer Probability</div>
    </div>
  </div>

  <div class="report-body">

    <div class="info-grid">
      <div class="info-box">
        <h4>Patient Information</h4>
        <div class="info-row"><span class="label">Patient ID</span><span class="value">{patient_id}</span></div>
        <div class="info-row"><span class="label">Age / Sex</span><span class="value">{age_display} / {sex}</span></div>
        <div class="info-row"><span class="label">BMI</span><span class="value">{bmi_display}</span></div>
        <div class="info-row"><span class="label">True Status</span><span class="value">{true_label}</span></div>
      </div>
      <div class="info-box">
        <h4>Sample &amp; Analysis</h4>
        <div class="info-row"><span class="label">Source Cohort</span><span class="value">{group}</span></div>
        <div class="info-row"><span class="label">Sample Type</span><span class="value">Fasting urine (mid-stream)</span></div>
        <div class="info-row"><span class="label">SERS Replicates</span><span class="value">5 (medoid selected)</span></div>
        <div class="info-row"><span class="label">Operating Mode</span><span class="value">{mode.capitalize()} (threshold={threshold:.4f})</span></div>
        <div class="info-row"><span class="label">Report Date</span><span class="value">{report_date}</span></div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">
        <span class="sec-num">1</span>
        Risk Assessment
      </div>
      <div class="risk-gauge">
        <div class="gauge-circle">
          <div class="gauge-value">{prob*100:.1f}%</div>
          <div class="gauge-unit">probability</div>
        </div>
        <div class="gauge-detail">
          <h3>{risk} RISK</h3>
          <p>{'This result exceeds the screening threshold. Pancreatic cancer-associated metabolite patterns were detected in the urine SERS spectrum.' if is_positive else 'This result is below the screening threshold. No pancreatic cancer-associated patterns detected.'}</p>
        </div>
      </div>

      <div style="position:relative; padding: 0 10px;">
        <div class="threshold-bar">
          <div class="threshold-marker" style="left:{threshold*100:.1f}%;"></div>
          <div class="threshold-label" style="left:{threshold*100:.1f}%;">Threshold<br>{threshold:.2f}</div>
          <div class="patient-marker" style="left:calc({min(prob*100, 99):.1f}% - 10px);"></div>
        </div>
        <div class="bar-labels">
          <span>0% &mdash; Normal</span>
          <span>Pancreatic Cancer &mdash; 100%</span>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">
        <span class="sec-num">2</span>
        Model Performance Summary
      </div>
      <table class="perf-table">
        <thead>
          <tr>
            <th>Operating Mode</th>
            <th>Threshold</th>
            <th>CV Sensitivity</th>
            <th>CV Specificity</th>
          </tr>
        </thead>
        <tbody>
          <tr {"class='active'" if mode == "screening" else ""}>
            <td>Screening</td>
            <td>{experiment_log['operating_modes']['screening']['threshold']:.4f}</td>
            <td>{experiment_log['operating_modes']['screening']['cv_sensitivity']*100:.1f}%</td>
            <td>{experiment_log['operating_modes']['screening']['cv_specificity']*100:.1f}%</td>
          </tr>
          <tr {"class='active'" if mode == "balanced" else ""}>
            <td>Balanced</td>
            <td>{experiment_log['operating_modes']['balanced']['threshold']:.4f}</td>
            <td>{experiment_log['operating_modes']['balanced']['cv_sensitivity']*100:.1f}%</td>
            <td>{experiment_log['operating_modes']['balanced']['cv_specificity']*100:.1f}%</td>
          </tr>
          <tr {"class='active'" if mode == "confirmatory" else ""}>
            <td>Confirmatory</td>
            <td>{experiment_log['operating_modes']['confirmatory']['threshold']:.4f}</td>
            <td>{experiment_log['operating_modes']['confirmatory']['cv_sensitivity']*100:.1f}%</td>
            <td>{experiment_log['operating_modes']['confirmatory']['cv_specificity']*100:.1f}%</td>
          </tr>
        </tbody>
      </table>
      <div style="margin-top:8px; font-size:0.65rem; color:var(--text3);">
        Validation: 5-fold stratified group CV | Test AUC: {test_metrics['auc']} | CV AUC: {cv_auc}
      </div>
    </div>

    <div class="section">
      <div class="section-title">
        <span class="sec-num">3</span>
        Clinical Recommendations
      </div>
      <ul class="rec-list">
        {rec_text}
      </ul>
    </div>

  </div>

  <div class="report-footer">
    <div class="disclaimer">
      <strong>Disclaimer:</strong> This report is generated by the AECD Platform (SOLUM Healthcare) for
      pancreatic cancer screening purposes only. Results are based on SERS urine metabolite profiling
      and should not be used as a sole diagnostic criterion. Clinical correlation and confirmatory
      testing are required. This test has been validated under MFDS single-cancer diagnostic approval.
      <br><strong>Limitation:</strong> Multi-institution data was used for model training. Batch effects
      between institutions (SMCXD04/SMCXD06/SMCXD03) have been identified and mitigated through
      regularization, but prospective validation with harmonized protocols is recommended.
    </div>
    <div class="footer-row">
      <div class="footer-meta">
        AECD Platform v2.0 &mdash; Pancreatic Cancer Module<br>
        Model: LR (C=0.1, balanced) | Features: 933 SERS | Preprocessing: SNV<br>
        Generated: {report_date} | SOLUM Healthcare Co., Ltd.
      </div>
    </div>
  </div>

</div>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="screening", choices=["screening", "balanced", "confirmatory"])
    parser.add_argument("--split", default="test", choices=["test", "train", "all"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_log = load_experiment()

    # Load predictions
    if args.split == "all":
        dfs = []
        for s in ["test", "train"]:
            d = pd.read_csv(EXPERIMENT_DIR / f"{s}_predictions.csv")
            d["split"] = s
            dfs.append(d)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(EXPERIMENT_DIR / f"{args.split}_predictions.csv")
        df["split"] = args.split

    print(f"Generating {len(df)} reports (mode={args.mode}, split={args.split})...")

    for _, row in df.iterrows():
        html = generate_report_html(row, experiment_log, mode=args.mode)
        fname = f"{row['patient_key']}.html"
        (OUTPUT_DIR / fname).write_text(html, encoding="utf-8")

    print(f"Done. Reports saved to: {OUTPUT_DIR}/")
    print(f"  Total: {len(df)} reports")
    print(f"  Positive: {(df['probability'] > experiment_log['operating_modes'][args.mode]['threshold']).sum()}")
    print(f"  Negative: {(df['probability'] <= experiment_log['operating_modes'][args.mode]['threshold']).sum()}")


if __name__ == "__main__":
    main()
