---
applyTo: "scripts/analysis/plot_*.py,scripts/analysis/figure*.py,publications/**/*.py"
---

# Figure 생성 스크립트 규칙

시각화 figure를 생성하는 스크립트를 만들거나 수정할 때, **재실행 CLI 명령어를 vault에 반드시 기록**할 것.

## Why
다음 세션에서 figure를 다시 생성하거나 수정해야 할 때, 스크립트 위치와 실행 방법을 즉시 알 수 있어야 함. 매번 파일 찾는 시간 낭비 방지.

## How to apply

### 스크립트 작성 시
스크립트 docstring에도 CLI 예시 포함:

```python
"""
Plot STK-V2 confusion matrices for 5 cancer aggregation modes.

CLI:
    cd /home/user/SERS-AI
    python scripts/analysis/plot_stk_v2_confusion_matrices.py

Output:
    figures/stk-v2/confusion_matrix_*.png  (5 files)
    figures/stk-v2/per_class_f1_barchart.png
"""
```

### 작업 완료 시
다음 vault 노트에 CLI 정보 추가:
- `01_Projects/SERS-AI/Experiments/figures/<figure-name>-cli.md`
- 또는 관련 분석 노트 안의 "## CLI" 섹션

기록 양식:
```markdown
## CLI
\`\`\`bash
cd /home/user/SERS-AI
python scripts/analysis/plot_xxx.py --option value
\`\`\`

## Output
- `figures/.../*.png`
- `results/.../*.csv` (있을 경우)

## Notes
- 입력 데이터: `results/...`
- 의존성: matplotlib, seaborn 등
```
