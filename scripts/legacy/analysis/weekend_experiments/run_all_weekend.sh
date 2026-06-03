#!/bin/bash
# ============================================================================
# Weekend Parallel Experiments Runner
# ============================================================================
# Launches all 6 experiments in parallel with nohup.
# Each experiment saves checkpoints and can resume after crash.
#
# Usage:
#   bash scripts/analysis/weekend_experiments/run_all_weekend.sh [--dry-run]
#
# Monitor:
#   tail -f results/weekend_experiments/logs/*.log
#   ls results/weekend_experiments/exp*/checkpoints/ | wc -l
#
# Check status:
#   ps aux | grep weekend_experiments
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/results/weekend_experiments/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DRY_RUN=""

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "[DRY-RUN MODE] Each experiment will run a minimal subset only."
fi

cd "$PROJECT_ROOT"

echo "=============================================="
echo "  Weekend Experiments — 6 Parallel Jobs"
echo "  Started: $(date)"
echo "  Logs:    $LOG_DIR"
echo "=============================================="
echo ""

# Exp 1: Bootstrap Confidence Intervals (~12h)
nohup python "$SCRIPT_DIR/bootstrap_confidence_intervals.py" $DRY_RUN \
    > "$LOG_DIR/exp1_bootstrap_${TIMESTAMP}.log" 2>&1 &
PID1=$!
echo "  [Exp1] Bootstrap CI              PID=$PID1"

# Exp 2: Leave-One-Hospital-Out (~14h)
nohup python "$SCRIPT_DIR/loho_validation.py" $DRY_RUN \
    > "$LOG_DIR/exp2_loho_${TIMESTAMP}.log" 2>&1 &
PID2=$!
echo "  [Exp2] LOHO Validation            PID=$PID2"

# Exp 3: Contrastive Pretraining Sweep (~12h, GPU)
nohup python "$SCRIPT_DIR/contrastive_sweep.py" $DRY_RUN \
    > "$LOG_DIR/exp3_contrastive_${TIMESTAMP}.log" 2>&1 &
PID3=$!
echo "  [Exp3] Contrastive Sweep          PID=$PID3"

# Exp 4: Permutation Feature Importance (~12h)
nohup python "$SCRIPT_DIR/permutation_importance.py" $DRY_RUN \
    > "$LOG_DIR/exp4_importance_${TIMESTAMP}.log" 2>&1 &
PID4=$!
echo "  [Exp4] Permutation Importance     PID=$PID4"

# Exp 5: Stacking/Meta-Learner Optimization (~14h)
nohup python "$SCRIPT_DIR/stacking_optimization.py" $DRY_RUN \
    > "$LOG_DIR/exp5_stacking_${TIMESTAMP}.log" 2>&1 &
PID5=$!
echo "  [Exp5] Stacking Optimization      PID=$PID5"

# Exp 6: Normalization × Feature Search (~12h)
nohup python "$SCRIPT_DIR/normalization_feature_search.py" $DRY_RUN \
    > "$LOG_DIR/exp6_norm_feature_${TIMESTAMP}.log" 2>&1 &
PID6=$!
echo "  [Exp6] Norm × Feature Search      PID=$PID6"

echo ""
echo "  All 6 experiments launched."
echo ""
echo "  PIDs: $PID1 $PID2 $PID3 $PID4 $PID5 $PID6"
echo ""
echo "  Monitor all:  tail -f $LOG_DIR/*_${TIMESTAMP}.log"
echo "  Check alive:  ps -p $PID1,$PID2,$PID3,$PID4,$PID5,$PID6 -o pid,etime,comm"
echo ""

# Save PIDs for easy killing later
echo "$PID1 $PID2 $PID3 $PID4 $PID5 $PID6" > "$LOG_DIR/pids_${TIMESTAMP}.txt"
echo "  PIDs saved to: $LOG_DIR/pids_${TIMESTAMP}.txt"
echo "  Kill all:      kill \$(cat $LOG_DIR/pids_${TIMESTAMP}.txt)"
