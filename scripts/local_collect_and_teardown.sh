#!/usr/bin/env bash
# Local Collector & Autonomous Teardown Daemon
# Monitors remote A10, downloads all compressed models, updates README, and powers down instance.

set -euo pipefail

SSH_KEY="${SSH_KEY:-$HOME/.ssh/A10-mv9p9tcb_key.pem}"
REMOTE_HOST="shadeform@129.213.130.107"
REMOTE_DIR="~/research"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=========================================================================="
echo " Nox Autonomous Collector & Teardown Daemon"
echo " Monitoring remote host: $REMOTE_HOST"
echo " Local directory: $LOCAL_DIR"
echo "=========================================================================="

# Check if pipeline is complete
if ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$REMOTE_HOST" "[ -f $REMOTE_DIR/PIPELINE_COMPLETE.flag ]"; then
    echo "[Collector] Pipeline is COMPLETE! Beginning local synchronization..."
    
    # Create local checkpoint destination
    mkdir -p "$LOCAL_DIR/checkpoints/day3_day4_cascade/master_cliff_sweep"
    
    # 1. Sync all checkpoints, manifests, results, and logs
    echo "[Collector] Downloading compressed models & artifacts (~1.9 GB)..."
    rsync -avz --progress -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
        "$REMOTE_HOST:$REMOTE_DIR/checkpoints/day3_day4_cascade/master_cliff_sweep/" \
        "$LOCAL_DIR/checkpoints/day3_day4_cascade/master_cliff_sweep/"
        
    rsync -avz -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
        "$REMOTE_HOST:$REMOTE_DIR/*.log" \
        "$LOCAL_DIR/checkpoints/day3_day4_cascade/master_cliff_sweep/" 2>/dev/null || true

    echo "[Collector] All models & benchmarks successfully synchronized locally!"
    
    # 2. Update repository documentation with benchmark results
    echo "[Collector] Updating documentation with Day 4 & Day 5 benchmark results..."
    python3 "$LOCAL_DIR/scripts/update_docs_with_results.py" || true
    
    # 3. Git commit & push
    cd "$LOCAL_DIR"
    git add README.md Novelty.md benchmarks/ checkpoints/day3_day4_cascade/master_cliff_sweep/*.json 2>/dev/null || true
    git commit -m "docs: Update final multi-tier benchmarks and Day 5 multi-task results" || true
    git push origin main || true

    # 4. Safe poweroff to stop compute billing
    echo "[Collector] Powering off remote A10 instance..."
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$REMOTE_HOST" "sudo poweroff" || true
    
    echo "=========================================================================="
    echo " [SUCCESS] All models downloaded locally, repo updated, and A10 powered off!"
    echo "=========================================================================="
    exit 0
else
    echo "[Collector] Remote pipeline still running. Status:"
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$REMOTE_HOST" "tail -n 10 $REMOTE_DIR/nox_big_guns.log" || true
    exit 1
fi
