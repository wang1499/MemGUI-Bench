#!/bin/bash
# Unified session check script
# Usage: ./check.sh [session_path]
#   Default session: /data2/wcl/MemGUI-Bench/results/session-memgui-v26050315-new-owl-v2

SESSION_PATH="${1:-/data2/wcl/MemGUI-Bench/results/session-memgui-v26050510-new-owl-all}"

echo "=========================================="
echo "Session Check"
echo "=========================================="
echo "Session: $SESSION_PATH"
echo ""

python scripts/check_session.py "$SESSION_PATH"

echo ""
echo "Done! Output saved to: scripts/output/$(basename $SESSION_PATH)/"
