#!/bin/bash
# Rebuild Manhattan coverage from the locally cached GPS recordings and refresh the
# site's local preview (http://localhost:8765/manhattan/). Nothing here touches Strava
# or GitHub; outputs stay in ~/.cache and the site's gitignored manhattan/_dev/.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-$HOME/.cache/endur-manhattan/local}"
SITE="${SITE:-$HOME/repos/harveybarnhard.github.io}"
.venv/bin/python endur/manhattan/update.py --local "$OUT" | tail -n 3
.venv/bin/python endur/manhattan/planner.py --if-stale "$OUT" | tail -n 1
mkdir -p "$SITE/manhattan/_dev"
cp "$OUT"/coverage.json "$OUT"/runs.json "$OUT"/planner.json data/manhattan/geo.json "$SITE/manhattan/_dev/"
echo "Preview updated: http://localhost:8765/manhattan/ (hard-refresh)"
