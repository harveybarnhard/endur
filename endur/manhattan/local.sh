#!/bin/bash
# Run the Manhattan pipeline on this machine and refresh the site's local preview
# (http://localhost:8765/manhattan/). Nothing here pushes to GitHub; outputs stay in
# ~/.cache and the site's gitignored manhattan/_dev/.
#
#   local.sh login         sign in to Strava once (saved to ~/.config/endur-manhattan)
#   local.sh sync          the real pipeline: fetch new activities from Strava, update coverage
#   local.sh sync --fresh  the same, starting over from nothing (re-downloads everything)
#   local.sh               rebuild offline from the recordings saved by sync (no Strava calls)
set -euo pipefail
cd "$(dirname "$0")/../.."
CACHE="$HOME/.cache/endur-manhattan"
SITE="${SITE:-$HOME/repos/harveybarnhard.github.io}"
PY=.venv/bin/python

preview() {
    $PY endur/manhattan/planner.py --if-stale "$1" | tail -n 1
    mkdir -p "$SITE/manhattan/_dev"
    cp "$1"/coverage.json "$1"/runs.json "$1"/planner.json data/manhattan/geo.json "$SITE/manhattan/_dev/"
    echo "Preview updated from $1: http://localhost:8765/manhattan/ (hard-refresh)"
}

case "${1:-}" in
login)
    $PY endur/manhattan/strava_auth.py login
    ;;
sync)
    OUT="$CACHE/api"
    if [ "${2:-}" = "--fresh" ]; then rm -rf "$OUT"; fi
    # keep a copy of the recordings fetched earlier through the chat connector, for comparison
    if [ -d "$CACHE/streams" ] && [ ! -e "$CACHE/streams-connector" ]; then
        cp -r "$CACHE/streams" "$CACHE/streams-connector"
        cp "$CACHE/dev_polylines.json" "$CACHE/dev_polylines-connector.json"
    fi
    $PY -u endur/manhattan/update.py --auth --out "$OUT" --save-streams --backfill
    preview "$OUT"
    ;;
"")
    OUT="${OUT:-$CACHE/local}"
    $PY endur/manhattan/update.py --local "$OUT" | tail -n 3
    preview "$OUT"
    ;;
*)
    sed -n '6,9p' "$0" | sed 's/^# *//'
    exit 1
    ;;
esac
