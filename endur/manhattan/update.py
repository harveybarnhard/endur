"""Incrementally update Manhattan street coverage from Strava.

    python endur/manhattan/update.py                 # cron: new activities only (capped)
    python endur/manhattan/update.py --backfill      # no cap (first run / catch-up)
    python endur/manhattan/update.py --rebuild       # reset state, reprocess everything
    python endur/manhattan/update.py --local OUTDIR  # local: cached full recordings (see local.sh)
    python endur/manhattan/update.py --auth --out DIR --save-streams  # local: the real thing (local.sh sync)

Reads the Strava access token from $STRAVA_TOKENS (a decrypted strava_tokens.json;
this job never refreshes it), or with --auth from a local sign-in (strava_auth.py).
GPS streams are held in memory only unless --save-streams keeps them in ~/.cache.
Writes data/manhattan/{state,coverage,runs}.json; none of them contain GPS points.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import polyline
import requests
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from geo import DATA, Projection, load_geo  # noqa: E402
from match import Matcher, PARAMS, is_complete, total, union  # noqa: E402
import strava_auth  # noqa: E402

FOOT = {"Run", "TrailRun", "Walk", "Hike"}
API = "https://www.strava.com/api/v3"
CRON_CAP = 60
START = "2024-11-01"  # project start (local date); earlier activities don't count
CACHE = os.path.expanduser("~/.cache/endur-manhattan")


# --- Strava access -----------------------------------------------------------
class Strava:
    def __init__(self, token):
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"

    def get(self, path, **params):
        for _ in range(4):
            r = self.s.get(API + path, params=params, timeout=60)
            usage = r.headers.get("X-ReadRateLimit-Usage") or r.headers.get("X-RateLimit-Usage")
            limit = r.headers.get("X-ReadRateLimit-Limit") or r.headers.get("X-RateLimit-Limit")
            if r.status_code == 429 or (usage and limit and
                                        int(usage.split(",")[0]) >= int(limit.split(",")[0]) - 3):
                wait = 900 - (time.time() % 900) + 5  # next 15-minute window
                print(f"  rate limit ({usage}/{limit}); sleeping {wait:.0f}s", flush=True)
                time.sleep(wait)
                if r.status_code == 429:
                    continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()

    def activities(self, after):
        page, out = 1, []
        while True:
            batch = self.get("/athlete/activities", after=int(after), per_page=200, page=page)
            if not batch:
                return out
            out += batch
            page += 1

    def track(self, aid):
        st = self.get(f"/activities/{aid}/streams", keys="latlng,time", key_by_type="true")
        if "latlng" not in st:
            return None, None
        return st["latlng"]["data"], st.get("time", {}).get("data")


def load_token():
    path = os.environ.get("STRAVA_TOKENS", "./data/strava_tokens.json")
    with open(path) as f:
        tok = json.load(f)
    if tok.get("expires_at", 0) < time.time() + 300:
        print("Access token expired or about to; skipping this run (build job refreshes it).")
        sys.exit(0)
    return tok["access_token"]


# --- helpers -----------------------------------------------------------------
class Network:
    """Coverage-independent lookups over the street network."""

    def __init__(self, geo):
        self.geo = geo
        self.proj = Projection(**{k: geo["proj"][k] for k in ("lat0", "lon0", "rot")})
        self.matcher = Matcher(geo["seg_xy"], cls=geo["segs"]["cls"], name=geo["segs"]["n"])
        self.length = self.matcher.length
        pts = np.vstack([np.asarray(s, dtype=float) for s in geo["seg_xy"]])
        self.tree = cKDTree(pts)

    def xy(self, latlng):
        return self.proj.fwd_many([(lon, lat) for lat, lon in latlng])

    def touches(self, latlng, within=40.0, min_frac=0.05):
        """Does a (possibly simplified) track run on Manhattan streets at all?"""
        if not latlng:
            return False
        d, _ = self.tree.query(self.xy(latlng), distance_upper_bound=within)
        return np.isfinite(d).mean() >= min_frac


STATE_PARAMS = {**PARAMS, "start": START}  # changing any of these rebuilds coverage


def empty_state(v):
    return {"v": v, "params": STATE_PARAMS, "last_start": None, "processed": {}, "iv": {}, "done": {}}


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


# --- core --------------------------------------------------------------------
def apply(net, state, runs, act, latlng, t):
    """Match one activity and fold it into state/runs. Returns new metres."""
    got = net.matcher.match(net.xy(latlng), t)
    idx = len(runs)
    day = act["start"][:10]
    new_m, newly = 0.0, []
    for s, iv in got.items():
        key = str(s)
        before = state["iv"].get(key, [])
        after = union(before, iv)
        gain = total(after) - total(before)
        if gain > 0.5:
            new_m += gain
            state["iv"][key] = after
        if key not in state["done"] and is_complete(total(after), net.length[s]):
            state["done"][key] = [day, idx]
            newly.append(s)
    cum = (runs[-1]["cum_m"] if runs else 0) + new_m
    runs.append({"id": act["id"], "d": day, "name": act["name"], "type": act["type"],
                 "mi": round(act["dist"] / 1609.344, 2), "new_m": round(new_m),
                 "cum_m": round(cum), "new": sorted(newly)})
    state["processed"][act["id"]] = idx
    if not state["last_start"] or act["start"] > state["last_start"]:
        state["last_start"] = act["start"]
    return new_m


def outputs(net, state, runs, outdir):
    n = len(net.length)
    f = [0] * n
    done = [-1] * n
    by = [-1] * n
    covered = 0.0
    day0 = min((r["d"] for r in runs), default=None)
    d0 = dt.date.fromisoformat(day0) if day0 else None
    for key, iv in state["iv"].items():
        s = int(key)
        c = min(total(iv), net.length[s])
        covered += c
        f[s] = int(100 * c / net.length[s]) if net.length[s] else 0
    for key, (day, idx) in state["done"].items():
        s = int(key)
        done[s] = (dt.date.fromisoformat(day) - d0).days
        by[s] = idx
        f[s] = 100
    updated = (state["last_start"] or "")[:10] or None  # last activity, so idle days don't diff
    cov = {"v": net.geo["v"], "updated": updated, "day0": day0,
           "total_m": round(float(net.length.sum())), "covered_m": round(covered),
           "f": f, "done": done, "by": by}
    os.makedirs(outdir, exist_ok=True)
    write_json(os.path.join(outdir, "coverage.json"), cov)
    write_json(os.path.join(outdir, "runs.json"), runs)
    write_json(os.path.join(outdir, "state.json"), state)
    return cov


def local_source(net, missing):
    """Local iteration: the activity index and full GPS recordings cached in ~/.cache
    (fetched once; never committed). Same recordings the Action downloads, so results
    match. Activities without a cached recording are listed in `missing`, not guessed at."""
    acts = load_json(os.path.join(CACHE, "dev_polylines.json"), [])
    for a in sorted(acts, key=lambda a: a["start"]):
        if a["start"] < START or a["type"] not in FOOT:
            continue
        path = os.path.join(CACHE, "streams", a["id"] + ".json")
        if not os.path.exists(path):
            if a.get("poly") and net.touches(polyline.decode(a["poly"])):
                missing.append(a)
            continue
        st = load_json(path, {})
        latlng, t = st["location"], st["time"]
        if net.touches(latlng):
            yield {"id": a["id"], "start": a["start"], "name": a["name"], "type": a["type"],
                   "dist": a["dist"]}, latlng, t


def list_manhattan(api, net, after, skip=()):
    """Foot activities since START that run on Manhattan streets, oldest first."""
    listed = api.activities(after)
    print(f"  listed {len(listed)} activities since {dt.datetime.fromtimestamp(after, dt.UTC):%Y-%m-%d}")
    todo = []
    for a in listed:
        poly = (a.get("map") or {}).get("summary_polyline")
        if (str(a["id"]) in skip or a.get("sport_type", a.get("type")) not in FOOT
                or a["start_date_local"] < START or a.get("manual") or a.get("trainer") or not poly):
            continue
        if net.touches(polyline.decode(poly)):
            todo.append(a)
    return sorted(todo, key=lambda a: a["start_date_local"])


def save_stream(a, latlng, t):
    """Keep a downloaded recording (and its index entry) in the local cache for `--local` rebuilds."""
    aid = str(a["id"])
    os.makedirs(os.path.join(CACHE, "streams"), exist_ok=True)
    write_json(os.path.join(CACHE, "streams", aid + ".json"), {"location": latlng, "time": t})
    path = os.path.join(CACHE, "dev_polylines.json")
    index = {e["id"]: e for e in load_json(path, [])}
    index[aid] = {"id": aid, "start": a["start_date_local"].rstrip("Z"), "name": a["name"],
                  "type": a.get("sport_type", a.get("type")), "dist": a["distance"],
                  "poly": a["map"]["summary_polyline"]}
    write_json(path, sorted(index.values(), key=lambda e: e["start"]))


def api_source(net, state, cap, token, keep=False):
    api = Strava(token)
    after = dt.datetime.fromisoformat(START).replace(tzinfo=dt.UTC).timestamp() - 86400
    if state["last_start"]:
        last = dt.datetime.fromisoformat(state["last_start"].replace("Z", "+00:00")).timestamp()
        after = max(after, last - 7 * 86400)
    todo = list_manhattan(api, net, after, skip=state["processed"])
    if cap and len(todo) > cap:
        print(f"  {len(todo)} new Manhattan activities; processing the first {cap}")
        todo = todo[:cap]
    elif not todo:
        print("  no new Manhattan activities")
    for a in todo:
        latlng, t = api.track(a["id"])
        if not latlng:
            continue
        if keep:
            save_stream(a, latlng, t)
        yield {"id": str(a["id"]), "start": a["start_date_local"], "name": a["name"],
               "type": a.get("sport_type", a.get("type")), "dist": a["distance"]}, latlng, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--local", metavar="OUTDIR", help="use cached recordings; write outputs to OUTDIR")
    ap.add_argument("--auth", nargs="?", const=strava_auth.CREDS, metavar="FILE",
                    help="use a local Strava sign-in (default %(const)s), refreshing it as needed")
    ap.add_argument("--out", metavar="DIR", help="write outputs to DIR instead of data/manhattan")
    ap.add_argument("--save-streams", action="store_true",
                    help="keep downloaded recordings in ~/.cache for --local rebuilds")
    args = ap.parse_args()

    geo = load_geo()
    net = Network(geo)
    outdir = args.local or args.out or DATA
    state = load_json(os.path.join(outdir, "state.json"), None)
    runs = load_json(os.path.join(outdir, "runs.json"), [])
    if (args.rebuild or args.local or not state or state.get("v") != geo["v"]
            or state.get("params") != STATE_PARAMS):
        if state and not (args.rebuild or args.local):
            print("Network or matcher parameters changed: rebuilding coverage from scratch.")
        state, runs = empty_state(geo["v"]), []
    fresh = not state["processed"]

    missing = []
    if args.local:
        source = local_source(net, missing)
    else:
        token = strava_auth.access_token(args.auth) if args.auth else load_token()
        source = api_source(net, state, cap=None if (args.backfill or args.rebuild or fresh) else CRON_CAP,
                            token=token, keep=args.save_streams)
    n, gained = 0, 0.0
    for act, latlng, t in source:
        g = apply(net, state, runs, act, latlng, t)
        gained += g
        n += 1
        print(f"  {act['start'][:10]} {act['type']:<5} +{g / 1609.344:5.2f} mi  {act['name'][:40]}")
        if n % 25 == 0:  # checkpoint: a long backfill that times out keeps its progress
            outputs(net, state, runs, outdir)
    cov = outputs(net, state, runs, outdir)
    pct = 100 * cov["covered_m"] / cov["total_m"]
    summary = (f"Processed {n} activities (+{gained / 1609.344:.1f} new mi). "
               f"Coverage {pct:.2f}% ({cov['covered_m'] / 1609.344:.1f} of {cov['total_m'] / 1609.344:.1f} mi), "
               f"{sum(1 for d in cov['done'] if d >= 0)} of {len(cov['done'])} segments complete.")
    print(summary)
    if missing:
        print(f"WARNING: {len(missing)} Manhattan activities since {START} have no cached recording and were "
              f"skipped: " + ", ".join(f"{a['start'][:10]} {a['name']} ({a['id']})" for a in missing[:10])
              + (" ..." if len(missing) > 10 else ""))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(f"### Manhattan coverage\n\n{summary}\n")


if __name__ == "__main__":
    main()
