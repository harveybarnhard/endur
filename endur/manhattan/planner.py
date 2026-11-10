"""Suggest next runs: zigzags through a few neighbouring columns of the grid.

    python endur/manhattan/planner.py [DATADIR]

Reads geo.json and coverage.json, writes planner.json. A column is the strip between
two neighbouring avenues. A column is run as a serpentine: east along one street, one
block up the avenue, west along the next street, one block up the other avenue, and
so on, which never crosses an avenue, where the long red lights are. A route is a
chunk: up to four side-by-side columns over the same range of streets, zigzagging
north up the first, south down the next, and so on, so a neighbourhood gets finished
in one visit. Each column's rule fits in one line ("1st to 2nd Ave, north from 14th
to 34th"), so the run needs no map. Routes start and end at a 1/2/3 station where
one is in reach, and other lines otherwise.
"""
import argparse
import collections
import hashlib
import heapq
import json
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from geo import DATA, load_geo  # noqa: E402

N_ROUTES = 6
MAX_LEN = 17500.0      # routes of about 10 mi, station legs included: up to ~10.9 mi ...
MIN_LEN = 12500.0      # ... and at least ~7.8 mi
MAX_COLS = 4           # columns per chunk
MIN_ROWS = 3           # streets per column
MIN_NEW = 3200.0       # not worth suggesting below ~2 new mi
LEN_COST = 0.3         # score = new metres - LEN_COST * metres - END_COST per end off the 1/2/3
HOME_LINES = {"1", "2", "3"}
END_COST = 1500.0      # an end at another line costs as much as this many new metres ...
HOME_SLACK = 2500.0    # ... and is used only when the nearest 1/2/3 is this much farther (m)
SPREAD = 800.0         # keep suggestions apart: route centres at least this far apart (m)
SNAP = 30.0            # join a dead end to a node this close (m)
AVE_MIN = 500.0        # a named road with this much north-south length is an avenue (m)
MAX_GAP = 350.0        # a longer gap between consecutive streets ends the column (m)
NOT_AVENUES = {"FDR Dr", "Harlem River Dr", "Henry Hudson Pkwy", "West St",
               "12th Ave", "Joe DiMaggio Hwy", "W Side Hwy"}
SAME_AS = {"México-Tenochtitlan Ave": "2nd Ave"}  # co-names OSM uses on stretches of an avenue
ALGO = 4             # bump when planning logic changes, to force a re-plan


class Graph:
    def __init__(self, geo):
        segs = geo["seg_xy"]
        self.segs = segs
        self.length = np.array(geo["segs"]["len"], dtype=float)
        key = {}
        self.xy = []

        def node(p):
            p = (int(p[0]), int(p[1]))
            if p not in key:
                key[p] = len(self.xy)
                self.xy.append(p)
            return key[p]

        self.ends = np.array([(node(s[0]), node(s[-1])) for s in segs])
        self.xy = np.array(self.xy, dtype=float)
        n = len(self.xy)
        rows, cols, w = [], [], []
        self.edge = {}  # (u, v) -> ("seg", index) or ("snap", cost); cheapest kept
        for i, (u, v) in enumerate(self.ends):
            L = max(self.length[i], 1.0)
            if u != v and ((u, v) not in self.edge or self.length[self.edge[(u, v)][1]] > L):
                self.edge[(u, v)] = self.edge[(v, u)] = ("seg", i)
            rows += [u, v]
            cols += [v, u]
            w += [L, L]
        tree = cKDTree(self.xy)
        # Park paths often meet the grid only through sidewalks, which aren't counted. Join each
        # dead end to its nearest other node rather than every close pair of nodes, which would
        # let routes cut straight through buildings and fences.
        deg = np.bincount(self.ends.ravel(), minlength=n)
        for u in np.flatnonzero(deg == 1):
            u = int(u)
            dist, idx = tree.query(self.xy[u], k=8, distance_upper_bound=SNAP)
            for d, v in zip(dist, idx):
                v = int(v)
                if np.isfinite(d) and v != u and (u, v) not in self.edge:
                    c = float(d) * 1.2 + 1
                    self.edge[(u, v)] = self.edge[(v, u)] = ("snap", c)
                    rows += [u, v]
                    cols += [v, u]
                    w += [c, c]
                    break
        self.m = csr_matrix((w, (rows, cols)), shape=(n, n))
        self.ncomp, self.comp = connected_components(self.m, directed=False)
        self.tree = tree


class Row:
    """One street block across a column, from its west avenue corner to its east one."""

    def __init__(self, G, street, nodes, segs):
        if G.xy[nodes[0], 0] > G.xy[nodes[-1], 0]:
            nodes, segs = nodes[::-1], segs[::-1]
        self.street, self.nodes, self.segs = street, nodes, segs
        self.len = float(sum(G.length[i] for i in segs))
        self.y = float((G.xy[nodes[0], 1] + G.xy[nodes[-1], 1]) / 2)

    def end(self, side):  # side 0 = west corner, 1 = east corner
        return self.nodes[-1] if side else self.nodes[0]


def columns(G, geo):
    """Runs of rows per column: {(west avenue, east avenue): [(rows, legs), ...]} where
    rows are ordered south to north and legs[k][side] = (nodes, segs, metres) is the
    avenue block joining row k to row k + 1 on that side."""
    S = geo["segs"]
    names = geo["names"]
    canon = {names.index(a): names.index(b) for a, b in SAME_AS.items() if a in names and b in names}
    nm = np.array([canon.get(x, x) for x in S["n"]])
    cls = np.array(S["cls"])
    vert, horiz = collections.Counter(), collections.Counter()
    for i, (u, v) in enumerate(G.ends):
        if cls[i] == 2 or nm[i] < 0:
            continue
        dx, dy = np.abs(G.xy[v] - G.xy[u])
        if dy > 2 * dx:
            vert[nm[i]] += G.length[i]
        elif dx > 2 * dy:
            horiz[nm[i]] += G.length[i]
    ave = {k for k, v in vert.items()
           if v >= AVE_MIN and v > 3 * horiz[k] and names[k] not in NOT_AVENUES}

    node_ave = collections.defaultdict(set)
    ave_adj = collections.defaultdict(list)     # node -> [(other node, seg)] along avenues
    street_adj = collections.defaultdict(list)  # node -> [(other node, seg)] along streets
    for i, (u, v) in enumerate(G.ends):
        u, v = int(u), int(v)
        if cls[i] == 2 or nm[i] < 0 or u == v:
            continue
        if nm[i] in ave:
            node_ave[u].add(nm[i])
            node_ave[v].add(nm[i])
            ave_adj[u].append((v, i))
            ave_adj[v].append((u, i))
        else:
            street_adj[u].append((v, i))
            street_adj[v].append((u, i))

    def corner(u):  # the one avenue at this node, or None (no avenue, or two meet here)
        a = node_ave.get(u, ())
        return next(iter(a)) if len(a) == 1 else None

    # walk each street from one avenue corner to the next
    cols = collections.defaultdict(dict)
    for u in list(node_ave):
        a = corner(u)
        if a is None:
            continue
        for v, i in street_adj[u]:
            nodes, segs, name, L = [u, v], [i], nm[i], G.length[i]
            while v not in node_ave and L < 800:
                nxt = [(w, j) for w, j in street_adj[v] if nm[j] == name and j != segs[-1]]
                if len(nxt) != 1:
                    break
                v, j = nxt[0]
                nodes.append(v)
                segs.append(j)
                L += G.length[j]
            b = corner(v)
            if b is None or b == a or len(set(nodes)) != len(nodes):
                continue
            dx, dy = np.abs(G.xy[v] - G.xy[u])
            if not (dx > 2 * dy and 50 < dx < 700 and L < 1.3 * dx + 20):
                continue
            row = Row(G, int(name), nodes, segs)
            w_ave, e_ave = (a, b) if row.nodes[0] == u else (b, a)
            cols[(int(w_ave), int(e_ave))][(row.nodes[0], row.nodes[-1])] = row

    def ave_leg(a, p, q, limit):
        """Shortest walk from p to q along avenue a alone, or None."""
        dist, prev, heap = {p: 0.0}, {}, [(0.0, p)]
        while heap:
            d, x = heapq.heappop(heap)
            if x == q:
                nodes, segs = [q], []
                while nodes[-1] != p:
                    y, j = prev[nodes[-1]]
                    nodes.append(y)
                    segs.append(j)
                return nodes[::-1], segs[::-1], d
            if d > dist[x] or d > limit:
                continue
            for y, j in ave_adj[x]:
                if nm[j] == a and d + G.length[j] < dist.get(y, np.inf):
                    dist[y] = d + G.length[j]
                    prev[y] = (x, j)
                    heapq.heappush(heap, (dist[y], y))
        return None

    out = {}
    for (wa, ea), rows in cols.items():
        rows = sorted(rows.values(), key=lambda r: -r.y)  # y grows southwards: south first
        runs, cur, legs = [], [], []
        for r in rows:
            if cur:
                gap = cur[-1].y - r.y
                if gap < 30:  # same street seen twice, or a sliver: keep the first
                    continue
                leg = None
                if gap < MAX_GAP:
                    lim = 1.5 * gap + 40
                    leg = (ave_leg(wa, cur[-1].end(0), r.end(0), lim),
                           ave_leg(ea, cur[-1].end(1), r.end(1), lim))
                if leg is None or None in leg:
                    if len(cur) >= 3:
                        runs.append((cur, legs))
                    cur, legs = [], []
                else:
                    legs.append(leg)
            cur.append(r)
        if len(cur) >= 3:
            runs.append((cur, legs))
        if runs:
            out[(wa, ea)] = runs
    return out


def node_path(pred, x):
    """Nodes from x back to the source of a predecessor array (x first)."""
    out = [int(x)]
    while pred[out[-1]] >= 0:
        out.append(int(pred[out[-1]]))
    return out


def coords(G, nodes):
    pts = []
    for u, v in zip(nodes, nodes[1:]):
        kind, x = G.edge[(u, v)]
        if kind == "seg":
            c = G.segs[x]
            c = c if tuple(map(int, c[0])) == tuple(G.xy[u].astype(int)) else c[::-1]
        else:
            c = [G.xy[u], G.xy[v]]
        for p in c:
            p = (int(p[0]), int(p[1]))
            if not pts or pts[-1] != p:
                pts.append(p)
    flat = [pts[0][0], pts[0][1]]
    for a, b in zip(pts, pts[1:]):
        flat += [b[0] - a[0], b[1] - a[1]]
    return flat


class Run:
    """A run of rows in one column, with prefix sums for O(1) zigzag lengths."""

    def __init__(self, key, idx, rows, legs):
        self.key, self.idx, self.rows, self.legs = key, idx, rows, legs
        self.ys = np.array([r.y for r in rows])  # descending: south first
        n = len(rows)
        self.rlen = np.concatenate([[0.0], np.cumsum([r.len for r in rows])])
        par = np.arange(n - 1) % 2
        self.llen = {(s, p): np.concatenate([[0.0], np.cumsum([lg[s][2] for lg in legs] * (par == p))])
                     for s in (0, 1) for p in (0, 1)}

    def rescore(self, rem):
        n = len(self.rows)
        self.rnew = np.concatenate([[0.0], np.cumsum([rem[r.segs].sum() for r in self.rows])])
        par = np.arange(n - 1) % 2
        self.lnew = {(s, p): np.concatenate([[0.0], np.cumsum(np.array([rem[lg[s][1]].sum() for lg in self.legs])
                                                                * (par == p))])
                     for s in (0, 1) for p in (0, 1)}

    def _legs(self, tab, i, j, s, north):
        # leg k joins rows k and k + 1. Going north from row i, the first leg is on side 1 - s;
        # going south from row j, the first leg (k = j - 1) is.
        p = i % 2 if north else (j - 1) % 2
        return (tab[(1 - s, p)][j] - tab[(1 - s, p)][i]) + (tab[(s, 1 - p)][j] - tab[(s, 1 - p)][i])

    def length(self, i, j, s, north):
        return self.rlen[j + 1] - self.rlen[i] + self._legs(self.llen, i, j, s, north)

    def new(self, i, j, s, north):
        return self.rnew[j + 1] - self.rnew[i] + self._legs(self.lnew, i, j, s, north)

    @staticmethod
    def end_side(i, j, s):
        return s if (j - i + 1) % 2 == 0 else 1 - s

    def corners(self, i, j, s, north):
        e = self.end_side(i, j, s)
        a, b = (self.rows[i], self.rows[j]) if north else (self.rows[j], self.rows[i])
        return a.end(s), b.end(e)

    def walk(self, i, j, s, north):
        """Nodes and segments of the zigzag."""
        nodes, segs, side = [], [], s
        order = range(i, j + 1) if north else range(j, i - 1, -1)
        for t, k in enumerate(order):
            r = self.rows[k]
            seq = r.nodes if side == 0 else r.nodes[::-1]
            nodes += seq if not nodes else seq[1:]
            segs += r.segs
            side = 1 - side
            if t < j - i:
                leg = self.legs[k][side] if north else self.legs[k - 1][side]
                ln = leg[0] if north else leg[0][::-1]
                nodes += ln[1:]
                segs += leg[1]
        return nodes, segs


def chunks(runs_by_col):
    """Every chunk of 1..MAX_COLS side-by-side column pieces over a shared range of streets,
    as lists of (run, i, j) from west to east."""
    by_west = collections.defaultdict(list)
    for key in runs_by_col:
        by_west[key[0]].append(key)

    def piece(key, ylo, yhi):
        """The rows of this column within [ylo, yhi], or None. Uptown the columns break
        often, so a neighbour need only span half the range."""
        best = None
        for run in runs_by_col[key]:
            ys = run.ys
            i = int(np.searchsorted(-ys, -(yhi + 40)))
            j = int(np.searchsorted(-ys, -(ylo - 40), side="right")) - 1
            if j - i + 1 >= MIN_ROWS and ys[i] - ys[j] >= 0.5 * (yhi - ylo):
                if best is None or j - i > best[2] - best[1]:
                    best = (run, i, j)
        return best

    for key, runs in runs_by_col.items():
        for run in runs:
            n = len(run.rows)
            for i in range(n):
                for j in range(i + MIN_ROWS - 1, n):
                    if run.length(i, j, 0, True) > MAX_LEN:
                        break
                    yhi, ylo = run.ys[i], run.ys[j]
                    stack = [[(run, i, j)]]
                    while stack:
                        ch = stack.pop()
                        yield ch
                        if len(ch) == MAX_COLS:
                            continue
                        for k2 in by_west[ch[-1][0].key[1]]:
                            p = piece(k2, ylo, yhi)
                            if p:
                                stack.append(ch + [p])


def plan(geo, cov):
    G = Graph(geo)
    names = geo["names"]
    f = np.array(cov["f"], dtype=float)
    rem = G.length * (1 - f / 100)
    st = geo["stations"]
    st_xy = np.array([(s["x"], s["y"]) for s in st], dtype=float)
    st_node = G.tree.query(st_xy)[1]
    home = np.array([bool(set(s["routes"].split()) & HOME_LINES) for s in st])
    node_st = {}
    for k, nd in enumerate(st_node):  # prefer a 1/2/3 station when two share a node
        if int(nd) not in node_st or home[k]:
            node_st[int(nd)] = k
    # distance from every node to the nearest 1/2/3 station and to the nearest of any line
    d_home, p_home = dijkstra(G.m, directed=False, indices=st_node[home], return_predecessors=True,
                              min_only=True)[:2]
    d_any, p_any = dijkstra(G.m, directed=False, indices=st_node, return_predecessors=True,
                            min_only=True)[:2]

    def station(nd):  # (metres, score cost, predecessors) of the leg to the chosen station
        if d_home[nd] <= d_any[nd] + HOME_SLACK:
            return d_home[nd], 0.0, p_home
        return d_any[nd], END_COST, p_any

    runs_by_col = {key: [Run(key, ri, rows, legs) for ri, (rows, legs) in enumerate(rs)]
                   for key, rs in columns(G, geo).items()}

    # Lengths don't depend on coverage, so enumerate and length-filter the candidates once.
    # A candidate: pieces in running order, each (run, i, j, start side, north?).
    cands = []
    for ch in chunks(runs_by_col):
        k = len(ch)
        for order in (ch, ch[::-1]):
            east = order is ch  # moving west to east across the columns
            nxt, prv = (1, 0) if east else (0, 1)
            for north0 in (True, False):
                firsts = (0, 1) if k == 1 else (None,)
                for s_first in firsts:
                    pieces, L, prev_end = [], 0.0, None
                    for t, (run, i, j) in enumerate(order):
                        north = north0 if t % 2 == 0 else not north0
                        m = j - i + 1
                        if t == 0:
                            s = s_first if k == 1 else (nxt if m % 2 == 0 else 1 - nxt)
                        else:
                            s = prv
                        a, b = run.corners(i, j, s, north)
                        if prev_end is not None:
                            dx, dy = np.abs(G.xy[a] - G.xy[prev_end])
                            L += dx + dy  # grid distance to the next column's first corner
                        L += run.length(i, j, s, north)
                        pieces.append((run, i, j, s, north))
                        prev_end = b
                    start = pieces[0][0].corners(*pieces[0][1:])[0]
                    if not (np.isfinite(d_home[start]) and np.isfinite(d_home[prev_end])):
                        continue
                    sa, ca, _ = station(start)
                    sb, cb, _ = station(prev_end)
                    total = L + sa + sb
                    if MIN_LEN <= total <= MAX_LEN:
                        cands.append((pieces, total, ca + cb, start, prev_end))

    routes, used, centres = [], set(), []
    for _ in range(N_ROUTES):
        for rs in runs_by_col.values():
            for run in rs:
                run.rescore(rem)
        scored = []
        for c in cands:
            pieces, total, cost = c[:3]
            new = sum(run.new(i, j, s, north) for run, i, j, s, north in pieces)
            if new >= MIN_NEW:
                scored.append((new - LEN_COST * total - cost, new, c))
        scored.sort(key=lambda x: -x[0])
        pick = None
        for score, new, c in scored:
            pieces = c[0]
            rowset = {(run.key, run.idx, r) for run, i, j, *_ in pieces for r in range(i, j + 1)}
            if rowset & used:
                continue
            xs = [G.xy[run.rows[r].nodes[0], 0] for run, i, j, *_ in pieces for r in (i, j)]
            ys = [run.ys[r] for run, i, j, *_ in pieces for r in (i, j)]
            ctr = np.array([np.mean(xs), np.mean(ys)])
            if any(np.hypot(*(ctr - p)) < SPREAD for p in centres):
                continue
            pick = (c, rowset, ctr)
            break
        if pick is None:
            break
        (pieces, total, cost, start, end), rowset, ctr = pick
        used |= rowset
        centres.append(ctr)

        _, _, pred_a = station(start)
        _, _, pred_b = station(end)
        nodes = node_path(pred_a, start)[::-1]  # station -> first corner
        segs, steps = [], []
        for run, i, j, s, north in pieces:
            zn, zs = run.walk(i, j, s, north)
            if nodes[-1] != zn[0]:  # step over to the next column
                _, pr = dijkstra(G.m, directed=False, indices=nodes[-1], return_predecessors=True,
                                 limit=3000)
                nodes += node_path(pr, zn[0])[::-1][1:]
            nodes += zn[1:]
            segs += zs
            lo, hi = run.rows[i], run.rows[j]
            steps.append({"west": names[run.key[0]], "east": names[run.key[1]],
                          "dir": "north" if north else "south",
                          "from": names[(lo if north else hi).street],
                          "to": names[(hi if north else lo).street],
                          "streets": j - i + 1, "side": "west" if s == 0 else "east"})
        nodes += node_path(pred_b, end)[1:]
        s_i, t_i = node_st[nodes[0]], node_st[nodes[-1]]
        new_m = float(rem[sorted(set(segs))].sum())
        L = float(sum(G.length[G.edge[(u, v)][1]] if G.edge[(u, v)][0] == "seg"
                      else G.edge[(u, v)][1] for u, v in zip(nodes, nodes[1:])))
        routes.append({"from": st[s_i]["name"], "from_routes": st[s_i]["routes"],
                       "to": st[t_i]["name"], "to_routes": st[t_i]["routes"],
                       "home": bool(home[s_i] and home[t_i]), "steps": steps,
                       "len_m": round(L), "new_m": round(new_m),
                       "segs": sorted(set(int(x) for x in segs)), "path": coords(G, nodes)})
        rem[segs] = 0  # a later route gets no credit for the same blocks
    routes.sort(key=lambda r: (not r["home"], -r["new_m"]))
    return routes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datadir", nargs="?", default=DATA)
    ap.add_argument("--if-stale", action="store_true",
                    help="skip when planner.json was built from this exact coverage and algorithm")
    args = ap.parse_args()
    d = args.datadir
    geo = load_geo()
    with open(os.path.join(d, "coverage.json")) as f:
        cov = json.load(f)
    # fingerprint of what the plan depends on: the network, the coverage itself, the algorithm
    sig = hashlib.sha1(json.dumps([geo["v"], cov["f"], cov["by"], ALGO]).encode()).hexdigest()[:12]
    if args.if_stale:
        try:
            with open(os.path.join(d, "planner.json")) as f:
                old = json.load(f)
            if old.get("sig") == sig:
                print("Routes are up to date.")
                return
        except (FileNotFoundError, ValueError):
            pass
    routes = plan(geo, cov)
    out = {"v": geo["v"], "updated": cov["updated"], "sig": sig, "routes": routes}
    with open(os.path.join(d, "planner.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    for r in routes:
        print(f"  {r['from']} ({r['from_routes']}) -> {r['to']} ({r['to_routes']}): "
              f"{r['len_m'] / 1609.344:.1f} mi, {r['new_m'] / 1609.344:.1f} new mi")
        for t in r["steps"]:
            print(f"      {t['west']} / {t['east']}: {t['dir']} {t['from']} to {t['to']}, {t['side']} side")


if __name__ == "__main__":
    main()
