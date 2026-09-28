"""Suggest next runs: station-to-station routes through clusters of unrun streets.

    python endur/manhattan/planner.py [DATADIR]

Reads geo.json and coverage.json, writes planner.json. Each route starts at the
subway station nearest the south end of a cluster of uncovered segments, covers
every one of them (rural-postman approximation: connect the cluster with an MST
of shortest paths, pair up odd-degree nodes with a minimum-weight matching, walk
an Euler trail), and ends at the station nearest the north end.
"""
import json
import os
import sys

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from geo import DATA, load_geo  # noqa: E402

N_ROUTES = 6
TARGET = 13000.0       # aim for routes up to ~8 mi ...
MIN_LEN = 5000.0       # ... and at least ~3 mi
SEED_SPACING = 1500.0  # keep suggestions spread across the island
SNAP = 30.0            # join network pieces whose ends are this close (m)


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
        self.edge = {}  # (u, v) -> ("seg", index) or ("snap", None); cheapest kept
        for i, (u, v) in enumerate(self.ends):
            L = max(self.length[i], 1.0)
            if u != v and ((u, v) not in self.edge or self.length[self.edge[(u, v)][1]] > L):
                self.edge[(u, v)] = self.edge[(v, u)] = ("seg", i)
            rows += [u, v]
            cols += [v, u]
            w += [L, L]
        tree = cKDTree(self.xy)
        for u, v in tree.query_pairs(SNAP):
            if (u, v) not in self.edge:
                d = float(np.hypot(*(self.xy[u] - self.xy[v]))) * 1.2 + 1
                self.edge[(u, v)] = self.edge[(v, u)] = ("snap", d)
                rows += [u, v]
                cols += [v, u]
                w += [d, d]
        self.m = csr_matrix((w, (rows, cols)), shape=(n, n))
        self.ncomp, self.comp = connected_components(self.m, directed=False)
        self.tree = tree

    def cost(self, u, v):
        kind, x = self.edge[(u, v)]
        return self.length[x] if kind == "seg" else x

    def path(self, pred, src_row, t):
        out = [t]
        while out[-1] != src_row:
            p = pred[out[-1]]
            if p < 0:
                return None
            out.append(p)
        return out[::-1]


def rpp(G, required, s_node, t_node):
    """Open trail s_node -> t_node traversing every required segment. Returns
    (node sequence, metres) or None."""
    M = nx.MultiGraph()
    for i in required:
        u, v = G.ends[i]
        M.add_edge(int(u), int(v), seg=int(i), w=G.length[i])
    # connect components of the required graph (+ stations) with an MST of shortest paths
    M.add_node(int(s_node))
    M.add_node(int(t_node))
    comps = [list(c) for c in nx.connected_components(M)]
    deadhead = []
    if len(comps) > 1:
        srcs = [c for comp in comps for c in comp]
        D, P = dijkstra(G.m, directed=False, indices=srcs, return_predecessors=True, limit=8000)
        row = {n: r for r, n in enumerate(srcs)}
        K = nx.Graph()
        for a in range(len(comps)):
            for b in range(a + 1, len(comps)):
                best = None
                for x in comps[a]:
                    dx = D[row[x], comps[b]]
                    j = int(np.argmin(dx))
                    if np.isfinite(dx[j]) and (best is None or dx[j] < best[0]):
                        best = (dx[j], x, comps[b][j])
                if best:
                    K.add_edge(a, b, w=best[0], ends=best[1:])
        if not nx.is_connected(K) or K.number_of_nodes() < len(comps):
            return None
        for a, b, d in nx.minimum_spanning_edges(K, weight="w", data=True):
            x, y = d["ends"]
            deadhead.append(G.path(P[row[x]], x, y))
    for p in deadhead:
        for u, v in zip(p, p[1:]):
            M.add_edge(int(u), int(v), w=G.cost(u, v))
    M.add_edge(int(s_node), int(t_node), virtual=True, w=0.0)
    odd = [n for n, d in M.degree() if d % 2]
    if odd:
        D, P = dijkstra(G.m, directed=False, indices=odd, return_predecessors=True, limit=8000)
        K = nx.Graph()
        for a in range(len(odd)):
            for b in range(a + 1, len(odd)):
                d = D[a, odd[b]]
                if np.isfinite(d):
                    K.add_edge(a, b, w=d)
        match = nx.min_weight_matching(K, weight="w")
        if len(match) * 2 != len(odd):
            return None
        for a, b in match:
            p = G.path(P[a], odd[a], odd[b])
            for u, v in zip(p, p[1:]):
                M.add_edge(int(u), int(v), w=G.cost(u, v))
    if not nx.is_eulerian(M):
        return None
    circuit = list(nx.eulerian_circuit(M, source=int(s_node), keys=True))
    # rotate so the virtual edge is last, then drop it: trail from one station to the other
    vi = next(i for i, (u, v, k) in enumerate(circuit) if M.edges[u, v, k].get("virtual"))
    trail = circuit[vi + 1:] + circuit[:vi]
    if trail and trail[0][0] != s_node:
        trail = [(v, u, k) for u, v, k in reversed(trail)]
    total = sum(M.edges[u, v, k]["w"] for u, v, k in trail)
    return trail, total, M


def trail_coords(G, trail, M):
    pts = []
    for u, v, k in trail:
        d = M.edges[u, v, k]
        if "seg" in d:
            c = G.segs[d["seg"]]
            c = c if tuple(map(int, c[0])) == tuple(G.xy[u].astype(int)) else c[::-1]
        else:
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


def plan(geo, cov):
    G = Graph(geo)
    f = np.array(cov["f"])
    done = np.array(cov["done"])
    todo = np.flatnonzero(done < 0)
    mids = np.array([np.mean(G.xy[G.ends[i]], axis=0) for i in range(len(G.ends))])
    main = np.bincount(G.comp).argmax()
    st = geo["stations"]
    st_xy = np.array([(s["x"], s["y"]) for s in st], dtype=float)
    st_node = G.tree.query(st_xy)[1]
    st_ok = G.comp[st_node] == main
    # only plan within the main connected network (Governors Island has no subway)
    todo = todo[G.comp[G.ends[todo, 0]] == main]
    left = set(todo.tolist())
    tt = cKDTree(mids[todo])
    remaining_len = np.array([G.length[i] * (1 - f[i] / 100) for i in todo])
    seeds, routes = [], []
    for _ in range(N_ROUTES * 3):
        if len(routes) >= N_ROUTES or not left:
            break
        # seed: densest pocket of remaining street, away from earlier seeds
        best, best_score = None, -1
        for j in range(0, len(todo), 7):
            if todo[j] not in left:
                continue
            p = mids[todo[j]]
            if any(np.hypot(*(p - s)) < SEED_SPACING for s in seeds):
                continue
            near = tt.query_ball_point(p, 600)
            score = sum(remaining_len[k] for k in near if todo[k] in left)
            if score > best_score:
                best, best_score = p, score
        if best is None:
            break
        seeds.append(best)
        lo, hi, pick = 150.0, 1600.0, None
        for _ in range(7):  # binary search the cluster radius to hit the length target
            r = (lo + hi) / 2
            req = [todo[k] for k in tt.query_ball_point(best, r) if todo[k] in left]
            if not req:
                lo = r
                continue
            ys = mids[req][:, 1]
            south = mids[req][np.argmax(ys)]  # screen y grows southwards
            north = mids[req][np.argmin(ys)]
            cand = np.flatnonzero(st_ok)
            s_i = cand[np.argmin(np.hypot(*(st_xy[cand] - south).T))]
            t_i = cand[np.argmin(np.hypot(*(st_xy[cand] - north).T))]
            res = rpp(G, req, st_node[s_i], st_node[t_i])
            if res is None:
                hi = r
                continue
            trail, L, M = res
            if L > TARGET:
                hi = r
            else:
                pick = (req, s_i, t_i, trail, L, M)
                lo = r
        if not pick or pick[4] < MIN_LEN:
            continue
        req, s_i, t_i, trail, L, M = pick
        new_m = float(sum(G.length[i] * (1 - f[i] / 100) for i in req))
        routes.append({"from": st[s_i]["name"], "from_routes": st[s_i]["routes"],
                       "to": st[t_i]["name"], "to_routes": st[t_i]["routes"],
                       "len_m": round(L), "new_m": round(new_m), "segs": sorted(int(i) for i in req),
                       "path": trail_coords(G, trail, M)})
        left -= set(req)
    routes.sort(key=lambda r: -r["new_m"] / r["len_m"])
    return routes


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else DATA
    geo = load_geo()
    with open(os.path.join(d, "coverage.json")) as f:
        cov = json.load(f)
    routes = plan(geo, cov)
    out = {"v": geo["v"], "updated": cov["updated"], "routes": routes}
    with open(os.path.join(d, "planner.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    for r in routes:
        print(f"  {r['from']} -> {r['to']}: {r['len_m'] / 1609.344:.1f} mi, "
              f"{r['new_m'] / 1609.344:.1f} new mi, {len(r['segs'])} blocks")


if __name__ == "__main__":
    main()
