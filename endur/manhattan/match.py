"""Match a GPS track to Manhattan street segments.

Road-centric proximity with a bearing check: every counted segment is cut into
~10 m pieces; a piece is covered when a track point passes within R metres
travelling roughly parallel to it. Crossing a street at an intersection does not
count, and because only counted segments are candidates, sidewalks can't
capture the match. Keep the public surface to `Matcher.match` so an HMM matcher
can be swapped in later if validation calls for it.
"""
import math

import numpy as np
from scipy.spatial import cKDTree

PARAMS = {
    "R": 30.0,          # max distance from track to street centreline (m); Midtown GPS drifts 15-25 m
    "piece": 10.0,      # target piece length (m)
    "angle": 35.0,      # max bearing difference (deg)
    "slack": 8.0,       # a point credits only the nearest matching way (plus any within this much further);
                        # streets compete only with streets, paths with everything
    "short": 20.0,      # blocks shorter than this (intersection pieces) skip the bearing check
    "same_name": True,  # parallel carriageways of the same named street don't exclude each other
    "resample": 5.0,    # track resampling step (m)
    "gap_fill": 30.0,   # fill uncovered gaps up to this long within a segment (m)
    "graze": 20.0,      # drop covered runs shorter than this (m) ...
    "snap": 10.0,       # ... and extend runs this close to a segment end
    "break": 100.0,     # don't interpolate across raw-track gaps longer than this (m)
    "max_speed": 10.0,  # ... or implying more than this speed (m/s)
    "complete_frac": 0.85,   # a segment counts as done at 85% of its length ...
    "complete_left": 20.0,   # ... or when at most 20 m are left
}


def is_complete(covered, length):
    return covered >= PARAMS["complete_frac"] * length or length - covered <= PARAMS["complete_left"]


def union(a, b):
    """Union of two sorted interval lists [[s, e], ...]."""
    out = []
    for s, e in sorted(a + b):
        if out and s <= out[-1][1] + 0.5:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(iv):
    return sum(e - s for s, e in iv)


def _along(pts, cum, d):
    """Points at distances d along a polyline with cumulative lengths cum."""
    x = np.interp(d, cum, pts[:, 0])
    y = np.interp(d, cum, pts[:, 1])
    return np.column_stack([x, y])


class Matcher:
    def __init__(self, seg_xy, cls=None, name=None, **params):
        """`cls` gives each segment's class (0 major road, 1 street, 2 path; default all streets);
        `name` its street-name id (-1 unnamed), so both carriageways of one street share credit."""
        self.p = {**PARAMS, **params}
        A, B, seg, o0, o1, first = [], [], [], [], [], []
        self.length = np.zeros(len(seg_xy))
        for s, pts in enumerate(seg_xy):
            pts = np.asarray(pts, dtype=float)
            cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
            L = cum[-1]
            self.length[s] = L
            n = max(1, round(L / self.p["piece"]))
            edges = np.linspace(0.0, L, n + 1)
            first.append(len(seg))
            A.append(_along(pts, cum, edges[:-1]))
            B.append(_along(pts, cum, edges[1:]))
            seg += [s] * n
            o0.append(edges[:-1])
            o1.append(edges[1:])
        first.append(len(seg))
        self.A, self.B = np.vstack(A), np.vstack(B)
        self.seg = np.asarray(seg)
        self.o0, self.o1 = np.concatenate(o0), np.concatenate(o1)
        self.first = np.asarray(first)
        cls = np.zeros(len(seg_xy), dtype=int) if cls is None else np.asarray(cls)
        self.path = (cls == 2)[self.seg]                             # per piece
        self.short = (self.length < self.p["short"])[self.seg]      # per piece
        name = np.full(len(seg_xy), -1) if name is None else np.asarray(name)
        self.name = name[self.seg]                                   # per piece
        d = self.B - self.A
        self.plen = np.hypot(d[:, 0], d[:, 1])
        self.pbear = np.mod(np.arctan2(d[:, 1], d[:, 0]), math.pi)
        self.tree = cKDTree((self.A + self.B) / 2)
        self.reach = self.p["R"] + self.plen.max() / 2

    # --- track preparation ---
    def _chunks(self, xy, t=None):
        """Split a raw track at implausible gaps, then resample each chunk."""
        xy = np.asarray(xy, dtype=float)
        if len(xy) < 2:
            return []
        step = np.hypot(*np.diff(xy, axis=0).T)
        brk = step > self.p["break"]
        if t is not None:
            dt = np.maximum(np.diff(np.asarray(t, dtype=float)), 1.0)
            brk |= (step / dt > self.p["max_speed"]) & (step > 25)
        cuts = np.flatnonzero(brk) + 1
        out = []
        for c in np.split(np.arange(len(xy)), cuts):
            if len(c) < 2:
                continue
            pts = xy[c]
            if t is not None and len(pts) >= 3:  # light smoothing of raw GPS
                sm = pts.copy()
                sm[1:-1] = (pts[:-2] + pts[1:-1] + pts[2:]) / 3
                pts = sm
            cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
            if cum[-1] < self.p["resample"]:
                continue
            d = np.arange(0.0, cum[-1] + 1e-9, self.p["resample"])
            out.append(_along(pts, cum, d))
        return out

    def match(self, xy, t=None):
        """Return {segment index: [[start_m, end_m], ...]} covered by this track.

        `xy` is an (N, 2) array in the projected frame; pass raw-stream times `t`
        (seconds) to enable smoothing and speed-based gap breaking. Simplified
        polylines should be passed without `t`.
        """
        hit = np.zeros(len(self.seg), dtype=bool)
        R = self.p["R"]
        max_ang = math.radians(self.p["angle"])
        for pts in self._chunks(xy, t):
            n = len(pts)
            k = 2  # bearing over +-10 m
            fwd = pts[np.minimum(np.arange(n) + k, n - 1)] - pts[np.maximum(np.arange(n) - k, 0)]
            tbear = np.mod(np.arctan2(fwd[:, 1], fwd[:, 0]), math.pi)
            cand = self.tree.query_ball_point(pts, self.reach)
            ii = np.repeat(np.arange(n), [len(c) for c in cand])
            if not len(ii):
                continue
            pp = np.fromiter((p for c in cand for p in c), dtype=np.int64, count=len(ii))
            # point-to-piece distance
            a, b, q = self.A[pp], self.B[pp], pts[ii]
            ab = b - a
            L2 = np.maximum((ab ** 2).sum(1), 1e-9)
            u = np.clip(((q - a) * ab).sum(1) / L2, 0, 1)
            dist = np.hypot(*(a + ab * u[:, None] - q).T)
            dang = np.abs(tbear[ii] - self.pbear[pp])
            dang = np.minimum(dang, math.pi - dang)
            ok = (dist <= R) & ((dang <= max_ang) | self.short[pp])
            # Exclusive, so parallel ways aren't double-counted: a park path 15 m from the one
            # you ran on is a different path, and so is the far carriageway of a divided avenue.
            # Streets compete only with streets, though: running the park-side path along
            # Central Park West, or the bike lane beside an avenue, still covers the street.
            # Both carriageways of one named street (Park Ave, Riverside Dr) count as one street.
            street = ok & ~self.path[pp]
            near_street = np.full(n, np.inf)
            near_any = np.full(n, np.inf)
            np.minimum.at(near_street, ii[street], dist[street])
            np.minimum.at(near_any, ii[ok], dist[ok])
            limit = np.where(self.path[pp], near_any[ii], near_street[ii]) + self.p["slack"]
            if self.p["same_name"]:
                # the nearest street's name, per point; pieces sharing it are never excluded
                order = np.lexsort((dist, ii))
                first = np.ones(len(order), dtype=bool)
                first[1:] = ii[order][1:] != ii[order][:-1]
                sorted_street = order[street[order]]
                nn = np.full(n, -2)
                fs = np.ones(len(sorted_street), dtype=bool)
                fs[1:] = ii[sorted_street][1:] != ii[sorted_street][:-1]
                nn[ii[sorted_street[fs]]] = self.name[pp[sorted_street[fs]]]
                same = (~self.path[pp]) & (self.name[pp] >= 0) & (self.name[pp] == nn[ii])
                ok &= (dist <= limit) | same
            else:
                ok &= dist <= limit
            hit[pp[ok]] = True
        return self._intervals(hit)

    def _intervals(self, hit):
        out = {}
        for s in np.unique(self.seg[hit]):
            lo, hi = self.first[s], self.first[s + 1]
            h = hit[lo:hi]
            o0, o1 = self.o0[lo:hi], self.o1[lo:hi]
            L = self.length[s]
            runs = []
            i = 0
            while i < len(h):
                if h[i]:
                    j = i
                    while j + 1 < len(h) and h[j + 1]:
                        j += 1
                    runs.append([o0[i], o1[j]])
                    i = j + 1
                else:
                    i += 1
            merged = []
            for r in runs:  # fill short gaps
                if merged and r[0] - merged[-1][1] <= self.p["gap_fill"]:
                    merged[-1][1] = r[1]
                else:
                    merged.append(r)
            keep = [r for r in merged if r[1] - r[0] >= min(self.p["graze"], 0.5 * L)]
            if not keep:
                continue
            if keep[0][0] <= self.p["snap"]:
                keep[0][0] = 0.0
            if L - keep[-1][1] <= self.p["snap"]:
                keep[-1][1] = L
            out[int(s)] = [[round(a, 1), round(b, 1)] for a, b in keep]
        return out
