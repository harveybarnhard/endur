"""Shared projection and geo.json helpers for the Manhattan pipeline.

Coordinates are metres in a local equirectangular frame centred on Manhattan,
rotated so the street grid's avenues (bearing ~29 deg) point straight up, with
y increasing downwards (screen convention). The page uses the same frame, so it
only needs to scale; `inv` exists for GPX export and debugging.
"""
import json
import math
import os

ROT_DEG = 29.0
LAT0, LON0 = 40.78, -73.97

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "manhattan")


class Projection:
    def __init__(self, lat0=LAT0, lon0=LON0, rot=ROT_DEG):
        self.lat0, self.lon0, self.rot = lat0, lon0, rot
        p = math.radians(lat0)
        self.my = 111132.954 - 559.822 * math.cos(2 * p) + 1.175 * math.cos(4 * p)
        self.mx = 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p)
        self.c, self.s = math.cos(math.radians(rot)), math.sin(math.radians(rot))

    def fwd(self, lon, lat):
        e = (lon - self.lon0) * self.mx
        n = (lat - self.lat0) * self.my
        return (e * self.c - n * self.s, -(e * self.s + n * self.c))

    def fwd_many(self, lonlat):
        """Vectorised fwd for an (N, 2) numpy array of lon/lat."""
        import numpy as np
        a = np.asarray(lonlat, dtype=float)
        e = (a[:, 0] - self.lon0) * self.mx
        n = (a[:, 1] - self.lat0) * self.my
        return np.column_stack([e * self.c - n * self.s, -(e * self.s + n * self.c)])

    def inv(self, x, y):
        yu = -y
        e = x * self.c + yu * self.s
        n = -x * self.s + yu * self.c
        return (self.lon0 + e / self.mx, self.lat0 + n / self.my)

    def params(self):
        return {"lat0": self.lat0, "lon0": self.lon0, "rot": self.rot,
                "mx": round(self.mx, 4), "my": round(self.my, 4)}


def decode_coords(flat):
    """Delta-encoded [x0, y0, dx1, dy1, ...] -> [(x, y), ...]."""
    x, y = flat[0], flat[1]
    pts = [(x, y)]
    for i in range(2, len(flat), 2):
        x += flat[i]
        y += flat[i + 1]
        pts.append((x, y))
    return pts


def load_geo(path=None):
    with open(path or os.path.join(DATA, "geo.json")) as f:
        geo = json.load(f)
    geo["seg_xy"] = [decode_coords(c) for c in geo["segs"]["c"]]
    return geo
