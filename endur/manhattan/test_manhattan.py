"""Tests for the Manhattan coverage pipeline: `.venv/bin/python -m pytest endur/manhattan`."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from geo import Projection, decode_coords  # noqa: E402
from match import Matcher, is_complete, total, union  # noqa: E402


def test_union_merges_and_sorts():
    assert union([[0, 10], [20, 30]], [[5, 22]]) == [[0, 30]]
    assert union([[20, 30]], [[0, 10]]) == [[0, 10], [20, 30]]
    assert total([[0, 10], [20, 30]]) == 20


def test_completion_rule():
    assert is_complete(90, 100)
    assert not is_complete(80, 100)
    assert is_complete(10, 24)          # short stub: <= 15 m left
    assert not is_complete(200, 250)    # long block: 50 m left is not done


def test_projection_round_trip_and_grid_rotation():
    P = Projection()
    lon, lat = -73.9857, 40.7484
    x, y = P.fwd(lon, lat)
    lon2, lat2 = P.inv(x, y)
    assert abs(lon - lon2) < 1e-9 and abs(lat - lat2) < 1e-9
    # a step along an avenue (bearing ~29 deg) should point straight up (negative y)
    d = 0.001
    x2, y2 = P.fwd(lon + d * math.sin(math.radians(29)) / math.cos(math.radians(40.78)),
                   lat + d * math.cos(math.radians(29)))
    assert abs(x2 - x) < 1.0 and y2 < y


def test_decode_coords():
    assert decode_coords([10, 20, 5, -5, 1, 1]) == [(10, 20), (15, 15), (16, 16)]


def grid():
    """A toy grid: one 'avenue' running north (x=0) crossed by three streets."""
    ave = [[(0, -y0), (0, -y0 - 80)] for y0 in (0, 80, 160)]
    streets = [[(0, -y), (250, -y)] for y in (0, 80, 160, 240)]
    return ave + streets


def test_running_an_avenue_does_not_credit_cross_streets():
    segs = grid()
    M = Matcher(segs)
    rng = np.random.default_rng(0)
    ys = np.arange(0, 240, 3.0)
    track = np.column_stack([rng.normal(6, 4, len(ys)), -ys])  # sidewalk, noisy
    got = M.match(track, t=np.arange(len(ys)))
    for i in range(3):  # the three avenue blocks are covered end to end
        assert i in got and got[i][0][0] == 0 and got[i][-1][1] == 80
    for i in range(3, 7):  # the streets are only crossed, never credited
        assert i not in got


def test_running_a_street_credits_it_and_skips_gps_jumps():
    segs = grid()
    M = Matcher(segs)
    xs = np.arange(0, 250, 3.0)
    track = np.column_stack([xs, np.full(len(xs), -84.0)])
    got = M.match(track)
    assert 4 in got and got[4] == [[0.0, 250.0]]
    # a teleport (e.g. a GPS gap) must not draw a straight line along the avenue
    jump = np.array([[5, 0], [5, -2], [5, -238], [5, -240]], dtype=float)
    got = M.match(jump, t=[0, 1, 2, 3])
    assert not got
