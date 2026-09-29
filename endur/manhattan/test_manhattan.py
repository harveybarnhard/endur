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
    assert is_complete(86, 100)
    assert not is_complete(160, 200)     # 80% with 40 m left is not done
    assert is_complete(10, 28)          # short stub: <= 20 m left
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


def test_parallel_paths_credit_only_the_one_run():
    # two park paths 16 m apart, and an avenue with a bike lane 5 m beside it
    segs = [[(0, 0), (0, -300)], [(16, 0), (16, -300)],
            [(200, 0), (200, -300)], [(205, 0), (205, -300)]]
    M = Matcher(segs)
    rng = np.random.default_rng(1)
    ys = np.arange(0, 300, 3.0)
    on_path = np.column_stack([rng.normal(0, 3, len(ys)), -ys])
    got = M.match(on_path, t=np.arange(len(ys)))
    assert 0 in got and total(got[0]) > 270
    assert 1 not in got or total(got[1]) < 30
    sidewalk = np.column_stack([rng.normal(211, 3, len(ys)), -ys])  # 6-11 m from both
    got = M.match(sidewalk, t=np.arange(len(ys)))
    assert total(got.get(2, [])) > 250 and total(got.get(3, [])) > 250


def test_streets_compete_only_with_streets_and_carriageways_share_credit():
    # an avenue (street) with a bike lane (path) 6 m east; a divided street whose two
    # carriageways (same name) are 22 m apart; and an unrelated street 40 m further on
    segs = [[(0, 0), (0, -300)], [(6, 0), (6, -300)],
            [(200, 0), (200, -300)], [(222, 0), (222, -300)], [(262, 0), (262, -300)]]
    cls = [1, 2, 0, 0, 1]
    name = [0, -1, 1, 1, 2]
    M = Matcher(segs, cls=cls, name=name)
    rng = np.random.default_rng(2)
    ys = np.arange(0, 300, 3.0)
    t = np.arange(len(ys))
    # running in the bike lane: nearer the path, 6-8 m further from the avenue, still covers both
    got = M.match(np.column_stack([rng.normal(8, 2, len(ys)), -ys]), t=t)
    assert total(got.get(0, [])) > 250 and total(got.get(1, [])) > 250
    # running beside the west carriageway covers both carriageways, not the street 40 m on
    got = M.match(np.column_stack([rng.normal(196, 2, len(ys)), -ys]), t=t)
    assert total(got.get(2, [])) > 250 and total(got.get(3, [])) > 250
    assert 4 not in got


def test_list_manhattan_keeps_foot_activities_on_manhattan_streets():
    import polyline
    import update

    on, off = polyline.encode([(40.75, -73.98)]), polyline.encode([(40.69, -73.94)])

    class Net:
        def touches(self, latlng):
            return latlng[0][0] > 40.7

    def act(i, start, kind="Run", poly=on, **kw):
        return {"id": i, "start_date_local": start + "T07:00:00Z", "sport_type": kind,
                "map": {"summary_polyline": poly}, **kw}

    listed = [act(1, "2025-03-02"), act(2, "2025-03-01", "Walk"), act(3, "2024-10-30"),
              act(4, "2025-03-03", "Ride"), act(5, "2025-03-04", poly=off),
              act(6, "2025-03-05", manual=True), act(7, "2025-03-06", poly=None), act(8, "2025-03-07")]

    class Api:
        def activities(self, after):
            return listed

    got = update.list_manhattan(Api(), Net(), after=0, skip={"8": 0})
    assert [a["id"] for a in got] == [2, 1]  # oldest first; no rides, pre-START, Brooklyn, manual, no-GPS or done


def test_access_token_refreshes_and_saves_a_rotated_refresh_token(tmp_path, monkeypatch):
    import json
    import time
    import strava_auth

    path = str(tmp_path / "strava.json")
    strava_auth.save(path, {"client_id": "1", "client_secret": "s", "refresh_token": "old",
                            "access_token": "stale", "expires_at": time.time() + 60})
    sent = []

    class Resp:
        status_code = 200

        def json(self):
            return {"access_token": "fresh", "refresh_token": "new", "expires_at": time.time() + 21600}

    monkeypatch.setattr(strava_auth.requests, "post", lambda url, data, timeout: sent.append(data) or Resp())
    assert strava_auth.access_token(path) == "fresh"
    assert sent[0]["grant_type"] == "refresh_token" and sent[0]["refresh_token"] == "old"
    saved = json.load(open(path))
    assert saved["refresh_token"] == "new" and oct(os.stat(path).st_mode & 0o777) == "0o600"
    assert strava_auth.access_token(path) == "fresh" and len(sent) == 1  # still valid: no second call


def test_touches_sees_sparse_summary_lines_along_a_street():
    import update

    P = Projection()
    geo = {"proj": {"lat0": P.lat0, "lon0": P.lon0, "rot": P.rot},
           "seg_xy": [[(0, 0), (0, -400)]], "segs": {"cls": [1], "n": [0]}}
    net = update.Network(geo)

    def latlng(pts):
        return [P.inv(x, y)[::-1] for x, y in pts]

    # a short walk whose summary line is two points mid-block, 100 m from either corner
    assert net.touches(latlng([(3, -100), (3, -300)]))
    assert not net.touches(latlng([(100, -100), (100, -300)]))  # a block away
