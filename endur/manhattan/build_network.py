"""Build the static Manhattan street network used by the coverage pipeline and page.

Run locally and rarely (street network changes slowly):
    .venv/bin/python endur/manhattan/build_network.py

Downloads OSM highways, park polygons and route relations for Manhattan from
Overpass, keeps the ways Wandrer's foot map counts, splits them into
intersection-to-intersection segments, and writes data/manhattan/geo.json.
Raw downloads are cached in ~/.cache/endur-manhattan so filters can be
iterated offline.
"""
import collections
import hashlib
import json
import math
import os
import re
import sys
import time

import requests
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

sys.path.insert(0, os.path.dirname(__file__))
from geo import Projection, ROT_DEG  # noqa: E402

AREA_ID = 3600000000 + 8398124  # OSM relation "Manhattan" (New York County)
OVERPASS = "https://overpass-api.de/api/interpreter"
CACHE = os.path.expanduser("~/.cache/endur-manhattan")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "manhattan", "geo.json")

# --- Wandrer foot-map rules (wandrer.earth/filters, as quoted on talk-gb 2023-09) ---
FOOT_OK = {"yes", "designated", "allowed", "permissive", "official"}
FOOT_BAD = {"use_sidepath", "private", "destination", "no"}
ACCESS_BAD = {"private", "customers", "military", "no"}
HW_REMOVED = {
    "motorway", "motorway_link", "steps", "escalator", "elevator", "construction",
    "proposed", "demolished", "escape", "bus_guideway", "sidewalk", "crossing",
    "bus_stop", "traffic_signals", "stop", "give_way", "milestone", "platform",
    "speed_camera", "raceway", "rest_area", "traffic_island", "services", "yes",
    "no", "drain", "street_lamp", "razed", "corridor", "busway", "abandoned",
    "disused",
}
HW_NEEDS_FOOT = {"trunk", "trunk_link", "service", "bridleway"}
# railway=* values that mean actual track or platforms. The High Line's footways carry
# railway=adjacent / abandoned, a note about the old line, and must stay.
RAIL = {"rail", "subway", "light_rail", "tram", "monorail", "narrow_gauge", "funicular",
        "miniature", "preserved", "platform", "station"}
PARKLIKE = {
    "leisure": ["park", "garden", "nature_reserve"],
    "natural": ["wetland", "wood", "scrub", "heath", "grassland", "fell", "tundra"],
    "landuse": ["farmland", "cemetery", "forest", "meadow"],
    "amenity": ["grave_yard"],
}
PATH_HW = {"footway", "path", "pedestrian", "cycleway", "track", "bridleway", "steps"}
MAJOR_HW = {"primary", "primary_link", "secondary", "secondary_link", "trunk", "trunk_link"}


def overpass(name, query):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"  overpass: {name} ...", flush=True)
    for attempt in range(6):
        r = requests.post(OVERPASS, data={"data": query}, timeout=600,
                          headers={"User-Agent": "endur-manhattan (harveybarnhard.com)"})
        if r.status_code not in (429, 504):
            break
        time.sleep(30 * (attempt + 1))
    r.raise_for_status()
    data = r.json()
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def fetch_json(name, url):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".json")
    if not os.path.exists(path):
        print(f"  fetch: {name} ...", flush=True)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, "w") as f:
            f.write(r.text)
    with open(path) as f:
        return json.load(f)


def boundary_polygon():
    """Manhattan borough (OSM relation 8398124), via Nominatim; cached."""
    path = os.path.join(CACHE, "boundary.geojson")
    if not os.path.exists(path):
        import osmnx as ox
        ox.settings.cache_folder = os.path.join(CACHE, "osmnx")
        gdf = ox.geocode_to_gdf("R8398124", by_osmid=True)
        with open(path, "w") as f:
            f.write(gdf.to_json())
    with open(path) as f:
        return shape(json.load(f)["features"][0]["geometry"])


def counted(tags, in_route, in_park):
    """Wandrer's foot-map decision for one way. Returns (keep, reason)."""
    hw = tags.get("highway")
    if tags.get("area") == "yes":
        return False, "area"
    if (tags.get("motorroad") == "yes" or tags.get("indoor") == "yes"
            or tags.get("tunnel") in ("yes", "building_passage")
            or tags.get("golf_cart") in ("yes", "designated", "private")
            or tags.get("railway") in RAIL or "waterway" in tags or tags.get("route") == "ferry"):
        return False, "misc"
    foot = tags.get("foot")
    # Deviation from Wandrer: NYC mappers put use_sidepath on ordinary streets
    # whose sidewalks are mapped separately (W 55th, W 57th, W 34th...). Those
    # are runnable streets; dropping them leaves holes in the grid. Only keep
    # the rule for trunk/motorway-class roads.
    sidepath = foot == "use_sidepath" or tags.get("bicycle") == "use_sidepath"
    if sidepath and (hw or "").startswith(("trunk", "motorway")):
        return False, "use_sidepath (trunk)"
    if foot in FOOT_BAD - {"use_sidepath"}:
        return False, f"foot={foot}"
    if hw in HW_REMOVED:
        return False, f"highway={hw}"
    if foot in FOOT_OK or "foot:conditional" in tags:
        return True, "foot ok"
    if in_route:
        return True, "route"
    if tags.get("access") in ACCESS_BAD:
        return False, f"access={tags.get('access')}"
    if hw in HW_NEEDS_FOOT:
        if hw == "service" and "name" in tags and "service" not in tags:
            return True, "named service"
        return False, f"highway={hw}"
    if hw == "footway":
        if tags.get("footway") in ("sidewalk", "crossing", "traffic_island"):
            return False, f"footway={tags.get('footway')}"
        return (True, "park footway") if in_park else (False, "footway outside park")
    return True, "default"


ABBR = [
    (r"\bWest\b", "W"), (r"\bEast\b", "E"), (r"\bNorth\b", "N"), (r"\bSouth\b", "S"),
    (r"\bStreet\b", "St"), (r"\bAvenue\b", "Ave"), (r"\bBoulevard\b", "Blvd"),
    (r"\bPlace\b", "Pl"), (r"\bDrive\b", "Dr"), (r"\bRoad\b", "Rd"), (r"\bLane\b", "Ln"),
    (r"\bSquare\b", "Sq"), (r"\bTerrace\b", "Ter"), (r"\bParkway\b", "Pkwy"),
    (r"\bPlaza\b", "Plz"), (r"\bSaint\b", "St"), (r"\bJunior\b", "Jr"),
]
NUMBERED = re.compile(r"^(?:East |West )?(\d+)(?:st|nd|rd|th) Street$")


def short_name(name):
    for pat, rep in ABBR:
        name = re.sub(pat, rep, name)
    return name


def main():
    proj = Projection()
    print("Downloading OSM data")
    hw = overpass("highways", f"""[out:json][timeout:600];
area(id:{AREA_ID})->.a;
way["highway"](area.a);
out body; >; out skel qt;""")
    routes = overpass("routes", f"""[out:json][timeout:300];
area(id:{AREA_ID})->.a;
(rel["route"~"^(foot|hiking|running|walking)$"](area.a);
 rel["network"~"^(lwn|rwn|nwn|iwn)$"](area.a););
out body;""")
    parkq = "".join(
        f'way["{k}"~"^({"|".join(v)})$"](area.a);rel["{k}"~"^({"|".join(v)})$"](area.a);'
        for k, v in PARKLIKE.items())
    parks = overpass("parks", f"""[out:json][timeout:300];
area(id:{AREA_ID})->.a;
({parkq});
out geom;""")
    ntas = fetch_json("ntas", "https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson?boroname=Manhattan&$limit=100")
    stations = fetch_json("stations", "https://data.ny.gov/resource/39hk-dx4f.json?borough=M&$limit=1000")

    # --- geometry helpers ---
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in hw["elements"] if e["type"] == "node"}
    ways = [e for e in hw["elements"] if e["type"] == "way"]
    route_ways = {m["ref"] for r in routes["elements"] for m in r.get("members", []) if m["type"] == "way"}

    park_polys = []
    for e in parks["elements"]:
        try:
            if e["type"] == "way" and len(e.get("geometry", [])) >= 4:
                ring = [(p["lon"], p["lat"]) for p in e["geometry"]]
                if ring[0] == ring[-1]:
                    park_polys.append(shape({"type": "Polygon", "coordinates": [ring]}).buffer(0))
            elif e["type"] == "relation":
                lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                         for m in e.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
                from shapely.ops import polygonize
                park_polys.extend(p.buffer(0) for p in polygonize(unary_union(lines)))
        except Exception:
            pass
    park_tree = STRtree(park_polys)
    print(f"  {len(ways)} highway ways, {len(park_polys)} park polygons, {len(route_ways)} route ways")

    manhattan = boundary_polygon()
    print(f"  boundary: {manhattan.geom_type}, {len(getattr(manhattan, 'geoms', [manhattan]))} parts")

    # --- filter ---
    reasons = collections.Counter()
    kept, outside = [], []
    for w in ways:
        t = w.get("tags", {})
        in_park = False
        if t.get("highway") == "footway":
            line = LineString([nodes[n] for n in w["nodes"] if n in nodes])
            in_park = any(park_polys[i].intersects(line) for i in park_tree.query(line))
        keep, why = counted(t, w["id"] in route_ways, in_park)
        if not keep and why == "footway outside park" and t.get("name"):
            outside.append(w)
            continue
        reasons[("keep " if keep else "drop ") + why] += 1
        if keep:
            kept.append(w)
    # Deviation from Wandrer: named footways outside parks (the Brooklyn Bridge Promenade's
    # ramp, esplanades, pedestrian bridges) are real routes, so they count too, unless they
    # share a counted street's name, which marks a sidewalk mapped without footway=sidewalk.
    street_names = {short_name(w["tags"]["name"]) for w in kept
                    if w["tags"].get("highway") not in PATH_HW and w["tags"].get("name")}
    for w in outside:
        keep = short_name(w["tags"]["name"]) not in street_names
        reasons["keep named footway" if keep else "drop footway named like a street"] += 1
        if keep:
            kept.append(w)
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:30]:
        print(f"    {v:6d}  {k}")

    # --- split into segments at shared nodes ---
    use = collections.Counter()
    for w in kept:
        ns = w["nodes"]
        for n in ns:
            use[n] += 1
        use[ns[0]] += 1
        use[ns[-1]] += 1
    segs = []  # (way, [node ids])
    for w in kept:
        ns = [n for n in w["nodes"] if n in nodes]
        start = 0
        for i in range(1, len(ns)):
            if use[ns[i]] >= 2 or i == len(ns) - 1:
                if i > start:
                    segs.append((w, ns[start:i + 1]))
                start = i

    # --- clip to boundary, project ---
    out = []
    for w, ns in segs:
        line = LineString([nodes[n] for n in ns])
        if line.length == 0:
            continue
        a_node, b_node = ns[0], ns[-1]
        if not manhattan.contains(line):
            clipped = manhattan.intersection(line)
            parts = [g for g in getattr(clipped, "geoms", [clipped]) if g.geom_type == "LineString" and not g.is_empty]
            if not parts:
                continue
            line = max(parts, key=lambda g: g.length)
            if Point(nodes[a_node]).distance(Point(line.coords[0])) > 1e-7:
                a_node = None
            if Point(nodes[b_node]).distance(Point(line.coords[-1])) > 1e-7:
                b_node = None
        xy = [proj.fwd(lon, lat) for lon, lat in line.coords]
        length = sum(math.dist(xy[i], xy[i + 1]) for i in range(len(xy) - 1))
        if length < 1:
            continue
        out.append({"way": w["id"], "tags": w.get("tags", {}), "a": a_node, "b": b_node, "xy": xy, "len": length})
    print(f"  {len(out)} segments, {sum(s['len'] for s in out) / 1609.344:.1f} mi")

    by_class = collections.Counter()
    for s in out:
        by_class[s["tags"].get("highway")] += s["len"] / 1609.344
    for k, v in by_class.most_common():
        print(f"    {v:7.1f} mi  highway={k}")

    # --- attributes ---
    names, name_idx = [], {}

    def nidx(name):
        if not name:
            return -1
        s = short_name(name)
        if s not in name_idx:
            name_idx[s] = len(names)
            names.append(s)
        return name_idx[s]

    node_names = collections.defaultdict(set)
    node_deg = collections.Counter()
    for s in out:
        nm = s["tags"].get("name")
        for n in (s["a"], s["b"]):
            if n is not None:
                node_deg[n] += 1
                if nm:
                    node_names[n].add(nm)

    nta_feats = ntas["features"]
    nta_polys = [shape(f["geometry"]) for f in nta_feats]
    nta_tree = STRtree(nta_polys)

    def nta_of(lonlat):
        p = Point(lonlat)
        for i in nta_tree.query(p):
            if nta_polys[i].contains(p):
                return int(i)
        return int(nta_tree.nearest(p))

    cols = collections.defaultdict(list)
    ways_out = []  # OSM way id per segment; debugging only, not published
    for s in out:
        t = s["tags"]
        hwv = t.get("highway")
        nm = t.get("name")
        mid = LineString(s["xy"]).interpolate(0.5, normalized=True)
        m = NUMBERED.match(nm or "")
        cols["n"].append(nidx(nm))
        cols["cls"].append(2 if hwv in PATH_HW else 0 if hwv in MAJOR_HW else 1)
        cols["len"].append(round(s["len"]))
        cols["nta"].append(nta_of(proj.inv(mid.x, mid.y)))
        cols["num"].append(int(m.group(1)) if m else 0)
        dead = hwv not in PATH_HW and any(n is not None and node_deg[n] == 1 for n in (s["a"], s["b"]))
        cols["dead"].append(1 if dead else 0)
        for key, n in (("xa", s["a"]), ("xb", s["b"])):
            others = sorted(x for x in node_names.get(n, ()) if x != nm) if n is not None else []
            cols[key].append(nidx(others[0]) if others else -1)
        ways_out.append(s["way"])
        pts = [(round(x), round(y)) for x, y in LineString(s["xy"]).simplify(1.0).coords]
        dedup = [pts[0]] + [p for i, p in enumerate(pts[1:], 1) if p != pts[i - 1]]
        if len(dedup) == 1:
            dedup.append(dedup[0])
        flat = [dedup[0][0], dedup[0][1]]
        for i in range(1, len(dedup)):
            flat += [dedup[i][0] - dedup[i - 1][0], dedup[i][1] - dedup[i - 1][1]]
        cols["c"].append(flat)

    # NTA rings (simplified, projected) for small multiples
    nta_out = []
    for f, poly in zip(nta_feats, nta_polys):
        rings = []
        for part in getattr(poly, "geoms", [poly]):
            pp = LineString([proj.fwd(*c) for c in part.exterior.coords]).simplify(8)
            rings.append([[round(x), round(y)] for x, y in pp.coords])
        nta_out.append({"code": f["properties"]["nta2020"], "name": f["properties"]["ntaname"],
                        "park": f["properties"].get("ntatype") != "0", "rings": rings})

    # Subway stations, merged by complex
    cx = collections.defaultdict(lambda: {"names": [], "routes": set(), "pts": []})
    for st in stations:
        c = cx[st["complex_id"]]
        c["names"].append(st["stop_name"])
        c["routes"].update(st.get("daytime_routes", "").split())
        c["pts"].append(proj.fwd(float(st["gtfs_longitude"]), float(st["gtfs_latitude"])))
    st_out = []
    for c in cx.values():
        x = sum(p[0] for p in c["pts"]) / len(c["pts"])
        y = sum(p[1] for p in c["pts"]) / len(c["pts"])
        name = collections.Counter(c["names"]).most_common(1)[0][0]
        st_out.append({"name": name, "routes": " ".join(sorted(c["routes"])), "x": round(x), "y": round(y)})
    st_out.sort(key=lambda s: s["y"])

    xs = [x for s in out for x, _ in s["xy"]]
    ys = [y for s in out for _, y in s["xy"]]
    geo = {
        "proj": proj.params(),
        "bbox": [math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys))],
        "total_m": round(sum(s["len"] for s in out)),
        "names": names,
        "ntas": nta_out,
        "stations": st_out,
        "segs": dict(cols),
    }
    body = json.dumps(geo, separators=(",", ":"))
    geo = {"v": hashlib.sha1(body.encode()).hexdigest()[:10], **geo}
    with open(os.path.join(CACHE, "seg_ways.json"), "w") as f:
        json.dump(ways_out, f)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(geo, f, separators=(",", ":"))
    print(f"Wrote {os.path.relpath(OUT)}: {os.path.getsize(OUT) / 1e3:.0f} KB, v={geo['v']}, "
          f"{len(names)} names, {len(st_out)} stations, rotation {ROT_DEG} deg")


if __name__ == "__main__":
    main()
