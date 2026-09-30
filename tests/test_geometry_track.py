import math

import numpy as np
import pytest

from autocross import geometry as g
from autocross import trackgen
from autocross.track import build, load


def test_projection_sign_and_station():
    line = g.Polyline(np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]))
    s, lat, _ = line.project(np.array([5.0, 2.0]))
    assert s == pytest.approx(5.0) and lat == pytest.approx(2.0)
    s, lat, _ = line.project(np.array([15.0, -1.0]), s_hint=14.0, window=5.0)
    assert s == pytest.approx(15.0) and lat == pytest.approx(-1.0)


def test_curvature_of_circle():
    th = np.linspace(0, 2 * math.pi, 400, endpoint=False)
    pts = 20.0 * np.column_stack((np.cos(th), np.sin(th)))
    assert np.allclose(g.curvature(pts, closed=True), 1 / 20.0, rtol=1e-3)


def test_frames_roundtrip():
    p = np.array([[3.0, -2.0], [10.0, 5.0]])
    b = g.world_to_body(p, 1.0, 2.0, 0.7)
    assert np.allclose(g.body_to_world(b, 1.0, 2.0, 0.7), p)


@pytest.mark.parametrize("name", ["practice_basic", "practice_loop", "practice_obstacles"])
def test_shipped_tracks_valid(root, name):
    t = load(root / "tracks" / f"{name}.yaml")
    assert t.validate() == []
    info = t.info(True, 20.0)
    assert len(info.centerline) == len(info.width_left) and len(info.gates) > 5
    blind = t.info(False, 20.0)
    assert len(blind.centerline) == 0 and len(blind.cones) == 0


def test_arc_segment_geometry():
    t = build({"segments": [{"type": "straight", "length": 50}, {"type": "arc", "radius": 20, "angle": 90},
                            {"type": "straight", "length": 50}]})
    assert t.center[-1] == pytest.approx([70.0, 70.0], abs=0.2)  # (50 + R, R + 50)


def test_generators_deterministic():
    a, b = trackgen.generate_open(5), trackgen.generate_open(5)
    assert a == b and build(a).validate() == []
    assert build(trackgen.generate_closed(2)).validate() == []


def test_overlapping_course_rejected():
    # 270 deg loop whose exit straight runs back across the entry straight
    t = build({"segments": [{"type": "straight", "length": 40}, {"type": "arc", "radius": 15, "angle": 270},
                            {"type": "straight", "length": 60}]})
    assert any("overlaps" in p for p in t.validate())
