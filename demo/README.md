# TIDE demos

Two self-contained web pages (React + Three.js from CDN, no build step). Open either file in a
browser, or serve the folder: `python3 -m http.server 8000` then http://localhost:8000/demo/

- **tide_arena.html** - the autocross planner live in 3D. Pick a course (HackBama test, technical,
  speedway, random), click the road to drop cone groups, inject steering bias or low grip, and watch
  the planned line re-plan every 50 ms.
- **tide_walk.html** - the same planner core as an assistive path planner for a white-cane user.
  A pedestrian walks a kerbed sidewalk between buildings; drop a cone, person, e-scooter or box in
  the way (or click the pavement) and the planner routes round it. The bottom card shows what the
  cane handle would tell the user: direction by haptic pattern, urgency by colour; it says Stop when
  the way is blocked.

Both are browser ports of `controllers/tide/controller.py` (corridor from perceived edges,
minimum-curvature path, curvature-limited speed profile, obstacle side selection, look-ahead
tracking) on simplified physics. The scored controller is the Python one.
