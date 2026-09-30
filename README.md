# mil-sim: EcoCAR Autocross Simulation Harness

A lightweight, CARLA-based simulation framework for testing autonomous **path planning and control** on autocross courses. It is built for the EcoCAR autocross hackathon and as a foundation for model-in-the-loop (MIL) and software-in-the-loop (SIL) testing with ideal ground-truth data.

It mirrors the EIC autonomous autocross event: **the controller never gets a map.** Each control step it receives a processed perception output (a *road model*) describing only the course visible ahead. From that it must plan a path and output steering, throttle and brake, all within an embedded-style compute budget. Teams do not write perception or sensor fusion.

The harness also provides:

- **Two backends:** CARLA 0.9.16 (full physics) and a calibrated dynamic bicycle model for fast iteration without CARLA. The two agree within about 2.5% on lap time.
- **Flat autocross pad:** generated on the fly from a small OpenDRIVE road, with knockable physics cones. No Unreal map editing is needed.
- **Sandboxed controllers:** each controller runs in its own process with 1 CPU core, no GPU, a memory ceiling and a 50 ms deadline per step.
- **Scoring:** rules, penalties, metrics, rubric scoring, leaderboard, robustness sweeps and real-time pacing.
- **Full ground-truth telemetry** logged at 40 Hz for every run.

Designed to run on a laptop (tested with an RTX 30-series GPU). A 40 s run takes about 3 s headless in CARLA, and about 1 s on the bicycle model.

---

## Quick start

### Requirements

- Windows 10/11 or Linux, Python **3.10–3.12** (the CARLA client wheels support these versions)
- [CARLA 0.9.16](https://github.com/carla-simulator/carla/releases/tag/0.9.16), only needed for the CARLA backend

### Setup (one time)

```powershell
git clone https://ecocar-gitlab.ua.edu/eic/cav/mil-sim.git
cd mil-sim
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1      # Linux: ./scripts/setup.sh
```

The setup script creates `.venv`, installs the harness and the `carla==0.9.16` client, then runs a self-check. If CARLA is not in `%USERPROFILE%\CARLA_0.9.16` or `C:\CARLA_0.9.16`, pass `-CarlaRoot <path>` or set the `CARLA_ROOT` environment variable.

Activate the environment with `.venv\Scripts\Activate.ps1` (Linux: `source .venv/bin/activate`). If PowerShell blocks the activation script, call `.venv\Scripts\python.exe` directly instead of `python`.

### See it run

```powershell
# No CARLA needed: bicycle model with a live top-down view
python -m autocross run -c controllers\baseline_pure_pursuit.py -t tracks\practice_basic.yaml --backend bicycle --view

# Full CARLA
scripts\start_carla.ps1                  # windowed 1280x720, Low quality; add -Headless for no window
python -m autocross check                # confirms the server is reachable
python -m autocross run -c controllers\baseline_pure_pursuit.py -t tracks\practice_obstacles.yaml --realtime
```

In CARLA the spectator follows the car, and the controller's planned path is drawn in cyan. Every run prints a score summary and writes a folder under `runs/`. Open `report.png` there first.

---

## Writing a controller

Copy `controllers/template_controller.py`. A controller is one Python file with one class that derives from `autocross.api.Controller`:

```python
from autocross.api import Command, Controller

class MyController(Controller):
    def on_start(self, track, vehicle):
        super().on_start(track, vehicle)          # vehicle: limits (mass, wheelbase, max_steer, mu, ...)

    def step(self, obs):                          # called every 50 ms
        road = obs.road                           # what the perception stack sees ahead
        ...
        return Command(steer=s, throttle=t, brake=b,          # steer in [-1, 1], + = left
                       path=my_path, path_frame="vehicle")   # the path you intend to drive
```

### What the controller receives

| | |
|---|---|
| `on_start(track, vehicle)` | **No map** in event mode (`track` arrays are empty). `vehicle` holds the published limits: 1,845 kg, 3.0 m wheelbase, 35° max steer, steering-rate limit, μ ≈ 0.95, 5.2 m/s² acceleration, 9 m/s² braking, and the control period. |
| `obs.road` (`RoadModel`) | The course up to `road_model.range_m` (40 m) ahead, in the vehicle frame (x forward, y left): `lateral_offset`, `heading_error`, `curvature`, `dist_to_left/right_boundary`, `preview_s` / `preview_curvature` / `preview_centerline` / `preview_left/right_boundary`, `corners` (direction, entry/apex/exit distance, min radius, turn angle), `obstacles` (distance, lateral extent), `finish_distance`. |
| Vehicle state | Ground truth: `obs.x, y, yaw, vx, vy, yaw_rate, ax, ay, speed`, and the actuator values actually applied. |
| `obs.cones` (optional) | Raw cone detections (vehicle frame, colour class) within 30 m and a 140° field of view. |

Full definitions are in [`autocross/api.py`](autocross/api.py). The baseline ([`controllers/baseline_pure_pursuit.py`](controllers/baseline_pure_pursuit.py)) shows a complete road-model controller. It uses pure pursuit on the perceived centerline, speed from the curvature preview, and a throttle limit based on how much grip is left for cornering. It ignores obstacles. It finishes cleanly but slowly, and is there to be beaten.

### Rules the harness enforces

- **Compute:** `step()` must return within 50 ms, or the previous command is held and an overrun is logged. The controller runs on 1 core, with no GPU and a 1 GB memory ceiling. `on_start()` may take up to 60 s.
- **Actuators:** steering is slew-rate limited, and optional actuation delay can be added.
- **Invalid output:** an exception, NaN or wrong return type counts as an error. 20 errors in a row is a DNF (did not finish).
- **Dependencies:** controllers run in the harness's Python environment, so use only packages installed there (numpy is always available).

---

## Scoring

Rules are evaluated from ground truth at every physics tick, so results are the same on both backends.

| Penalty (per occurrence) | Points |
|---|---|
| Cone hit (each cone counted once) | −2 |
| Off course (all four wheels outside the cones, once per excursion) | −5 |
| Missed gate (passing a gate station outside the corridor) | −10 |
| Controller error (up to 10) | −1 |

A run is a **DNF** if the car stays off course for more than 5 s, gets more than 15 m from the course, drives the wrong way, is stuck for 8 s, rolls over, or hits the 240 s timeout.

| Category | Weight | Measured as |
|---|---|---|
| Completion time | 30 | vs. a reference time (the course's `reference_time`, or an automatic friction-limited centerline estimate) |
| Path deviation | 15 | mean/peak distance from the path the controller published (centerline if none) |
| Compute | 15 | p99 step time vs. deadline, overruns, peak memory |
| Physical realism | 15 | grip over-demand, rear-axle sliding, steering-rate saturation, pedal overlap |
| Innovation | 25 | judged; enter with `score --judged innovation=<0-100>` |

Weights and thresholds are in [`config/scoring.yaml`](config/scoring.yaml).

---

## Commands

All commands are `python -m autocross <command> --help`.

| Command | Purpose |
|---|---|
| `run -c CTRL -t TRACK` | One run. Useful flags: `--backend bicycle`, `--view`, `--realtime [FACTOR]`, `--no-render`, `--team NAME`, `--controller-config cfg.yaml`, `--perturb KEY=VALUE`, `--set section.key=value` |
| `batch -c CTRL` | Robustness sweep over courses × perturbations × seeds ([`config/robustness.yaml`](config/robustness.yaml)): perception and road-model noise, shorter sight distance, low grip, extra mass, actuator latency, steering bias, start offsets |
| `score RUN_DIR` | Re-score a run, optionally adding `--judged innovation=80` |
| `leaderboard [runs] --award` | Best run per team and course, plus the award categories (fastest, most realistic, most efficient, most precise) |
| `gen-track --seed N --out F.yaml [--closed] [--obstacles K] --preview` | Generate a random valid course |
| `show-track F.yaml` | Validate a course, print stats and reference time, and save a preview PNG |
| `plot RUN_DIR` | Regenerate `report.png` |
| `calibrate --backend carla` | Measure the vehicle's acceleration, braking and lateral grip limits |
| `check` | Verify the install and CARLA connectivity |

### Run outputs (`runs/<timestamp>_<team>_<course>/`)

- `telemetry.csv`: ground truth at physics rate (pose, body velocities and accelerations, measured steering angle, progress, lateral offset, boundary margin, path error, commanded vs. applied actuation, step timing, and `dbg_*` values from `Command.debug`)
- `events.json`: start, finish, cone hits, off-course excursions, missed gates and DNF
- `summary.json`: metrics, score, vehicle parameters, config snapshot, controller and course hashes, and machine info
- `report.png`: course map with the speed-coloured trajectory and events, plus speed, error, actuation and compute plots

---

## Courses

Courses are YAML files built from segments (`straight`, `arc`, `slalom`) or from explicit centerline points. Cones, gates and start/finish lines are generated automatically. `obstacles` places cone groups that are **not** on the map and appear only in perception.

```yaml
name: practice_basic
closed: false            # point-to-point; closed: true + laps: N for loop courses
width: 5.5
segments:
  - {type: straight, length: 35}
  - {type: arc, radius: 15, angle: -110}      # + = left
  - {type: slalom, cycles: 3, wavelength: 26, swing: 2.5}
obstacles:
  - {s: 120, offset: 1.0, width: 2.0}          # station along the course, lateral offset, width
```

Practice courses are in `tracks/`. Keep event courses in `tracks/private/`, which is git-ignored.

---

## Configuration

[`config/sim.yaml`](config/sim.yaml) holds the official event settings. Any value can be overridden per run with `--set`.

| Section | Contents |
|---|---|
| `sim` | physics step (25 ms), control rate (20 Hz), timeout, seed, `realtime_factor` |
| `carla` | host/port, rendering, spectator, cone models, weather, recorder |
| `vehicle` | blueprint, steering limit, CARLA tire friction / brake torque / torque scale, published limits, bicycle-model parameters |
| `road_model` | preview range and step, corner detection, lane-model noise (0 = ideal) |
| `perception` | raw cone sensor range, field of view, noise, dropout, misclassification |
| `track_knowledge` | `perception` (event: no map) or `full` (map given in `on_start`, for debugging) |
| `compute_budget` | enforce/measure mode, cores, deadline, memory, GPU access |
| `rules` | off-course rule, DNF thresholds, gate spacing, cone-hit margin |

**Vehicle calibration.** CARLA's stock Model 3 is unrealistic for this use:

| | Stock CARLA | Configured |
|---|---|---|
| Lateral grip | ≈ 2.4 g | tire friction reduced to give ≈ 0.95 g |
| Braking | ≈ 4 m/s² | brake torque raised to give ≈ 9 m/s² |
| Throttle mid-corner | spins the tires (CARLA has no traction control) | torque scaled to 75% |
| Max road-wheel steer | 70° | 35° |

Re-run `python -m autocross calibrate --backend carla` after changing the vehicle, and update the published limits.

---

## Toward MIL / SIL

The `Controller` interface is where a real controller plugs in. The simulator, rules, logging and scoring stay the same:

- **MIL:** wrap a Simulink model exported as an FMU (e.g. with `fmpy`) or called through the MATLAB Engine in a `Controller` subclass. Set `compute_budget.mode: measure`, since the loop is lockstep and slow models just slow the run.
- **SIL:** compile Embedded Coder output to a shared library and call its step function via `ctypes`. Keep `mode: enforce` with ECU-like `cpu_cores` / `step_deadline_ms`. This approximates ECU timing only; true timing needs PIL/HIL.
- **Ground truth:** every run's telemetry is suitable as a reference for comparing against vehicle logs. `carla.record: true` also saves a CARLA replay log.
- **Not yet implemented:** camera/lidar sensor models (perception here is derived from ground truth) and an asynchronous real-time loop for HIL.

---

## Repository layout

```
autocross/            harness package
  api.py              team-facing types: Controller, Observation, RoadModel, Command, ...
  runner.py           closed loop, actuator model, logging, batch sweeps
  sandbox.py          controller process + compute budget
  perception.py       road model and cone sensor (ground truth + configurable noise)
  monitor.py          race rules and events
  metrics.py, score.py, plot.py, viewer.py
  track.py, trackgen.py, geometry.py
  backends/           carla_backend.py, bicycle.py
  cli.py, calibrate.py, config.py
config/               sim.yaml, scoring.yaml, robustness.yaml
controllers/          baseline_pure_pursuit.py, template_controller.py
tracks/               practice courses (private/ is git-ignored)
scripts/              setup and CARLA launch scripts (PowerShell + bash)
tests/                pytest suite (bicycle backend, no CARLA needed)
```

## Development

```powershell
python -m pytest            # 29 tests, ~15 s, no CARLA required
```

The official event scoring should be run on a single reference machine, because the 50 ms deadline depends on the machine.
