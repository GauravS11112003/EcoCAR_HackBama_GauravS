# TIDE controller — demo notes

## Results (bicycle backend, nominal settings)

| Course | Baseline | TIDE |
|---|---|---|
| hackathon_test | 56.0 s / 61.9 | 44.4 s / 73.7 |
| practice_basic | 35.9 s / 66.2 | 27.5 s / 74.7 |
| practice_loop (2 laps) | 92.9 s / 66.5 | 79.6 s / 73.1 |
| practice_obstacles | 61.2 s / 59.5 (3 cones) | 47.2 s / 73.7 (0 cones) |
| practice_technical | 39.5 s / 57.5 (4 cones) | 32.2 s / 72.8 (0 cones) |
| **Mean technical / 75** | **62.3** | **73.7** |

- ~18% faster everywhere, 9 cone hits -> 0 across 8 courses (5 public + 3 random with obstacles)
- 0.6 ms per step of the 50 ms budget (1.2%), 36 MB of 1 GB
- Robustness sweep (harness `batch`): unaffected by 1 m / 6 deg start offset, 25 m sight instead of 40,
  15% extra mass, noisy / misclassified cones, 100 ms actuator lag, 3% steering bias.
  Degrades gracefully with 10% less grip (70.8 on technical, 0 cones).

## Pipeline (every 50 ms, from `obs.road` only - no map)

1. **Corridor** - cone lines minus a car-width margin (grows slightly with distance: trust the far
   preview less); each obstacle group blocked out on the side with more room (side remembered).
2. **Racing line** - minimum-curvature lateral offsets inside the corridor. Exact QP: banded
   `np.linalg.solve` + active-set loop, 41 unknowns, ~0.2 ms. Anchored at the car; warm-started.
3. **Speed profile** - `v = sqrt(a_lat / kappa)` along the line; forward pass (power-limited accel x
   friction circle); backward pass (trail-braking at 70% x friction circle); unseen road past 40 m
   treated as a 9 m hairpin, planned at full straight-line braking.
4. **Longitudinal** - acceleration that meets the plan 0.25 s ahead; pedals capped by the friction
   circle against the *planned* lateral load; drive/brake force model; never throttle+brake together.
5. **Lateral** - pure pursuit + Stanley on the racing line, slew-limited; steering-bias observer
   (applied steer vs command); grip observer (rear-slip / sideslip -> lower the grip budget).

## Steering: Stanley / pure pursuit vs MPC (talking point)

- **Stanley** (Stanford, DARPA 2005): `steer = heading_error + atan(k * lateral_error / speed)`.
  Front-axle error, one formula, no car model.
- **Pure pursuit**: steer the arc through a point on the path a speed-scaled distance ahead.
- Both: *one error now -> one steering angle now*. Reactive; no knowledge of actuator lag.
- **MPC**: predict the car ~0.75 s ahead with an explicit model (wheelbase, 60 ms steering lag, slew
  limit), choose the steering *sequence* that best follows the line, apply only the first command,
  repeat at 20 Hz. Predictive: starts turning before the error exists, leads the actuator lag.
- Analogy: Stanley drives by watching the lane line at the front wheel; MPC drives by looking through
  the windscreen and planning the hands' motion for the next second.
- Our LTV-MPC: lateral + heading error + wheel angle as state, linearised kinematic bicycle along the
  planned path at the planned speed, 15-step horizon, quadratic cost on errors / steering / steering
  rate -> one 15x15 linear solve per step (< 1 ms, numpy only). Fits the embedded budget.

## MPC vs reinforcement learning (likely judge question)

Not RL. MPC optimises a *known model* online; nothing is learned, no training data, deterministic,
verifiable - the same every run. RL learns a policy from trial and error offline and is a black box
at runtime. Both optimise a cost over time, which is the only overlap ("model-based RL" is the
bridge). We chose MPC because: 50 ms hard deadline, no GPU, no training data for a course nobody has
seen, and the judges can read every line of what the car will do - which is what a production ADAS
stack needs.

## Why this matters for the EcoCAR Innovation Challenge

Same architecture a real Blazer EV would run: perception -> road model -> planner -> MPC -> actuators.
The `Controller` interface is the MIL/SIL plug-in point; everything here is numpy and could be ported
to Simulink / Embedded Coder.

## "Would RL have been better?" (likely judge question)

For this problem, on this timeline: **MPC, clearly.**

| | MPC | RL |
|---|---|---|
| Accuracy on an *unseen* course | car model is given (`VehicleParams`): nothing to learn, same error seen or unseen | must discover the physics from laps; generalisation to new geometry is RL's weakest point; tends to overfit sim quirks that don't exist in CARLA |
| Compute per step | one 15x15 solve, ~0.3 ms | small net, ~0.1 ms (but hand-rolled in numpy: no torch/GPU allowed); training is 1000x the cost |
| Robustness | handles lag / bias / noise / grip loss by construction (verified with the harness sweep) | only if those cases were in the training distribution |
| Explainability | every line readable; deterministic | black box |
| Ceiling | limited by the linear model | theoretically higher (could learn tyre nonlinearity) - but weeks of reward shaping |

Where RL genuinely wins: when the model is *unknown or wrong* (real Blazer EV tyres in the wet).
Framing: **RL earns its place where physics is uncertain; MPC owns the layer where physics is known.**
Future-work slide: MPC with a data-learned residual model (the hybrid research teams actually build).
Context: UA EcoCAR's own Argonne AI Innovation Award project was DDPG (RL) for cooperative ACC -
the judges know RL; the contrast above is the honest reason we chose MPC for lateral control.

## Shipped configuration (verified on the development Mac, 73.7 / 75 mean, zero cones)

- **Tracker: speed-scheduled hybrid.** LTV-MPC contributes up to 30% of the steering command at or
  below 6 m/s, blending out by 10 m/s; pure pursuit + Stanley always keeps at least 70% and takes
  over fully above 10 m/s or whenever the car is far from the MPC's linearisation point (lateral
  error > 0.5 m, heading error > 10 deg, or sideslip > 6 deg). MPC is active on ~13-16% of control
  steps on the open courses (hairpins, slow slaloms). Why capped: MPC alone tracks tighter but is a
  linearised controller - under road-model noise an upset at low speed left it in control with an
  invalid model and it could not recover; the cap and gate keep nominal performance identical and
  robustness within ~2 points of pure pursuit (harness perturbation batch). `dbg_w_mpc` in telemetry.
- **Robustness lesson of the day:** a throttle-circle tweak worth +0.1 nominal cost 30 points under
  road-model noise. Caught only because every change was re-run through the 60-run perturbation
  batch. Reverted.
- **Lap memory as a horizon prior.** The controller stitches everything it sees into a world-frame
  map on every lap (verified exact against the true track offline). On later laps of a closed course
  the remembered road past the 40 m preview replaces the blind "assume a hairpin" speed limit
  (8.5 m/s) with the remembered corner speed, planned with margin. Active on 52% of lap 2 of
  practice_loop (max horizon speed 25.7 m/s); never changes a decision on an open course.
  `dbg_v_hor_used` in telemetry. Full preview extension (150 m) was prototyped and is selectable
  (`memory_mode: extend`) but destabilised the planner on one hard course, so it is not shipped.

## MPC alone vs pure pursuit (the data behind the hybrid; `lateral: mpc` / `lateral: pp`)

Head-to-head, 8 courses (5 public + 3 random), nominal settings, zero cones in all three:

| Tracker | Tech / 75 | Lap-time ratio | Deviation score |
|---|---|---|---|
| Pure pursuit + Stanley (shipped) | **73.5** | 0.975 | 96.3 |
| LTV-MPC | 71.6 | 1.053 | **98.1** |
| MPC on practice_technical alone | 72.8 (= pure pursuit) | | 98.8 |

- MPC tracks the planned line ~2 points better and runs in 0.66 ms (one 15x15 linear solve).
- It loses ~2 points of lap time because it drives the *planned* line exactly, and the planned line is
  conservative (car-width margin, min-curvature). Pure pursuit effectively "cuts" that line within
  the tolerance the scorer allows. With a precise tracker the speed cap also has to respect true
  curvature peaks (smoothing under-rates slaloms) - the MPC *exposed* that optimism.
- Closing the gap is a planner problem (less conservative line, tracked exactly), not a controller
  problem. That is the next step for the real car; we chose to ship the faster combination today.
- What MPC needs to beat pure pursuit here: the actuator slew limit inside the QP (constrained
  solve), and an unanchored reference line with a plan-consistency term. Both prototyped; both
  need more tuning time than a hackathon allows.
