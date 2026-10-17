"""TIDE controller: look-ahead planning on the perceived road model.

Pipeline, every 50 ms (vehicle frame, x forward / y left):
  1. Corridor:      usable lateral band around the centerline = cone line minus a car-width margin,
                    with each obstacle group blocked out on the side we decide to pass it.
  2. Racing line:   lateral offsets e[i] inside the band that minimise path curvature (quadratic
                    cost on second differences), warm-started from last step's solution so the
                    published plan stays consistent from step to step.
  3. Speed profile: friction-circle limited: v_cap = sqrt(a_lat_max / |kappa|) along the racing
                    line, forward pass (power + grip limited accel), backward pass (grip limited
                    braking), unknown road past the horizon treated as a possible hairpin.
  4. Longitudinal:  feedforward accel from the profile + PI on speed, mapped through the drive /
                    brake force model to pedals; never throttle and brake together.
  5. Lateral:       pure pursuit on the racing line with a speed-scaled lookahead plus curvature
                    feedforward, clamped to the grip circle and slew-rate limited.

Only numpy + autocross.api. Nothing is read from disk, no map is used.
"""
from __future__ import annotations

import math

import numpy as np

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams

G = 9.81


class TideController(Controller):
    # ------------------------------------------------------------------ setup
    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        c = self.config
        vp = vehicle
        # --- physical envelope we allow ourselves (fractions of the published limits)
        self.f_lat = float(c.get("lat_frac", 0.85))          # lateral grip used for cornering
        self.f_dec = float(c.get("decel_frac", 0.85))        # braking
        self.f_acc = float(c.get("accel_frac", 0.95))        # throttle
        self.a_lat = self.f_lat * vp.mu * G
        self.a_dec = self.f_dec * vp.max_decel
        self.a_acc = self.f_acc * vp.max_accel
        self.v_max = float(c.get("v_max", vp.max_speed))
        self.v_horizon = float(c.get("v_horizon", 8.5))      # assume a hairpin right past the horizon
        # --- corridor geometry
        half_w = vp.width / 2.0
        self.margin = float(c.get("boundary_margin", half_w + 0.35))   # car centre to cone line
        self.obs_clear = float(c.get("obstacle_clearance", half_w + 0.45))
        self.obs_len = float(c.get("obstacle_length", 4.0))  # how far before/after an obstacle we hold the side
        # --- racing line solver
        self.w_curv = float(c.get("w_curv", 1.0))
        self.w_smooth = float(c.get("w_smooth", 0.05))      # keeps the offset steady where curvature doesn't care
        self.w_center = float(c.get("w_center", 0.0003))    # tiny tie-break toward the centerline
        self.w_prev = float(c.get("w_prev", 0.0))           # plan consistency: cost of moving away from last step's line
        self.n_iter = int(c.get("line_iters", 12))
        # --- lateral control
        self.k_ld = float(c.get("lookahead_gain", 0.32))
        self.ld_min = float(c.get("lookahead_min", 3.5))
        self.k_ff = float(c.get("ff_gain", 0.5))
        self.steer_rate = vp.max_steer_rate * vp.control_dt * 0.95  # per-step slew we allow ourselves
        # --- longitudinal control
        self.look_t = float(c.get("look_t", 0.25))          # s, lookahead for the speed law
        self.plan_dec_frac = float(c.get("plan_decel_frac", 0.7))   # fraction of a_dec the plan assumes for visible corners
        self.hor_dec_frac = float(c.get("horizon_decel_frac", 1.0))  # ... and for the unseen road past the horizon (straight-line braking)
        self.ki_v = float(c.get("ki_v", 0.15))
        self.i_err = 0.0
        # --- drive/brake model (published numbers; the bicycle backend has similar values)
        self.mass = vp.mass
        self.f_drive = self.mass * vp.max_accel          # N at low speed
        self.p_max = float(c.get("max_power_w", 200000.0))
        self.f_brake = self.mass * vp.max_decel
        # --- state
        self.prev_e = None          # last lateral-offset solution (K,)
        self.prev_s0 = None         # obs.progress when prev_e was computed
        self.prev_steer = 0.0
        self.side_memory: dict[int, int] = {}   # obstacle station (rounded) -> +1 pass left / -1 pass right

    # ------------------------------------------------------------------ step
    def step(self, obs: Observation) -> Command:
        road, vp = obs.road, self.vehicle
        if road is None or not road.valid or len(road.preview_s) < 4:
            return Command(brake=1.0, steer=self.prev_steer)
        v = max(obs.speed, 0.0)
        s = road.preview_s
        cl = road.preview_centerline
        K = len(s)
        ds = float(np.mean(np.diff(s))) if K > 1 else 1.0

        # centerline heading / normal in vehicle frame
        tang = np.gradient(cl, axis=0)
        tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-6)[:, None]
        nrm = np.column_stack((-tang[:, 1], tang[:, 0]))   # left normal

        # ------------------------------------------------ 1. corridor band
        lo = -(road.preview_width_right - self.margin)     # most-right allowed offset (negative)
        hi = road.preview_width_left - self.margin         # most-left allowed offset
        narrow = hi < lo
        if narrow.any():                                   # corridor narrower than the car: hold centre
            mid = 0.5 * (lo[narrow] + hi[narrow])
            lo[narrow], hi[narrow] = mid, mid
        self._apply_obstacles(road, obs.progress, s, lo, hi)

        # hedge the unknown: near the horizon pull the band toward the centre
        tail = s > s[-1] - 6.0
        lo[tail] *= 0.5
        hi[tail] *= 0.5

        # ------------------------------------------------ 2. racing line (lateral offsets)
        e0 = float(road.lateral_offset)
        e = self._solve_line(s, cl, nrm, lo, hi, e0, obs.progress, ds)
        path = cl + nrm * e[:, None]

        # ------------------------------------------------ 3. speed profile along the path
        kappa = _curvature(path)
        kappa_s = np.convolve(np.pad(kappa, 2, mode="edge"), np.ones(5) / 5.0, mode="valid")
        v_prof = self._speed_profile(path, kappa_s, s, v, road)

        # ------------------------------------------------ 4. longitudinal command
        self.ay_meas = obs.ay
        self._path = path
        ps_tmp = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))))
        q = np.linspace(0.0, max(v * 0.4, 1.0), 5)
        k_q = np.abs(np.interp(q, ps_tmp, kappa_s))
        v_q = np.interp(q, ps_tmp, v_prof)
        self.a_lat_ahead = float(np.max(k_q * v_q * v_q))     # lateral load the plan will actually put on the tyres
        throttle, brake, v_tgt, a_ff = self._longitudinal(v, v_prof, s, vp)

        # ------------------------------------------------ 5. lateral command
        steer, ld = self._lateral(path, s, kappa, v, vp, obs)

        return Command(steer=steer, throttle=throttle, brake=brake, path=path, path_frame="vehicle",
                       target_speed=v_tgt,
                       debug={"v_tgt": v_tgt, "a_ff": a_ff, "ld": ld, "e0": e0,
                              "vpmin": float(np.min(v_prof[1:])), "ivpmin": float(np.argmin(v_prof[1:]) + 1),
                              "kmax": float(np.max(np.abs(kappa_s))), "ikmax": float(np.argmax(np.abs(kappa_s))),
                              "kc_max": float(np.max(np.abs(road.preview_curvature))),
                              "vp10": float(v_prof[min(10, K - 1)]), "vp20": float(v_prof[min(20, K - 1)]),
                              "kappa_path": float(kappa[min(3, K - 1)])})

    # ------------------------------------------------------------------ corridor
    def _apply_obstacles(self, road, progress, s, lo, hi):
        for ob in road.obstacles:
            d = float(ob.distance)
            key = int(round((progress + d) / 2.0))        # stable id for the same obstacle across steps
            room_left = (road.preview_width_left[min(max(int(d), 0), len(s) - 1)] - self.margin) - (ob.lateral_max + self.obs_clear)
            room_right = (ob.lateral_min - self.obs_clear) - (-(road.preview_width_right[min(max(int(d), 0), len(s) - 1)] - self.margin))
            side = self.side_memory.get(key)
            if side is None:
                # prefer the side with more room; tie-break toward where our line already is
                if room_left > room_right + 0.3:
                    side = +1
                elif room_right > room_left + 0.3:
                    side = -1
                else:
                    cur = 0.0 if self.prev_e is None else float(np.interp(d, s, self.prev_e))
                    side = +1 if cur >= 0.5 * (ob.lateral_min + ob.lateral_max) else -1
                self.side_memory[key] = side
            m = (s >= d - self.obs_len) & (s <= d + self.obs_len)
            if not m.any():
                continue
            if side > 0:
                lo[m] = np.maximum(lo[m], ob.lateral_max + self.obs_clear)
            else:
                hi[m] = np.minimum(hi[m], ob.lateral_min - self.obs_clear)
            bad = lo[m] > hi[m]
            if bad.any():                                   # no room on that side: squeeze to whatever is left
                mid = 0.5 * (lo[m][bad] + hi[m][bad])
                tmp_lo, tmp_hi = lo[m], hi[m]
                tmp_lo[bad], tmp_hi[bad] = mid, mid
                lo[m], hi[m] = tmp_lo, tmp_hi

    # ------------------------------------------------------------------ racing line
    def _solve_line(self, s, cl, nrm, lo, hi, e0, progress, ds):
        K = len(s)
        # warm start: shift the previous solution by the distance driven
        wp = np.zeros(K)
        if self.prev_e is not None and self.prev_s0 is not None and len(self.prev_e) == K:
            shift = progress - self.prev_s0
            e = np.interp(s + shift, s, self.prev_e, left=self.prev_e[0], right=self.prev_e[-1])
            wp[(s + shift) <= s[-1]] = self.w_prev     # only where the old plan actually had road
        else:
            e = 0.5 * (lo + hi)
        e = np.clip(e, lo, hi)
        e[0] = e0
        # the path must leave the car along its heading: second point follows current heading
        # (vehicle frame heading is 0 → y stays at 0 relative to the car), expressed as an offset
        y1 = 0.0   # car's lateral position one metre ahead along its own heading
        e1_target = y1 - cl[1, 1] if K > 1 else e0
        e1_target = e1_target / max(nrm[1, 1], 0.5) if K > 1 else e0
        # projected gradient descent on J = w_curv * sum (D2 path)^2 + w_center * sum e^2
        # Path second difference along the corridor ≈ (cl + n e)'' ; treat n as locally constant:
        # d/de_i J = 2 w_curv * (D2^T D2 (cl_n + e))_i + 2 w_center e_i, with cl_n the centerline's own
        # lateral curvature term.  Using the centerline's curvature as a constant "bias" makes the
        # optimum the classic minimum-curvature line.
        kap_c = _curvature(cl)                     # centerline curvature (1/m)
        b = kap_c * ds * ds                        # the centerline's own second difference along the left normal
        # Cost J = w_c * sum_{j=1..K-2} (e[j-1] - 2 e[j] + e[j+1] + b[j])^2 + w_0 * sum e^2.
        # Projected Gauss-Seidel in three colours (the stencil spans +-2, so i and i+3 are independent).
        wc, w0 = self.w_curv, self.w_center
        e1 = float(np.clip(e1_target, lo[1], hi[1])) if K > 1 else e0
        # Quadratic form: J = (D e + b)^T W (D e + b) + w0 e^T e  ->  H e = -D^T W b,  H = wc D^T D + w0 I
        D = self._d2_matrix(K)
        D1 = self._d1_matrix(K)
        H = wc * (D.T @ D) + self.w_smooth * (D1.T @ D1) + np.diag(w0 + wp)
        g = wc * (D.T @ b) - wp * e                    # e here is the warm start = last plan shifted by distance driven
        # active-set loop: solve with the currently fixed points, clip, fix violators, repeat
        fixed_val = np.full(K, np.nan)
        fixed_val[0] = e0
        if K > 1:
            fixed_val[1] = e1
        for _ in range(6):
            Hs, gs = H.copy(), -g.copy()
            fx = np.isfinite(fixed_val)
            if fx.any():
                # move fixed columns to the right-hand side, then pin those rows
                gs = gs - H[:, fx] @ fixed_val[fx]
                Hs[fx, :] = 0.0
                Hs[:, fx] = 0.0
                Hs[np.ix_(fx, fx)] = np.eye(int(fx.sum()))
                gs[fx] = fixed_val[fx]
            try:
                e_new = np.linalg.solve(Hs, gs)
            except np.linalg.LinAlgError:
                break
            viol_lo = e_new < lo - 1e-6
            viol_hi = e_new > hi + 1e-6
            viol = (viol_lo | viol_hi) & ~fx
            e = np.clip(e_new, lo, hi)
            if not viol.any():
                break
            fixed_val[viol_lo & ~fx] = lo[viol_lo & ~fx]
            fixed_val[viol_hi & ~fx] = hi[viol_hi & ~fx]
        e = np.clip(e, lo, hi)
        e[0] = e0
        if K > 1:
            e[1] = e1
        self.prev_e, self.prev_s0 = e.copy(), progress
        return e

    _d2_cache: dict = {}
    _d1_cache: dict = {}

    @classmethod
    def _d1_matrix(cls, K: int) -> np.ndarray:
        D = cls._d1_cache.get(K)
        if D is None:
            D = np.zeros((K, K))
            for j in range(K - 1):
                D[j, j], D[j, j + 1] = -1.0, 1.0
            cls._d1_cache[K] = D
        return D

    @classmethod
    def _d2_matrix(cls, K: int) -> np.ndarray:
        D = cls._d2_cache.get(K)
        if D is None:
            D = np.zeros((K, K))
            for j in range(1, K - 1):
                D[j, j - 1], D[j, j], D[j, j + 1] = 1.0, -2.0, 1.0
            cls._d2_cache[K] = D
        return D

    # ------------------------------------------------------------------ speed profile
    def _speed_profile(self, path, kappa, s, v_now, road):
        K = len(s)
        seg = np.hypot(*np.diff(path, axis=0).T)
        seg = np.append(seg, seg[-1] if len(seg) else 1.0)
        v_cap = np.minimum(np.sqrt(self.a_lat / np.maximum(np.abs(kappa), 1e-4)), self.v_max)
        # the finish line is in sight: no need to slow for anything beyond it
        fd = road.finish_distance
        if np.isfinite(fd):
            v_cap[s >= fd - 1.0] = self.v_max
        # unknown road past the horizon: assume the worst corner the rules allow. That is a straight-
        # line braking problem (nothing to turn into until the corner comes into view), so it may use
        # the full brake; visible corners below are planned at the gentler trail-braking rate.
        if not (np.isfinite(fd) and s[-1] >= fd - 1.0):
            a_hor = self.a_dec * self.hor_dec_frac
            s_back = s[-1] - s
            v_cap = np.minimum(v_cap, np.sqrt(self.v_horizon ** 2 + 2.0 * a_hor * np.maximum(s_back, 0.0)))
        v = v_cap.copy()
        v[0] = v_now
        # forward pass: power/traction limited acceleration, reduced by lateral use (friction circle)
        for i in range(1, K):
            a_lat_use = min(abs(kappa[i - 1]) * v[i - 1] ** 2 / self.a_lat, 1.0)
            circle = math.sqrt(max(1.0 - a_lat_use ** 2, 0.0))
            a = min(self.a_acc, self.p_max / self.mass / max(v[i - 1], 1.0)) * circle
            a = max(a, 0.3)
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * a * seg[i - 1]))
        # backward pass: planned braking (a fraction of the limit), reduced by lateral use;
        # the car's current point is left alone (that is reality, not a plan)
        a_plan = self.a_dec * self.plan_dec_frac
        for i in range(K - 2, 0, -1):
            a_lat_use = min(abs(kappa[i + 1]) * v[i + 1] ** 2 / self.a_lat, 1.0)
            circle = math.sqrt(max(1.0 - a_lat_use ** 2, 0.0))
            a = max(a_plan * circle, 0.3 * a_plan)
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a * seg[i]))
        return v

    # ------------------------------------------------------------------ longitudinal
    def _longitudinal(self, v, v_prof, s, vp):
        # acceleration that brings us from v to the plan's speed at a short lookahead. The backward
        # pass already carried every far constraint into v_prof at the planned decel, so this both
        # follows the plan and corrects any excess speed.
        ps = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(self._path, axis=0).T))))
        look = max(v * self.look_t, 2.0)
        cand = np.linspace(2.0, max(look, 2.5), 4)
        v_l = np.interp(cand, ps, v_prof)
        a_cand = (v_l ** 2 - v * v) / (2.0 * cand)
        a_req = float(np.min(a_cand))
        v_tgt = float(v_l[-1])
        a_ff = a_req
        a_req = float(np.clip(a_req, -self.a_dec, self.a_acc))
        # resistive losses to compensate for (drag + rolling, rough numbers)
        f_res = 0.5 * 1.2 * 0.55 * v * v + 0.012 * self.mass * G
        # friction circle on the pedals: lateral grip in use (measured now, or planned for the next
        # ~0.4 s, whichever is larger) is not available for braking / driving
        a_lat_plan = max(abs(self.ay_meas), self.a_lat_ahead)
        lat_use = min(a_lat_plan / (vp.mu * G), 0.98)
        circle = math.sqrt(max(1.0 - lat_use * lat_use, 0.0))
        a_req = float(np.clip(a_req, -self.a_dec * max(circle, 0.2), self.a_acc * max(circle, 0.15)))
        if a_req >= 0.0:
            f_avail = min(self.f_drive, self.p_max / max(v, 1.0))
            throttle = (a_req * self.mass + f_res) / max(f_avail, 1.0)
            throttle, brake = float(np.clip(throttle, 0.0, 1.0)), 0.0
        else:
            brake = (-a_req * self.mass - f_res) / self.f_brake
            throttle, brake = 0.0, float(np.clip(brake, 0.0, 1.0))
        return throttle, brake, v_tgt, a_ff

    # ------------------------------------------------------------------ lateral
    def _lateral(self, path, s, kappa, v, vp, obs):
        ld = max(self.ld_min, self.k_ld * v + 1.5)
        ps = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))))
        tx, ty = float(np.interp(ld, ps, path[:, 0])), float(np.interp(ld, ps, path[:, 1]))
        alpha = math.atan2(ty, max(tx, 0.1))
        dist = max(math.hypot(tx, ty), 1.0)
        delta_pp = math.atan2(2.0 * vp.wheelbase * math.sin(alpha), dist)
        # blend pure pursuit with a Stanley-style term on the racing line: curvature feedforward at a
        # short preview (covers actuator lag) + heading + lateral error feedback
        k_ff = float(np.interp(max(v * 0.2, 1.0), ps, kappa))
        delta_ff = math.atan(vp.wheelbase * k_ff)
        i_near = int(np.argmin(np.hypot(path[:, 0], path[:, 1])))
        p_near = path[i_near]
        t_near = path[min(i_near + 1, len(path) - 1)] - path[max(i_near - 1, 0)]
        psi_path = math.atan2(t_near[1], max(t_near[0], 1e-3))
        e_lat = -p_near[0] * math.sin(psi_path) + p_near[1] * math.cos(psi_path)  # + = path is left of car
        delta_st = delta_ff + 0.8 * psi_path + math.atan2(1.2 * e_lat, v + 2.0)
        w = self.k_ff
        delta = w * delta_st + (1.0 - w) * delta_pp
        # grip-circle clamp on the commanded wheel angle (keeps the 'grip demand' metric honest)
        if v > 3.0:
            d_max = math.atan(1.05 * vp.mu * G * vp.wheelbase / (v * v))
            delta = float(np.clip(delta, -d_max, d_max))
        steer = float(np.clip(delta / vp.max_steer, -1.0, 1.0))
        # slew limit
        steer = float(np.clip(steer, self.prev_steer - self.steer_rate, self.prev_steer + self.steer_rate))
        self.prev_steer = steer
        return steer, ld


# ---------------------------------------------------------------------- helpers
def _curvature(p: np.ndarray) -> np.ndarray:
    """Signed curvature (1/m, + left) of a polyline sampled ~uniformly."""
    if len(p) < 3:
        return np.zeros(len(p))
    dx, dy = np.gradient(p[:, 0]), np.gradient(p[:, 1])
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    denom = np.power(dx * dx + dy * dy, 1.5)
    return (dx * ddy - dy * ddx) / np.maximum(denom, 1e-6)
