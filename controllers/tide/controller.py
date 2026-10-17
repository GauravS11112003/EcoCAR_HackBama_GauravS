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
        self.a_lat_nom = self.f_lat * vp.mu * G
        self.a_lat = self.a_lat_nom
        self.mu_scale = 1.0                                   # grip observer: 1 = nominal, lowered when the car slides
        self.i_lat = 0.0                                      # lateral-error integrator (cancels steering bias)
        self.ki_lat = float(c.get("ki_lat", 0.0))             # steer units per (m*s); off by default
        self.bias_est = 0.0                                   # actuator steering bias, measured from applied vs commanded
        self.cmd_hist = []                                    # commands sent, oldest first (for delay alignment)
        self.delay_s = vp.actuation_delay_steps * vp.control_dt
        self.margin_slope = float(c.get("margin_slope", 0.004))   # extra margin per metre of preview distance
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
        self.w_head = float(c.get("w_head", -1.0))          # <0: hard heading anchor on e[1]; >=0: soft penalty weight
        self.n_iter = int(c.get("line_iters", 12))
        # --- lateral control
        self.k_ld = float(c.get("lookahead_gain", 0.32))
        self.ld_min = float(c.get("lookahead_min", 3.5))
        self.k_ff = float(c.get("ff_gain", 0.5))
        # --- LTV-MPC lateral controller (mode "mpc") or the pure-pursuit/Stanley blend (mode "pp")
        self.lat_mode = str(c.get("lateral", "hybrid"))     # pp | mpc | hybrid (MPC at low speed, pure pursuit at high)
        self.mpc_v_lo = float(c.get("mpc_v_low", 6.0))     # hybrid: full MPC at or below this speed [m/s]
        self.mpc_v_hi = float(c.get("mpc_v_high", 10.0))    # hybrid: full pure pursuit at or above this speed
        self.mpc_N = int(c.get("mpc_horizon", 15))
        self.mpc_qy = float(c.get("mpc_q_y", 2.0))            # lateral error [1/m^2]
        self.mpc_qpsi = float(c.get("mpc_q_psi", 3.0))        # heading error [1/rad^2]
        self.mpc_r = float(c.get("mpc_r", 0.3))               # steering effort [1/rad^2]
        self.mpc_rd = float(c.get("mpc_r_rate", 8.0))         # steering rate [1/rad^2 per step]
        self.mpc_qterm = float(c.get("mpc_q_terminal", 4.0))  # terminal weight multiplier
        self.mpc_tau = float(c.get("mpc_steer_tau", 0.06))    # steering actuator time constant [s]
        self.mpc_smooth_kappa = bool(c.get("mpc_smooth_kappa", True))   # feed the MPC the smoothed path curvature (noise robustness)
        self.mpc_max_share = float(c.get("mpc_max_share", 0.3))        # pure pursuit always keeps at least 70% of the wheel
        self.mpc_u_prev = 0.0
        self.anchor_heading = bool(c.get("anchor_heading", True))   # pin the path's start to the car's position and heading
        self.cap_mode = str(c.get("cap_mode", "peak" if self.lat_mode == "mpc" else "smooth"))  # speed cap on smoothed or true-peak curvature
        self.steer_rate = vp.max_steer_rate * vp.control_dt * 0.95  # per-step slew we allow ourselves
        # --- longitudinal control
        self.look_t = float(c.get("look_t", 0.25))          # s, lookahead for the speed law
        self.acc_circle_pow = float(c.get("acc_circle_pow", 1.0))
        # --- lap memory: stitch what we see into a world-frame map; on later laps of a closed course
        #     extend the preview from memory (the controller never *receives* a map; it builds one)
        self.use_memory = bool(c.get("lap_memory", True))
        self.mem_mode = str(c.get("memory_mode", "prior"))   # prior: remembered speed past the horizon; extend: longer preview (experimental)
        self.mem_range = float(c.get("memory_range_m", 150.0))
        self.mem_lat_scale = float(c.get("memory_lat_scale", 0.85))   # grip fraction multiplier for the remembered road
        self.mem = {}                 # station [m, int] -> (x, y, w_left, w_right, seen_from_m)
        self.mem_keys = np.zeros(0, int)
        self.mem_xy = np.zeros((0, 2))
        self.mem_dirty = False   # 1 = true friction circle for throttle; <1 = allow more throttle while cornering
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
        self.prev_s = None
        self.prev_s0 = None         # obs.progress when prev_e was computed
        self.prev_steer = 0.0
        self.side_memory: dict[int, int] = {}   # obstacle station (rounded) -> +1 pass left / -1 pass right

    # ------------------------------------------------------------------ step
    def step(self, obs: Observation) -> Command:
        road, vp = obs.road, self.vehicle
        if road is None or not road.valid or len(road.preview_s) < 4:
            return Command(brake=1.0, steer=self.prev_steer)
        v = max(obs.speed, 0.0)
        self._observe_grip(obs, vp, v)
        self._observe_steer_bias(obs, vp)
        self._v_hor_mem = None
        if self.use_memory:
            self._update_map(obs, road)
            ext = self._extend_preview(obs, road)
            if ext is not None and self.mem_mode == "prior":
                # use memory only to replace the blanket hairpin assumption past the live preview
                s_m, cl_m, _, _ = ext
                k_m = np.abs(_curvature(cl_m))
                beyond = s_m > road.preview_s[-1]
                if beyond.sum() >= 10:
                    v_m = np.sqrt(self.a_lat * self.mem_lat_scale / np.maximum(k_m[beyond], 1e-4))
                    s_b = s_m[beyond] - road.preview_s[-1]
                    # the speed we may carry at the horizon so that every remembered corner is still
                    # reachable at the planned braking rate (same reach law as the live profile)
                    a_plan = self.a_dec * self.plan_dec_frac
                    v_reach = np.sqrt(v_m ** 2 + 2.0 * a_plan * s_b)
                    v_end = float(np.min(np.append(v_reach, np.sqrt(self.v_horizon ** 2 + 2.0 * a_plan * s_b[-1]))))
                    self._v_hor_mem = float(np.clip(v_end, self.v_horizon, self.v_max))
                ext = None
        else:
            ext = None
        if ext is not None:
            s, cl, wl, wr = ext
            extended = True
        else:
            s, cl, wl, wr = road.preview_s, road.preview_centerline, road.preview_width_left, road.preview_width_right
            extended = False
        K = len(s)
        self._live_K = len(road.preview_s) if extended else None
        ds = float(np.mean(np.diff(s))) if K > 1 else 1.0

        # centerline heading / normal in vehicle frame
        tang = np.gradient(cl, axis=0)
        tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-6)[:, None]
        nrm = np.column_stack((-tang[:, 1], tang[:, 0]))   # left normal

        # ------------------------------------------------ 1. corridor band
        marg = self.margin + self.margin_slope * s          # trust the far end of the preview a little less
        lo = -(wr - marg)                                  # most-right allowed offset (negative)
        hi = wl - marg                                     # most-left allowed offset
        narrow = hi < lo
        if narrow.any():                                   # corridor narrower than the car: hold centre
            mid = 0.5 * (lo[narrow] + hi[narrow])
            lo[narrow], hi[narrow] = mid, mid
        self._apply_obstacles(road, obs.progress, s, lo, hi, wl, wr)

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
        kap_c_dbg = _curvature(cl)
        kappa_s = np.convolve(np.pad(kappa, 2, mode="edge"), np.ones(5) / 5.0, mode="valid")
        # the speed cap must respect the true curvature peaks (smoothing alone under-rates slaloms)
        if self.cap_mode == "peak":
            kappa_s = np.where(np.abs(kappa) > np.abs(kappa_s), kappa, kappa_s)
        v_prof = self._speed_profile(path, kappa_s, s, v, road)

        # ------------------------------------------------ 4. longitudinal command
        self.ay_meas = obs.ay
        self._path = path
        self._v_prof = v_prof
        self._kappa_s = kappa_s
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
                       debug={"v_tgt": v_tgt, "a_ff": a_ff, "ld": ld, "e0": e0, "mem_ext": float(extended),
                              "mem_n": float(len(self.mem)), "w_mpc": getattr(self, "_dbg_wmpc", 0.0), "v_hor_used": getattr(self, "_dbg_vhor", 0.0), "e5": float(np.interp(5, s, e)), "e20": float(np.interp(20, s, e)), "e40": float(np.interp(40, s, e)), "e60": float(np.interp(60, s, e)),
                              "lo5": float(np.interp(5, s, lo)), "hi5": float(np.interp(5, s, hi)),
                              **{f"vp{d}": float(np.interp(d, s, v_prof)) for d in (0, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140)},
                              **{f"kw{a}": float(np.max(np.abs(kappa_s[(s >= a) & (s < a + 20)]))) if ((s >= a) & (s < a + 20)).any() else 0.0 for a in (0, 20, 40, 60, 80, 100, 120)},
                              **{f"kc{a}": float(np.max(np.abs(kap_c_dbg[(s >= a) & (s < a + 20)]))) if ((s >= a) & (s < a + 20)).any() else 0.0 for a in (0, 20, 40, 60, 80, 100, 120)}, "mem_err": getattr(self, "_dbg_memerr", (0,0,0))[0], "L_est": getattr(self, "_dbg_memerr", (0,0,0))[1], "match_d": getattr(self, "_dbg_memerr", (0,0,0))[2],
                              "ext_maxseg": getattr(self, "_dbg_ext", (0,0,0,0,0))[0], "ext_maxturn": getattr(self, "_dbg_ext", (0,0,0,0,0))[1],
                              "ext_len": getattr(self, "_dbg_ext", (0,0,0,0,0))[2], "ext_gap": getattr(self, "_dbg_ext", (0,0,0,0,0))[3], "ext_k0": getattr(self, "_dbg_ext", (0,0,0,0,0))[4],
                              "mu_scale": self.mu_scale, "i_lat": self.i_lat, "track_err": getattr(self, "track_err", 0.0), "bias_est": self.bias_est,
                              "vpmin": float(np.min(v_prof[1:])), "ivpmin": float(np.argmin(v_prof[1:]) + 1),
                              "kmax": float(np.max(np.abs(kappa_s))), "ikmax": float(np.argmax(np.abs(kappa_s))),
                              "kc_max": float(np.max(np.abs(road.preview_curvature))),
                              "vp10": float(v_prof[min(10, K - 1)]), "vp20": float(v_prof[min(20, K - 1)]),
                              "kappa_path": float(kappa[min(3, K - 1)])})

    # ------------------------------------------------------------------ lap memory
    def _update_map(self, obs, road):
        c, s_ = math.cos(obs.yaw), math.sin(obs.yaw)
        cl = road.preview_centerline
        xw = obs.x + c * cl[:, 0] - s_ * cl[:, 1]
        yw = obs.y + s_ * cl[:, 0] + c * cl[:, 1]
        st = np.round(obs.progress + road.preview_s).astype(int)
        for i in range(len(st)):
            k = int(st[i])
            old = self.mem.get(k)
            if old is None or road.preview_s[i] < old[4]:     # keep the sample seen from closest
                self.mem[k] = (float(xw[i]), float(yw[i]), float(road.preview_width_left[i]),
                               float(road.preview_width_right[i]), float(road.preview_s[i]))
                self.mem_dirty = True

    def _extend_preview(self, obs, road):
        if len(self.mem) < 150:
            return None
        if self.mem_dirty:
            self.mem_keys = np.array(sorted(self.mem.keys()), int)
            self.mem_xy = np.array([self.mem[k][:2] for k in self.mem_keys])
            self.mem_dirty = False
        # world position of the end of the live preview
        c, s_ = math.cos(obs.yaw), math.sin(obs.yaw)
        pe = road.preview_centerline[-1]
        end_w = np.array([obs.x + c * pe[0] - s_ * pe[1], obs.y + s_ * pe[0] + c * pe[1]])
        # candidates: samples recorded on an earlier pass (station well behind the current one)
        earlier = self.mem_keys < obs.progress - 50.0
        if not earlier.any():
            return None
        d = np.hypot(*(self.mem_xy[earlier] - end_w).T)
        j = int(np.argmin(d))
        if d[j] > 1.0:
            return None
        k0 = int(self.mem_keys[earlier][j])
        # diagnostic: lap length estimate and map error vs live preview at the same stations
        L_est = (obs.progress + road.preview_s[-1]) - k0
        errs = []
        for i in range(0, len(road.preview_s), 8):
            k = int(round(obs.progress + road.preview_s[i] - L_est))
            m = self.mem.get(k)
            if m is not None:
                pw = road.preview_centerline[i]
                xw = obs.x + c * pw[0] - s_ * pw[1]; yw = obs.y + s_ * pw[0] + c * pw[1]
                errs.append(math.hypot(m[0] - xw, m[1] - yw))
        self._dbg_memerr = (float(np.mean(errs)) if errs else -1.0, float(L_est), float(d[j]))
        # walk forward through consecutive stored stations
        need = int(self.mem_range - road.preview_s[-1])
        pts, wls, wrs = [], [], []
        for k in range(k0 + 1, k0 + 1 + need):
            m = self.mem.get(k)
            if m is None:
                break
            pts.append(m[:2]); wls.append(m[2]); wrs.append(m[3])
        if len(pts) < 10:
            return None
        pts = np.array(pts)
        # to vehicle frame
        dx, dy = pts[:, 0] - obs.x, pts[:, 1] - obs.y
        bx = c * dx + s_ * dy
        by = -s_ * dx + c * dy
        ext_cl = np.column_stack((bx, by))
        # continuity check: the first memory point must continue the live preview smoothly
        if np.hypot(*(ext_cl[0] - road.preview_centerline[-1])) > 2.0:
            return None
        # join, drop memory points that sit on top of the preview end, resample to a uniform grid
        # keep only memory points that lie ahead of the preview end along its tangent (no folds)
        pe_v = road.preview_centerline[-1]
        tv = road.preview_centerline[-1] - road.preview_centerline[-3]
        tv = tv / max(np.hypot(*tv), 1e-6)
        along = (ext_cl - pe_v) @ tv
        keep = along > 0.5
        if keep.sum() < 10:
            return None
        cl_raw = np.vstack((road.preview_centerline, ext_cl[keep]))
        wl_raw = np.concatenate((road.preview_width_left, np.array(wls)[keep]))
        wr_raw = np.concatenate((road.preview_width_right, np.array(wrs)[keep]))
        seg = np.hypot(*np.diff(cl_raw, axis=0).T)
        arc = np.concatenate(([0.0], np.cumsum(seg)))
        step = float(np.mean(np.diff(road.preview_s))) if len(road.preview_s) > 1 else 1.0
        s = np.arange(0.0, arc[-1] + 1e-6, step)
        cl = np.column_stack((np.interp(s, arc, cl_raw[:, 0]), np.interp(s, arc, cl_raw[:, 1])))
        wl = np.interp(s, arc, wl_raw)
        wr = np.interp(s, arc, wr_raw)
        h = np.unwrap(np.arctan2(*np.diff(cl_raw, axis=0).T[::-1]))
        self._dbg_ext = (float(seg.max()), float(np.abs(np.diff(h)).max()), float(arc[-1]), float(np.hypot(*(ext_cl[keep][0] - road.preview_centerline[-1]))), int(k0))
        return s, cl, wl, wr

    # ------------------------------------------------------------------ steering-bias observer
    def _observe_steer_bias(self, obs, vp):
        """The applied steering (obs.steer, i.e. the steering-angle sensor) minus the command that
        should be applied now is the actuator's bias. Low-passed, ignored while slew-limited."""
        n = vp.actuation_delay_steps + 1
        if len(self.cmd_hist) >= n:
            expected = self.cmd_hist[-n]
            resid = obs.steer - expected
            if abs(resid) < 0.12:
                self.bias_est = 0.96 * self.bias_est + 0.04 * resid
        self.cmd_hist = self.cmd_hist[-(n + 2):]

    # ------------------------------------------------------------------ grip observer
    def _observe_grip(self, obs, vp, v):
        """Lower the grip budget when the car is sliding (rear-axle slip or body sideslip beyond what
        rolling tyres produce). Recovers slowly once the car is settled again."""
        if v > 4.0:
            rear_slip = math.degrees(math.atan2(obs.vy - vp.cg_to_rear * obs.yaw_rate, max(abs(obs.vx), 0.1)))
            sideslip = math.degrees(math.atan2(obs.vy, max(abs(obs.vx), 0.1)))
            if abs(rear_slip) > 8.0 or abs(sideslip) > 9.0:
                self.mu_scale = max(self.mu_scale - 0.015, 0.75)
            else:
                self.mu_scale = min(self.mu_scale + 0.0015, 1.0)
        self.a_lat = self.a_lat_nom * self.mu_scale

    # ------------------------------------------------------------------ corridor
    def _apply_obstacles(self, road, progress, s, lo, hi, wl, wr):
        for ob in road.obstacles:
            d = float(ob.distance)
            key = int(round((progress + d) / 2.0))        # stable id for the same obstacle across steps
            i_d = min(max(int(d), 0), len(s) - 1)
            room_left = (wl[i_d] - self.margin) - (ob.lateral_max + self.obs_clear)
            room_right = (ob.lateral_min - self.obs_clear) - (-(wr[i_d] - self.margin))
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
        if self.prev_e is not None and self.prev_s0 is not None:
            shift = progress - self.prev_s0
            e = np.interp(s + shift, self.prev_s, self.prev_e, left=self.prev_e[0], right=self.prev_e[-1])
            wp[(s + shift) <= self.prev_s[-1]] = self.w_prev     # only where the old plan actually had road
            self.track_err = float(e[0] - e0)          # + = the plan we committed to is left of where we are
        else:
            e = 0.5 * (lo + hi)
            self.track_err = 0.0
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
        soft_head = self.anchor_heading and self.w_head >= 0.0 and K > 1
        if soft_head:
            H[1, 1] += self.w_head
            g[1] -= self.w_head * e1
        # active-set loop: solve with the currently fixed points, clip, fix violators, repeat
        fixed_val = np.full(K, np.nan)
        if self.anchor_heading:
            fixed_val[0] = e0
        if K > 1 and self.anchor_heading and not soft_head:
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
        if self.anchor_heading:
            e[0] = e0
        if K > 1 and self.anchor_heading and not soft_head:
            e[1] = e1
        self.prev_e, self.prev_s0, self.prev_s = e.copy(), progress, s.copy()
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
        a_lat_s = np.full(K, self.a_lat)
        if self._live_K is not None and K > self._live_K:          # remembered road: plan it with margin
            a_lat_s[self._live_K:] *= self.mem_lat_scale
        v_cap = np.minimum(np.sqrt(a_lat_s / np.maximum(np.abs(kappa), 1e-4)), self.v_max)
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
            v_h = self._v_hor_mem if self._v_hor_mem is not None else self.v_horizon
            self._dbg_vhor = v_h
            v_cap = np.minimum(v_cap, np.sqrt(v_h ** 2 + 2.0 * a_hor * np.maximum(s_back, 0.0)))
        v = v_cap.copy()
        v[0] = v_now
        # forward pass: power/traction limited acceleration, reduced by lateral use (friction circle)
        for i in range(1, K):
            a_lat_use = min(abs(kappa[i - 1]) * v[i - 1] ** 2 / self.a_lat, 1.0)
            circle = math.sqrt(max(1.0 - a_lat_use ** 2, 0.0)) ** self.acc_circle_pow
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
        look = max(v * (self.look_t + self.delay_s), 2.0)
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
        circle_acc = circle ** self.acc_circle_pow
        a_req = float(np.clip(a_req, -self.a_dec * max(circle, 0.2), self.a_acc * max(circle_acc, 0.15)))
        if a_req >= 0.0:
            f_avail = min(self.f_drive, self.p_max / max(v, 1.0))
            throttle = (a_req * self.mass + f_res) / max(f_avail, 1.0)
            throttle, brake = float(np.clip(throttle, 0.0, 1.0)), 0.0
        else:
            brake = (-a_req * self.mass - f_res) / self.f_brake
            throttle, brake = 0.0, float(np.clip(brake, 0.0, 1.0))
        return throttle, brake, v_tgt, a_ff

    # ------------------------------------------------------------------ LTV-MPC
    def _mpc(self, path, kappa, v_prof, v, vp, obs):
        """Linear time-varying MPC on the path-relative error state x = [e_y, e_psi, delta].
        Kinematic bicycle linearised along the planned path at the planned speed, first-order
        steering actuator, N steps of dt. Quadratic cost, no inequality constraints -> one linear
        solve. Returns the first commanded wheel angle [rad]."""
        N, dt, L, tau = self.mpc_N, vp.control_dt, vp.wheelbase, self.mpc_tau
        ps = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))))
        # where along the path each horizon step lands, at the planned speed
        v_k = np.empty(N)
        s_k = np.empty(N)
        s_acc = 0.0
        for k in range(N):
            vk = max(float(np.interp(s_acc, ps, v_prof)), 1.0)
            v_k[k], s_k[k] = vk, s_acc
            s_acc += vk * dt
        kap_k = np.interp(s_k, ps, kappa)
        # initial error state: the path leaves the car at its own position and heading, so the
        # initial errors are the (small) residuals of that anchoring; the wheel angle is measured
        t0 = path[1] - path[0]
        psi0 = math.atan2(t0[1], max(t0[0], 1e-3))
        # lateral error of the car from the path's first segment, in the path frame (+ = car left of path)
        e_y0 = -path[0, 1] * math.cos(psi0) + path[0, 0] * math.sin(psi0)
        x0 = np.array([e_y0, -psi0, obs.steer * vp.max_steer])
        # batch prediction: x_{k+1} = A_k x_k + B u_k + c_k
        nx = 3
        Phi = np.zeros((nx * N, nx))          # effect of x0 on x_1..x_N
        Gam = np.zeros((nx * N, N))           # effect of u_0..u_{N-1}
        off = np.zeros(nx * N)                # effect of the affine (curvature) terms
        A_prod = np.eye(nx)
        a = 1.0 - dt / tau if tau > dt else 0.0
        bT = dt / tau if tau > dt else 1.0
        x_aff = np.zeros(nx)
        As, cs = [], []
        for k in range(N):
            A = np.array([[1.0, dt * v_k[k], 0.0],
                          [0.0, 1.0, dt * v_k[k] / L],
                          [0.0, 0.0, a]])
            c = np.array([0.0, -dt * v_k[k] * kap_k[k], 0.0])
            As.append(A); cs.append(c)
        B = np.array([0.0, 0.0, bT])
        # forward recursion
        Phi_k = np.eye(nx)
        for k in range(N):
            Phi_k = As[k] @ Phi_k
            Phi[nx * k:nx * (k + 1)] = Phi_k
        for j in range(N):
            M = B.copy()
            for k in range(j, N):
                if k > j:
                    M = As[k] @ M
                Gam[nx * k:nx * (k + 1), j] = M
        xa = np.zeros(nx)
        for k in range(N):
            xa = As[k] @ xa + cs[k]
            off[nx * k:nx * (k + 1)] = xa
        # cost: sum_k x_k^T Q x_k + r u_k^2 + rd (u_k - u_{k-1})^2, terminal weight on x_N
        q = np.tile([self.mpc_qy, self.mpc_qpsi, 0.0], N)
        q[-nx:] *= self.mpc_qterm
        Q = np.diag(q)
        Dm = np.eye(N) - np.eye(N, k=-1)      # u_k - u_{k-1}; row 0 is u_0 - u_prev handled below
        H = Gam.T @ Q @ Gam + self.mpc_r * np.eye(N) + self.mpc_rd * (Dm.T @ Dm)
        xfree = Phi @ x0 + off
        g = Gam.T @ Q @ xfree
        g[0] -= self.mpc_rd * self.mpc_u_prev
        try:
            u = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            return None
        u0 = float(u[0])
        self.mpc_u_prev = u0
        return u0

    # ------------------------------------------------------------------ lateral
    def _lateral(self, path, s, kappa, v, vp, obs):
        ld = max(self.ld_min, self.k_ld * v + 1.5) + v * self.delay_s   # actuator delay: look further ahead
        ps = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))))
        tx, ty = float(np.interp(ld, ps, path[:, 0])), float(np.interp(ld, ps, path[:, 1]))
        alpha = math.atan2(ty, max(tx, 0.1))
        dist = max(math.hypot(tx, ty), 1.0)
        delta_pp = math.atan2(2.0 * vp.wheelbase * math.sin(alpha), dist)
        # blend pure pursuit with a Stanley-style term on the racing line: curvature feedforward at a
        # short preview (covers actuator lag) + heading + lateral error feedback
        k_ff = float(np.interp(max(v * (0.2 + self.delay_s), 1.0), ps, kappa))
        delta_ff = math.atan(vp.wheelbase * k_ff)
        i_near = int(np.argmin(np.hypot(path[:, 0], path[:, 1])))
        p_near = path[i_near]
        t_near = path[min(i_near + 1, len(path) - 1)] - path[max(i_near - 1, 0)]
        psi_path = math.atan2(t_near[1], max(t_near[0], 1e-3))
        # cross-track error to the line we committed to last step (the published path is anchored
        # at the car, so its own nearest point is useless as an error signal)
        e_lat = float(np.clip(getattr(self, "track_err", 0.0), -2.0, 2.0))
        delta_st = delta_ff + 0.8 * psi_path + math.atan2(1.2 * e_lat, v + 2.0)
        w = self.k_ff
        delta = w * delta_st + (1.0 - w) * delta_pp
        w_mpc = 0.0
        if self.lat_mode == "mpc":
            w_mpc = 1.0
        elif self.lat_mode == "hybrid":
            w_mpc = float(np.clip((self.mpc_v_hi - v) / max(self.mpc_v_hi - self.mpc_v_lo, 0.1), 0.0, 1.0))
        # the MPC is a linearised controller: hand over to pure pursuit whenever the car is far from
        # the linearisation point (large lateral / heading error, or sliding), where recovery matters
        near = (abs(e_lat) < 0.5 and abs(psi_path) < 0.17
                and abs(math.degrees(math.atan2(obs.vy, max(abs(obs.vx), 0.1)))) < 6.0)
        if not near:
            w_mpc = 0.0
        w_mpc = min(w_mpc, self.mpc_max_share)
        if w_mpc > 0.0 and v > 2.0:
            u0 = self._mpc(path, self._kappa_s if self.mpc_smooth_kappa else kappa, self._v_prof, v, vp, obs)
            if u0 is not None and math.isfinite(u0):
                delta = w_mpc * u0 + (1.0 - w_mpc) * delta
        else:
            self.mpc_u_prev = delta
        self._dbg_wmpc = w_mpc
        # slow integral on lateral error: cancels a constant steering bias / crosswind-type offset
        if v > 3.0:
            self.i_lat = float(np.clip(self.i_lat + self.ki_lat * e_lat * vp.control_dt, -0.08, 0.08))
        delta += self.i_lat * vp.max_steer
        # grip-circle clamp on the commanded wheel angle (keeps the 'grip demand' metric honest)
        if v > 3.0:
            d_max = math.atan(1.05 * vp.mu * G * vp.wheelbase / (v * v))
            delta = float(np.clip(delta, -d_max, d_max))
        steer = float(np.clip(delta / vp.max_steer - self.bias_est, -1.0, 1.0))
        # slew limit
        steer = float(np.clip(steer, self.prev_steer - self.steer_rate, self.prev_steer + self.steer_rate))
        self.prev_steer = steer
        self.cmd_hist.append(steer)
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
