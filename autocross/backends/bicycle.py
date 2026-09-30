"""Dynamic single-track (bicycle) model with saturating tyres and a friction circle.

Development backend only: fast, deterministic and dependency-free, so controllers can be
iterated and unit-tested without a CARLA server. Parameters default to the values measured
from the CARLA vehicle (see ``autocross calibrate``) but it is NOT a substitute for the
scored CARLA runs."""
from __future__ import annotations

import math

import numpy as np

from .base import Backend, PhysicalParams, VehicleState

G = 9.81


class BicycleBackend(Backend):
    name = "bicycle"

    def __init__(self, sim_cfg: dict):
        self.dt = float(sim_cfg["physics_dt"])
        self.sub = max(int(round(self.dt / 0.0025)), 1)

    def setup(self, track, vehicle_cfg: dict, perturb: dict, rng) -> PhysicalParams:
        b = vehicle_cfg.get("bicycle", {})
        self.m = float(b.get("mass", 1845.0)) * float(perturb.get("mass_scale", 1.0))
        self.L = float(b.get("wheelbase", 3.0))
        self.lf = float(b.get("cg_to_front", 1.45))
        self.lr = self.L - self.lf
        self.h = float(b.get("cg_height", 0.5))
        self.iz = float(b.get("yaw_inertia_factor", 1.0)) * self.m * self.lf * self.lr
        self.mu = float(b.get("mu", 1.0)) * float(perturb.get("friction_scale", 1.0))
        self.c_norm = float(b.get("cornering_stiffness_per_load", 14.0))  # (N/rad)/N
        self.f_drive = float(b.get("max_drive_force", 11000.0))
        self.p_max = float(b.get("max_power_w", 200000.0))
        self.f_brake = float(b.get("max_brake_force", 15000.0))
        self.cda = float(b.get("drag_area", 0.55))
        self.crr = float(b.get("rolling_resistance", 0.012))
        self.tau_steer = float(b.get("steer_time_constant", 0.06))
        self.front_drive = float(b.get("front_drive_fraction", 0.4))
        self.slip_limit = float(b.get("traction_limit", 0.85))
        self.max_steer = math.radians(float(vehicle_cfg.get("max_steer_deg", 35.0)))
        self.cmd = (0.0, 0.0, 0.0)
        return PhysicalParams(self.m, self.L, float(b.get("track_width", 1.67)), float(b.get("length", 4.79)),
                              float(b.get("width", 2.16)), self.lf, self.lr, self.max_steer)

    def reset(self, x, y, yaw) -> VehicleState:
        self.s = np.array([x, y, yaw, 0.0, 0.0, 0.0])  # x y yaw vx vy r
        self.delta = 0.0
        self.t = 0.0
        self.acc = (0.0, 0.0)
        return self._state()

    def apply(self, steer, throttle, brake):
        self.cmd = (steer, throttle, brake)

    def tick(self) -> VehicleState:
        h = self.dt / self.sub
        v0 = self.s[3:5].copy()
        for _ in range(self.sub):
            self._substep(h)
        self.t += self.dt
        vx, vy, r = self.s[3:6]
        dv = (self.s[3:5] - v0) / self.dt
        self.acc = (float(dv[0] - r * vy), float(dv[1] + r * vx))
        return self._state()

    def _substep(self, h):
        steer, throttle, brake = self.cmd
        target = float(np.clip(steer, -1, 1)) * self.max_steer
        self.delta += (target - self.delta) * min(h / self.tau_steer, 1.0)
        x, y, yaw, vx, vy, r = self.s
        d = self.delta
        # longitudinal forces
        f_trac = throttle * min(self.f_drive, self.p_max / max(vx, 1.0))
        f_brk = brake * self.f_brake
        f_res = 0.5 * 1.2 * self.cda * vx * abs(vx) + self.crr * self.m * G * (1.0 if vx > 0.05 else 0.0)
        # load transfer from previous-step accel
        ax_prev = self.acc[0]
        fzf = self.m * (G * self.lr - ax_prev * self.h) / self.L
        fzr = self.m * G - fzf
        # axle longitudinal demand (AWD split, 60/40 brakes); traction control / ABS keep the
        # longitudinal force below slip_limit * mu * Fz so some lateral grip always remains
        fxf = self.front_drive * f_trac - 0.6 * f_brk
        fxr = (1.0 - self.front_drive) * f_trac - 0.4 * f_brk
        lim = self.slip_limit * self.mu
        fxf = float(np.clip(fxf, -lim * fzf, lim * fzf))
        fxr = float(np.clip(fxr, -lim * fzr, lim * fzr))
        if vx > 1.0:
            af = math.atan2(vy + self.lf * r, vx) - d
            ar = math.atan2(vy - self.lr * r, vx)
            fyf = self._lat(af, fzf, fxf)
            fyr = self._lat(ar, fzr, fxr)
            vx_dot = (fxr + fxf * math.cos(d) - fyf * math.sin(d) - f_res) / self.m + r * vy
            vy_dot = (fyr + fyf * math.cos(d) + fxf * math.sin(d)) / self.m - r * vx
            rdot = (self.lf * (fyf * math.cos(d) + fxf * math.sin(d)) - self.lr * fyr) / self.iz
            vx += vx_dot * h
            vy += vy_dot * h
            r += rdot * h
        # low speed: kinematic model (blended in over 1..3 m/s for continuity)
        w_kin = float(np.clip((3.0 - vx) / 2.0, 0.0, 1.0))
        if w_kin > 0:
            fx_net = fxr + fxf - f_res
            vk = vx + fx_net / self.m * h if vx <= 1.0 else vx
            if vk < 0 and throttle <= 0.0:
                vk = 0.0
            rk = vk * math.tan(d) / self.L
            vyk = rk * self.lr
            vx = w_kin * vk + (1 - w_kin) * vx
            vy = w_kin * vyk + (1 - w_kin) * vy
            r = w_kin * rk + (1 - w_kin) * r
        vx = max(vx, 0.0)
        c, s = math.cos(yaw), math.sin(yaw)
        x += (vx * c - vy * s) * h
        y += (vx * s + vy * c) * h
        yaw += r * h
        self.s = np.array([x, y, yaw, vx, vy, r])

    def _lat(self, alpha, fz, fx):
        fmax = self.mu * fz
        avail = math.sqrt(max(fmax ** 2 - fx ** 2, 0.0))
        c = self.c_norm * fz
        return -avail * math.tanh(c * alpha / max(avail, 1.0))

    def _state(self) -> VehicleState:
        x, y, yaw, vx, vy, r = self.s
        return VehicleState(self.t, float(x), float(y), float(math.atan2(math.sin(yaw), math.cos(yaw))),
                            float(vx), float(vy), float(r), self.acc[0], self.acc[1], steer_angle=self.delta)
