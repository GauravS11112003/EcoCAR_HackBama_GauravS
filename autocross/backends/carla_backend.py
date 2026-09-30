"""CARLA backend. Builds a flat asphalt pad from a generated OpenDRIVE road sized to the
course, lays physics-enabled cones, and steps the world in synchronous mode.

Frame conversion: the harness uses right-handed ENU; CARLA is left-handed (y right, yaw
clockwise, degrees). OpenDRIVE shares the ENU convention, so pad coordinates need no
conversion: ENU (x, y, yaw) <-> CARLA (x, -y, -yaw).
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np

from ..api import CONE_ORANGE
from .base import Backend, PhysicalParams, VehicleState

try:
    import carla
except ImportError as e:  # pragma: no cover
    raise ImportError("The CARLA backend needs the 'carla' Python package matching your server "
                      "(pip install carla==0.9.16). Use --backend bicycle to run without CARLA.") from e

PAD_TEMPLATE = """<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="{name}" version="1.00" north="0" south="0" east="0" west="0"/>
  <road name="pad" length="{L:.3f}" id="1" junction="-1">
    <link/>
    <planView><geometry s="0" x="{x0:.3f}" y="{yc:.3f}" hdg="0" length="{L:.3f}"><line/></geometry></planView>
    <elevationProfile><elevation s="0" a="0" b="0" c="0" d="0"/></elevationProfile>
    <lateralProfile/>
    <lanes><laneSection s="0">
      <left><lane id="1" type="driving" level="false"><link/><width sOffset="0" a="{hw:.3f}" b="0" c="0" d="0"/></lane></left>
      <center><lane id="0" type="none" level="false"><link/></lane></center>
      <right><lane id="-1" type="driving" level="false"><link/><width sOffset="0" a="{hw:.3f}" b="0" c="0" d="0"/></lane></right>
    </laneSection></lanes>
  </road>
</OpenDRIVE>"""


def to_carla(x, y, yaw, z=0.0):
    return carla.Transform(carla.Location(x=float(x), y=float(-y), z=float(z)),
                           carla.Rotation(yaw=float(-math.degrees(yaw))))


class CarlaBackend(Backend):
    name = "carla"

    def __init__(self, sim_cfg: dict, carla_cfg: dict):
        self.dt = float(sim_cfg["physics_dt"])
        self.cfg = carla_cfg
        self.client = carla.Client(carla_cfg.get("host", "localhost"), int(carla_cfg.get("port", 2000)))
        self.client.set_timeout(float(carla_cfg.get("timeout_s", 30.0)))
        self._wait_for_server(float(carla_cfg.get("connect_wait_s", 60.0)))
        self.world = None
        self.vehicle = None
        self.cone_ids: list[int] = []
        self._sync_set = False
        self.cmd = carla.VehicleControl()
        self.recording = False

    def _wait_for_server(self, wait):
        t0 = time.time()
        while True:
            try:
                sv, cv = self.client.get_server_version(), self.client.get_client_version()
                if sv.split("-")[0] != cv.split("-")[0]:
                    print(f"[carla] WARNING: server {sv} != client {cv}; install the matching carla wheel")
                return
            except RuntimeError:
                if time.time() - t0 > wait:
                    raise RuntimeError(f"No CARLA server at {self.cfg.get('host', 'localhost')}:{self.cfg.get('port', 2000)}. "
                                       "Start it with scripts/start_carla.ps1 (or .sh).")
                time.sleep(2.0)

    # ------------------------------------------------------------------ world
    def _ensure_pad(self, track):
        margin = float(self.cfg.get("pad_margin_m", 40.0))
        x0, y0, x1, y1 = track.bounds(margin)
        L, hw, yc = max(x1 - x0, 150.0), max((y1 - y0) / 2, 75.0), (y0 + y1) / 2
        key = hashlib.sha1(f"{x0:.1f},{L:.1f},{yc:.1f},{hw:.1f}".encode()).hexdigest()[:10]
        name = f"autocross_pad_{key}"
        xodr = PAD_TEMPLATE.format(name=name, L=L, x0=x0, yc=yc, hw=hw)
        world = self.client.get_world()
        self._set_async(world)
        try:
            same = name in world.get_map().to_opendrive()
        except RuntimeError:
            same = False
        if not same:
            params = carla.OpendriveGenerationParameters(
                vertex_distance=4.0, max_road_length=500.0, wall_height=0.0, additional_width=0.0,
                smooth_junctions=True, enable_mesh_visibility=True)
            world = self.client.generate_opendrive_world(xodr, params)
        else:
            self._clear_world(world)
        return world

    @staticmethod
    def _set_async(world):
        s = world.get_settings()
        if s.synchronous_mode:
            s.synchronous_mode = False
            s.fixed_delta_seconds = None
            world.apply_settings(s)

    def _clear_world(self, world):
        ids = [a.id for a in world.get_actors() if a.type_id.startswith(("vehicle.", "static.prop.", "sensor."))]
        if ids:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids])

    def setup(self, track, vehicle_cfg, perturb, rng) -> PhysicalParams:
        self.world = self._ensure_pad(track)
        w = self.world
        weather = getattr(carla.WeatherParameters, str(self.cfg.get("weather", "ClearNoon")), carla.WeatherParameters.ClearNoon)
        w.set_weather(weather)
        s = w.get_settings()
        self._sync_set = True
        s.synchronous_mode = True
        s.fixed_delta_seconds = self.dt
        s.substepping = True
        s.max_substep_delta_time = min(0.01, self.dt)
        s.max_substeps = max(int(math.ceil(self.dt / s.max_substep_delta_time)), 1)
        s.no_rendering_mode = not bool(self.cfg.get("render", True))
        w.apply_settings(s)
        self.render = not s.no_rendering_mode
        bl = w.get_blueprint_library()
        # cones (visual + physical); scoring uses their surveyed positions, not the actors
        if self.cfg.get("spawn_cones", True):
            pos, col, _ = track.all_cones()
            bp_lane = bl.find(self.cfg.get("cone_blueprint", "static.prop.constructioncone"))
            bp_orange = bl.find(self.cfg.get("marker_cone_blueprint", "static.prop.trafficcone01"))
            batch = [carla.command.SpawnActor(bp_orange if c == CONE_ORANGE else bp_lane,
                                              to_carla(p[0], p[1], float(rng.uniform(-math.pi, math.pi)), 0.05))
                     for p, c in zip(pos, col)]
            res = self.client.apply_batch_sync(batch, False)
            self.cone_ids = [r.actor_id for r in res if not r.error]
            if self.cfg.get("cone_physics", True):
                self.client.apply_batch_sync([carla.command.SetSimulatePhysics(i, True) for i in self.cone_ids], False)
        # vehicle
        bp = bl.find(vehicle_cfg.get("blueprint", "vehicle.tesla.model3"))
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "hero")
        if bp.has_attribute("color"):
            bp.set_attribute("color", str(vehicle_cfg.get("color", "180,20,30")))
        x, y, yaw = track.spawn_pose()
        self.vehicle = w.spawn_actor(bp, to_carla(x, y, yaw, 0.4))
        pc = self.vehicle.get_physics_control()
        max_steer = float(vehicle_cfg.get("max_steer_deg", 35.0))
        fscale = float(perturb.get("friction_scale", 1.0))
        wheels = pc.wheels
        brake_nm = vehicle_cfg.get("carla_brake_torque_nm")
        for i, wh in enumerate(wheels):
            if i < 2:
                wh.max_steer_angle = max_steer
            if brake_nm:
                wh.max_brake_torque = float(brake_nm)
            base_mu = vehicle_cfg.get("carla_tire_friction")
            wh.tire_friction = (float(base_mu) if base_mu else wh.tire_friction) * fscale
        pc.wheels = wheels
        pc.mass = pc.mass * float(perturb.get("mass_scale", 1.0))
        tq = float(vehicle_cfg.get("carla_torque_scale", 1.0))
        if tq != 1.0:  # stock torque overwhelms the tyres (CARLA has no traction control)
            pc.torque_curve = [carla.Vector2D(p.x, p.y * tq) for p in pc.torque_curve]
        pc.steering_curve = [carla.Vector2D(0.0, 1.0), carla.Vector2D(300.0, 1.0)]  # no speed-dependent steer
        self.vehicle.apply_physics_control(pc)
        w.tick()
        pc = self.vehicle.get_physics_control()
        tr = self.vehicle.get_transform()
        cy, sy = math.cos(math.radians(tr.rotation.yaw)), math.sin(math.radians(tr.rotation.yaw))

        def body_x(wh):  # wheel position (world, cm) -> vehicle-frame longitudinal offset (m)
            return (wh.position.x / 100.0 - tr.location.x) * cy + (wh.position.y / 100.0 - tr.location.y) * sy

        fl, fr, rl, rr = pc.wheels
        x_front, x_rear = 0.5 * (body_x(fl) + body_x(fr)), 0.5 * (body_x(rl) + body_x(rr))
        wheelbase = float(x_front - x_rear)
        track_w = float(math.hypot(fl.position.x - fr.position.x, fl.position.y - fr.position.y) / 100.0)
        bb = self.vehicle.bounding_box
        cg_to_front = float(np.clip(x_front - pc.center_of_mass.x, 0.2 * wheelbase, 0.8 * wheelbase))
        if self.cfg.get("record", False):
            self.client.start_recorder(str(self.cfg.get("record_path", "autocross_run.log")), True)
            self.recording = True
        return PhysicalParams(float(pc.mass), wheelbase, track_w, 2 * bb.extent.x, 2 * bb.extent.y,
                              cg_to_front, wheelbase - cg_to_front, math.radians(max_steer))

    # ------------------------------------------------------------------ stepping
    def reset(self, x, y, yaw) -> VehicleState:
        self.vehicle.set_transform(to_carla(x, y, yaw, 0.3))
        self.vehicle.set_target_velocity(carla.Vector3D())
        self.vehicle.set_target_angular_velocity(carla.Vector3D())
        self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(int(1.0 / self.dt)):  # let the suspension settle
            self.world.tick()
        self.t0 = self.world.get_snapshot().timestamp.elapsed_seconds
        self._update_spectator()
        return self._state()

    def apply(self, steer, throttle, brake):
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=float(throttle), steer=float(-steer), brake=float(brake), hand_brake=False, reverse=False))

    def tick(self) -> VehicleState:
        self.world.tick()
        self._update_spectator()
        return self._state()

    def _state(self) -> VehicleState:
        snap = self.world.get_snapshot()
        a = snap.find(self.vehicle.id)
        tr, v, av, acc = a.get_transform(), a.get_velocity(), a.get_angular_velocity(), a.get_acceleration()
        yaw = -math.radians(tr.rotation.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        vxw, vyw = v.x, -v.y
        axw, ayw = acc.x, -acc.y
        steer = float("nan")
        if self.cfg.get("measure_wheel_angle", True):
            fl = self.vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
            fr = self.vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
            steer = -math.radians(0.5 * (fl + fr))
        return VehicleState(
            t=snap.timestamp.elapsed_seconds - self.t0 if hasattr(self, "t0") else 0.0,
            x=tr.location.x, y=-tr.location.y, yaw=yaw,
            vx=c * vxw + s * vyw, vy=-s * vxw + c * vyw, yaw_rate=-math.radians(av.z),
            ax=c * axw + s * ayw, ay=-s * axw + c * ayw,
            roll=math.radians(tr.rotation.roll), pitch=math.radians(tr.rotation.pitch), steer_angle=steer)

    def _update_spectator(self):
        if not self.render or self.cfg.get("spectator", "chase") != "chase":
            return
        tr = self.vehicle.get_transform()
        yaw = math.radians(tr.rotation.yaw)
        d, hgt = float(self.cfg.get("chase_distance_m", 9.0)), float(self.cfg.get("chase_height_m", 4.0))
        loc = carla.Location(tr.location.x - d * math.cos(yaw), tr.location.y - d * math.sin(yaw), tr.location.z + hgt)
        self.world.get_spectator().set_transform(carla.Transform(loc, carla.Rotation(pitch=-18.0, yaw=tr.rotation.yaw)))

    def draw_path(self, path, life):
        if not self.render or not self.cfg.get("draw_path", True) or path is None or len(path) < 2:
            return
        step = max(len(path) // 25, 1)
        p = path[::step]
        dbg = self.world.debug
        col = carla.Color(0, 200, 255)
        for a, b in zip(p[:-1], p[1:]):
            dbg.draw_line(carla.Location(float(a[0]), float(-a[1]), 0.15), carla.Location(float(b[0]), float(-b[1]), 0.15),
                          thickness=0.08, color=col, life_time=life)

    def close(self):
        try:
            if self.recording:
                self.client.stop_recorder()
            ids = list(self.cone_ids) + ([self.vehicle.id] if self.vehicle else [])
            if ids:
                self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], False)
            if self.world is not None and self._sync_set:
                s = self.world.get_settings()
                s.synchronous_mode = False
                s.fixed_delta_seconds = None
                s.no_rendering_mode = False
                self.world.apply_settings(s)
        except RuntimeError as e:
            print(f"[carla] cleanup warning: {e}")
