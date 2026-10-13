"""Runs a team controller in a separate process under an embedded-style compute budget.

Budget enforcement (``compute_budget.mode``):
  * enforce - the step must answer within ``step_deadline_ms``; otherwise the previous
              command is held (like a missed ECU task deadline) and an overrun is logged.
              Late answers are discarded.
  * measure - always wait for the answer (up to ``hard_timeout_s``) and log overruns.
Independently of mode, the process is pinned to ``cpu_cores`` logical cores, BLAS/OpenMP
thread pools are limited to that count, CUDA devices are hidden unless ``allow_gpu``, and
CPU time and peak memory are sampled.

The controller may not open course files or earlier run outputs (``sandbox.protected_paths``
plus the course being driven): a Python audit hook in the controller process refuses them,
so an evaluation course cannot be read from disk.
"""
from __future__ import annotations

import importlib.util
import inspect
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass
class StepStats:
    wall_ms: float = float("nan")   # time inside step(), measured in the controller process
    cpu_ms: float = float("nan")    # process CPU time consumed by step()
    latency_ms: float = float("nan")  # round trip seen by the harness (incl. IPC)
    overrun: bool = False
    error: str | None = None


def load_controller_class(path: str, class_name: str | None = None):
    path = str(Path(path).resolve())
    spec = importlib.util.spec_from_file_location(f"team_controller_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(mod)
    from .api import Controller
    if class_name:
        return getattr(mod, class_name)
    found = [c for _, c in inspect.getmembers(mod, inspect.isclass)
             if issubclass(c, Controller) and c is not Controller and c.__module__ == mod.__name__]
    if len(found) != 1:
        raise RuntimeError(f"{path}: expected exactly one Controller subclass, found {[c.__name__ for c in found]}; "
                           "pass --controller-class")
    return found[0]


def _install_file_guard(protected: list[str]) -> None:
    """Refuse open/listdir/scandir on the protected files and directories. Audit hooks cannot be
    removed, so this holds for the life of the controller process."""
    roots = [os.path.normcase(os.path.abspath(p)) for p in protected]
    if not roots:
        return
    events = {"open", "os.listdir", "os.scandir"}

    def hook(event, args):
        if event not in events or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        try:
            p = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        except (TypeError, ValueError):
            return
        for r in roots:
            if p == r or p.startswith(r.rstrip(os.sep) + os.sep):
                raise PermissionError(f"controllers may not access course or run files: {os.fsdecode(args[0])}")

    sys.addaudithook(hook)


def _worker(conn, path, class_name, ctrl_cfg, protected=()):
    try:
        _install_file_guard(list(protected))
        cls = load_controller_class(path, class_name)
        ctrl = cls(ctrl_cfg)
        conn.send(("ready", 0, None, 0.0, 0.0))
    except Exception:
        conn.send(("error", 0, traceback.format_exc(), 0.0, 0.0))
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        kind, seq, payload = msg
        t0, c0 = time.perf_counter(), time.process_time()
        try:
            if kind == "start":
                out = ctrl.on_start(*payload)
            elif kind == "step":
                out = ctrl.step(payload)
            elif kind == "finish":
                ctrl.on_finish()
                conn.send(("ok", seq, None, 0.0, 0.0))
                return
            else:
                continue
            conn.send(("ok", seq, out, (time.perf_counter() - t0) * 1e3, (time.process_time() - c0) * 1e3))
        except Exception:
            conn.send(("error", seq, traceback.format_exc(), (time.perf_counter() - t0) * 1e3,
                       (time.process_time() - c0) * 1e3))


class ControllerProcess:
    def __init__(self, path: str, budget: dict, class_name: str | None = None, ctrl_cfg: dict | None = None,
                 protected_paths: list[str] | None = None):
        self.budget = budget
        self.mode = str(budget.get("mode", "enforce"))
        self.deadline = float(budget.get("step_deadline_ms", 50.0)) / 1e3
        self.hard_timeout = float(budget.get("hard_timeout_s", 10.0))
        self.seq = 0
        self.peak_rss = 0
        self.dead = False
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        cores = int(budget.get("cpu_cores", 1))
        env_overrides = {k: str(cores) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
        if not budget.get("allow_gpu", False):
            env_overrides["CUDA_VISIBLE_DEVICES"] = "-1"
        saved = {k: os.environ.get(k) for k in env_overrides}
        os.environ.update(env_overrides)
        try:
            self.proc = ctx.Process(target=_worker, args=(child, path, class_name, ctrl_cfg or {},
                                                          [str(p) for p in protected_paths or []]), daemon=True)
            self.proc.start()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.ps = psutil.Process(self.proc.pid)
        try:
            n = psutil.cpu_count(logical=True) or 1
            self.cores = list(range(max(n - cores, 0), n))  # leave core 0 for the simulator/harness
            self.ps.cpu_affinity(self.cores)
        except (AttributeError, psutil.Error):
            self.cores = []
        kind, _, payload, _, _ = self._recv(float(budget.get("init_timeout_s", 60.0)))
        if kind != "ready":
            self.close()
            raise RuntimeError(f"controller failed to load:\n{payload}")

    def _recv(self, timeout):
        if not self.conn.poll(timeout):
            return ("timeout", -1, None, float("nan"), float("nan"))
        return self.conn.recv()

    def _call(self, kind, payload, timeout):
        self.seq += 1
        seq = self.seq
        self.conn.send((kind, seq, payload))
        end = time.perf_counter() + timeout
        while True:
            msg = self._recv(max(end - time.perf_counter(), 0.0))
            if msg[0] == "timeout" or msg[1] == seq:
                return msg
            # stale reply from an earlier overrun: drop it

    def start(self, track_info, vehicle_params) -> None:
        kind, _, payload, wall, _ = self._call("start", (track_info, vehicle_params),
                                               float(self.budget.get("init_timeout_s", 60.0)))
        self._sample_memory()
        if kind == "timeout":
            raise TimeoutError(f"on_start exceeded init_timeout_s={self.budget.get('init_timeout_s', 60.0)}")
        if kind == "error":
            raise RuntimeError(f"controller on_start raised:\n{payload}")
        self.init_ms = wall

    def step(self, obs):
        """Return (command or None, StepStats). None means: hold the previous command."""
        if self.dead:
            return None, StepStats(error="controller process not running", overrun=True)
        t0 = time.perf_counter()
        timeout = self.deadline if self.mode == "enforce" else self.hard_timeout
        kind, _, payload, wall, cpu = self._call("step", obs, timeout)
        lat = (time.perf_counter() - t0) * 1e3
        if obs.step % 10 == 0:
            self._sample_memory()
        if kind == "timeout":
            if self.mode != "enforce" or not self.proc.is_alive():
                self.dead = True
            return None, StepStats(latency_ms=lat, overrun=True)
        if kind == "error":
            return None, StepStats(wall, cpu, lat, True, error=payload)
        return payload, StepStats(wall, cpu, lat, wall > self.deadline * 1e3)

    def _sample_memory(self):
        try:
            mi = self.ps.memory_info()
            self.peak_rss = max(self.peak_rss, getattr(mi, "peak_wset", 0) or 0, mi.rss)
        except psutil.Error:
            pass

    def close(self):
        try:
            if self.proc.is_alive():
                self._call("finish", None, 2.0)
        except (OSError, BrokenPipeError):
            pass
        self._sample_memory()
        if self.proc.is_alive():
            self.proc.terminate()
        self.proc.join(timeout=2.0)
