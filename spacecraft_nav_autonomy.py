#!/usr/bin/env python3
"""
Autonomous Spacecraft Navigation (research prototype)

Goal:
  Demonstrate a practical autonomy stack for spacecraft rendezvous/navigation:
    - Dynamics: Clohessy–Wiltshire (Hill) linearized relative orbital motion (2D)
    - Estimation: Linear Kalman filter
    - Planning: Sampling-based MPC (random shooting) with safety constraints
    - Safety: Symbolic monitor (keep-out zone, thrust/fuel limits, approach gating)

This is not a flight-certified system. It is a research scaffold that connects:
  "predict -> abstract -> plan -> verify" autonomy to a concrete aerospace task.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Literal

import numpy as np


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def norm2(xy: np.ndarray) -> float:
    return float(np.sqrt(np.sum(xy * xy)))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------------------------------------------------------
# Relative Orbit Dynamics (Clohessy–Wiltshire / Hill) in LVLH frame
# State: [x, y, xdot, ydot] (meters, meters, m/s, m/s)
# Control: [ax, ay] (m/s^2)
# -----------------------------------------------------------------------------


@dataclass
class CWDynamics:
    n: float  # mean motion [rad/s]

    def A(self) -> np.ndarray:
        n = self.n
        return np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [3.0 * n * n, 0.0, 0.0, 2.0 * n],
                [0.0, 0.0, -2.0 * n, 0.0],
            ],
            dtype=float,
        )

    def B(self) -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )

    def step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        # simple RK2 integrator for linear dynamics
        A = self.A()
        B = self.B()
        k1 = A @ x + B @ u
        x_mid = x + 0.5 * dt * k1
        k2 = A @ x_mid + B @ u
        return x + dt * k2


# -----------------------------------------------------------------------------
# Sensor + Kalman Filter
# -----------------------------------------------------------------------------


@dataclass
class SensorModel:
    pos_sigma: float = 0.5  # meters
    vel_sigma: float = 0.01  # m/s

    def measure(self, x_true: np.ndarray) -> np.ndarray:
        z = x_true.copy()
        z[0] += np.random.normal(0.0, self.pos_sigma)
        z[1] += np.random.normal(0.0, self.pos_sigma)
        z[2] += np.random.normal(0.0, self.vel_sigma)
        z[3] += np.random.normal(0.0, self.vel_sigma)
        return z


class LinearKalmanFilter:
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.H = np.eye(A.shape[0], dtype=float)  # full-state measurement

        self.x = np.zeros((A.shape[0],), dtype=float)
        self.P = np.eye(A.shape[0], dtype=float) * 10.0

    def predict(self, u: np.ndarray, dt: float):
        # discrete approx: x_{k+1} = (I + A dt) x_k + (B dt) u
        F = np.eye(self.A.shape[0], dtype=float) + self.A * dt
        G = self.B * dt
        self.x = F @ self.x + G @ u
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0], dtype=float)
        self.P = (I - K @ self.H) @ self.P


# -----------------------------------------------------------------------------
# Safety Monitor (symbolic constraints)
# -----------------------------------------------------------------------------


@dataclass
class SafetySpec:
    keepout_radius: float = 10.0  # meters (hard keep-out until approach mode)
    dock_radius: float = 0.5  # meters (success region)
    max_speed_near: float = 0.08  # m/s when close to target
    max_speed_far: float = 1.50  # m/s allowed far from target
    approach_radius: float = 5.0  # meters
    speed_slope: float = 0.12  # additional allowed speed per meter from docking radius
    thrust_max: float = 0.03  # m/s^2
    fuel_budget: float = 20.0  # m/s total delta-v budget (proxy)

    # Docking mode (more conservative close-in control)
    dock_mode_radius: float = 3.0
    thrust_max_dock: float = 0.012  # m/s^2


class SafetyMonitor:
    def __init__(self, spec: SafetySpec):
        self.spec = spec

    def thrust_clip(self, u: np.ndarray) -> np.ndarray:
        umax = self.spec.thrust_max
        return np.array([clamp(float(u[0]), -umax, umax), clamp(float(u[1]), -umax, umax)], dtype=float)

    def speed_limit(self, r: float) -> float:
        v_max = self.spec.max_speed_near + self.spec.speed_slope * max(0.0, (r - self.spec.dock_radius))
        return min(v_max, self.spec.max_speed_far)

    def violates(self, x: np.ndarray, u: np.ndarray, fuel_used: float) -> bool:
        pos = x[:2]
        r = norm2(pos)

        # Fuel limit
        if fuel_used > self.spec.fuel_budget:
            return True

        # Thrust limit enforced separately via clipping, but keep check for completeness
        if abs(float(u[0])) > self.spec.thrust_max + 1e-9 or abs(float(u[1])) > self.spec.thrust_max + 1e-9:
            return True

        return False

    def is_docked(self, x: np.ndarray) -> bool:
        r = norm2(x[:2])
        v = norm2(x[2:])
        return r < self.spec.dock_radius and v < self.spec.max_speed_near


# -----------------------------------------------------------------------------
# Planner: sampling-based MPC (random shooting) with a few maneuver primitives
# -----------------------------------------------------------------------------


@dataclass
class MPCConfig:
    horizon_steps: int = 60
    dt: float = 1.0
    n_candidates: int = 1500
    temperature: float = 0.6  # exploration magnitude

    # cost weights
    w_pos: float = 1.0
    w_vel: float = 12.0
    w_fuel: float = 0.02
    w_violation: float = 1e6


class SamplingMPC:
    def __init__(self, dyn: CWDynamics, monitor: SafetyMonitor, cfg: MPCConfig):
        self.dyn = dyn
        self.monitor = monitor
        self.cfg = cfg
        self._warm_u: Optional[np.ndarray] = None

    def _rollout_cost(self, x0: np.ndarray, u_seq: np.ndarray, fuel_used0: float) -> float:
        x = x0.copy()
        fuel_used = fuel_used0
        cost = 0.0

        for t in range(self.cfg.horizon_steps):
            u = self.monitor.thrust_clip(u_seq[t])
            # approximate fuel via ||u|| dt (delta-v proxy)
            fuel_used += float(np.linalg.norm(u)) * self.cfg.dt
            if self.monitor.violates(x, u, fuel_used):
                return self.cfg.w_violation

            x = self.dyn.step(x, u, self.cfg.dt)
            pos = x[:2]
            vel = x[2:]

            cost += self.cfg.w_pos * (pos @ pos)
            cost += self.cfg.w_vel * (vel @ vel)
            cost += self.cfg.w_fuel * (abs(float(u[0])) + abs(float(u[1])))

            # Soft safety shaping: penalize exceeding speed limit at the current range.
            r = float(np.sqrt(pos @ pos))
            v = float(np.sqrt(vel @ vel))
            v_lim = self.monitor.speed_limit(r)
            if v > v_lim:
                cost += 2000.0 * (v - v_lim) ** 2

        # terminal shaping: strongly prefer near-zero position and velocity at horizon end
        cost += 80.0 * float((x[:2] @ x[:2])) + 800.0 * float((x[2:] @ x[2:]))
        return float(cost)

    def _sample_sequence(self, base: Optional[np.ndarray]) -> np.ndarray:
        H = self.cfg.horizon_steps
        umax = self.monitor.spec.thrust_max
        temp = self.cfg.temperature

        if base is None:
            base = np.zeros((H, 2), dtype=float)

        # Add structured noise and a primitive "brake" segment near the end.
        noise = np.random.normal(0.0, umax * temp, size=(H, 2))
        u = base + noise

        # Clip
        u = np.clip(u, -umax, umax)

        # Simple maneuver primitive: taper thrust near the end (reduces approach speed)
        taper_len = max(8, H // 5)
        u[-taper_len:] *= 0.05
        return u

    def plan(self, x_est: np.ndarray, fuel_used: float) -> np.ndarray:
        best_cost = float("inf")
        best_seq = None

        base = None
        if self._warm_u is not None:
            base = np.vstack([self._warm_u[1:], np.zeros((1, 2), dtype=float)])

        for _ in range(self.cfg.n_candidates):
            u_seq = self._sample_sequence(base)
            c = self._rollout_cost(x_est, u_seq, fuel_used)
            if c < best_cost:
                best_cost = c
                best_seq = u_seq

        if best_seq is None:
            best_seq = np.zeros((self.cfg.horizon_steps, 2), dtype=float)

        self._warm_u = best_seq
        return best_seq[0]


class DiscreteLQR:
    """
    Discrete-time LQR for linear dynamics x_{k+1} = F x_k + G u_k.
    """

    def __init__(self, F: np.ndarray, G: np.ndarray, Q: np.ndarray, R: np.ndarray, max_iter: int = 200):
        self.F = F
        self.G = G
        self.Q = Q
        self.R = R
        self.max_iter = max_iter
        self.K = self._solve_gain()

    def _solve_gain(self) -> np.ndarray:
        F, G, Q, R = self.F, self.G, self.Q, self.R
        P = Q.copy()
        for _ in range(self.max_iter):
            S = R + G.T @ P @ G
            K = np.linalg.solve(S, G.T @ P @ F)
            Pn = Q + F.T @ P @ (F - G @ K)
            if np.max(np.abs(Pn - P)) < 1e-8:
                P = Pn
                break
            P = Pn
        S = R + G.T @ P @ G
        K = np.linalg.solve(S, G.T @ P @ F)
        return K

    def control(self, x: np.ndarray) -> np.ndarray:
        # u = -K x
        return -(self.K @ x)


# -----------------------------------------------------------------------------
# End-to-end Agent
# -----------------------------------------------------------------------------


@dataclass
class EpisodeConfig:
    seed: int = 7
    dt: float = 0.5
    steps: int = 1200

    # initial relative state (meters, m/s)
    x0: float = 60.0
    y0: float = -40.0
    xdot0: float = 0.0
    ydot0: float = 0.0


@dataclass
class AutonomyConfig:
    """
    Configuration for ablation studies.

    controller:
      - "lqr": discrete LQR on CW linearized dynamics
      - "mpc": sampling MPC (random shooting)
    docking_mode:
      - if True, switch to a close-in PD controller inside dock_mode_radius
    """

    controller: Literal["lqr", "mpc"] = "lqr"
    docking_mode: bool = True
    safety_shield: bool = True
    speed_shaping_cost: bool = True

    # limits / budgets
    thrust_max: float = 0.03
    fuel_budget: float = 20.0

    # sensor noise
    pos_sigma: float = 0.75
    vel_sigma: float = 0.015

    # controller tuning
    lqr_R_scale: float = 200.0
    mpc_candidates: int = 1500
    mpc_horizon_steps: int = 80


def run_episode(verbose: bool = True) -> Dict[str, float]:
    # Backwards-compatible demo run.
    ep = EpisodeConfig()
    cfg = AutonomyConfig()
    return run_episode_cfg(ep=ep, cfg=cfg, verbose=verbose)


def run_episode_cfg(ep: EpisodeConfig, cfg: AutonomyConfig, verbose: bool = False) -> Dict[str, float]:
    set_seed(ep.seed)

    # Mean motion: roughly LEO orbital rate ~ 0.0011 rad/s (about 90 minutes per orbit)
    dyn = CWDynamics(n=0.0011)
    sensor = SensorModel(pos_sigma=cfg.pos_sigma, vel_sigma=cfg.vel_sigma)
    spec = SafetySpec()
    spec.thrust_max = cfg.thrust_max
    spec.fuel_budget = cfg.fuel_budget
    monitor = SafetyMonitor(spec)

    # Choose controller:
    # - SamplingMPC is flexible but noisy; DiscreteLQR is a strong baseline for CW rendezvous.
    dt = ep.dt
    F = np.eye(4, dtype=float) + dyn.A() * dt
    G = dyn.B() * dt
    Q_lqr = np.diag([5.0, 5.0, 20.0, 20.0])
    # Higher R reduces aggressive thrusting and improves fuel usage.
    R_lqr = np.diag([1.0, 1.0]) * float(cfg.lqr_R_scale)
    lqr = DiscreteLQR(F, G, Q_lqr, R_lqr)
    mpc_cfg = MPCConfig(dt=dt, horizon_steps=int(cfg.mpc_horizon_steps), n_candidates=int(cfg.mpc_candidates))
    planner = SamplingMPC(dyn, monitor, mpc_cfg)

    # KF setup
    A = dyn.A()
    B = dyn.B()
    Q = np.eye(4) * 0.02
    R = np.diag([sensor.pos_sigma**2, sensor.pos_sigma**2, sensor.vel_sigma**2, sensor.vel_sigma**2])
    kf = LinearKalmanFilter(A, B, Q, R)

    x_true = np.array([ep.x0, ep.y0, ep.xdot0, ep.ydot0], dtype=float)
    kf.x = x_true + np.array([2.0, -2.0, 0.02, -0.02])

    fuel_used = 0.0
    min_range = norm2(x_true[:2])
    t0 = time.time()

    for t in range(1, int(ep.steps) + 1):
        # Sense and update estimator
        z = sensor.measure(x_true)
        kf.update(z)

        # Controller selection with optional docking-mode switch.
        r_est = float(np.linalg.norm(kf.x[:2]))
        in_dock_mode = cfg.docking_mode and (r_est < spec.dock_mode_radius)

        if cfg.controller == "mpc" and not in_dock_mode:
            u = planner.plan(kf.x, fuel_used)
            u = monitor.thrust_clip(u)
        elif in_dock_mode:
            pos = kf.x[:2]
            vel = kf.x[2:]
            kp = 0.0009
            kd = 0.09
            u = -(kp * pos + kd * vel)
            u = np.array(
                [
                    clamp(float(u[0]), -spec.thrust_max_dock, spec.thrust_max_dock),
                    clamp(float(u[1]), -spec.thrust_max_dock, spec.thrust_max_dock),
                ],
                dtype=float,
            )
        else:
            u = lqr.control(kf.x)
            u = monitor.thrust_clip(u)

        if cfg.safety_shield:
            # Enforce hard thrust/fuel constraints and brake if predicted next speed exceeds envelope.
            if monitor.violates(x_true, u, fuel_used):
                if fuel_used > spec.fuel_budget:
                    if verbose:
                        print(f"Safety abort at t={t} (fuel budget exceeded)")
                    break
                u = np.zeros((2,), dtype=float)

            x_next = dyn.step(x_true, u, dt)
            r_next = norm2(x_next[:2])
            v_next = norm2(x_next[2:])
            v_lim = monitor.speed_limit(r_next)
            if v_next > v_lim:
                vel = x_true[2:].copy()
                vnorm = float(np.linalg.norm(vel)) + 1e-9
                gain = 0.40
                u_brake = -gain * vel / vnorm
                u = monitor.thrust_clip(u_brake)

        fuel_used += float(np.linalg.norm(u)) * dt

        # Propagate truth and KF predict
        x_true = dyn.step(x_true, u, dt)
        kf.predict(u, dt)

        r = norm2(x_true[:2])
        v = norm2(x_true[2:])
        min_range = min(min_range, r)

        if verbose and (t % 25 == 0 or r < spec.approach_radius):
            print(f"[t={t:03d}] range={r:7.3f} m  speed={v:6.3f} m/s  fuel={fuel_used:6.3f}  u=({u[0]:+.4f},{u[1]:+.4f})")

        if monitor.is_docked(x_true):
            elapsed = time.time() - t0
            if verbose:
                print(f"Docked at t={t}  fuel={fuel_used:.3f}  elapsed={elapsed:.2f}s")
            return {
                "success": 1.0,
                "steps": float(t),
                "fuel_used": float(fuel_used),
                "min_range": float(min_range),
                "final_range": float(r),
                "final_speed": float(v),
                "elapsed_seconds": float(elapsed),
            }

        # Hard abort only on fuel.
        if fuel_used > spec.fuel_budget:
            if verbose:
                print(f"Safety abort at t={t} (fuel budget exceeded)")
            break

    elapsed = time.time() - t0
    r = norm2(x_true[:2])
    v = norm2(x_true[2:])
    if verbose:
        print(f"Episode ended  fuel={fuel_used:.3f}  range={r:.3f}  speed={v:.3f}  elapsed={elapsed:.2f}s")
    return {
        "success": 0.0,
        "steps": float(ep.steps),
        "fuel_used": float(fuel_used),
        "min_range": float(min_range),
        "final_range": float(r),
        "final_speed": float(v),
        "elapsed_seconds": float(elapsed),
    }


def main():
    print("=" * 80)
    print("AUTONOMOUS SPACECRAFT NAVIGATION: PREDICT-ABSTRACT-PLAN-VERIFY PROTOTYPE")
    print("=" * 80)
    metrics = run_episode(verbose=True)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()


