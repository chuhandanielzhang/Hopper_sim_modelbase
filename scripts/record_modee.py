#!/usr/bin/env python3
"""
Record a clean, isolated demo of:

  Event-based hop phases (touchdown/compress/push/flight/apex)
    -> SRB WBC-QP (solve f_contact_w + tri-rotor thrusts, with actuator constraints)
    -> Joint torques via Hopper4-style mapping: tau = J^T * (-f_contact_w)
       (yaw not controlled)

This is "modee" (SRB-QP + joint torque mapping, no MPC).
"""

from __future__ import annotations

import os
import sys
import csv
import argparse
import subprocess
from dataclasses import dataclass

import numpy as np
import cv2
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "utils"))

# External reference controller (flight SE3 attitude), per user request:
#   /home/abc/Hopper/hopper_driver-master-reference/hopper_driver-master/Quadrotor_SE3_Control-main
_SE3_REF_DIR = "/home/abc/Hopper/hopper_driver-master-reference/hopper_driver-master/Quadrotor_SE3_Control-main"
_se3_geometry = None
_se3_controller = None
try:
    if os.path.isdir(_SE3_REF_DIR):
        sys.path.insert(0, _SE3_REF_DIR)
        import geometry as _se3_geometry  # type: ignore
        import se3_controller as _se3_controller  # type: ignore
except Exception:
    _se3_geometry = None
    _se3_controller = None

from utils.mujoco_interface import MuJoCoInterface
from controllers.motor_utils import MotorTableModel
from controllers.wbc_qp_osqp import WBCQP, WBCQPConfig
from controllers.mpc import MITCondensedWrenchMPC, MITCondensedWrenchMPCConfig

# Hopper4 delta/3-RSR kinematics (for MPC actuator feasibility on the real leg)
_HOPPER4_KIN_DIR = "/home/abc/Hopper/Hopping-Robot-master/hopper_controller"
_hopper4_inverse_jacobian_cls = None
try:
    if os.path.isdir(_HOPPER4_KIN_DIR):
        sys.path.insert(0, _HOPPER4_KIN_DIR)
        from forward_kinematics import InverseJacobian as _hopper4_inverse_jacobian_cls  # type: ignore
except Exception:
    _hopper4_inverse_jacobian_cls = None


def _vee_so3(A: np.ndarray) -> np.ndarray:
    """vee-map for a 3x3 skew-symmetric matrix."""
    A = np.asarray(A, dtype=float).reshape(3, 3)
    return np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=float)


def _Rz(yaw: float) -> np.ndarray:
    c = float(np.cos(float(yaw)))
    s = float(np.sin(float(yaw)))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _Rx(roll: float) -> np.ndarray:
    c = float(np.cos(float(roll)))
    s = float(np.sin(float(roll)))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _Ry(pitch: float) -> np.ndarray:
    c = float(np.cos(float(pitch)))
    s = float(np.sin(float(pitch)))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _serial_leg_fk_jac(q_roll: float, q_pitch: float, q_shift: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Encoder-only kinematics for the *serial-equivalent* leg in `mjcf/hopper_serial.xml`.

    NOTE:
    - This is NOT Hopper4's delta/3-RSR kinematics (see `Hopping-Robot-master/hopper_controller/forward_kinematics.py`).
      Hopper4's file models the real parallel delta leg, while our MuJoCo model here is the serial-equivalent roll/pitch/shift.
    - We keep kinematics independent from MuJoCo state (real-robot style), matching the "kinematics-only" intent.

    Returns:
      foot_b: (3,) foot body origin position in base frame
      J_body: (3,3) Jacobian mapping qdot -> foot_vrel_b in base frame
    """
    # From `hopper_serial.xml`:
    # base_link -> base_roll_link translation
    p0 = np.array([0.0, 0.0, -0.0416], dtype=float)
    # Leg_Link -> Foot_Link translation (body origin)
    foot_z = 0.5237

    Rr = _Rx(q_roll)
    Rp = _Ry(q_pitch)
    R = (Rr @ Rp).astype(float)

    # Vector from hip origin to foot (in pitch frame), then rotate to base frame.
    v = np.array([0.0, 0.0, float(q_shift) - float(foot_z)], dtype=float)
    foot_rel = (R @ v.reshape(3)).reshape(3)
    foot_b = (p0 + foot_rel).reshape(3)

    # Jacobian columns:
    # roll (about base X)
    axis_roll = np.array([1.0, 0.0, 0.0], dtype=float)
    # pitch axis is in the roll-rotated frame
    axis_pitch = (Rr @ np.array([0.0, 1.0, 0.0], dtype=float).reshape(3)).reshape(3)
    # prismatic axis is +Z in the roll/pitch frame
    axis_shift = R[:, 2].reshape(3)

    J0 = np.cross(axis_roll, foot_rel)
    J1 = np.cross(axis_pitch, foot_rel)
    J2 = axis_shift
    J_body = np.stack([J0, J1, J2], axis=1).astype(float)
    return foot_b, J_body


def _quat_to_R_wb(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in np.asarray(q_wxyz, dtype=float).reshape(4)]
    return np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ],
        dtype=float,
    )


def _quat_normalize(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return (q / n).astype(float)


def _quat_mul(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> np.ndarray:
    # Hamilton product, wxyz convention.
    w1, x1, y1, z1 = [float(v) for v in np.asarray(q1_wxyz, dtype=float).reshape(4)]
    w2, x2, y2, z2 = [float(v) for v in np.asarray(q2_wxyz, dtype=float).reshape(4)]
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _add_frame_axes_to_scene(
    scene: mujoco.MjvScene,
    origin_w: np.ndarray,
    R_wb: np.ndarray,
    axis_len_m: float = 0.12,
    width_px: float = 3.0,
    rgba_x: np.ndarray | None = None,
    rgba_y: np.ndarray | None = None,
    rgba_z: np.ndarray | None = None,
) -> None:
    """Add a 3-axis coordinate frame (x,y,z) to the current MuJoCo scene as colored line geoms.

    This is *visualization only* (no effect on physics/control). Coordinates are in WORLD.
    """
    try:
        origin = np.asarray(origin_w, dtype=float).reshape(3)
        R = np.asarray(R_wb, dtype=float).reshape(3, 3)
        L = float(max(0.0, float(axis_len_m)))
        if L <= 1e-9:
            return
        # Default colors: RGB for xyz
        if rgba_x is None:
            rgba_x = np.array([1.0, 0.0, 0.0, 0.90], dtype=np.float32)
        if rgba_y is None:
            rgba_y = np.array([0.0, 1.0, 0.0, 0.90], dtype=np.float32)
        if rgba_z is None:
            rgba_z = np.array([0.0, 0.3, 1.0, 0.90], dtype=np.float32)

        def _add_line(p0: np.ndarray, p1: np.ndarray, rgba: np.ndarray) -> None:
            if int(scene.ngeom) >= int(scene.maxgeom):
                return
            geom = scene.geoms[int(scene.ngeom)]
            size = np.zeros(3, dtype=np.float64)
            pos = np.zeros(3, dtype=np.float64)
            mat = np.zeros(9, dtype=np.float64)
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, size, pos, mat, np.asarray(rgba, dtype=np.float32).reshape(4))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, float(width_px), np.asarray(p0, dtype=np.float64).reshape(3), np.asarray(p1, dtype=np.float64).reshape(3))
            scene.ngeom = int(scene.ngeom) + 1

        ex = origin + L * R[:, 0]
        ey = origin + L * R[:, 1]
        ez = origin + L * R[:, 2]
        _add_line(origin, ex, rgba_x)
        _add_line(origin, ey, rgba_y)
        _add_line(origin, ez, rgba_z)
    except Exception:
        # Never break rendering due to debug visualization.
        return


def _quat_from_omega_dt(omega_b: np.ndarray, dt: float) -> np.ndarray:
    w = np.asarray(omega_b, dtype=float).reshape(3)
    th = float(np.linalg.norm(w) * float(dt))
    if th < 1e-12:
        # small-angle
        dq = np.array([1.0, 0.5 * float(w[0]) * float(dt), 0.5 * float(w[1]) * float(dt), 0.5 * float(w[2]) * float(dt)], dtype=float)
        return _quat_normalize(dq)
    axis = w / float(np.linalg.norm(w))
    half = 0.5 * th
    s = float(np.sin(half))
    return np.array([float(np.cos(half)), float(axis[0]) * s, float(axis[1]) * s, float(axis[2]) * s], dtype=float)


def _R_to_rpy_xyz(R_wb: np.ndarray) -> np.ndarray:
    R = np.asarray(R_wb, dtype=float).reshape(3, 3)
    # xyz (roll-pitch-yaw)
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    pitch = float(np.arcsin(float(np.clip(-R[2, 0], -1.0, 1.0))))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=float)


class SimpleIMUAttitudeEstimator:
    """
    Minimal 'real robot style' attitude estimator:
      - integrate gyro (body rates)
      - correct roll/pitch using accelerometer direction (complementary / Mahony-like, no mag)
    Yaw is unobservable -> will drift (that's OK; yaw is not controlled).
    """

    def __init__(
        self,
        *,
        kp_acc: float = 2.0,
        acc_g_min: float = 0.7,
        acc_g_max: float = 1.3,
        acc_lpf_tau: float = 0.20,
    ):
        self.kp_acc = float(kp_acc)
        self.acc_g_min = float(acc_g_min)
        self.acc_g_max = float(acc_g_max)
        self.acc_lpf_tau = float(acc_lpf_tau)
        self.q_wb = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # body->world
        self._inited = False
        self._acc_lp_b: np.ndarray | None = None

    def reset(self, q_wb: np.ndarray | None = None):
        self.q_wb = _quat_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if q_wb is None else q_wb)
        self._inited = True
        self._acc_lp_b = None

    def update(self, *, omega_b: np.ndarray, acc_b: np.ndarray, dt: float) -> np.ndarray:
        dt = float(dt)
        omega = np.asarray(omega_b, dtype=float).reshape(3)
        acc = np.asarray(acc_b, dtype=float).reshape(3)
        q = np.asarray(self.q_wb, dtype=float).reshape(4)
        q = _quat_normalize(q)

        # Initialize roll/pitch from accelerometer (yaw=0) if not yet initialized.
        if not bool(self._inited):
            n = float(np.linalg.norm(acc))
            if n > 1e-6:
                a = acc / n
                # We want body +Z to align with measured 'up' direction a (since at rest acc≈+g*z_b).
                # Construct a rotation that maps body z to a, with yaw=0 preference.
                # Use simple roll/pitch from a:
                pitch = float(np.arctan2(-a[0], max(1e-9, float(np.sqrt(a[1] * a[1] + a[2] * a[2])))))
                roll = float(np.arctan2(a[1], a[2]))
                yaw = 0.0
                cr, sr = float(np.cos(roll * 0.5)), float(np.sin(roll * 0.5))
                cp, sp = float(np.cos(pitch * 0.5)), float(np.sin(pitch * 0.5))
                cy, sy = float(np.cos(yaw * 0.5)), float(np.sin(yaw * 0.5))
                # xyz
                q = np.array(
                    [
                        cr * cp * cy + sr * sp * sy,
                        sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy,
                    ],
                    dtype=float,
                )
                q = _quat_normalize(q)
                self._inited = True

        # Accelerometer correction (roll/pitch only).
        # IMPORTANT: use a low-pass filtered accelerometer for gravity direction; raw acc includes
        # large non-gravitational components during hopping ("shooting") and can destabilize rpy_hat.
        omega_corr = omega.copy()
        # update LPF
        if self._acc_lp_b is None or (not np.all(np.isfinite(self._acc_lp_b))):
            self._acc_lp_b = acc.copy()
        tau = float(max(1e-6, self.acc_lpf_tau))
        a_lpf = float(np.clip(dt / (tau + dt), 0.0, 1.0))
        self._acc_lp_b = (1.0 - a_lpf) * self._acc_lp_b + a_lpf * acc

        acc_use = np.asarray(self._acc_lp_b, dtype=float).reshape(3)
        nacc = float(np.linalg.norm(acc_use))
        if nacc > 1e-6 and np.all(np.isfinite(acc_use)):
            g = 9.81
            ratio = float(nacc / g)
            if (ratio >= self.acc_g_min) and (ratio <= self.acc_g_max):
                a_dir_b = acc_use / nacc
                R = _quat_to_R_wb(q)
                # predicted up direction in body: R_bw * e_z
                up_pred_b = (R.T @ np.array([0.0, 0.0, 1.0], dtype=float)).reshape(3)
                # Mahony-style: error direction is measured x estimated (drives estimate toward measurement).
                err = np.cross(a_dir_b, up_pred_b)
                omega_corr = omega_corr + float(self.kp_acc) * err

        # Integrate
        dq = _quat_from_omega_dt(omega_corr, dt)
        q = _quat_mul(q, dq)
        q = _quat_normalize(q)
        self.q_wb = q
        return q.copy()

def _export_h264(path_mp4: str) -> str | None:
    try:
        out_h264 = path_mp4.replace(".mp4", "_h264.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                path_mp4,
                "-c:v",
                "libx264",
                "-profile:v",
                "baseline",
                "-level",
                "3.0",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                out_h264,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return out_h264 if os.path.exists(out_h264) else None
    except Exception:
        return None


@dataclass
class ModeEConfig:
    # recording
    duration_s: float = 20.0
    fps: int = 50
    width: int = 1280
    height: int = 720
    log_hz: float = 100.0

    # command
    vx_cmd: float = 0.0
    vy_cmd: float = 0.0

    # hop target
    # NOTE: in stance we intentionally compress before push-off, so liftoff height is typically a bit lower than
    # the touchdown height. Using a slightly lower z0 makes the takeoff velocity target large enough to reach
    # apex≈0.80m consistently.
    hop_z0: float = 0.55
    hop_peak_z: float = 0.81
    stance_T: float = 0.38
    stance_min_T: float = 0.10
    flight_min_T: float = 0.10

    # contact-eff thresholds (world foot height)
    # NOTE (real robot): no foot contact sensor.
    # Phase switching uses ONLY leg length + leg speed (encoder-based).
    # The following gates are on the prismatic "shift" joint:
    # - q_shift increases -> leg SHORTENS (compression)
    # - q_shift decreases -> leg LENGTHENS (extension)
    # Touchdown gates (leg-only).
    # Keep a strict "near full extension" gate so we don't false-trigger in mid-air.
    # We lower the qd_shift threshold so we don't miss soft landings (important for deep compression).
    td_q_shift_gate: float = 0.25    # m
    # Touchdown detection is purely leg-based (no foot sensor). In practice the prismatic velocity spike can be
    # small (especially with strong flight extension PD), so keep these thresholds permissive but gated by
    # td_q_shift_gate (leg near extension).
    td_qd_shift_gate: float = 0.01   # m/s
    td_dq_shift_gate: float = 5e-5   # m, debounce on q_shift increase
    # Touchdown should not trigger while the base is still moving upward in flight.
    # Use IMU-propagated vertical velocity to reject mid-air false touchdowns caused by swing-leg motion.
    td_vz_hat_max: float = 0.00      # m/s
    # (Optional) extra touchdown gates could be added here if needed, but the default implementation
    # intentionally uses ONLY leg length/speed (no foot sensors).
    # Liftoff should only happen when the leg is VERY close to full extension; otherwise we may switch
    # to flight too early while the foot is still loaded, and the flight extension PD will create a huge impulse
    # (causing apex >> 0.8m).
    lo_q_shift_gate: float = 0.05    # m, near full extension -> allow liftoff
    lo_qd_shift_gate: float = -0.20  # m/s, extension speed threshold for liftoff

    # pre-compression (stroke)
    # For forward running, keep compression moderate; deep compression drives very large fz spikes and roll/pitch coupling.
    # User requirement: compression depth should NOT exceed 15cm.
    compress_depth_m: float = 0.15
    # Give compression more of the stance budget so we can reliably reach the target stroke.
    compress_s_max: float = 0.60
    # With moderate stroke, touchdown vertical speed can be large; keep v_des moderately aggressive
    # so we can reach the target without bottoming out.
    compress_v_gain: float = 10.0
    compress_v_max: float = 1.5
    compress_kv: float = 15.0
    # Allow a bit more downward accel so the leg can reach the target stroke before push-off.
    az_compress_mag: float = 8.0
    az_compress_brake_max: float = 12.0
    # If the leg starts extending (qd_shift sufficiently negative), end compression early and start push.
    # This avoids commanding "more compression" while the leg is already rebounding, which can drive fz -> 0
    # and destabilize roll/pitch.
    compress_end_qd_shift: float = -0.05  # m/s
    z_guard: float = 0.35

    # push impulse shaping (vertical)
    impulse_shape: str = "sin2"  # "sin" or "sin2"
    dv_cmd_max: float = 1.60
    # Avoid collapsing takeoff speed too low (causes “no hop / sink”).
    v_to_min: float = 1.60
    v_to_max: float = 2.20

    # swing foot (flight)
    swing_clear_z: float = 0.08

    # Raibert S2S (step-to-step) touchdown foot placement for forward/lateral speed tracking.
    # We compute a desired touchdown horizontal offset in WORLD frame, then rotate into BODY frame
    # using the IMU quaternion, and command hip roll/pitch to realize it.
    use_s2s: bool = True
    s2s_k: float = 0.05  # [s] gain on (v - v_des)
    # Optional feedforward term scale on v*(T/2). For force/MPC-driven hopping, set 0 to avoid over-aggressive
    # placement; keep only the feedback term k*(v - v_des).
    # Use the classic Raibert feedforward term ~ v*(T/2) to keep the foot landing ahead as speed increases.
    # Without this term, the single-leg hopper tends to land "behind" at higher speed, driving pitch-up and runaway vx.
    s2s_ff_scale: float = 1.0
    # Lateral feedforward scale on v_y*(T/2). Helps arrest sustained vy drift without relying purely on a small
    # feedback gain (which was producing only ~cm-level offsets right before the fall).
    s2s_ff_scale_y: float = 0.7
    s2s_step_lim: float = 0.20  # [m] clamp on |x|,|y| touchdown offset
    # Lateral (y) foot placement: use a smaller feedback-only correction (no v*T/2 feedforward) to avoid
    # injecting large roll moments during high-fz compression/push, while still cancelling vy drift.
    # NOTE: v58 shows vy reaching ~-0.9m/s while off_y only ~2-3cm, which is too weak to arrest lateral drift.
    # Increase lateral authority, but keep it moderate to avoid aggressive mid-air retargeting that can
    # amplify slip/estimation error.
    s2s_step_lim_y: float = 0.18
    s2s_ky_scale: float = 1.20
    s2s_lpf_tau: float = 0.06  # [s] low-pass on touchdown offset command
    s2s_max_tilt_deg: float = 25.0  # [deg] clamp on commanded hip roll/pitch

    # props
    # Running: a bit more total thrust gives more differential moment authority for roll/pitch,
    # while still far below mg (assist-only, never becomes a quadrotor).
    prop_base_thrust_ratio: float = 0.22

    # Horizontal velocity regulation (stance). Keep moderate; aggressive values cause slip and estimator
    # divergence. When MPC vx/vy tracking is enabled, MPC should provide most of the horizontal authority.
    axy_damp: float = 0.5

    # PI-style velocity correction (helps remove accumulated drift).
    # NOTE: for vx_cmd=0.8 this integrator can wind up and flip the sign of fy in stance,
    # causing vy runaway and roll blow-up. Keep it OFF; use damping + S2S instead.
    ki_xy: float = 0.0
    v_int_max: float = 0.30

    # Torque smoothing time constant for the unified QP torque output (flight swing task vs stance).
    # Smaller -> more responsive, larger -> smoother (less ms-level jitter at liftoff/touchdown).
    tau_ref_tau: float = 0.0  # [s]

    # Velocity estimator (stance): scale the no-slip contact fusion on vx.
    # 1.0 means full fusion (same weight as vy/vz). Lower values can help if forward slip biases the
    # no-slip measurement, but too low will cause IMU drift. Keep 1.0 by default.
    v_fuse_vx_scale: float = 1.0

    # Yaw: do NOT control yaw angle (and do not demand yaw torque by default).
    # NOTE: MODEE intentionally does NOT include any dedicated yaw suppression controller.
    yaw_rate_damp: float = 0.0  # [N*m / (rad/s)]
    tau_yaw_max: float = 0.0   # [N*m]

    # MPC (SRB condensed wrench MPC)
    mpc_dt: float = 0.02
    mpc_N: int = 12


def _make_mpc(cfg: ModeEConfig) -> MITCondensedWrenchMPC:
    # SRB condensed wrench MPC (MIT-style).
    # Yaw is NOT controlled: yaw weights = 0, yaw rate ref = 0.
    return MITCondensedWrenchMPC(
        MITCondensedWrenchMPCConfig(
            dt=float(cfg.mpc_dt),
            N=int(cfg.mpc_N),
            mu=0.8,
            # In MPC we run mainly during push; keep a higher minimum normal force to keep stance robust
            # (more friction margin for "leg strongly controls attitude" under forward command).
            # Lower fz_min so the stance doesn't inject excessive upward impulse (which makes apex > 0.8m).
            fz_min=28.0,
            fz_max=220.0,
            # For running (vx_cmd ~0.8): allow enough horizontal GRF so MPC can reject lateral drift (vy)
            # and avoid frequent infeasibility. This bound is on |fxy| relative to (m*g).
            fxy_max_ratio=0.12,
            w_py=0.0,
            w_pz=80.0,
            w_vx=8.0,
            w_vy=16.0,
            w_vz=220.0,
            # Let WBC/QP handle roll/pitch; yaw remains free in MPC.
            w_roll=0.0,
            w_pitch=0.0,
            w_yaw=0.0,
            w_wx=0.0,
            w_wy=0.0,
            w_wz=0.0,
            w_tsum_ref=0.0,
            # Make MPC leg-torque aware (approx): discourage GRFs that drive large joint torques.
            # We keep this soft by default; hard torque constraints can be enabled later if desired.
            w_tau=0.0,
            enforce_tau_limits=True,
            alpha_u_f=1e-4,
            alpha_u_t=6e-4,
            rp_limit_deg=None,
            re_setup_each_solve=True,
        )
    )


class ModeESim:
    def __init__(self, cfg: ModeEConfig):
        self.cfg = cfg
        self.gravity = 9.81

        model_path = os.path.join(PROJECT_ROOT, "mjcf", "hopper_serial.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.interface = MuJoCoInterface(self.model, self.data)

        # IMU attitude estimator (no mag): body->world quaternion estimate
        # For hopping, raw accelerometer contains large non-gravity components; use a conservative gravity correction.
        self.att = SimpleIMUAttitudeEstimator(kp_acc=0.6, acc_g_min=0.90, acc_g_max=1.10, acc_lpf_tau=0.25)

        # SE3 attitude controller (reference implementation).
        # We use it for roll/pitch stabilization in ALL phases (compression/push/flight).
        # Yaw remains free: we set the desired yaw to the CURRENT yaw and zero any yaw torque output.
        self.se3_att = None
        if _se3_controller is not None:
            try:
                self.se3_att = _se3_controller.SE3Controller()
                # Disable position/velocity outer loop inside the reference controller; we only use its SO(3) part.
                self.se3_att.kx = 0.0
                self.se3_att.kv = 0.0
            except Exception:
                self.se3_att = None

        # Nominal foot vector in base frame at neutral leg angles (used for S2S target z).
        self._foot_b_nom, _ = _serial_leg_fk_jac(0.0, 0.0, 0.0)

        # S2S touchdown offset command state (BODY XY)
        self._s2s_off_b = np.zeros(2, dtype=float)
        # Persist the last S2S desired foot position (base frame) so touchdown rows can be compared
        # against the *commanded* landing point from the preceding flight.
        self._foot_b_des_last = np.asarray(self._foot_b_nom, dtype=float).copy()
        self._s2s_active_last = False
        # Apex reached flag: only start swing leg (S2S) after apex is detected
        self._apex_reached = False

        self.dt = float(self.model.opt.timestep)
        self.sim_time = 0.0

        # motor table (for thrust caps + PWM realism)
        self.motor_table = MotorTableModel.default_from_table()
        self.max_thrust_per_motor = float(np.max(self.motor_table.thrust_n_bp))
        # total mass for base thrust target
        self.total_mass = float(np.sum(self.model.body_mass))
        self.base_thrust = float(cfg.prop_base_thrust_ratio * self.total_mass * self.gravity)

        # 3-RSR (delta) inverse Jacobian (Hopper4) for MPC torque feasibility.
        # This is used ONLY to constrain MPC's GRF so that the *real* 3-RSR actuators (10 Nm each) are feasible,
        # while the MuJoCo simulation still uses the serial-equivalent roll/pitch/shift leg.
        self._kin3rsr = None
        try:
            if _hopper4_inverse_jacobian_cls is not None:
                self._kin3rsr = _hopper4_inverse_jacobian_cls(use_simulink_filter=False, forgetting_factor=0.95, dt=float(self.dt))
        except Exception:
            self._kin3rsr = None

        # SRB-QP WBC (solve for: contact force f + tri-rotor thrusts t, with actuator constraints)
        thrust_ratio_hw_max = float((3.0 * float(self.max_thrust_per_motor)) / max(1e-9, float(self.total_mass) * float(self.gravity)))
        self.wbc = WBCQP(
            WBCQPConfig(
                mu=1.5,
                fz_min=0.0,
                fz_max=220.0,
                # Allow enough horizontal GRF to actually cancel roll/pitch moments (single-leg) and reject vy drift.
                fxy_abs_max=4.0,
                # Keep each rotor spinning a bit to preserve roll/pitch moment authority in flight.
                thrust_min_each=0.5,
                thrust_total_ratio_max=thrust_ratio_hw_max,
                # For strong stance attitude: prioritize torque tracking, allow limited force mismatch if needed.
                # Allow horizontal force mismatch so the solver can choose fy needed for roll/pitch torque
                # without being forced to track a potentially conflicting F_des_xy during stance.
                w_slack_Fxy=1e4,
                w_slack_Fz=1e9,
                # Torque slack: keep high so the solver respects Tau_des for roll/pitch stabilization.
                # (Yaw is still "uncontrolled" because Tau_des.z=0; but we keep the yaw torque equation tight
                # enough to avoid runaway yaw acceleration from reaction torques.)
                # Allow some torque mismatch to prevent the solver from generating extreme horizontal GRF
                # (fy saturation) just to satisfy Tau_des, which can destabilize higher-speed runs.
                w_slack_tau=1e5,
                w_slack_tau_flight=1e5,
                # In flight, relax force dynamics so the QP can temporarily increase total thrust (within bounds)
                # to gain attitude moment authority.
                w_slack_Fxy_flight=1e3,
                w_slack_Fz_flight=1e3,
                w_f=1e-4,
                w_t=1e-4,
                w_f_ref=5.0,
                # Do NOT strongly track a thrust split; let the QP allocate for attitude.
                # Add a small thrust smoothing term to reduce liftoff PWM chatter.
                w_t_ref=1e-3,
                # Keep this small; we need enough flight thrust range to hit apex≈0.80m at vx_cmd=0.8.
                w_tsum_ref=1e-2,
                # Joint torque smoothing/tracking (normalized by tau_cmd_max inside the QP).
                w_tau_ref=200.0,
            )
        )

        # MPC (SRB condensed wrench MPC)
        self.mpc = _make_mpc(cfg)
        self._mpc_last_t = -1e9
        self._mpc_f0 = np.zeros(3, dtype=float)
        self._mpc_t0 = np.ones(3, dtype=float) * (float(self.base_thrust) / 3.0)
        self._mpc_status = "init"
        self._mpc_mode = "init"  # "compress" | "push" (for avoiding stale MPC output across phase switches)

        # WBC fallback (robustness): hold last feasible wrench solution if OSQP fails.
        self._wbc_last_f = np.zeros(3, dtype=float)
        self._wbc_last_t = np.ones(3, dtype=float) * (float(self.base_thrust) / 3.0)

        # phase state
        self._stance = False
        self._td_t: float | None = None
        self._lo_t: float | None = None
        # touchdown height estimate (world z) from leg kinematics (assume foot on ground)
        self._td_z_est: float = float(cfg.hop_z0)
        self._vz_td: float = 0.0
        self._q_shift_td: float | None = None
        self._q_shift_prev: float | None = None  # pre-impact latch (1kHz)
        self._qd_shift_prev: float | None = None
        self._push_started: bool = False
        self._t_push: float = 0.0
        self._T_push: float = float(cfg.stance_T)
        self._vz_push: float = 0.0
        self._prev_vz: float | None = None

        dz0 = max(1e-6, float(cfg.hop_peak_z) - float(cfg.hop_z0))
        self._v_to_cmd = float(np.sqrt(2.0 * self.gravity * dz0))
        # filtered contact force for torque mapping
        self._f_contact_w_filt: np.ndarray | None = None
        # Previous commanded joint torque (for smooth tau_ref blending across phase switches)
        self._tau_cmd_prev = np.zeros(3, dtype=float)
        # --- step-by-step "real sensor pipeline" ---
        # Start with a simple velocity estimate:
        #   - stance: use contact constraint + leg kinematics (no-slip) to estimate base velocity
        #   - flight: integrate IMU specific force to propagate velocity
        self._v_hat_w = np.zeros(3, dtype=float)
        self._v_hat_inited = False
        # Contact-velocity fusion time constant.
        # NOTE: too small -> noise/jerk in v_hat; too large -> drift (especially vy_hat) over many hops.
        self._v_hat_lpf_tau = 0.02  # seconds
        # Integral of horizontal velocity error (world XY). Helps remove slow drift at higher vx_cmd.
        self._v_int_xy = np.zeros(2, dtype=float)
        self._p_hat_w = np.array([0.0, 0.0, float(cfg.hop_z0)], dtype=float)
        self._z_hat_contact_filt: float | None = None

    def _touchdown_ok(self) -> bool:
        if self._lo_t is None:
            return True
        return (float(self.sim_time) - float(self._lo_t)) >= float(self.cfg.flight_min_T)

    def _liftoff_ok(self) -> bool:
        if self._td_t is None:
            return True
        return (float(self.sim_time) - float(self._td_t)) >= float(self.cfg.stance_min_T)

    def step(self, desired_v: np.ndarray) -> tuple[dict, np.ndarray, dict]:
        desired_v = np.asarray(desired_v, dtype=float).reshape(2)
        state = self.interface.get_state()

        imu_gyro_b = np.asarray(state.get("imu_gyro", state.get("body_ang_vel", np.zeros(3))), dtype=float).reshape(3)
        imu_acc_b = np.asarray(state.get("imu_acc", np.zeros(3)), dtype=float).reshape(3)  # specific force in body

        # ----- Full sensor pipeline (control uses estimates only) -----
        # Keep MuJoCo truth for logging/debug only (do not use for control).
        pos_true = np.asarray(state.get("body_pos", np.zeros(3)), dtype=float).reshape(3)
        vel_true = np.asarray(state.get("body_vel", np.zeros(3)), dtype=float).reshape(3)
        rpy_true = np.asarray(state.get("body_rpy", np.zeros(3)), dtype=float).reshape(3)

        # IMU attitude estimate (body->world)
        q_hat = self.att.update(omega_b=imu_gyro_b, acc_b=imu_acc_b, dt=float(self.dt))
        R_wb_hat = _quat_to_R_wb(q_hat)
        rpy_hat = _R_to_rpy_xyz(R_wb_hat)
        z_w = np.asarray(R_wb_hat[:, 2], dtype=float).reshape(3)

        # shift (stroke proxy)
        try:
            qj = np.asarray(state.get("joint_pos", np.zeros(3)), dtype=float).reshape(-1)
            q_shift = float(qj[2]) if qj.size >= 3 else float("nan")
        except Exception:
            q_shift = float("nan")
        try:
            qd = np.asarray(state.get("joint_vel", np.zeros(3)), dtype=float).reshape(-1)
            qd_shift = float(qd[2]) if qd.size >= 3 else float("nan")
        except Exception:
            qd_shift = float("nan")

        # Encoder-only foot kinematics (no MuJoCo "truth" kinematics).
        q_roll = float(qj[0]) if qj.size >= 1 else 0.0
        q_pitch = float(qj[1]) if qj.size >= 2 else 0.0
        q_shift_val = float(qj[2]) if qj.size >= 3 else 0.0
        qd_roll = float(qd[0]) if qd.size >= 1 else 0.0
        qd_pitch = float(qd[1]) if qd.size >= 2 else 0.0
        qd_shift_val = float(qd[2]) if qd.size >= 3 else 0.0

        foot_b, J_body = _serial_leg_fk_jac(q_roll, q_pitch, q_shift_val)
        qd_leg = np.array([qd_roll, qd_pitch, qd_shift_val], dtype=float)
        foot_vrel_b = (J_body @ qd_leg.reshape(3)).reshape(3)

        # For logging only: estimated foot height in world
        foot_w_hat = (np.asarray(self._p_hat_w, dtype=float).reshape(3) + (R_wb_hat @ foot_b.reshape(3))).reshape(3)
        foot_h = float(foot_w_hat[2])

        # ===== IMU propagation (needed for touchdown gating) =====
        g_w = np.array([0.0, 0.0, -float(self.gravity)], dtype=float)
        a_w = (R_wb_hat @ imu_acc_b.reshape(3)) + g_w
        if not bool(self._v_hat_inited):
            self._v_hat_w = np.zeros(3, dtype=float)
            self._v_hat_inited = True
        v_pred = (np.asarray(self._v_hat_w, dtype=float).reshape(3) + a_w * float(self.dt)).reshape(3)

        touchdown_evt = False
        liftoff_evt = False
        apex_evt = False

        # Apex event will be computed AFTER we update vz_hat (IMU+leg-based velocity estimate),
        # to avoid relying on MuJoCo truth velocity for the "real sensor pipeline".

        # ===== touchdown detection (NO foot sensor): leg length + leg speed only =====
        # Touchdown signature: shift joint starts moving in the + direction (compression) with sufficient speed.
        if (not bool(self._stance)) and self._touchdown_ok():
            dq = 0.0
            if (self._q_shift_prev is not None) and np.isfinite(float(self._q_shift_prev)) and np.isfinite(q_shift):
                dq = float(q_shift - float(self._q_shift_prev))
            if np.isfinite(q_shift) and np.isfinite(qd_shift):
                # Only allow touchdown when the leg is already near-max-extended.
                # This avoids false touchdown triggers due to commanded prismatic motion in mid-air.
                if (
                    (float(q_shift) <= float(self.cfg.td_q_shift_gate))
                    and (float(qd_shift) > float(self.cfg.td_qd_shift_gate))
                    and (float(dq) > float(self.cfg.td_dq_shift_gate))
                    and (float(v_pred[2]) <= float(self.cfg.td_vz_hat_max))
                ):
                    touchdown_evt = True
                    self._stance = True
                    self._td_t = float(self.sim_time)
                    # Reset apex flag on touchdown (new hop cycle)
                    self._apex_reached = False
                    # Touchdown height estimate from leg kinematics (assume foot is on the ground plane).
                    z_td_est = -float((R_wb_hat @ foot_b.reshape(3))[2])
                    self._td_z_est = float(z_td_est)
                    # Correct vertical position estimate at touchdown (contact provides a strong measurement).
                    try:
                        self._p_hat_w[2] = float(self._td_z_est)
                        self._z_hat_contact_filt = float(self._td_z_est)
                    except Exception:
                        pass
                    # Recompute desired takeoff speed for the target apex each hop:
                    # ballistic: v_to = sqrt(2*g_eff*(z_apex - z_td))
                    m_tot = float(self.total_mass)
                    g_eff = float(self.gravity - (float(z_w[2]) * float(self.base_thrust)) / max(1e-6, m_tot))
                    g_eff = float(max(1e-3, g_eff))
                    dz_tgt = float(max(0.0, float(self.cfg.hop_peak_z) - float(self._td_z_est)))
                    v_to = float(np.sqrt(2.0 * g_eff * dz_tgt))
                    self._v_to_cmd = float(np.clip(v_to, float(self.cfg.v_to_min), float(self.cfg.v_to_max)))

                    self._vz_td = float(self._v_hat_w[2]) if bool(self._v_hat_inited) else 0.0
                    # pre-impact latch for shift (avoid instant touchdown spike)
                    if (self._q_shift_prev is not None) and np.isfinite(float(self._q_shift_prev)):
                        self._q_shift_td = float(self._q_shift_prev)
                    else:
                        self._q_shift_td = float(q_shift) if np.isfinite(q_shift) else None
                    self._push_started = False
                    self._vz_push = float(self._v_hat_w[2]) if bool(self._v_hat_inited) else 0.0
                    self._t_push = float(self.sim_time)
                    self._T_push = float(self.cfg.stance_T)

        # track previous shift every 1kHz (for the pre-impact latch)
        if np.isfinite(q_shift):
            self._q_shift_prev = float(q_shift)
        if np.isfinite(qd_shift):
            self._qd_shift_prev = float(qd_shift)

        # ===== liftoff detection (NO foot sensor): leg length only (STRICT l0) =====
        # User-defined: l0 ≡ maximum leg length in this model, which corresponds to q_shift == 0 (joint lower limit).
        # We switch to FLIGHT ONLY when the leg has re-extended back to l0 (no epsilon margin).
        if bool(self._stance) and self._liftoff_ok() and np.isfinite(q_shift):
            if float(q_shift) <= 0.0:
                liftoff_evt = True
                self._stance = False
                self._lo_t = float(self.sim_time)

        # ===== Step-1 estimator: base velocity v_hat (world) =====
        # Real-robot style: always propagate with IMU, then (in stance) fuse a no-slip contact measurement.
        # When the foot is slipping, the no-slip constraint becomes invalid; detect slip and down-weight it.
        if bool(self._stance):
            # stance measurement: v_base_body ≈ -(v_foot_rel + ω×r_foot)  (no-slip contact)
            v_meas_b = -(foot_vrel_b + np.cross(imu_gyro_b, foot_b))
            v_meas_w = (R_wb_hat @ v_meas_b.reshape(3)).reshape(3)

            # slip metric: predicted foot velocity in world should be ~0 if no-slip holds
            omega_w_hat = (R_wb_hat @ imu_gyro_b.reshape(3)).reshape(3)
            # 只使用横向(roll/pitch)角速度判据；纯 yaw 自旋不应被当成“打滑”。
            omega_noyaw = omega_w_hat.copy()
            omega_noyaw[2] = 0.0
            r_foot_w_rel = (R_wb_hat @ foot_b.reshape(3)).reshape(3)
            v_foot_w_pred = (
                v_pred
                + np.cross(omega_noyaw, r_foot_w_rel)
                + (R_wb_hat @ foot_vrel_b.reshape(3))
            ).reshape(3)
            # Slip metric: predicted world foot velocity should be ~0 if no-slip holds.
            # We already removed pure yaw-spin contribution via omega_noyaw.
            # Keep full 3D norm here (more robust to contact transients than XY-only).
            slip_speed = float(np.linalg.norm(v_foot_w_pred))
            # If slip_ref is too small we almost always down-weight the contact constraint and v_hat drifts.
            slip_ref = 1.0  # [m/s]
            gate = float(np.clip(np.exp(-slip_speed / max(1e-6, slip_ref)), 0.20, 1.0))

            tau = float(self._v_hat_lpf_tau)
            a = float(np.clip(float(self.dt) / (tau + float(self.dt)), 0.0, 1.0)) if tau > 1e-9 else 1.0
            a_eff = float(a * gate)
            # Component-wise contact fusion: keep vy/vz tightly fused (for lateral drift suppression and
            # kinematic z correction), but optionally reduce/disable vx fusion to avoid forward-slip bias.
            vx_scale = float(np.clip(float(getattr(self.cfg, "v_fuse_vx_scale", 1.0)), 0.0, 1.0))
            a_eff_vx = float(a_eff * vx_scale)
            v_new = np.asarray(v_pred, dtype=float).reshape(3).copy()
            v_new[0] = float((1.0 - a_eff_vx) * float(v_pred[0]) + a_eff_vx * float(v_meas_w[0]))
            v_new[1] = float((1.0 - a_eff) * float(v_pred[1]) + a_eff * float(v_meas_w[1]))
            v_new[2] = float((1.0 - a_eff) * float(v_pred[2]) + a_eff * float(v_meas_w[2]))
            self._v_hat_w = v_new.reshape(3)

            # PI velocity integrator update (stance only, slip-gated)
            try:
                err_xy = (np.asarray(desired_v, dtype=float).reshape(2) - np.asarray(self._v_hat_w[0:2], dtype=float).reshape(2)).reshape(2)
                v_int_max = float(max(0.0, float(self.cfg.v_int_max)))
                # Only integrate VY error to suppress lateral drift. Integrating VX tends to cause overshoot spikes at vx_cmd=0.8.
                self._v_int_xy[1] = float(self._v_int_xy[1] + float(err_xy[1]) * float(self.dt) * float(gate))
                self._v_int_xy[1] = float(np.clip(self._v_int_xy[1], -v_int_max, +v_int_max))
                self._v_int_xy[0] = 0.0
            except Exception:
                pass
        else:
            self._v_hat_w = v_pred.copy()
            # flight: slowly decay the integrator so it doesn't wind up in the air
            self._v_int_xy = (0.995 * self._v_int_xy).astype(float)

        # position estimate: integrate v_hat; in stance, correct z using kinematic contact constraint
        self._p_hat_w = self._p_hat_w + self._v_hat_w * float(self.dt)
        if bool(self._stance):
            z_meas = -float((R_wb_hat @ foot_b.reshape(3))[2])  # assume foot is on ground (z=0)
            if self._z_hat_contact_filt is None:
                self._z_hat_contact_filt = float(z_meas)
            z_tau = 0.05
            az = float(np.clip(float(self.dt) / (z_tau + float(self.dt)), 0.0, 1.0))
            self._z_hat_contact_filt = (1.0 - az) * float(self._z_hat_contact_filt) + az * float(z_meas)
            self._p_hat_w[2] = float(self._z_hat_contact_filt)

        # apex detection (sensor-style): use vz_hat sign change in flight
        vz_hat = float(self._v_hat_w[2])
        if self._prev_vz is None:
            self._prev_vz = float(vz_hat)
        if (not bool(self._stance)) and (float(self._prev_vz) > 0.0) and (float(vz_hat) <= 0.0):
            apex_evt = True
            # Mark apex as reached: now allow swing leg (S2S) to start
            self._apex_reached = True
        self._prev_vz = float(vz_hat)

        # ===== stance: compression -> push =====
        az_des = -self.gravity  # default (flight)
        compress_active = False
        depth_now = 0.0
        # Hard safety cap: user requirement says compression must not exceed 15cm.
        depth_cap = 0.15
        depth_tgt = float(np.clip(float(self.cfg.compress_depth_m), 0.0, float(depth_cap)))
        # Practical guard band to prevent tiny overshoot due to discrete-time update / touchdown inertia.
        # This keeps measured comp_m <= 0.15m in practice.
        depth_guard = 0.010
        depth_tgt_act = float(max(0.0, depth_tgt - depth_guard))

        if bool(self._stance):
            t_td = float(self._td_t) if (self._td_t is not None) else float(self.sim_time)
            s = float(np.clip((float(self.sim_time) - t_td) / max(1e-6, float(self.cfg.stance_T)), 0.0, 1.0))

            # compression depth from shift (IMPORTANT: q_shift increases -> compression / shorter leg)
            q_shift_td = self._q_shift_td
            if q_shift_td is None or (not np.isfinite(float(q_shift_td))) or (not np.isfinite(q_shift)):
                depth_now = 0.0
            else:
                depth_now = float(max(0.0, float(q_shift) - float(q_shift_td)))
                # cap target by remaining stroke to joint max
                q_max = 0.4
                depth_tgt = float(min(depth_tgt, max(0.0, float(q_max) - float(q_shift_td))))

            z_now = float(self._p_hat_w[2])
            if (not bool(self._push_started)) and (depth_tgt > 1e-6):
                extending = bool(np.isfinite(qd_shift)) and (float(qd_shift) < float(self.cfg.compress_end_qd_shift))
                if (z_now >= float(self.cfg.z_guard)) and (s < float(self.cfg.compress_s_max)) and (depth_now < depth_tgt_act) and (not extending):
                    compress_active = True

            if compress_active:
                err = float(depth_tgt - depth_now)
                v_des = float(np.clip(-float(self.cfg.compress_v_gain) * err, -abs(float(self.cfg.compress_v_max)), 0.0))
                # Step-2: use estimated vertical velocity (closer to real robot pipeline)
                az_cmd = float(float(self.cfg.compress_kv) * (v_des - float(vz_hat)))
                az_des = float(np.clip(az_cmd, -abs(float(self.cfg.az_compress_mag)), abs(float(self.cfg.az_compress_brake_max))))
            else:
                # latch push-start once per stance
                if not bool(self._push_started):
                    self._push_started = True
                    self._vz_push = float(vz_hat)
                    self._t_push = float(self.sim_time)
                    t_used = float(self._t_push - t_td)
                    self._T_push = float(max(1e-6, float(self.cfg.stance_T) - t_used))

                v_to = float(np.clip(float(self._v_to_cmd), float(self.cfg.v_to_min), float(self.cfg.v_to_max)))
                dv_cmd = float(np.clip(v_to - float(self._vz_push), 0.0, float(self.cfg.dv_cmd_max)))
                T_push = float(max(1e-6, float(self._T_push)))
                sp = float(np.clip((float(self.sim_time) - float(self._t_push)) / T_push, 0.0, 1.0))

                if str(self.cfg.impulse_shape) == "sin":
                    a0 = float(dv_cmd * np.pi / (2.0 * T_push))  # area matches Δv
                    az_des = float(a0 * np.sin(np.pi * sp))
                else:
                    a0 = float(2.0 * dv_cmd / T_push)  # ∫ sin^2 = T/2
                    az_des = float(a0 * (np.sin(np.pi * sp) ** 2))

            # Bottom-out safety: never command additional downward acceleration once we hit the 15cm limit.
            if depth_now >= float(depth_tgt) - 1e-9:
                az_des = float(max(float(az_des), 0.0))

        # ===== MPC (SRB condensed wrench MPC) =====
        m = float(self.total_mass)
        thrust_sum_ref = float(self.base_thrust)

        # Build moment arms about COM (world) for MPC and WBC
        try:
            bid = int(self.interface.base_body_id)
            com_b = np.asarray(self.model.body_ipos[bid], dtype=float).reshape(3)
        except Exception:
            bid = int(self.interface.base_body_id)
            com_b = np.zeros(3, dtype=float)
        # Use encoder-based foot_b (base frame) and IMU attitude estimate for lever arms
        r_foot_w = (R_wb_hat @ (foot_b - com_b).reshape(3)).reshape(3)
        prop_pos_b = np.asarray(self.interface.prop_positions_body, dtype=float).reshape(3, 3)
        prop_r_w = (R_wb_hat @ (prop_pos_b - com_b.reshape(1, 3)).T).T.copy()

        # Joint torque mapping for both MPC (approx) and WBC-QP:
        #  - Serial-equivalent leg (MuJoCo actuation): tau_serial = (-J_serial^T * R_wb^T) * f_foot_w
        #  - Real 3-RSR leg (Hopper4 delta):           tau_3rsr    = A_3rsr * f_foot_w, |tau_3rsr|<=10Nm
        A_tau_f_serial = None
        tau_cmd_max_serial = None
        A_tau_f_3rsr = None
        tau_cmd_max_3rsr = None
        try:
            A_tau_f_serial = -np.asarray(J_body, dtype=float).reshape(3, 3).T @ np.asarray(R_wb_hat, dtype=float).reshape(3, 3).T
            tau_cmd_max_serial = np.array([27.0, 27.0, 2500.0], dtype=float)
        except Exception:
            A_tau_f_serial = None
            tau_cmd_max_serial = None

        # 3-RSR torque feasibility for MPC (10 Nm each).
        # Build A_tau_f_3rsr (maps world GRF -> motor torques) from Hopper4 inverse Jacobian at current foot position.
        try:
            if self._kin3rsr is not None:
                robot2vicon = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)
                # Hopper4 uses an additional +0.03m offset (Foot_Link origin -> contact site)
                x3 = (robot2vicon @ np.asarray(foot_b, dtype=float).reshape(3)).reshape(3)
                x3[2] = float(x3[2] + 0.03)
                # Clamp into a reasonable delta workspace to avoid numerical blow-ups when serial-equivalent geometry differs.
                x3[0] = float(np.clip(x3[0], -0.27, +0.27))
                x3[1] = float(np.clip(x3[1], -0.27, +0.27))
                x3[2] = float(np.clip(x3[2], 0.22, 0.49))

                # Hopper4 returns J_inv such that: thetadot = J_inv * xdot
                # Torque mapping: tau = inv(J_inv^T) * f
                J_inv, _ = self._kin3rsr.inverse_jacobian(x3, np.zeros(3, dtype=float), theta=None)
                J_inv = np.asarray(J_inv, dtype=float).reshape(3, 3)
                inv_Jt = np.linalg.inv(J_inv.T)

                # f_w -> f_b (MuJoCo base frame) -> Hopper4/delta frame (robot2vicon)
                A_tau_f_3rsr = -(inv_Jt @ robot2vicon @ np.asarray(R_wb_hat, dtype=float).reshape(3, 3).T).astype(float)
                tau_cmd_max_3rsr = np.array([10.0, 10.0, 10.0], dtype=float)
        except Exception:
            A_tau_f_3rsr = None
            tau_cmd_max_3rsr = None

        # MPC uses the 3-RSR torque map (paper-style actuator feasibility), while WBC-QP uses serial map.
        A_tau_f_mpc = A_tau_f_3rsr
        tau_cmd_max_mpc = tau_cmd_max_3rsr

        # ---- References for SRB-QP executor ----
        # We do NOT force a per-rotor thrust split; instead we (softly) smooth thrusts inside the QP via w_t_ref.
        thrust_ref = None
        f_ref = np.zeros(3, dtype=float)
        mpc_used = False
        # debug (log what the controller *intended* for horizontal force)
        fx_cmd_dbg = 0.0
        fy_cmd_dbg = 0.0
        vy_used_dbg = float("nan")

        # ===== Swing leg torque reference (flight) =====
        # Compute a flight torque reference using the existing swing PD logic,
        # but do NOT apply it directly. Instead, we pass it into the unified QP as tau_ref,
        # and we always command tau from QP (one closed-loop solution).
        tau_swing_ref = np.zeros(3, dtype=float)
        if not bool(self._stance):
            qj = np.asarray(state.get("joint_pos", np.zeros(3)), dtype=float).reshape(-1)
            qd = np.asarray(state.get("joint_vel", np.zeros(3)), dtype=float).reshape(-1)
            q_roll = float(qj[0]) if qj.size >= 1 else 0.0
            q_pitch = float(qj[1]) if qj.size >= 2 else 0.0
            q_shift = float(qj[2]) if qj.size >= 3 else 0.0
            qd_roll = float(qd[0]) if qd.size >= 1 else 0.0
            qd_pitch = float(qd[1]) if qd.size >= 2 else 0.0
            qd_shift = float(qd[2]) if qd.size >= 3 else 0.0

            kp_rp, kd_rp = 60.0, 2.0
            q_shift_des = 0.0
            q_roll_des = 0.0
            q_pitch_des = 0.0
            # Only start swing leg (S2S) after apex is reached
            if bool(self.cfg.use_s2s) and bool(self._apex_reached):
                v_b = (R_wb_hat.T @ np.asarray(self._v_hat_w, dtype=float).reshape(3)).reshape(3)
                v_xy_b = np.asarray(v_b[0:2], dtype=float).reshape(2)
                v_des_w = np.asarray(desired_v, dtype=float).reshape(2)
                v_des_b = (R_wb_hat.T @ np.array([float(v_des_w[0]), float(v_des_w[1]), 0.0], dtype=float).reshape(3)).reshape(3)
                v_des_xy_b = np.asarray(v_des_b[0:2], dtype=float).reshape(2)
                T = float(self.cfg.stance_T)
                off_ff_x = float(self.cfg.s2s_ff_scale) * float(v_xy_b[0]) * (0.5 * T)
                off_ff_y = float(self.cfg.s2s_ff_scale_y) * float(v_xy_b[1]) * (0.5 * T)
                off_fb = float(self.cfg.s2s_k) * (v_xy_b - v_des_xy_b)
                off_b_xy = np.array([off_ff_x + off_fb[0], off_ff_y + float(self.cfg.s2s_ky_scale) * off_fb[1]], dtype=float)
                lim_x = float(max(0.0, float(self.cfg.s2s_step_lim)))
                lim_y = float(max(0.0, float(self.cfg.s2s_step_lim_y)))
                off_b_xy[0] = float(np.clip(off_b_xy[0], -lim_x, +lim_x))
                off_b_xy[1] = float(np.clip(off_b_xy[1], -lim_y, +lim_y))
                tau_lpf = float(max(0.0, float(self.cfg.s2s_lpf_tau)))
                a_lpf = float(np.clip(float(self.dt) / (tau_lpf + float(self.dt)), 0.0, 1.0)) if tau_lpf > 1e-9 else 1.0
                self._s2s_off_b = (1.0 - a_lpf) * self._s2s_off_b + a_lpf * off_b_xy

                foot_b_des = np.asarray(self._foot_b_nom, dtype=float).copy()
                foot_b_des[0] = float(self._s2s_off_b[0])
                foot_b_des[1] = float(self._s2s_off_b[1])

                J_rp = np.asarray(J_body[:, 0:2], dtype=float).reshape(3, 2)
                e = (foot_b_des - foot_b.reshape(3)).astype(float)
                e[2] = 0.0
                lam = 1e-4
                H = (J_rp.T @ J_rp) + (lam * np.eye(2))
                g = (J_rp.T @ e.reshape(3))
                try:
                    dq = np.linalg.solve(H, g).reshape(2)
                except Exception:
                    dq = (J_rp.T @ e.reshape(3)).reshape(2) * 0.01
                q_roll_des = float(q_roll + dq[0])
                q_pitch_des = float(q_pitch + dq[1])
                tilt_lim = float(np.deg2rad(float(self.cfg.s2s_max_tilt_deg)))
                q_roll_des = float(np.clip(q_roll_des, -tilt_lim, +tilt_lim))
                q_pitch_des = float(np.clip(q_pitch_des, -tilt_lim, +tilt_lim))
                self._foot_b_des_last = np.asarray(foot_b_des, dtype=float).reshape(3).copy()
                self._s2s_active_last = True
            else:
                self._foot_b_des_last = np.asarray(self._foot_b_nom, dtype=float).reshape(3).copy()
                self._s2s_active_last = False

            kp_s, kd_s = 8000.0, 200.0
            tau_swing_ref = np.array(
                [
                    kp_rp * (q_roll_des - q_roll) + kd_rp * (0.0 - qd_roll),
                    kp_rp * (q_pitch_des - q_pitch) + kd_rp * (0.0 - qd_pitch),
                    kp_s * (q_shift_des - q_shift) + kd_s * (0.0 - qd_shift),
                ],
                dtype=float,
            )
            tau_swing_ref[0] = float(np.clip(tau_swing_ref[0], -27.0, 27.0))
            tau_swing_ref[1] = float(np.clip(tau_swing_ref[1], -27.0, 27.0))
            tau_swing_ref[2] = float(np.clip(tau_swing_ref[2], -2500.0, 2500.0))

        # Compression stage: MPC (task stays "compression" via az_des-based vertical reference).
        if bool(self._stance) and bool(compress_active):
            # Avoid using stale MPC output from the push stage.
            if str(self._mpc_mode) != "compress":
                self._mpc_mode = "compress"
                self._mpc_last_t = -1e9
                self._mpc_status = "init"

            if (float(self.sim_time) - float(self._mpc_last_t)) >= float(self.mpc.cfg.dt) - 1e-9:
                # inertia (body frame, diagonal -> matrix)
                try:
                    I_diag = np.asarray(self.model.body_inertia[bid], dtype=float).reshape(3)
                    I_body = np.diag(I_diag)
                except Exception:
                    I_body = np.diag(np.ones(3, dtype=float))

                pos_com_hat = (np.asarray(self._p_hat_w, dtype=float).reshape(3) + (R_wb_hat @ com_b.reshape(3))).reshape(3)
                vel_hat = np.asarray(self._v_hat_w, dtype=float).reshape(3)
                yaw = float(rpy_hat[2])
                omega_w_hat = (R_wb_hat @ imu_gyro_b.reshape(3)).reshape(3)
                x0 = np.array(
                    [
                        pos_com_hat[0],
                        pos_com_hat[1],
                        pos_com_hat[2],
                        vel_hat[0],
                        vel_hat[1],
                        vel_hat[2],
                        rpy_hat[0],
                        rpy_hat[1],
                        rpy_hat[2],
                        omega_w_hat[0],
                        omega_w_hat[1],
                        omega_w_hat[2],
                        yaw,
                    ],
                    dtype=float,
                )

                N = int(self.mpc.cfg.N)
                dtm = float(self.mpc.cfg.dt)
                sched = np.ones(N, dtype=int)  # stance-only MPC

                # Compression reference: follow a locally consistent (pz,vz) trajectory under az_des.
                vz0 = float(vel_hat[2])
                a_des = float(az_des)
                x_ref_seq = np.zeros((N, self.mpc.nx), dtype=float)
                for k in range(N):
                    tk = float((k + 1) * dtm)
                    vz_ref = float(vz0 + a_des * tk)
                    pz_ref = float(pos_com_hat[2] + vz0 * tk + 0.5 * a_des * (tk**2))
                    x_ref_seq[k, :] = np.array(
                        [
                            pos_com_hat[0],
                            pos_com_hat[1],
                            pz_ref,
                            # Compression task: do NOT fight horizontal speed here.
                            float(vel_hat[0]),
                            float(vel_hat[1]),
                            vz_ref,
                            0.0,
                            0.0,
                            yaw,
                            0.0,
                            0.0,
                            0.0,
                            yaw,
                        ],
                        dtype=float,
                    )

                T_base = float(thrust_sum_ref)
                sol_mpc = self.mpc.solve(
                    x0=x0,
                    x_ref_seq=x_ref_seq,
                    contact_schedule=sched,
                    m=float(m),
                    g=float(self.gravity),
                    I_body=I_body,
                    r_foot_w=r_foot_w,
                    prop_r_w=prop_r_w,
                    z_w=z_w,
                    thrust_max_each=float(self.max_thrust_per_motor),
                    yaw_rate_ref=0.0,
                    thrust_sum_ref=float(T_base),
                    thrust_sum_target=float(T_base),
                    A_tau_f=A_tau_f_mpc,
                    tau_cmd_max=tau_cmd_max_mpc,
                )
                self._mpc_status = str(sol_mpc.get("status", ""))
                if self._mpc_status in ("solved", "solved_inaccurate"):
                    self._mpc_f0 = np.asarray(sol_mpc.get("f0", np.zeros(3)), dtype=float).reshape(3).copy()
                    self._mpc_t0 = np.asarray(sol_mpc.get("t0", np.ones(3) * (T_base / 3.0)), dtype=float).reshape(3).copy()
                self._mpc_last_t = float(self.sim_time)

            if (str(self._mpc_mode) == "compress") and (str(self._mpc_status) in ("solved", "solved_inaccurate")):
                f_ref = np.asarray(self._mpc_f0, dtype=float).reshape(3).copy()
                thrust_ref = np.asarray(self._mpc_t0, dtype=float).reshape(3).copy()
                # Safety clamp by a friction-consistent bound.
                fx_cmd_dbg = float(f_ref[0])
                fy_cmd_dbg = float(f_ref[1])
                vy_used_dbg = float(self._v_hat_w[1])
                mu = 0.8
                fz_for_fric = float(max(0.0, float(f_ref[2])))
                fxy_lim = float(mu * fz_for_fric)
                f_ref[0] = float(np.clip(float(f_ref[0]), -fxy_lim, fxy_lim))
                f_ref[1] = float(np.clip(float(f_ref[1]), -fxy_lim, fxy_lim))
                mpc_used = True
            else:
                # fallback: hop shaper compression (soft landing + hit target compression)
                vx_used = float(self._v_hat_w[0])
                vy_used = float(self._v_hat_w[1])
                fx = float(m * float(self.cfg.axy_damp) * (float(desired_v[0]) - vx_used))
                fy = float(m * float(self.cfg.axy_damp) * (float(desired_v[1]) - vy_used)) + float(m * float(self.cfg.ki_xy) * float(self._v_int_xy[1]))
                fx_cmd_dbg = float(fx)
                fy_cmd_dbg = float(fy)
                vy_used_dbg = float(vy_used)
                F_des_z = float(m * (float(az_des) + float(self.gravity)))
                fz = float(F_des_z - float(z_w[2]) * float(thrust_sum_ref))
                fz = float(np.clip(fz, 0.0, 220.0))
                mu = 0.8
                fxy_lim = float(mu * fz)
                fx = float(np.clip(fx, -fxy_lim, fxy_lim))
                fy = float(np.clip(fy, -fxy_lim, fxy_lim))
                f_ref = np.array([fx, fy, fz], dtype=float)
                mpc_used = False

        # Push stage: run SRB-MPC (MIT condensed wrench MPC) to hit target takeoff vertical velocity and vx/vy.
        elif bool(self._stance):
            # Avoid using stale MPC output from the compression stage.
            if str(self._mpc_mode) != "push":
                self._mpc_mode = "push"
                self._mpc_last_t = -1e9
                self._mpc_status = "init"
            if (float(self.sim_time) - float(self._mpc_last_t)) >= float(self.mpc.cfg.dt) - 1e-9:
                # inertia (body frame, diagonal -> matrix)
                try:
                    I_diag = np.asarray(self.model.body_inertia[bid], dtype=float).reshape(3)
                    I_body = np.diag(I_diag)
                except Exception:
                    I_body = np.diag(np.ones(3, dtype=float))

                # COM position estimate for MPC state
                pos_com_hat = (np.asarray(self._p_hat_w, dtype=float).reshape(3) + (R_wb_hat @ com_b.reshape(3))).reshape(3)
                vel_hat = np.asarray(self._v_hat_w, dtype=float).reshape(3)
                yaw = float(rpy_hat[2])
                omega_w_hat = (R_wb_hat @ imu_gyro_b.reshape(3)).reshape(3)
                x0 = np.array(
                    [
                        pos_com_hat[0],
                        pos_com_hat[1],
                        pos_com_hat[2],
                        vel_hat[0],
                        vel_hat[1],
                        vel_hat[2],
                        rpy_hat[0],
                        rpy_hat[1],
                        rpy_hat[2],
                        omega_w_hat[0],
                        omega_w_hat[1],
                        omega_w_hat[2],
                        yaw,
                    ],
                    dtype=float,
                )

                N = int(self.mpc.cfg.N)
                dtm = float(self.mpc.cfg.dt)
                sched = np.ones(N, dtype=int)  # stance-only MPC

                # Reference: track vx/vy, and ramp vz to the desired takeoff velocity v_to_cmd.
                vz0 = float(vel_hat[2])
                v_to = float(np.clip(float(self._v_to_cmd), float(self.cfg.v_to_min), float(self.cfg.v_to_max)))
                # remaining stance time (for ramp)
                t_in_stance = float(self.sim_time - float(self._td_t)) if self._td_t is not None else 0.0
                T_rem = float(max(1e-3, float(self.cfg.stance_T) - t_in_stance))
                T_ramp = float(min(float(N) * dtm, T_rem))

                x_ref_seq = np.zeros((N, self.mpc.nx), dtype=float)
                for k in range(N):
                    tk = float((k + 1) * dtm)
                    a = float(np.clip(tk / T_ramp, 0.0, 1.0))
                    vz_ref = float(vz0 + a * (v_to - vz0))
                    x_ref_seq[k, :] = np.array(
                        [
                            pos_com_hat[0],
                            pos_com_hat[1],
                            pos_com_hat[2],
                            float(desired_v[0]),
                            float(desired_v[1]),
                            vz_ref,
                            0.0,
                            0.0,
                            yaw,
                            0.0,
                            0.0,
                            0.0,
                            yaw,
                        ],
                        dtype=float,
                    )

                T_base = float(thrust_sum_ref)
                sol_mpc = self.mpc.solve(
                    x0=x0,
                    x_ref_seq=x_ref_seq,
                    contact_schedule=sched,
                    m=float(m),
                    g=float(self.gravity),
                    I_body=I_body,
                    r_foot_w=r_foot_w,
                    prop_r_w=prop_r_w,
                    z_w=z_w,
                    thrust_max_each=float(self.max_thrust_per_motor),
                    yaw_rate_ref=0.0,
                    thrust_sum_ref=float(T_base),
                    thrust_sum_target=float(T_base),
                    A_tau_f=A_tau_f_mpc,
                    tau_cmd_max=tau_cmd_max_mpc,
                )
                self._mpc_status = str(sol_mpc.get("status", ""))
                if self._mpc_status in ("solved", "solved_inaccurate"):
                    self._mpc_f0 = np.asarray(sol_mpc.get("f0", np.zeros(3)), dtype=float).reshape(3).copy()
                    self._mpc_t0 = np.asarray(sol_mpc.get("t0", np.ones(3) * (T_base / 3.0)), dtype=float).reshape(3).copy()
                self._mpc_last_t = float(self.sim_time)

            # IMPORTANT robustness: if MPC is infeasible (or not solved yet), do NOT keep applying stale
            # forces. Fall back to the hop shaper + horizontal damping so we don't inject the wrong fy.
            if str(self._mpc_status) in ("solved", "solved_inaccurate"):
                f_ref = np.asarray(self._mpc_f0, dtype=float).reshape(3).copy()
                thrust_ref = np.asarray(self._mpc_t0, dtype=float).reshape(3).copy()
                # Safety clamp by a friction-consistent bound (should already be satisfied by MPC constraints).
                fx_cmd_dbg = float(f_ref[0])
                fy_cmd_dbg = float(f_ref[1])
                vy_used_dbg = float(self._v_hat_w[1])
                mu = 0.8
                fz_for_fric = float(max(0.0, float(f_ref[2])))
                fxy_lim = float(mu * fz_for_fric)
                f_ref[0] = float(np.clip(float(f_ref[0]), -fxy_lim, fxy_lim))
                f_ref[1] = float(np.clip(float(f_ref[1]), -fxy_lim, fxy_lim))
                mpc_used = True
            else:
                vx_used = float(self._v_hat_w[0])
                vy_used = float(self._v_hat_w[1])
                fx = float(m * float(self.cfg.axy_damp) * (float(desired_v[0]) - vx_used))
                fy = float(m * float(self.cfg.axy_damp) * (float(desired_v[1]) - vy_used)) + float(m * float(self.cfg.ki_xy) * float(self._v_int_xy[1]))
                fx_cmd_dbg = float(fx)
                fy_cmd_dbg = float(fy)
                vy_used_dbg = float(vy_used)
                F_des_z = float(m * (float(az_des) + float(self.gravity)))
                fz = float(F_des_z - float(z_w[2]) * float(thrust_sum_ref))
                fz = float(np.clip(fz, 0.0, 220.0))
                mu = 0.8
                fxy_lim = float(mu * fz)
                fx = float(np.clip(fx, -fxy_lim, fxy_lim))
                fy = float(np.clip(fy, -fxy_lim, fxy_lim))
                f_ref = np.array([fx, fy, fz], dtype=float)
                mpc_used = False

        # flight: no contact force
        else:
            f_ref[:] = 0.0

        # Build equality targets for SRB-QP:
        #   f + z_w*sum(t) = F_des
        F_des = f_ref + z_w * float(thrust_sum_ref)

        # Attitude stabilization (NO Euler-angle PD):
        # Use the external SE3 controller for ALL phases, yaw free.
        if bool(self._stance) and bool(compress_active):
            tau_rp_max = 6.0  # [N*m]
            kR, kW = 120.0, 15.0
        elif bool(self._stance):
            tau_rp_max = 4.0  # [N*m]
            kR, kW = 90.0, 10.0
        else:
            tau_rp_max = 3.0  # [N*m]
            kR, kW = 80.0, 10.0

        yaw = float(rpy_hat[2])
        # Reference controller uses its "forward" vector to define body Y axis (it assumes body X=right, Y=forward).
        # Our hopper uses the common convention body X=forward, Y=left, Z=up, so we pass the desired body-Y direction
        # (left) as the "forward" vector: y_des = [-sin(yaw), cos(yaw), 0].
        fwd = np.array([-float(np.sin(yaw)), float(np.cos(yaw)), 0.0], dtype=float)
        n = float(np.linalg.norm(fwd[0:2]))
        if n > 1e-9:
            fwd = (fwd / n).astype(float)
        else:
            fwd = np.array([0.0, 1.0, 0.0], dtype=float)

        if self.se3_att is not None:
            # Configure gains (SO(3) only).
            self.se3_att.kR = float(kR)
            self.se3_att.kw = float(kW)

            # Build SE3 controller states.
            # Note: reference controller expects quaternion in xyzw order; our estimator uses wxyz.
            q_xyzw = np.array([float(q_hat[1]), float(q_hat[2]), float(q_hat[3]), float(q_hat[0])], dtype=float)
            omega_b = np.asarray(imu_gyro_b, dtype=float).reshape(3)
            pos_hat = np.asarray(self._p_hat_w, dtype=float).reshape(3)
            vel_hat = np.asarray(self._v_hat_w, dtype=float).reshape(3)

            cur = _se3_controller.State(pos_hat, vel_hat, q_xyzw, omega_b)
            goal = _se3_controller.State(pos_hat, np.zeros(3, dtype=float), q_xyzw, np.zeros(3, dtype=float))
            cmd = self.se3_att.control_update(cur, goal, float(self.dt), fwd)
            tau_b = np.asarray(cmd.angular, dtype=float).reshape(3)
        else:
            # Fallback: Lee-style SO(3) error using vee-map (still SE3, no Euler PD).
            R_des = _Rz(yaw)
            E = (R_des.T @ R_wb_hat) - (R_wb_hat.T @ R_des)
            if _se3_geometry is not None:
                e_R = 0.5 * np.asarray(_se3_geometry.veemap(E), dtype=float).reshape(3)
            else:
                e_R = 0.5 * _vee_so3(E)
            omega_b = np.asarray(imu_gyro_b, dtype=float).reshape(3)
            tau_b = (-float(kR) * e_R) - (float(kW) * omega_b)

        # yaw remains free
        tau_b[2] = 0.0
        tau_w = (R_wb_hat @ tau_b.reshape(3)).reshape(3)
        Tau_des = np.array([float(tau_w[0]), float(tau_w[1]), 0.0], dtype=float)

        Tau_des[0] = float(np.clip(float(Tau_des[0]), -tau_rp_max, +tau_rp_max))
        Tau_des[1] = float(np.clip(float(Tau_des[1]), -tau_rp_max, +tau_rp_max))

        # Unified joint torque reference for QP (flight only):
        # We keep stance dynamics free of tau_ref smoothing so liftoff detection stays reliable.
        tau_ref = None
        if not bool(self._stance):
            tau_target = np.asarray(tau_swing_ref, dtype=float).reshape(3).copy()
            tau_tau = float(max(0.0, float(self.cfg.tau_ref_tau)))
            a_tau = float(np.clip(float(self.dt) / (tau_tau + float(self.dt)), 0.0, 1.0)) if tau_tau > 1e-9 else 1.0
            tau_ref = ((1.0 - a_tau) * np.asarray(self._tau_cmd_prev, dtype=float).reshape(3) + a_tau * tau_target).reshape(3)

        sol = self.wbc.update_and_solve(
            m=float(m),
            g=float(self.gravity),
            z_w=z_w,
            r_foot_w=r_foot_w,
            prop_r_w=prop_r_w,
            F_des=F_des,
            Tau_des=Tau_des,
            in_stance=bool(self._stance),
            thrust_sum_target=None,
            # In stance, allow props to cooperate with the leg for attitude by increasing total thrust
            # within a small band (still far below mg, so it never becomes a quadrotor).
            thrust_sum_bounds=(
                (float(thrust_sum_ref), 2.8 * float(thrust_sum_ref))
                if bool(self._stance)
                else (0.5 * float(thrust_sum_ref), 2.8 * float(thrust_sum_ref))
            ),
            thrust_sum_ref=thrust_sum_ref,
            thrust_max_each=float(self.max_thrust_per_motor),
            f_ref=f_ref,
            thrust_ref=thrust_ref,
            A_tau_f=A_tau_f_serial,
            tau_cmd_max=tau_cmd_max_serial,
            tau_ref=tau_ref,
        )
        f_contact_w = np.asarray(sol.get("f_foot_w", np.zeros(3)), dtype=float).reshape(3)
        thrusts = np.asarray(sol.get("thrusts", np.zeros(3)), dtype=float).reshape(3)
        tau_qp = np.asarray(sol.get("tau_cmd", np.zeros(3)), dtype=float).reshape(3)
        slack = np.asarray(sol.get("slack", np.zeros(6)), dtype=float).reshape(6)
        status = str(sol.get("status", ""))

        # Robust fallback: if OSQP does not return a clean solution, HOLD last solution instead of zeroing
        # (zeroing causes an instant drop in stance).
        ok = (status == "solved") and np.all(np.isfinite(f_contact_w)) and np.all(np.isfinite(thrusts))
        if ok:
            # Update hold state (stance uses both; flight only thrust matters)
            self._wbc_last_t = np.asarray(thrusts, dtype=float).reshape(3).copy()
            if bool(self._stance):
                self._wbc_last_f = np.asarray(f_contact_w, dtype=float).reshape(3).copy()
        else:
            thrusts = np.asarray(self._wbc_last_t, dtype=float).reshape(3).copy()
            if bool(self._stance):
                f_contact_w = np.asarray(self._wbc_last_f, dtype=float).reshape(3).copy()
            else:
                f_contact_w[:] = 0.0
            # keep slack for logging, but avoid NaNs
            slack = np.zeros_like(slack)
            status = f"fallback({status})"

        # --- Joint torque command ---
        tau = np.zeros(3, dtype=float)
        if bool(self._stance):
            # Use QP-solved joint torques directly (tau is a decision variable and is constrained in the QP).
            tau = np.asarray(tau_qp, dtype=float).reshape(3).copy()
            # Safety clamp (should rarely/never trigger; OSQP may return tiny infeasible values).
            tau[0] = float(np.clip(float(tau[0]), -27.0, 27.0))
            tau[1] = float(np.clip(float(tau[1]), -27.0, 27.0))
            tau[2] = float(np.clip(float(tau[2]), -2500.0, 2500.0))
        else:
            # flight: still command the QP-solved torques (single closed-loop solution).
            # The swing task enters via tau_ref (computed above) and w_tau_ref in the QP.
            tau = np.asarray(tau_qp, dtype=float).reshape(3).copy()
            tau[0] = float(np.clip(float(tau[0]), -27.0, 27.0))
            tau[1] = float(np.clip(float(tau[1]), -27.0, 27.0))
            tau[2] = float(np.clip(float(tau[2]), -2500.0, 2500.0))

        # store for next-step smoothing
        self._tau_cmd_prev = np.asarray(tau, dtype=float).reshape(3).copy()

        # apply joint torques
        self.interface.set_torque(tau)

        # apply prop forces (PWM realism)
        #
        # IMPORTANT: we intentionally do NOT inject propeller reaction torques here.
        # The MJCF visually models coaxial pairs (upper/lower discs on each arm), which cancel reaction torque.
        # With yaw free (no yaw control), injecting a non-zero motor reaction torque would cause runaway yaw spin
        # and can destabilize the estimator and flight attitude.
        thrusts_qp = np.asarray(thrusts, dtype=float).reshape(3).copy()
        pwm_us = self.motor_table.pwm_from_thrust(thrusts_qp)
        thrust_apply = self.motor_table.thrust_from_pwm(pwm_us)
        tau_react = np.zeros_like(thrust_apply)
        self.interface.apply_propeller_forces(thrust_apply, reaction_torques=tau_react, attitude_only=False)

        mujoco.mj_step(self.model, self.data)
        self.sim_time = float(self.sim_time + self.dt)

        # info for logging/overlay
        comp_now = 0.0
        if (self._q_shift_td is not None) and np.isfinite(q_shift):
            comp_now = float(max(0.0, float(q_shift) - float(self._q_shift_td)))
        info = {
            "contact_obs": int(bool(self._stance)),
            "touchdown": int(touchdown_evt),
            "liftoff": int(liftoff_evt),
            "apex": int(apex_evt),
            "foot_h": float(foot_h),
            "qd_shift": float(qd_shift) if np.isfinite(qd_shift) else 0.0,
            "az_des": float(az_des),
            "v_to_cmd": float(self._v_to_cmd),
            "push_started": int(bool(self._push_started)),
            "comp_m": float(comp_now),
            "comp_tgt_m": float(depth_tgt),
            # QP thrusts vs applied thrusts after PWM quantization (for jitter diagnosis)
            "thrusts_qp": np.asarray(thrusts_qp, dtype=float).copy(),
            "thrusts": np.asarray(thrust_apply, dtype=float).copy(),
            "pwm_us": np.asarray(pwm_us, dtype=float).copy(),
            "slackN": float(np.linalg.norm(slack)),
            "status": status,
            "f_contact_w": np.asarray(f_contact_w, dtype=float).copy(),
            # Commanded torques / moments for millisecond-level jitter diagnosis
            "tau_cmd": np.asarray(tau, dtype=float).copy(),
            "Tau_des": np.asarray(Tau_des, dtype=float).copy(),
            "v_hat_w": np.asarray(self._v_hat_w, dtype=float).copy(),
            "p_hat_w": np.asarray(self._p_hat_w, dtype=float).copy(),
            "rpy_hat": np.asarray(rpy_hat, dtype=float).copy(),
            "q_hat_wxyz": np.asarray(q_hat, dtype=float).copy(),
            "slip_speed": float(slip_speed) if bool(self._stance) else 0.0,
            "stance_int": int(bool(self._stance)),
            "compress_int": int(bool(compress_active)),
            "push_int": int(bool(self._push_started)),
            "mpc_used": int(bool(mpc_used)),
            "mpc_status": str(getattr(self, "_mpc_status", "")),
            "fy_cmd": float(fy_cmd_dbg) if np.isfinite(fy_cmd_dbg) else 0.0,
            "vy_used": float(vy_used_dbg) if np.isfinite(vy_used_dbg) else 0.0,
            # Foot placement debug (S2S): actual/desired foot position in base frame.
            "foot_b": np.asarray(foot_b, dtype=float).reshape(3).copy(),
            "foot_b_des": np.asarray(self._foot_b_des_last, dtype=float).reshape(3).copy(),
            "s2s_off_b": np.asarray(self._s2s_off_b, dtype=float).reshape(2).copy(),
            "s2s_active": int(bool(self._s2s_active_last)),
        }
        return state, tau, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration_s", type=float, default=20.0)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    # Task4 scenario: fwd -> zero -> lateral impulse (robustness test)
    ap.add_argument("--task4", action="store_true", help="Run Task4: fwd/zero/push with a one-shot lateral dvy impulse")
    ap.add_argument("--task4_fwd_s", type=float, default=10.0)
    ap.add_argument("--task4_zero_s", type=float, default=10.0)
    ap.add_argument("--task4_push_s", type=float, default=5.0)
    ap.add_argument("--task4_dvy", type=float, default=0.25, help="One-shot +dvy impulse applied to base velocity at start of push segment")
    ap.add_argument("--task4_impulse_t", type=float, default=-1.0, help="If >=0, apply impulse at this absolute sim time; else at start of push segment")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--log_hz", type=float, default=100.0)
    ap.add_argument("--tag", type=str, default="modee_inplace_comp10cm_apex0p8_v1")
    args = ap.parse_args()

    # If Task4 is enabled, duration is derived from segment lengths (ignore --duration_s).
    if bool(getattr(args, "task4", False)):
        args.duration_s = float(max(0.0, float(args.task4_fwd_s) + float(args.task4_zero_s) + float(args.task4_push_s)))

    cfg = ModeEConfig(
        duration_s=float(args.duration_s),
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
        log_hz=float(args.log_hz),
        vx_cmd=float(args.vx),
        vy_cmd=float(args.vy),
    )
    sim = ModeESim(cfg)

    out_dir = os.path.join(PROJECT_ROOT, "videos", "modee")
    os.makedirs(out_dir, exist_ok=True)
    out_mp4 = os.path.join(out_dir, f"{str(args.tag)}_vx{cfg.vx_cmd:.2f}_seg{cfg.duration_s:.0f}s.mp4")
    out_csv = os.path.join(out_dir, f"{str(args.tag)}_vx{cfg.vx_cmd:.2f}_seg{cfg.duration_s:.0f}s.csv")

    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(cfg.fps), (int(cfg.width), int(cfg.height)))
    renderer = mujoco.Renderer(sim.model, height=int(cfg.height), width=int(cfg.width))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.0
    cam.elevation = -15.0
    cam.azimuth = 120.0
    cam.lookat[:] = np.array([0.0, 0.0, 0.6], dtype=float)
    lookat_filt = None
    cam_tau = 0.15

    steps = int(float(cfg.duration_s) / float(sim.dt))
    stride = max(1, int(round((1.0 / float(cfg.fps)) / float(sim.dt))))
    log_stride = max(1, int(round((1.0 / float(cfg.log_hz)) / float(sim.dt))))

    f_csv = open(out_csv, "w", newline="")
    w = csv.writer(f_csv)
    w.writerow(
        [
            "t",
            "vx_cmd",
            "vx",
            "vx_hat",
            "px_hat",
            "vy",
            "vy_hat",
            "py_hat",
            "vz",
            "vz_hat",
            "pz_hat",
            "pz",
            "roll",
            "pitch",
            "yaw",
            "roll_hat",
            "pitch_hat",
            "yaw_hat",
            "contact_obs",
            "touchdown",
            "liftoff",
            "apex",
            "foot_h",
            "qd_shift",
            "az_des",
            "v_to_cmd",
            "q_shift",
            "comp_m",
            "comp_tgt_m",
            "prop_t0_qp",
            "prop_t1_qp",
            "prop_t2_qp",
            "prop_t0",
            "prop_t1",
            "prop_t2",
            "pwm0",
            "pwm1",
            "pwm2",
            "f_fx",
            "f_fy",
            "f_fz",
            "tau0",
            "tau1",
            "tau2",
            "Tau_des0",
            "Tau_des1",
            "Tau_des2",
            "fy_cmd",
            "vy_used",
            "foot_bx",
            "foot_by",
            "foot_bx_des",
            "foot_by_des",
            "s2s_off_bx",
            "s2s_off_by",
            "s2s_active",
            "slackN",
            "stance_int",
            "compress_int",
            "push_int",
            "mpc_used",
            "mpc_status",
            "status",
        ]
    )

    fell = False
    base_bid = int(sim.interface.base_body_id)
    q_shift_td = None
    task4_impulse_applied = False

    for i in range(steps):
        t = float(sim.sim_time)

        # Desired velocity command (world XY)
        vx_cmd_now = float(cfg.vx_cmd)
        vy_cmd_now = float(cfg.vy_cmd)
        seg_name = "CONST"
        if bool(getattr(args, "task4", False)):
            t1 = float(max(0.0, float(args.task4_fwd_s)))
            t2 = float(t1 + max(0.0, float(args.task4_zero_s)))
            t3 = float(t2 + max(0.0, float(args.task4_push_s)))

            if t < t1:
                vx_cmd_now = float(args.vx)
                vy_cmd_now = 0.0
                seg_name = "FWD"
            elif t < t2:
                vx_cmd_now = 0.0
                vy_cmd_now = 0.0
                seg_name = "ZERO"
            elif t < t3:
                vx_cmd_now = 0.0
                vy_cmd_now = 0.0
                seg_name = "PUSH"
            else:
                vx_cmd_now = 0.0
                vy_cmd_now = 0.0
                seg_name = "DONE"

            # Apply one-shot lateral impulse at start of push segment (or at a user-specified time).
            impulse_t = float(args.task4_impulse_t)
            if impulse_t < 0.0:
                impulse_t = float(t2)
            if (not task4_impulse_applied) and (t >= impulse_t):
                try:
                    sim.data.qvel[1] = float(sim.data.qvel[1] + float(args.task4_dvy))
                except Exception:
                    pass
                task4_impulse_applied = True

        des = np.array([vx_cmd_now, vy_cmd_now], dtype=float)
        state, _, info = sim.step(des)

        pos = np.asarray(state.get("body_pos", np.zeros(3)), dtype=float).reshape(3)
        vel = np.asarray(state.get("body_vel", np.zeros(3)), dtype=float).reshape(3)
        rpy = np.asarray(state.get("body_rpy", np.zeros(3)), dtype=float).reshape(3)

        try:
            qj = np.asarray(state.get("joint_pos", np.zeros(3)), dtype=float).reshape(-1)
            q_shift = float(qj[2]) if qj.size >= 3 else float("nan")
        except Exception:
            q_shift = float("nan")

        contact_obs = bool(int(info.get("contact_obs", 0)) == 1)
        touchdown = int(info.get("touchdown", 0))
        liftoff = int(info.get("liftoff", 0))
        apex = int(info.get("apex", 0))
        if touchdown and np.isfinite(q_shift):
            q_shift_td = float(q_shift)

        comp_now = float(info.get("comp_m", 0.0))
        comp_tgt = float(info.get("comp_tgt_m", float(cfg.compress_depth_m)))

        # fall detection
        if (not fell) and (pos[2] < 0.20 or abs(rpy[0]) > np.deg2rad(60.0) or abs(rpy[1]) > np.deg2rad(60.0)):
            fell = True

        if i % log_stride == 0:
            thrusts = np.asarray(info.get("thrusts", np.zeros(3)), dtype=float).reshape(3)
            thrusts_qp = np.asarray(info.get("thrusts_qp", thrusts), dtype=float).reshape(3)
            pwm = np.asarray(info.get("pwm_us", np.zeros(3)), dtype=float).reshape(3)
            f_contact_w = np.asarray(info.get("f_contact_w", np.zeros(3)), dtype=float).reshape(3)
            tau_cmd = np.asarray(info.get("tau_cmd", np.zeros(3)), dtype=float).reshape(3)
            Tau_des = np.asarray(info.get("Tau_des", np.zeros(3)), dtype=float).reshape(3)
            v_hat = np.asarray(info.get("v_hat_w", np.zeros(3)), dtype=float).reshape(3)
            p_hat = np.asarray(info.get("p_hat_w", np.zeros(3)), dtype=float).reshape(3)
            rpy_hat = np.asarray(info.get("rpy_hat", np.zeros(3)), dtype=float).reshape(3)
            w.writerow(
                [
                    f"{t:.6f}",
                    f"{vx_cmd_now:.3f}",
                    f"{vel[0]:.6f}",
                    f"{float(v_hat[0]):.6f}",
                    f"{float(p_hat[0]):.6f}",
                    f"{vel[1]:.6f}",
                    f"{float(v_hat[1]):.6f}",
                    f"{float(p_hat[1]):.6f}",
                    f"{vel[2]:.6f}",
                    f"{float(v_hat[2]):.6f}",
                    f"{float(p_hat[2]):.6f}",
                    f"{pos[2]:.6f}",
                    f"{rpy[0]:.6f}",
                    f"{rpy[1]:.6f}",
                    f"{rpy[2]:.6f}",
                    f"{float(rpy_hat[0]):.6f}",
                    f"{float(rpy_hat[1]):.6f}",
                    f"{float(rpy_hat[2]):.6f}",
                    int(contact_obs),
                    touchdown,
                    liftoff,
                    apex,
                    f"{float(info.get('foot_h', 0.0)):.6f}",
                    f"{float(info.get('qd_shift', 0.0)):.6f}",
                    f"{float(info.get('az_des', 0.0)):.6f}",
                    f"{float(info.get('v_to_cmd', 0.0)):.6f}",
                    f"{q_shift:.6f}",
                    f"{comp_now:.6f}",
                    f"{comp_tgt:.6f}",
                    f"{float(thrusts_qp[0]):.6f}",
                    f"{float(thrusts_qp[1]):.6f}",
                    f"{float(thrusts_qp[2]):.6f}",
                    f"{float(thrusts[0]):.6f}",
                    f"{float(thrusts[1]):.6f}",
                    f"{float(thrusts[2]):.6f}",
                    f"{float(pwm[0]):.2f}",
                    f"{float(pwm[1]):.2f}",
                    f"{float(pwm[2]):.2f}",
                    f"{float(f_contact_w[0]):.6f}",
                    f"{float(f_contact_w[1]):.6f}",
                    f"{float(f_contact_w[2]):.6f}",
                    f"{float(tau_cmd[0]):.6f}",
                    f"{float(tau_cmd[1]):.6f}",
                    f"{float(tau_cmd[2]):.6f}",
                    f"{float(Tau_des[0]):.6f}",
                    f"{float(Tau_des[1]):.6f}",
                    f"{float(Tau_des[2]):.6f}",
                    f"{float(info.get('fy_cmd', 0.0)):.6f}",
                    f"{float(info.get('vy_used', 0.0)):.6f}",
                    f"{float(np.asarray(info.get('foot_b', np.zeros(3)), dtype=float).reshape(3)[0]):.6f}",
                    f"{float(np.asarray(info.get('foot_b', np.zeros(3)), dtype=float).reshape(3)[1]):.6f}",
                    f"{float(np.asarray(info.get('foot_b_des', np.zeros(3)), dtype=float).reshape(3)[0]):.6f}",
                    f"{float(np.asarray(info.get('foot_b_des', np.zeros(3)), dtype=float).reshape(3)[1]):.6f}",
                    f"{float(np.asarray(info.get('s2s_off_b', np.zeros(2)), dtype=float).reshape(2)[0]):.6f}",
                    f"{float(np.asarray(info.get('s2s_off_b', np.zeros(2)), dtype=float).reshape(2)[1]):.6f}",
                    int(info.get("s2s_active", 0)),
                    f"{float(info.get('slackN', 0.0)):.6e}",
                    int(info.get("stance_int", 0)),
                    int(info.get("compress_int", 0)),
                    int(info.get("push_int", 0)),
                    int(info.get("mpc_used", 0)),
                    str(info.get("mpc_status", "")),
                    str(info.get("status", "")),
                ]
            )

        if i % stride == 0:
            # camera follow with LPF on lookat
            try:
                lookat = np.asarray(sim.data.xpos[base_bid], dtype=float).reshape(3).copy()
                if lookat_filt is None:
                    lookat_filt = lookat.copy()
                a = float(np.clip((float(stride) * float(sim.dt)) / (cam_tau + float(stride) * float(sim.dt)), 0.0, 1.0))
                lookat_filt = (1.0 - a) * lookat_filt + a * lookat
                cam.lookat[:] = lookat_filt
                cam.lookat[2] = float(cam.lookat[2] + 0.15)
            except Exception:
                pass

            renderer.update_scene(sim.data, camera=cam)
            # --- 3D coordinate frames (debug visualization) ---
            # Robot/base frame: true base_link axes (RGB).
            # IMU frame: attitude estimate axes (CMY), slightly offset upward for visibility.
            try:
                scene = renderer.scene
                base_pos = np.asarray(sim.data.xpos[base_bid], dtype=float).reshape(3)
                R_wb_true = np.asarray(sim.data.xmat[base_bid], dtype=float).reshape(3, 3)
                _add_frame_axes_to_scene(scene, base_pos, R_wb_true, axis_len_m=0.14, width_px=3.0)

                q_hat = np.asarray(info.get("q_hat_wxyz", np.array([1.0, 0.0, 0.0, 0.0], dtype=float)), dtype=float).reshape(4)
                R_wb_hat = _quat_to_R_wb(q_hat)
                imu_origin = (base_pos + np.array([0.0, 0.0, 0.18], dtype=float)).reshape(3)
                _add_frame_axes_to_scene(
                    scene,
                    imu_origin,
                    R_wb_hat,
                    axis_len_m=0.12,
                    width_px=3.0,
                    rgba_x=np.array([0.0, 1.0, 1.0, 0.85], dtype=np.float32),
                    rgba_y=np.array([1.0, 0.0, 1.0, 0.85], dtype=np.float32),
                    rgba_z=np.array([1.0, 1.0, 0.0, 0.85], dtype=np.float32),
                )
            except Exception:
                pass
            frame = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.putText(frame_bgr, "MODEE (SRB-QP + tau=J^T(-f), yaw free)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            v_hat = np.asarray(info.get("v_hat_w", np.zeros(3)), dtype=float).reshape(3)
            p_hat = np.asarray(info.get("p_hat_w", np.zeros(3)), dtype=float).reshape(3)
            rpy_hat = np.asarray(info.get("rpy_hat", np.zeros(3)), dtype=float).reshape(3)
            # Phase display (COMP/PUSH/FLIGHT + TD/LO/APEX)
            stance_int = int(info.get("stance_int", 0))
            comp_int = int(info.get("compress_int", 0))
            push_int = int(info.get("push_int", 0))
            td = int(info.get("touchdown", 0))
            lo = int(info.get("liftoff", 0))
            apx = int(info.get("apex", 0))
            if stance_int:
                phase_txt = "STANCE:PUSH" if push_int else "STANCE:COMP"
                if comp_int and (not push_int):
                    phase_txt = "STANCE:COMP"
            else:
                phase_txt = "FLIGHT"
            cv2.putText(frame_bgr, f"phase={phase_txt}  TD={td} LO={lo} AP={apx}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
            cv2.putText(
                frame_bgr,
                f"{seg_name}  vx_cmd={vx_cmd_now:.2f}  vx={vel[0]:+.2f}  vx_hat={float(v_hat[0]):+.2f}  vz_hat={float(v_hat[2]):+.2f}  z_hat={float(p_hat[2]):.2f}  z={pos[2]:.2f}  contact={int(contact_obs)}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame_bgr,
                f"r/p_hat={np.degrees(rpy_hat[0]):+.1f}/{np.degrees(rpy_hat[1]):+.1f}  yaw_hat={np.degrees(rpy_hat[2]):+.1f}  |  r/p={np.degrees(rpy[0]):+.1f}/{np.degrees(rpy[1]):+.1f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )
            cv2.putText(frame_bgr, f"shift={q_shift:+.3f}m  qd_shift={float(info.get('qd_shift',0.0)):+.2f}  comp={comp_now*100.0:.1f}cm / {comp_tgt*100.0:.0f}cm  slackN={float(info.get('slackN',0.0)):.2e}", (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            cv2.putText(frame_bgr, "frames: ROBOT xyz=RGB (base), IMU xyz=CMY (est, offset up)", (20, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            # Display prop PWM (like the older modex recordings)
            pwm_us = np.asarray(info.get("pwm_us", np.zeros(3)), dtype=float).reshape(-1)
            if pwm_us.size >= 3 and np.all(np.isfinite(pwm_us[:3])):
                cv2.putText(
                    frame_bgr,
                    f"PWM(us): [{pwm_us[0]:.0f}, {pwm_us[1]:.0f}, {pwm_us[2]:.0f}]",
                    (20, 184),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                )
            if fell:
                cv2.putText(frame_bgr, "FELL", (int(cfg.width * 0.45), int(cfg.height * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)

            vw.write(frame_bgr)

    renderer.close()
    vw.release()
    f_csv.close()

    print(f"Wrote: {out_mp4}")
    print(f"Wrote: {out_csv}")
    print(f"sim_time={sim.sim_time:.2f}s fell={fell}")
    out_h264 = _export_h264(out_mp4)
    if out_h264 is not None:
        print(f"Wrote: {out_h264}")


if __name__ == "__main__":
    main()


