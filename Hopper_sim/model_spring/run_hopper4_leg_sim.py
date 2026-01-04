#!/usr/bin/env python3
"""
Hopper4 (leg-only) demo runner for Hopper_sim (SERIAL MJCF).

This replaces the original Hopper4 delta-leg kinematics with a serial-leg LCM controller,
so the leg has real authority on the MuJoCo `hopper_serial.xml` plant.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import lcm
from scipy.spatial.transform import Rotation

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_LCM_TYPES_DIR = os.path.join(_CUR_DIR, "..", "hopper_lcm_types", "lcm_types")
sys.path.append(_LCM_TYPES_DIR)
sys.path.append(os.path.join(_CUR_DIR, "controllers"))

from python.hopper_cmd_lcmt import hopper_cmd_lcmt  # type: ignore  # noqa: E402
from python.hopper_data_lcmt import hopper_data_lcmt  # type: ignore  # noqa: E402
from python.hopper_imu_lcmt import hopper_imu_lcmt  # type: ignore  # noqa: E402

from raibert_controller import RaibertController  # type: ignore  # noqa: E402


def _quat_to_R_wb(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        q = q / n
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _Rx(a: float) -> np.ndarray:
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _Ry(a: float) -> np.ndarray:
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


@dataclass(frozen=True)
class SerialLegKinematics:
    # From `hopper_serial.xml`
    base_roll_off_z: float = -0.0416
    foot_link_z: float = -0.5237
    foot_site_z: float = -0.03

    def foot_pos_body(self, roll: float, pitch: float, shift: float) -> np.ndarray:
        """Foot contact site position in BODY frame (z up; foot below -> z negative)."""
        p_br = np.array([0.0, 0.0, float(self.base_roll_off_z)], dtype=float)
        p_local = np.array([0.0, 0.0, float(shift + self.foot_link_z + self.foot_site_z)], dtype=float)
        return p_br + (_Rx(roll) @ (_Ry(pitch) @ p_local.reshape(3))).reshape(3)

    def jacobian_body(self, roll: float, pitch: float, shift: float) -> np.ndarray:
        """3x3 Jacobian mapping qd=[roll_dot,pitch_dot,shift_dot] -> foot_vel in BODY."""
        p_br = np.array([0.0, 0.0, float(self.base_roll_off_z)], dtype=float)
        p = self.foot_pos_body(roll, pitch, shift)
        r = (p - p_br).reshape(3)
        a1 = np.array([1.0, 0.0, 0.0], dtype=float)  # roll axis in base
        a2 = (_Rx(roll) @ np.array([0.0, 1.0, 0.0], dtype=float)).reshape(3)  # pitch axis in base
        a3 = (_Rx(roll) @ (_Ry(pitch) @ np.array([0.0, 0.0, 1.0], dtype=float))).reshape(3)  # shift axis in base
        col1 = np.cross(a1, r)
        col2 = np.cross(a2, r)
        col3 = a3
        return np.column_stack([col1, col2, col3]).astype(float)

    def foot_vel_body(self, roll: float, pitch: float, shift: float, qd: np.ndarray) -> np.ndarray:
        J = self.jacobian_body(roll, pitch, shift)
        return (J @ np.asarray(qd, dtype=float).reshape(3)).reshape(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcm-url", type=str, default="udpm://239.255.76.67:7667?ttl=255")
    ap.add_argument("--ctrl-hz", type=float, default=500.0)
    ap.add_argument("--duration-s", type=float, default=0.0, help="Wall time seconds. <=0 means run until killed.")
    ap.add_argument("--vx-fwd", type=float, default=0.20, help="Forward vx during middle segment (m/s).")
    ap.add_argument("--t-inplace-s", type=float, default=3.0)
    ap.add_argument("--t-fwd-s", type=float, default=5.0)
    args = ap.parse_args()

    lc = lcm.LCM(str(args.lcm_url))
    kin = SerialLegKinematics()
    ctl = RaibertController()
    ctl.reset()

    # latest measurements
    q = np.zeros(3, dtype=float)
    qd = np.zeros(3, dtype=float)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    gyro = np.zeros(3, dtype=float)
    acc = np.zeros(3, dtype=float)
    rpy = np.zeros(3, dtype=float)
    have_q = False
    have_imu = False

    def _on_data(_chan: str, payload: bytes) -> None:
        nonlocal q, qd, have_q
        try:
            msg = hopper_data_lcmt.decode(payload)
            q = np.asarray(msg.q, dtype=float).reshape(3)
            qd = np.asarray(msg.qd, dtype=float).reshape(3)
            have_q = True
        except Exception:
            return

    def _on_imu(_chan: str, payload: bytes) -> None:
        nonlocal quat, gyro, acc, rpy, have_imu
        try:
            msg = hopper_imu_lcmt.decode(payload)
            quat = np.asarray(msg.quat, dtype=float).reshape(4)
            gyro = np.asarray(msg.gyro, dtype=float).reshape(3)
            acc = np.asarray(msg.acc, dtype=float).reshape(3)
            rpy = np.asarray(msg.rpy, dtype=float).reshape(3)
            have_imu = True
        except Exception:
            return

    lc.subscribe("hopper_data_lcmt", _on_data)
    lc.subscribe("hopper_imu_lcmt", _on_imu)

    ctrl_hz = float(max(1.0, float(args.ctrl_hz)))
    dt_nom = 1.0 / ctrl_hz
    t0 = time.time()
    t_prev = time.time()
    # Hopper4 controller expects a "VICON-style" body frame where +Z is DOWN.
    # We compute foot kinematics in MuJoCo BODY frame (+Z UP) and flip Z before feeding the controller.
    robot2vicon = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)

    # Simple body velocity estimate (from foot velocity in stance)
    body_vel_est = np.zeros(3, dtype=float)
    body_vel_lpf_tau = 0.05  # low-pass filter time constant

    def _desired_vx(t: float) -> float:
        t_in = float(max(0.0, float(args.t_inplace_s)))
        t_fwd = float(max(0.0, float(args.t_fwd_s)))
        if t < t_in:
            return 0.0
        if t < (t_in + t_fwd):
            return float(args.vx_fwd)
        return 0.0

    while True:
        # drain LCM quickly
        for _ in range(32):
            if lc.handle_timeout(0) <= 0:
                break

        now = time.time()
        if float(args.duration_s) > 0.0 and (now - t0) >= float(args.duration_s):
            break
        if not (have_q and have_imu):
            time.sleep(0.001)
            continue

        dt = float(max(1e-4, min(0.02, now - t_prev)))
        t_prev = now

        # --- serial leg kinematics (MuJoCo BODY frame, +Z up) ---
        roll, pitch, shift = [float(v) for v in q.reshape(3)]
        foot_pos_b = kin.foot_pos_body(roll, pitch, shift)
        foot_vel_b = kin.foot_vel_body(roll, pitch, shift, qd)
        # Convert to Hopper4 frame (+Z down) for the controller.
        foot_pos_h4 = (robot2vicon @ foot_pos_b.reshape(3)).reshape(3)
        foot_vel_h4 = (robot2vicon @ foot_vel_b.reshape(3)).reshape(3)
        gyro_h4 = (robot2vicon @ gyro.reshape(3)).reshape(3)

        # desired velocity profile
        t_rel = float(now - t0)
        vx_des = _desired_vx(t_rel)
        desired_vel = np.array([vx_des, 0.0, 0.0], dtype=float)

        # Estimate body velocity from foot velocity (works in stance when foot is on ground)
        # body_vel_world ≈ -R_wb @ foot_vel_body (foot fixed -> body moves opposite)
        quat_scipy = [quat[1], quat[2], quat[3], quat[0]]  # wxyz -> xyzw
        R_wb = Rotation.from_quat(quat_scipy).as_matrix()
        # foot_vel_b is in MuJoCo body frame (+Z up), convert to world
        foot_vel_world = (R_wb @ foot_vel_b.reshape(3)).reshape(3)
        # In stance, body velocity = -foot_vel_world (foot stationary on ground)
        body_vel_raw = -foot_vel_world
        # Low-pass filter to smooth the estimate
        alpha = float(dt / (body_vel_lpf_tau + dt))
        body_vel_est[0] = float(body_vel_est[0] + alpha * (body_vel_raw[0] - body_vel_est[0]))
        body_vel_est[1] = float(body_vel_est[1] + alpha * (body_vel_raw[1] - body_vel_est[1]))
        # Don't estimate Z velocity (keep 0)

        state = {
            "foot_pos": foot_pos_h4.reshape(3),
            "foot_vel": foot_vel_h4.reshape(3),
            "joint_pos": q.reshape(3),
            "joint_vel": qd.reshape(3),
            # Body velocity estimated from foot velocity
            "body_vel": body_vel_est.copy(),
            "body_quat": quat.reshape(4),
            "body_ang_vel": gyro_h4.reshape(3),
            "body_pos": np.zeros(3, dtype=float),
            "body_rpy": rpy.reshape(3),
        }

        info = ctl.update(state, desired_vel=desired_vel, dt=dt)
        tau = np.asarray(info.get("torque", np.zeros(3)), dtype=float).reshape(3)

        # publish joint torque command
        cmd = hopper_cmd_lcmt()
        cmd.tau_ff = [float(v) for v in tau.reshape(3)]
        lc.publish("hopper_cmd_lcmt", cmd.encode())

        # sleep to target control rate
        time.sleep(max(0.0, dt_nom - (time.time() - now)))

    # stop
    cmd = hopper_cmd_lcmt()
    cmd.tau_ff = [0.0, 0.0, 0.0]
    lc.publish("hopper_cmd_lcmt", cmd.encode())
    print("[hopper4_leg_sim] done")


if __name__ == "__main__":
    main()


