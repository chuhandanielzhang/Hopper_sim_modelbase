#!/usr/bin/env python3
"""
MuJoCo "fake robot" process that exposes the same IO channels as the real hopper **using real MIT LCM**.

Publishes (sensor side):
- `hopper_data_lcmt` : q, qd, tauIq
- `hopper_imu_lcmt`  : quat(wxyz), gyro(body), acc(body specific force), rpy(xyz)

Subscribes (command side):
- `hopper_cmd_lcmt`  : tau_ff (applied as joint torques)
- `motor_pwm_lcmt`   : 6-channel PWM (mapped -> thrust via MotorTableModel), `control_mode` gates output

Requires:
- `lcm` Python bindings installed (MIT LCM).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "utils"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lcm_types"))

import lcm  # noqa: E402

from utils.mujoco_interface import MuJoCoInterface  # noqa: E402
from controllers.motor_utils import MotorTableModel  # noqa: E402

from python.hopper_cmd_lcmt import hopper_cmd_lcmt  # noqa: E402
from python.hopper_data_lcmt import hopper_data_lcmt  # noqa: E402
from python.hopper_imu_lcmt import hopper_imu_lcmt  # noqa: E402
from python.motor_pwm_lcmt import motor_pwm_lcmt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=os.path.join(PROJECT_ROOT, "mjcf", "hopper_serial.xml"))
    ap.add_argument("--lcm_url", type=str, default="udpm://239.255.76.67:7667?ttl=255")
    ap.add_argument("--duration_s", type=float, default=20.0)
    ap.add_argument("--init_z", type=float, default=0.80)
    ap.add_argument("--init_shift", type=float, default=0.00)
    ap.add_argument("--arm", action="store_true", help="Force-arm outputs even if motor_pwm_lcmt.control_mode=0")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    interface = MuJoCoInterface(model, data)
    interface.reset(init_height=float(args.init_z), init_shift=float(args.init_shift))

    motor_table = MotorTableModel.default_from_table()

    lc = lcm.LCM(args.lcm_url)

    cmd_tau = np.zeros(3, dtype=float)
    cmd_pwm = np.ones(6, dtype=float) * 1000.0
    cmd_armed = False

    def _on_cmd(_chan: str, payload: bytes):
        nonlocal cmd_tau
        try:
            msg = hopper_cmd_lcmt.decode(payload)
            cmd_tau = np.asarray(msg.tau_ff, dtype=float).reshape(3).copy()
        except Exception:
            # keep last command
            return

    def _on_pwm(_chan: str, payload: bytes):
        nonlocal cmd_pwm, cmd_armed
        try:
            msg = motor_pwm_lcmt.decode(payload)
            cmd_pwm = np.asarray(msg.pwm_values, dtype=float).reshape(6).copy()
            cmd_armed = bool(int(getattr(msg, "control_mode", 0)))
        except Exception:
            return

    lc.subscribe("hopper_cmd_lcmt", _on_cmd)
    lc.subscribe("motor_pwm_lcmt", _on_pwm)

    dt = float(model.opt.timestep)
    steps = int(max(1, round(float(args.duration_s) / max(1e-9, dt))))

    for _k in range(steps):
        # Drain any pending inbound packets (non-blocking)
        for _ in range(8):
            if lc.handle_timeout(0) <= 0:
                break

        armed = bool(args.arm) or bool(cmd_armed)

        tau_apply = cmd_tau if armed else np.zeros(3, dtype=float)
        interface.set_torque(tau_apply)

        pwm_apply = cmd_pwm if armed else (np.ones(6, dtype=float) * 1000.0)
        thrusts6 = motor_table.thrust_from_pwm(pwm_apply)
        interface.apply_propeller_forces(thrusts6, reaction_torques=np.zeros_like(thrusts6), attitude_only=False)

        mujoco.mj_step(model, data)

        st = interface.get_state()

        # Publish joint data
        jd = hopper_data_lcmt()
        q = np.asarray(st.get("joint_pos", np.zeros(3)), dtype=float).reshape(3)
        qd = np.asarray(st.get("joint_vel", np.zeros(3)), dtype=float).reshape(3)
        jd.q = [float(v) for v in q]
        jd.qd = [float(v) for v in qd]
        jd.tauIq = [float(v) for v in np.asarray(tau_apply, dtype=float).reshape(3)]
        lc.publish("hopper_data_lcmt", jd.encode())

        # Publish IMU
        im = hopper_imu_lcmt()
        quat = np.asarray(st.get("body_quat", np.array([1.0, 0.0, 0.0, 0.0])), dtype=float).reshape(4)
        gyro = np.asarray(st.get("imu_gyro", np.zeros(3)), dtype=float).reshape(3)
        acc = np.asarray(st.get("imu_acc", np.zeros(3)), dtype=float).reshape(3)
        rpy = np.asarray(st.get("body_rpy", np.zeros(3)), dtype=float).reshape(3)
        im.quat = [float(v) for v in quat]
        im.gyro = [float(v) for v in gyro]
        im.acc = [float(v) for v in acc]
        im.rpy = [float(v) for v in rpy]
        lc.publish("hopper_imu_lcmt", im.encode())

        # keep CPU sane if someone runs with a tiny dt; simulation is still deterministic
        time.sleep(0.0)


if __name__ == "__main__":
    main()


