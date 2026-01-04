#!/usr/bin/env python3
"""
Non-interactive Hopper4 (prop-enabled) runner for Hopper_sim.

Runs the Hopper4 LCM controller for a fixed duration with propellers ARMED,
and exits cleanly (no interactive CLI, no matplotlib plotting).
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from Hopper4 import HopperLCMController


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, default=10.0, help="Run time (seconds).")
    args = ap.parse_args()

    ctl = HopperLCMController()

    # In-place hop in "velocity mode"
    ctl.desired_velocity = np.array([0.0, 0.0, 1.0], dtype=float)

    # Prop-enabled: arm props immediately (avoid needing gamepad A press)
    ctl.propeller_armed = True
    ctl.propeller_vector_mode = False

    # LCM handler thread
    t_lcm = threading.Thread(target=ctl.run_lcm_handler, daemon=True)
    t_lcm.start()

    # Stopper
    t0 = time.time()

    def _stopper() -> None:
        time.sleep(float(max(0.1, args.duration_s)))
        ctl.running = False

    threading.Thread(target=_stopper, daemon=True).start()

    try:
        ctl.run_controller()
    finally:
        ctl.running = False
        # Disarm props at exit
        try:
            for _ in range(10):
                ctl.send_motor_command(1000, 1000, 1000, False)
                time.sleep(0.01)
        except Exception:
            pass

    dt = time.time() - t0
    print(f"[hopper4_prop_sim] done (ran {dt:.2f}s)")


if __name__ == "__main__":
    main()


