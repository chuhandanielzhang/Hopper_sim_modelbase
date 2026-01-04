#!/usr/bin/env python3

import argparse
import threading
import time

from modee.lcm_controller import ModeELCMController, ModeELCMConfig
from modee.core import ModeEConfig


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ModeE controller for real robot (PC-side, via LCM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (internal solver uses nominal torque limits; output torque is safely capped)
  python3 run_modee.py

  # First-time bring-up (recommended): keep MPC/QP internal limits normal, but cap OUTPUT torque
  python3 run_modee.py --tau-out-max 1 --pwm-max 1100 --thrust-ratio 0.03 --thrust-max-each 4

  # Custom LCM URL (if not using default multicast)
  python3 run_modee.py --lcm-url "udpm://239.255.76.67:7667?ttl=1"

  # Print every loop (no throttling)
  python3 run_modee.py --print-hz 0

  # Verbose printing (5 Hz)
  python3 run_modee.py --print-hz 5
        """,
    )

    # Safety parameters (most important for bring-up)
    ap.add_argument(
        "--tau-max",
        type=float,
        default=None,
        help="INTERNAL torque limit (Nm) used by MPC/WBC-QP feasibility. Default: 20.0. "
        "For bring-up safety, prefer --tau-out-max/--tau-out-scale so the internal solver still behaves normally.",
    )
    ap.add_argument(
        "--tau-sign",
        type=float,
        default=None,
        help="Motor torque wiring sign (+1 or -1). Default: +1.0. Set -1 if your motors push the wrong way.",
    )
    ap.add_argument(
        "--pwm-max",
        type=float,
        default=None,
        help="Max PWM pulse width (us). Default: 1300. For first bring-up, use 1100-1200.",
    )
    ap.add_argument(
        "--pwm-min",
        type=float,
        default=None,
        help="Min PWM pulse width (us). Default: 1000 (disarmed/idle).",
    )
    ap.add_argument(
        "--thrust-ratio",
        type=float,
        default=None,
        help="Baseline prop thrust ratio (0.0-1.0). Default: 0.10. For first bring-up, use 0.0-0.05.",
    )
    ap.add_argument(
        "--thrust-max-each",
        type=float,
        default=None,
        help="Per-arm max thrust cap passed to MPC/WBC (N). Default: 10.0. For first bring-up, use ~2-6.",
    )
    ap.add_argument(
        "--thrust-min-each",
        type=float,
        default=None,
        help="Per-arm MIN thrust lower bound in WBC-QP (N). Helps avoid props hitting pwm_min (stop/start) which causes wobble. Default: 0.0",
    )
    ap.add_argument(
        "--wbc-w-t-ref",
        type=float,
        default=None,
        help="WBC-QP thrust smoothing weight. Higher -> less left/right thrust swapping (less wobble) but slower response. Suggested: 1e-2 to 1e-1.",
    )
    ap.add_argument(
        "--wbc-w-tsum-ref",
        type=float,
        default=None,
        help="WBC-QP total-thrust tracking weight. Higher -> sum(thrust) stays closer to baseline. Suggested: 0 to 1e-2.",
    )

    # LCM settings
    ap.add_argument(
        "--lcm-url",
        type=str,
        default=None,
        help='LCM URL. Default: "udpm://239.255.76.67:7667?ttl=255"',
    )

    # Control tuning
    ap.add_argument(
        "--max-cmd-vel",
        type=float,
        default=None,
        help="Max commanded velocity from gamepad stick (m/s). Default: 0.8",
    )
    ap.add_argument(
        "--tau-out-max",
        type=float,
        default=None,
        help="Output torque limit (Nm) applied in Python right before publishing (does NOT affect ModeECore/QP).",
    )
    ap.add_argument(
        "--tau-out-scale",
        type=float,
        default=None,
        help="Output torque scale applied in Python right before publishing (e.g. 0.1 for bring-up). Default: 1.0",
    )
    ap.add_argument(
        "--swing-kp-xy",
        type=float,
        default=None,
        help="Flight swing PERPENDICULAR (⊥ leg axis) position gain kp (N/m). Default: 300.0.",
    )
    ap.add_argument(
        "--swing-kd-xy",
        type=float,
        default=None,
        help="Flight swing PERPENDICULAR (⊥ leg axis) damping gain kd (N/(m/s)). Default: 0.0.",
    )
    ap.add_argument(
        "--swing-kp-z",
        type=float,
        default=None,
        help="Flight swing AXIAL (along leg axis) position gain kp (N/m). Default: 300.0.",
    )
    ap.add_argument(
        "--swing-kd-z",
        type=float,
        default=None,
        help="Flight swing AXIAL (along leg axis) damping gain kd (N/(m/s)). Default: 0.0.",
    )
    ap.add_argument(
        "--foot-vel-lpf",
        action="store_true",
        help="Enable foot velocity LPF (flight phase). Default: enabled when kd > 3.",
    )
    ap.add_argument(
        "--no-foot-vel-lpf",
        action="store_true",
        help="Disable foot velocity LPF (use raw Kalman-filtered motor qd).",
    )
    ap.add_argument(
        "--foot-vel-lpf-tau",
        type=float,
        default=None,
        help="Foot velocity LPF time constant (s). Smaller = more filtering. Default: 0.015 (~10Hz).",
    )
    ap.add_argument(
        "--print-hz",
        type=float,
        default=None,
        help="Print frequency (Hz). <=0 means print every loop. Default: 0 (every loop).",
    )
    ap.add_argument(
        "--force-arm",
        action="store_true",
        help="Force ARM on startup (for testing without gamepad). WARNING: Robot will start sending torques immediately!",
    )
    ap.add_argument(
        "--stance-use-props",
        action="store_true",
        help="Enable propellers in STANCE (helpful for balance in simulation). Default: disabled (leg-only stance).",
    )
    ap.add_argument(
        "--leg-model",
        type=str,
        default=None,
        help='Leg kinematics backend: "delta" (real robot) or "serial" (MuJoCo hopper_serial.xml).',
    )

    # Stance velocity convergence tuning (symmetric XY; user request)
    ap.add_argument(
        "--mpc-w-vxy",
        type=float,
        default=None,
        help="MPC velocity tracking weight applied to BOTH vx and vy (symmetric). Default: 12.0",
    )
    ap.add_argument(
        "--axy-damp",
        type=float,
        default=None,
        help="Fallback stance horizontal damping gain (N/(m/s)) used when MPC is infeasible. Default: 0.5",
    )
    ap.add_argument(
        "--ki-xy",
        type=float,
        default=None,
        help="Horizontal velocity integral gain (applied symmetrically to X and Y). Default: 0.0",
    )
    ap.add_argument(
        "--v-int-max",
        type=float,
        default=None,
        help="Horizontal velocity integrator clamp (m/s). Default: 0.30",
    )

    args = ap.parse_args()

    # Build ModeEConfig with overrides
    modee_cfg = ModeEConfig()
    if args.tau_max is not None:
        modee_cfg.tau_cmd_max_nm = (float(args.tau_max),) * 3
        print(f"[run_modee] tau_cmd_max_nm = {modee_cfg.tau_cmd_max_nm}")
    if args.tau_sign is not None:
        modee_cfg.tau_cmd_sign = (float(args.tau_sign),) * 3
        print(f"[run_modee] tau_cmd_sign = {modee_cfg.tau_cmd_sign}")
    if args.pwm_max is not None:
        modee_cfg.pwm_max_us = float(args.pwm_max)
        print(f"[run_modee] pwm_max_us = {modee_cfg.pwm_max_us}")
    if args.pwm_min is not None:
        modee_cfg.pwm_min_us = float(args.pwm_min)
        print(f"[run_modee] pwm_min_us = {modee_cfg.pwm_min_us}")
    if args.thrust_ratio is not None:
        modee_cfg.prop_base_thrust_ratio = float(args.thrust_ratio)
        print(f"[run_modee] prop_base_thrust_ratio = {modee_cfg.prop_base_thrust_ratio}")
    if args.thrust_max_each is not None:
        modee_cfg.thrust_max_each_n = float(args.thrust_max_each)
        print(f"[run_modee] thrust_max_each_n = {modee_cfg.thrust_max_each_n}")
    if args.thrust_min_each is not None:
        modee_cfg.wbc_thrust_min_each_n = float(args.thrust_min_each)
        print(f"[run_modee] wbc_thrust_min_each_n = {modee_cfg.wbc_thrust_min_each_n}")
    if args.wbc_w_t_ref is not None:
        modee_cfg.wbc_w_t_ref = float(args.wbc_w_t_ref)
        print(f"[run_modee] wbc_w_t_ref = {modee_cfg.wbc_w_t_ref}")
    if args.wbc_w_tsum_ref is not None:
        modee_cfg.wbc_w_tsum_ref = float(args.wbc_w_tsum_ref)
        print(f"[run_modee] wbc_w_tsum_ref = {modee_cfg.wbc_w_tsum_ref}")
    if args.swing_kp_xy is not None:
        modee_cfg.swing_kp_xy = float(args.swing_kp_xy)
        print(f"[run_modee] swing_kp_xy = {modee_cfg.swing_kp_xy}")
    if args.swing_kd_xy is not None:
        modee_cfg.swing_kd_xy = float(args.swing_kd_xy)
        print(f"[run_modee] swing_kd_xy = {modee_cfg.swing_kd_xy}")
    if args.swing_kp_z is not None:
        modee_cfg.swing_kp_z = float(args.swing_kp_z)
        print(f"[run_modee] swing_kp_z = {modee_cfg.swing_kp_z}")
    if args.swing_kd_z is not None:
        modee_cfg.swing_kd_z = float(args.swing_kd_z)
        print(f"[run_modee] swing_kd_z = {modee_cfg.swing_kd_z}")

    if bool(getattr(args, "stance_use_props", False)):
        modee_cfg.stance_use_props = True
        print(f"[run_modee] stance_use_props = {modee_cfg.stance_use_props}")

    if args.leg_model is not None:
        modee_cfg.leg_model = str(args.leg_model).strip().lower()
        print(f"[run_modee] leg_model = {modee_cfg.leg_model}")

    if args.mpc_w_vxy is not None:
        modee_cfg.mpc_w_vxy = float(args.mpc_w_vxy)
        print(f"[run_modee] mpc_w_vxy = {modee_cfg.mpc_w_vxy}")
    if args.axy_damp is not None:
        modee_cfg.axy_damp = float(args.axy_damp)
        print(f"[run_modee] axy_damp = {modee_cfg.axy_damp}")
    if args.ki_xy is not None:
        modee_cfg.ki_xy = float(args.ki_xy)
        print(f"[run_modee] ki_xy = {modee_cfg.ki_xy}")
    if args.v_int_max is not None:
        modee_cfg.v_int_max = float(args.v_int_max)
        print(f"[run_modee] v_int_max = {modee_cfg.v_int_max}")
    
    # Foot velocity LPF (flight phase noise rejection)
    if args.no_foot_vel_lpf:
        modee_cfg.use_foot_vel_lpf = False
        print("[run_modee] use_foot_vel_lpf = False (disabled)")
    elif args.foot_vel_lpf:
        modee_cfg.use_foot_vel_lpf = True
        print("[run_modee] use_foot_vel_lpf = True (enabled)")
    if args.foot_vel_lpf_tau is not None:
        modee_cfg.foot_vel_lpf_tau = float(args.foot_vel_lpf_tau)
        print(f"[run_modee] foot_vel_lpf_tau = {modee_cfg.foot_vel_lpf_tau}")

    # Build ModeELCMConfig with overrides
    lcm_cfg = ModeELCMConfig()
    if args.lcm_url is not None:
        lcm_cfg.lcm_url = str(args.lcm_url)
    if args.max_cmd_vel is not None:
        lcm_cfg.max_cmd_vel = float(args.max_cmd_vel)
    if args.print_hz is not None:
        lcm_cfg.print_hz = float(args.print_hz)
    if args.tau_out_max is not None:
        lcm_cfg.tau_out_max_nm = float(args.tau_out_max)
    if args.tau_out_scale is not None:
        lcm_cfg.tau_out_scale = float(args.tau_out_scale)

    print("=" * 70)
    print("HOPPER-AERO MODEE CONTROLLER (LCM)")
    print("- Always sends commands; underlying hopper_driver handles mode switching")
    print("- Right stick: desired vx/vy (same mapping as Hopper4.py)")
    print("- Press Y: start CSV log (saved under ~/hopper_logs/modee_csv/ by default). Log stops when program exits.")
    print("- Press POINT (the 'I' button): toggle zero-velocity HOLD (v_hat forced to 0; desired_v forced to 0).")
    print("  (Override directory: export MODEE_LOG_DIR=/your/path)")
    print("=" * 70)
    print(f"[run_modee] LCM URL: {lcm_cfg.lcm_url}")
    print(f"[run_modee] Safety limits:")
    print(f"  - tau_max: {modee_cfg.tau_cmd_max_nm} Nm")
    print(f"  - tau_sign: {modee_cfg.tau_cmd_sign}")
    print(f"  - tau_out_scale: {lcm_cfg.tau_out_scale}")
    print(f"  - tau_out_max: {lcm_cfg.tau_out_max_nm}")
    print(f"  - pwm_range: [{modee_cfg.pwm_min_us}, {modee_cfg.pwm_max_us}] us")
    print(f"  - thrust_ratio: {modee_cfg.prop_base_thrust_ratio}")
    print(f"  - thrust_max_each: {modee_cfg.thrust_max_each_n} N/arm")
    print(f"  - thrust_min_each: {modee_cfg.wbc_thrust_min_each_n} N/arm")
    print(f"  - wbc_w_t_ref: {modee_cfg.wbc_w_t_ref}")
    print(f"  - wbc_w_tsum_ref: {modee_cfg.wbc_w_tsum_ref}")
    print(f"  - swing_kp_xy: {modee_cfg.swing_kp_xy}, swing_kd_xy: {modee_cfg.swing_kd_xy} (XY)")
    print(f"  - swing_kp_z: {modee_cfg.swing_kp_z}, swing_kd_z: {modee_cfg.swing_kd_z} (Z)")
    print(f"  - use_foot_vel_lpf: {modee_cfg.use_foot_vel_lpf}")
    if modee_cfg.use_foot_vel_lpf:
        import math
        cutoff_hz = 1.0 / (2.0 * math.pi * modee_cfg.foot_vel_lpf_tau)
        print(f"  - foot_vel_lpf_tau: {modee_cfg.foot_vel_lpf_tau} s (~{cutoff_hz:.1f} Hz cutoff)")
    print("=" * 70)

    ctl = ModeELCMController(modee_cfg=modee_cfg, lcm_cfg=lcm_cfg)
    
    # Note: --force-arm flag is deprecated (always sends commands now)
    if args.force_arm:
        print("[run_modee] Note: --force-arm is deprecated; controller always sends commands")

    # Non-daemon threads so we can shutdown cleanly (close log file, flush buffers, etc.)
    lcm_thread = threading.Thread(target=ctl.run_lcm_handler, daemon=False)
    ctrl_thread = threading.Thread(target=ctl.run_controller, daemon=False)
    lcm_thread.start()
    ctrl_thread.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[modee] shutting down...")
    finally:
        ctl.running = False
        # Join to allow ModeELCMController.run_controller() to close the CSV log cleanly.
        try:
            ctrl_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            lcm_thread.join(timeout=2.0)
        except Exception:
            pass


if __name__ == "__main__":
    main()



