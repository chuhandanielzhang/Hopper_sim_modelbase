from __future__ import annotations

"""
ModeE core controller (real-robot version)
=========================================

This is the "modee" architecture used in MuJoCo:
  - Event-based hop phases (TD/COMP/PUSH/FLIGHT/APEX)
  - SRB condensed wrench MPC (stance) -> references (f_ref, thrust_ref)
  - SRB WBC-QP (OSQP) -> solves: GRF + 3-arm thrusts + (stance-only) motor torques
  - All control uses IMU + encoders only (no MuJoCo ground truth)

This file is MuJoCo-free and is meant to run on the real robot via LCM.
"""

from dataclasses import dataclass
import math
import numpy as np

# NOTE: hopper_controller is not a Python package by default (no __init__.py).
# Keep imports relative to the folder that runs the controller (same style as Hopper4.py).
from forward_kinematics import ForwardKinematics, InverseJacobian

from modee.controllers.motor_utils import MotorTableModel
from modee.controllers.wbc_qp_osqp import WBCQP, WBCQPConfig
from modee.controllers.mpc import MITCondensedWrenchMPC, MITCondensedWrenchMPCConfig


def _skew(r: np.ndarray) -> np.ndarray:
    x, y, z = [float(v) for v in np.asarray(r, dtype=float).reshape(3)]
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _vee_so3(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=float).reshape(3, 3)
    return np.array([E[2, 1], E[0, 2], E[1, 0]], dtype=float)


def _Rz(yaw: float) -> np.ndarray:
    c = float(math.cos(float(yaw)))
    s = float(math.sin(float(yaw)))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q = (q / n).astype(float)
    # keep w>=0 (avoid discontinuities)
    if float(q[0]) < 0.0:
        q = (-q).astype(float)
    return q


def _quat_to_R_wb(q_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix R_wb: body -> world. Quaternion is wxyz."""
    q = _quat_normalize_wxyz(q_wxyz)
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _R_to_rpy_xyz(R: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw (XYZ intrinsic) from R_wb."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    # roll
    roll = float(math.atan2(R[2, 1], R[2, 2]))
    # pitch
    pitch = float(math.atan2(-R[2, 0], math.sqrt(max(1e-12, R[2, 1] ** 2 + R[2, 2] ** 2))))
    # yaw
    yaw = float(math.atan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=float)


def _quat_from_omega_dt(omega_b: np.ndarray, dt: float) -> np.ndarray:
    w = np.asarray(omega_b, dtype=float).reshape(3)
    th = float(np.linalg.norm(w) * float(dt))
    if th < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = (w / np.linalg.norm(w)).astype(float)
    half = 0.5 * th
    return _quat_normalize_wxyz(np.array([math.cos(half), *(math.sin(half) * axis)], dtype=float))


def _quat_mul(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> np.ndarray:
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


class SimpleIMUAttitudeEstimator:
    """
    Minimal 'real robot style' attitude estimator:
      - propagate by gyro integration
      - correct tilt using accelerometer (no mag)
    """

    def __init__(self, kp_acc: float = 0.6, acc_g_min: float = 0.90, acc_g_max: float = 1.10, acc_lpf_tau: float = 0.25):
        self.kp_acc = float(kp_acc)
        self.acc_g_min = float(acc_g_min)
        self.acc_g_max = float(acc_g_max)
        self.acc_lpf_tau = float(acc_lpf_tau)
        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # wxyz, body->world
        self._acc_f = np.zeros(3, dtype=float)
        self._inited = False

    def reset(self) -> None:
        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self._acc_f = np.zeros(3, dtype=float)
        self._inited = False

    def update(self, *, omega_b: np.ndarray, acc_b: np.ndarray, dt: float) -> np.ndarray:
        dt = float(dt)
        omega_b = np.asarray(omega_b, dtype=float).reshape(3)
        acc_b = np.asarray(acc_b, dtype=float).reshape(3)

        # LPF accel (for tilt correction gate)
        if not bool(self._inited):
            self._acc_f = acc_b.copy()
            self._inited = True
        else:
            tau = float(max(1e-6, self.acc_lpf_tau))
            a = float(np.clip(dt / (tau + dt), 0.0, 1.0))
            self._acc_f = (1.0 - a) * self._acc_f + a * acc_b

        # gyro integration
        dq = _quat_from_omega_dt(omega_b, dt)
        self._q = _quat_mul(self._q, dq)
        self._q = _quat_normalize_wxyz(self._q)

        # accel correction (tilt only)
        a_norm = float(np.linalg.norm(self._acc_f))
        if a_norm > 1e-9:
            g = 9.81
            g_ratio = float(a_norm / g)
            if float(self.acc_g_min) <= g_ratio <= float(self.acc_g_max):
                # measured "down" direction in body
                a_b = (self._acc_f / a_norm).astype(float)
                # estimated "down" in body (from q): down_w = [0,0,-1], so down_b = R_bw * down_w = R_wb^T * down_w
                R_wb = _quat_to_R_wb(self._q)
                down_b = (R_wb.T @ np.array([0.0, 0.0, -1.0], dtype=float)).reshape(3)
                # tilt error axis ~ cross(down_b, a_b)
                e = np.cross(down_b, a_b)
                # small-angle correction in body frame
                omega_corr = float(self.kp_acc) * e
                dq2 = _quat_from_omega_dt(omega_corr, dt)
                self._q = _quat_mul(self._q, dq2)
                self._q = _quat_normalize_wxyz(self._q)

        return self._q.copy()


@dataclass
class ModeEConfig:
    # control rate
    dt: float = 0.002  # 500 Hz

    # physical params
    mass_kg: float = 3.75
    gravity: float = 9.81
    # COM offset in base frame (m). If unknown, keep zeros; tune later.
    # Computed from MuJoCo MJCF (`Hopper-modee-clean/mjcf/hopper_serial.xml`) at default pose.
    com_b: tuple[float, float, float] = (-2.79376456e-04, 1.68299070e-06, -5.72937376e-02)
    # Body inertia diagonal in BODY frame (kg*m^2). Needed for MPC linearization.
    # Computed from MuJoCo MJCF (`Hopper-modee-clean/mjcf/hopper_serial.xml`) as whole-body inertia about COM,
    # expressed in base/body frame (diagonal approximation; off-diagonals are small).
    I_body_diag: tuple[float, float, float] = (0.0716072799, 0.0716088488, 0.0579831725)

    # delta leg nominal "length" (vicon/delta z coordinate, meters)
    # Nominal "extended" leg length (m).
    # - delta: this is ||foot_vicon|| when the delta leg is at its nominal length.
    # - serial (MuJoCo roll/pitch/shift model): we will auto-override this to match the model geometry.
    leg_l0_m: float = 0.464

    # hop target (world z)
    hop_z0: float = 0.55
    hop_peak_z: float = 0.6  # Reduced from 0.7 to lower hop height
    stance_T: float = 0.38
    stance_min_T: float = 0.10
    flight_min_T: float = 0.10

    # ===== Debug / bring-up switches =====
    # User request: in STANCE phase, do NOT output any forces/torques (hardcode to zero).
    # This is for FLIGHT-only PD tuning; set to False when ready to tune stance.
    stance_zero_output: bool = False

    # ===== Falling cat (self-righting / recovery gating) =====
    # User request:
    # - When the robot is NOT upright / is highly tilted, keep leg target "factory/original" (0,0) and
    #   do NOT apply quaternion-mapped foot targeting.
    # - Once the robot is upright AND roll/pitch error is within 45deg, enable quaternion mapping again.
    # - While tilted/inverted, prioritize flipping upright (freeze vxy, disable MPC, disable push-catch).
    falling_cat_enable: bool = False
    # Only enable quaternion-mapped flight foot target when BOTH |roll| and |pitch| <= this (deg).
    falling_cat_quat_map_rp_deg: float = 45.0
    # Additionally require body +Z to point upward: z_w[2] > falling_cat_upright_z_w_min (0.0 => just >0).
    falling_cat_require_upright: bool = True
    falling_cat_upright_z_w_min: float = 0.0
    # While falling-cat is active (tilted/inverted), force desired_v_xy to zero.
    falling_cat_zero_desired_vxy: bool = True
    # While falling-cat is active in STANCE, force-disable MPC so QP focuses on attitude recovery.
    falling_cat_disable_mpc_in_stance: bool = True
    # While falling-cat is active, disable stance PUSH "v_to catch" to avoid large hop impulses while tilted.
    falling_cat_disable_push_catch: bool = True
    # Only enable quaternion mapping when in PWM mode (props enabled) AND angle < 45deg.
    # Otherwise keep original leg target (0,0,-l0) in all cases.
    falling_cat_require_pwm_mode: bool = True  # If True, quaternion mapping only when props are active

    # touchdown/liftoff detection on equivalent shift coordinate:
    #   q_shift = leg_length - leg_l0_m
    #   negative: compressed (stance phase, allow up to -0.02 m compression)
    #   positive: extended (flight phase)
    td_q_shift_gate: float = -0.02  # touchdown when q_shift <= -0.02 (compressed by at least 2cm)
    td_qd_shift_gate: float = -0.01  # touchdown when qd_shift < -0.01 (compressing)
    td_dq_shift_gate: float = -5e-5  # touchdown when dq < -5e-5 (compression increasing)
    td_vz_hat_max: float = 0.00

    # Liftoff: switch to flight when q_shift >= 0 (leg extended to or beyond nominal length)
    stance_lo_min_T: float = 0.10

    # ===== Signal conditioning (real-robot robustness) =====
    # 3-RSR delta Jacobian can amplify encoder noise near workspace edges, which shows up as:
    #   - jitter in foot velocity estimate (foot_vdot)
    #   - jitter in q_shift / qd_shift (phase machine chattering)
    #   - jitter in force->torque mapping (inv(J_inv^T))
    #
    # These simple filters make the controller much more tolerant to noisy qd/Jacobian.
    #
    # Low-pass on joint velocity used in kinematics/Jacobian (seconds). Set <=0 to disable.
    joint_vel_lpf_tau: float = 0.008
    # Low-pass on q_shift / qd_shift used for touchdown/liftoff detection (seconds). Set <=0 to disable.
    q_shift_lpf_tau: float = 0.010
    qd_shift_lpf_tau: float = 0.010
    # Debounce: require N consecutive samples to declare touchdown/liftoff.
    td_debounce_steps: int = 2
    lo_debounce_steps: int = 2

    # DLS / ridge regularization for delta Jacobian inversions.
    # When enabled, we compute a damped pseudo-inverse:
    #   A^+ = (A^T A + λ^2 I)^(-1) A^T
    # with λ = lambda_rel * ||A||_F.
    # This prevents inv(J_inv) / inv(J_inv^T) from exploding near singularities.
    delta_jacobian_dls_enable: bool = True
    delta_jacobian_dls_lambda_rel: float = 0.002

    # ===== Unified stance reference (single-mode; no COMP/PUSH switching) =====
    # When enabled, stance is controlled by a single smooth COM-z reference trajectory:
    #   (z_td, vz_td) -> (z_min, 0) -> (z_end, v_to)
    # where z_min (compression depth) is chosen adaptively from touchdown vertical speed to "soft land".
    use_unified_stance: bool = True
    # Approximate max upward deceleration (m/s^2) during landing. Smaller => deeper compression, softer landing.
    soft_land_a_max: float = 25.0
    # Time to reach max compression (s): t_comp ≈ |vz_td| / soft_land_a_max, clamped to keep numerics stable.
    soft_land_tc_min: float = 0.06
    soft_land_tc_max_ratio: float = 0.60  # t_comp <= ratio * stance_T
    # If True, retime the unified stance profile when the leg stops compressing earlier than planned
    # (qd_shift crosses from negative -> positive). This prevents the controller from continuing to
    # command a "braking" stance reference while the leg is already extending, which can cause MPC to
    # drop fz toward fz_min (and miss liftoff).
    stance_retime_on_qd_cross: bool = True
    stance_retime_qd_eps: float = 0.01
    stance_retime_early_margin_s: float = 0.02
    # Compression depth bounds (m) relative to touchdown height (base frame).
    soft_land_depth_min_m: float = 0.0
    # NOTE: for meaningful leg-only hopping, we need enough compression travel to generate takeoff velocity
    # without forcing an extra downward motion in the "push" segment. 0.12m was too small in practice.
    soft_land_depth_max_m: float = 0.25
    # Optional safety guard on minimum base height during stance reference generation (m).
    z_guard: float = 0.35

    # ===== Legacy COMP/PUSH shaping (kept for backward compatibility) =====
    # Used only when `use_unified_stance=False`.
    compress_depth_m: float = 0.05
    compress_s_max: float = 0.60
    compress_v_gain: float = 10.0
    compress_v_max: float = 1.5
    compress_kv: float = 15.0
    az_compress_mag: float = 8.0
    az_compress_brake_max: float = 12.0
    compress_end_qd_shift: float = -0.05
    impulse_shape: str = "sin2"
    dv_cmd_max: float = 1.60
    # Takeoff speed bounds used by the unified stance profile and PUSH "v_to catch" logic.
    # NOTE: Previous defaults (1.6~2.2 m/s) cannot reach hop_peak_z=0.81 from typical z_td (~0.44m).
    # Keep these as safety clamps (real robot), but set them high enough to not block the apex target.
    v_to_min: float = 0.0
    v_to_max: float = 3.0  # Reduced from 4.0 to limit maximum takeoff velocity

    # ===== Stance PUSH takeoff-velocity catch (leg-only hop) =====
    # Goal: when the leg returns to nominal length (q_shift -> 0), the upward COM/base velocity should reach v_to_cmd
    # (ballistic to hop_peak_z). In logs we often see MPC hugging fz_min (~40N) in PUSH, producing tiny hops.
    # This feature enforces a *dynamic* vertical force floor during PUSH so the leg must generate the needed impulse.
    stance_push_v_to_catch: bool = True
    # Small ramp-in time (s) to avoid a discontinuous force step exactly at PUSH start.
    stance_push_v_to_catch_ramp_s: float = 0.02
    # Numerical guard to avoid huge accelerations when dz_rem is tiny.
    stance_push_dz_min_m: float = 0.02

    # ===== Apex height convergence (per-hop outer loop) =====
    # Goal: use actual apex height from previous hop to correct takeoff velocity command.
    # This is the "robust/optimal" way: each hop updates v_to_cmd based on apex error (PI control).
    # Without this, model errors (energy loss, thrust estimation) cause persistent height errors.
    apex_use_feedback: bool = True
    # PI gains for apex height error -> v_to_cmd correction.
    # NOTE: kp/ki must be SMALL to avoid overshoot/oscillation (apex error is per-hop, not continuous).
    apex_kp: float = 0.05  # Reduced from 0.11: less aggressive height correction
    apex_ki: float = 0.01  # Reduced from 0.02: less aggressive integral correction
    apex_int_max: float = 0.5  # max integral accumulation (m/s) to prevent windup

    # ===== Flight foot placement =====
    # Flight foot placement uses Hopper4 Raibert (Kv/Kr) in heading frame.
    flight_kv: float = 0.13
    flight_kr: float = 0.09
    flight_stepper_lim_m: float = 0.12

    # swing (flight) foot-space torque reference (passed via QP tau_ref)
    # Hopper4-style decomposition:
    #   - Axial (along leg direction): kp_z/kd_z act on leg length + axial velocity
    #   - Perpendicular (⊥ leg direction): kp_xy/kd_xy act on foot Cartesian error, projected to ⊥ plane
    # NOTE: these are the exact knobs you want to tune to prevent flight over-extension.
    # Hopper4 defaults (see `Hopper-mujoco-standalone/hopper_controller/hopper_config.py`):
    #   k = 1000 N/m, b = 20 N/(m/s), Khp = 50, Khd = 1
    swing_kp_xy: float = 20.0   # Perpendicular position gain (N/m)
    swing_kd_xy: float = 1.8    # Perpendicular damping gain (N/(m/s))
    swing_kp_z: float = 1000.0  # Axial (virtual spring) stiffness (N/m), along leg direction
    swing_kd_z: float = 15.0    # Axial damping (N/(m/s)), along leg direction

    # Foot velocity low-pass filter (flight phase only, noise rejection for high kd)
    # When kd > 3, foot velocity noise (from motor encoder differentiation + Jacobian propagation)
    # can be amplified into significant damping force noise, causing torque jitter.
    # Enable this LPF to smooth foot velocity before feeding into the PD damping term.
    use_foot_vel_lpf: bool = True   # True = enable foot-space velocity LPF (recommended for kd > 3)
    foot_vel_lpf_tau: float = 0.015  # LPF time constant (s). Smaller = more filtering, more delay.
                                     # 0.015s → cutoff ~10Hz (good balance for kd=3~10)
                                     # 0.010s → cutoff ~15Hz (for kd > 10)

    # props / thrust
    # Treat 3 QP thrust variables as "per-arm total thrust" (N).
    # total baseline thrust = ratio*m*g
    # User request: props should handle more attitude work throughout stance/flight,
    # so the leg can focus more on velocity convergence.
    prop_base_thrust_ratio: float = 0.03
    # User request: stance attitude should be solvable by the LEG alone.
    # If False, props are disabled in STANCE (thrusts forced to 0; leg must provide all force/moments).
    # Flight still uses props for attitude.
    # NOTE (real robot bring-up): if you see pitch/roll diverge on hop 2+, enable props in stance.
    # Single-point contact forces couple velocity regulation ↔ attitude moments; props help decouple and prevent flips.
    stance_use_props: bool = True
    thrust_total_ratio_max: float = 0.50  # QP cap on sum(thrust) <= ratio*m*g (safety)
    # Per-arm thrust cap passed to MPC/WBC-QP (N).
    # NOTE: 10N/arm was not enough roll authority in logs (roll diverged while saturated).
    thrust_max_each_n: float = 25.0

    # ===== Stance prop usage policy =====
    # Minimum allowed total thrust during STANCE, as a ratio of (m*g).
    # - 0.0 means props are NOT forced on in stance; they can still be used if needed (up to max).
    # - >0.0 forces a baseline thrust sum in stance (useful if you want constant spin / baseline support).
    stance_thrust_sum_min_ratio: float = 0.0

    # prop geometry in base frame (meters); default is symmetric with GREEN on +X
    prop_arm_len_m: float = 0.569451

    # ===== Prop PWM channel mapping (REAL ROBOT) =====
    # We output a 6-length `pwm_us` vector that is sent directly to Betaflight via MSP_SET_MOTOR:
    #   motor_pwm_lcmt.pwm_values[0..5] -> motors M1..M6 on the flight controller.
    #
    # ModeE solves 3 thrust variables (one per *arm*) ordered consistently with `prop_positions_b`:
    #   arm 0 = RED, arm 1 = GREEN, arm 2 = BLUE
    #
    # Your real robot may have:
    # - 3 props total (one per arm), OR
    # - 6 props total (coaxial, two per arm), etc.
    #
    # We represent this as: for each arm, a tuple of 1+ PWM indices that belong to that arm.
    # The per-arm thrust is divided equally among its indices.
    #
    # Your measured order (top view, clockwise): GREEN -> PWM[1], then PWM[3], then PWM[2].
    # And you currently have ONLY 3 propellers installed, so we map:
    #   GREEN arm -> (1,)
    #   BLUE  arm -> (3,)   (clockwise next from GREEN)
    #   RED   arm -> (2,)   (clockwise next from BLUE)
    #
    # Unused PWM indices will get 0 thrust -> pwm_min_us (1000us).
    # If you re-wire / re-map, update this tuple.
    prop_pwm_idx_per_arm: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
        (2,),  # RED arm
        (1,),  # GREEN arm
        (3,),  # BLUE arm
    )

    # motor torque limits (delta 3-RSR), per user: 10 Nm
    # IMPORTANT:
    # This limit is used INSIDE ModeECore (MPC/WBC-QP feasibility).
    # For bring-up safety, prefer limiting torques at the output layer in `modee/lcm_controller.py`
    # (tau_out_max / tau_out_scale) so the internal solver still behaves normally.
    tau_cmd_max_nm: tuple[float, float, float] = (20.0, 20.0, 20.0)
    # Motor torque sign convention (real robot wiring/driver):
    # Keep this as a *motor wiring/driver sign* override:
    #   +1 means "as modeled", -1 flips the commanded motor torque direction.
    # This applies to BOTH: stance torque mapping (A_tau_f) and flight swing tau_ref.
    tau_cmd_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # QP weights
    # Weight for tracking swing spring tau_ref in flight phase.
    # Higher value = QP prioritizes following swing spring torque (better leg length control).
    # NOTE: Setting this too high (>>1) may cause QP to sacrifice other objectives (thrust smoothing, attitude balance).
    # Current value (1.0) is aggressive but should help prevent leg over-extension.
    wbc_w_tau_ref_flight: float = 1.0

    # ===== Prop thrust smoothing (reduce left-right wobble / chatter) =====
    # These do NOT change the physics model; they just bias the QP to choose smoother thrusts.
    # - wbc_w_t_ref: track previous thrust solution (or provided thrust_ref) to reduce rapid thrust swapping.
    # - wbc_w_tsum_ref: keep total thrust near thrust_sum_ref (optional).
    # - wbc_thrust_min_each_n: lower bound per-arm thrust (N). Setting a small >0 can prevent motors
    #   from hitting pwm_min (stop/start), which often causes wobble.
    wbc_w_t_ref: float = 1e-2
    wbc_w_tsum_ref: float = 0.0
    wbc_thrust_min_each_n: float = 0.5
    # - wbc_w_f_ref: track MPC contact force reference f_ref (higher -> follow MPC f_ref more strictly).
    # Lower value allows leg to deviate from f_ref to generate attitude torque (leg + props share balance).
    wbc_w_f_ref: float = 0.001  # Reduced from 0.01: allow leg to participate in attitude balance

    # ===== WBC-QP regularization (bias who does the work) =====
    # These weights bias the QP solution distribution between leg contact force vs prop thrust:
    # - Larger wbc_w_t => thrust is "expensive" => QP prefers using the leg when feasible.
    # - Larger wbc_w_f => contact force is "expensive" => QP prefers using thrust when feasible.
    #
    # For your desired behavior ("stance mostly leg, props as disturbance rejection"):
    #   keep wbc_w_t noticeably larger than wbc_w_f.
    wbc_w_f: float = 1e-5
    wbc_w_t: float = 1e-1

    # ===== Contact friction (controller-side) =====
    # Must match the ground/contact physics as closely as possible (e.g. MuJoCo friction).
    # This parameter is used by BOTH:
    # - condensed wrench MPC (friction cone)
    # - WBC-QP (friction pyramid / cone approximation)
    mu: float = 0.6

    # ===== Leg kinematics backend =====
    # - "delta": real-robot 3-RSR delta motor angles (uses `forward_kinematics.py`)
    # - "serial": MuJoCo serial-equivalent leg (roll/pitch/shift) used by hopper_serial.xml
    leg_model: str = "delta"

    # Serial leg geometry (must match hopper_serial.xml):
    # base_link -> hip origin offset (m), and hip -> foot body origin offset (m).
    serial_hip_z_off_m: float = 0.0416   # base_link to hip is at z=-0.0416
    serial_foot_z_m: float = 0.5237      # Leg_Link to Foot_Link offset magnitude along -Z

    # stance horizontal regulation fallback (when MPC infeasible)
    axy_damp: float = 0.5
    # Integral gain on horizontal velocity error (applied symmetrically to X and Y).
    ki_xy: float = 0.0
    v_int_max: float = 0.30
    # Velocity fusion scaling (applied symmetrically to X and Y).
    # (Kept name for backward-compat.)
    v_fuse_vx_scale: float = 1.0

    # Slip detection reference speed (m/s) for velocity fusion gating.
    # When foot velocity prediction exceeds this value, fusion weight is reduced.
    # Default 1.0 m/s is conservative; increase to 3.0-5.0 if leg retraction during
    # STANCE:PUSH causes false slip detection (preventing velocity fusion).
    # Set to a very large value (e.g., 100.0) to effectively disable slip detection
    # and use direct leg kinematics like Hopper4.
    v_slip_ref: float = 3.0
    
    # Enable slip detection gating for velocity fusion.
    # If False, always use leg kinematics measurement (like Hopper4, no gating).
    # If True, use slip detection to reduce fusion weight when foot velocity is large.
    v_use_slip_detection: bool = False

    # ===== MPC Velocity Filter (Paper-Grade Baseline) =====
    # Advanced adaptive velocity filter for MPC convergence baseline.
    # Uses adaptive filtering with phase-aware tuning, outlier rejection, and confidence-based fusion.
    # Reference: "Robust Velocity Estimation for Legged Robots via Complementary Filtering"
    # and "Adaptive Kalman Filtering for State Estimation in Contact-Rich Manipulation"
    use_mpc_velocity_filter: bool = False  # Enable advanced velocity filter for MPC input
    # Adaptive filter time constants (s). Larger = smoother but more delay.
    # In STANCE: trust leg kinematics more (smaller tau = faster response)
    # In FLIGHT: reduce noise from IMU drift (larger tau = smoother)
    mpc_v_filter_tau_stance: float = 0.2   # Stance phase: fast response (leg kinematics reliable)
    mpc_v_filter_tau_flight: float = 0   # Flight phase: smooth (reduce IMU noise/drift)
    # Outlier rejection threshold (m/s). Velocity changes exceeding this are heavily filtered.
    mpc_v_filter_outlier_th: float = 2.0    # Reject jumps > 2 m/s as outliers
    # Confidence weighting: weight of leg kinematics in stance (0=IMU only, 1=kinematics only)
    mpc_v_filter_kinematics_weight: float = 0.85  # In stance, trust kinematics 85%
    # Minimum filter coefficient (prevents filter from being too slow)
    mpc_v_filter_alpha_min: float = 0.10    # Minimum 10% new value per step
    # Maximum filter coefficient (prevents filter from being too fast)
    mpc_v_filter_alpha_max: float = 0.95    # Maximum 95% new value per step

    # MPC velocity tracking weight (applied symmetrically to vx and vy).
    # IMPORTANT: this is a cost weight and must be >= 0 (negative would make the optimization ill-posed).
    mpc_w_vxy: float = 15.0  # MIT Mini Cheetah default: w_vy=10.0, w_vx=12.0 (we use symmetric 12.0)

    # ===== Debug: MPC enable/disable for staged tuning =====
    # If False, stance uses fallback PD (axy_damp + ki_xy) instead of MPC.
    # This is useful for staged tuning:
    #   1. Disable MPC, tune attitude control (kR/kW, thrust) to get stable in-place hopping
    #   2. Enable MPC with low weights, tune MPC parameters
    #   3. Gradually increase MPC weights for better velocity convergence
    use_mpc_in_stance: bool = False  # Enable MPC (start with low weights)

    # MPC
    mpc_dt: float = 0.02
    mpc_N: int = 10
    # Expose a few MPC knobs for soft-landing / balance coupling experiments
    # For real-robot hopping, allowing MPC to choose fz < mg often causes an unintended "brake"
    # during early push/rebound. Set a conservative default >= mg; tune with care for your hardware.
    mpc_fz_min: float = 30.0
    # IMPORTANT (root cause of “越刹越快”):
    # If MPC is asked to track roll/pitch, it will use the single contact GRF (via r×f) to generate
    # attitude torque. With the foot ahead/behind the COM, the GRF needed for attitude can *fight*
    # velocity convergence and even create positive feedback on vx (force pushes in the same direction
    # as velocity error).
    #
    # In ModeE, attitude is already handled by SO(3) + WBC-QP (leg + props share Tau_des), so we keep
    # MPC roll/pitch weights at 0 by default and let MPC focus on velocity/height references.
    mpc_w_roll: float = 0.0
    mpc_w_pitch: float = 0.0

    # PWM limits
    pwm_min_us: float = 1000.0
    pwm_max_us: float = 1800.0

    # ===== Propeller PWM mapping method =====
    # If True: use Hopper4-style k_thrust square-root relationship (pwm = 1000 + sqrt(thrust / k_thrust))
    # If False: use MotorTableModel lookup table (interpolation from measured data)
    use_hopper4_pwm_mapping: bool = False
    # Hopper4 thrust coefficient (N per (pwm_delta)^2, where pwm_delta = pwm - 1000)
    # Default from Hopper4: k_thrust = 1.47e-4
    # Formula: thrust = k_thrust * (pwm - 1000)^2
    # Inverse: pwm = 1000 + sqrt(thrust / k_thrust)
    # NOTE: 1.47e-5 makes even ~5N/arm saturate at pwm_max=1300 (sqrt mapping), starving attitude torque.
    prop_k_thrust: float = 1.47e-4

    # Use FC quaternion directly (recommended for real robot) vs. re-estimate from gyro+acc
    use_fc_quat: bool = True

    # ===== Flight phase attitude control gains (SO(3) roll/pitch) =====
    # Separate gains for roll and pitch axes to allow independent tuning.
    # kR: attitude error gain (proportional, larger = stiffer response)
    # kW: angular velocity damping gain (derivative, larger = more damping/braking)
    # tau_rp_max: maximum desired roll/pitch torque (Nm)  pitchdebug
    # NOTE (real robot): keep flight attitude authority reasonably high; thrust limits still provide safety.
    flight_kR_roll: float = 50.0
    flight_kW_roll: float = 70.0
    flight_kR_pitch: float = 50.0
    flight_kW_pitch: float = 70.0
    flight_tau_rp_max: float = 130.0


def _make_mpc(cfg: ModeEConfig) -> MITCondensedWrenchMPC:
    return MITCondensedWrenchMPC(
        MITCondensedWrenchMPCConfig(
            dt=float(cfg.mpc_dt),
            N=int(cfg.mpc_N),
            mu=float(cfg.mu),
            fz_min=float(cfg.mpc_fz_min),
            fz_max=220.0,
            # Allow enough horizontal GRF so the leg can generate roll/pitch moments in stance
            # (single-leg attitude requires non-zero f_xy).
            fxy_max_ratio=1.0,
            w_py=0.0,
            w_pz=80.0,
            # User request: forbid XY asymmetry in velocity regulation.
            w_vx=float(cfg.mpc_w_vxy),
            w_vy=float(cfg.mpc_w_vxy),
            w_vz=220.0,
            w_roll=float(cfg.mpc_w_roll),
            w_pitch=float(cfg.mpc_w_pitch),
            w_yaw=0.0,
            w_wx=0.0,
            w_wy=0.0,
            w_wz=0.0,
            w_tsum_ref=0.0,
            w_tau=0.0,
            enforce_tau_limits=True,
            alpha_u_f=1e-4,  # MIT Mini Cheetah default: 1e-4
            alpha_u_t=2e-4,  # MIT Mini Cheetah default: 2e-4
            rp_limit_deg=None,
            re_setup_each_solve=True,
        )
    )


class ModeECore:
    """
    Pure controller core (no LCM, no MuJoCo).

    Inputs:
      - joint_pos, joint_vel: delta motor angles and velocities (3,)
      - imu_*: gyro/acc (body frame), optional quat (wxyz)
      - desired_v_xy: desired world velocity [vx, vy]

    Outputs:
      - tau_cmd (3,) motor torques (Nm)
      - pwm_us (6,) prop PWM microseconds
      - debug/info dict (phases, estimates, etc)
    """

    def __init__(self, cfg: ModeEConfig):
        self.cfg = cfg
        self.dt = float(cfg.dt)
        self.mass = float(cfg.mass_kg)
        self.gravity = float(cfg.gravity)
        self.com_b = np.asarray(cfg.com_b, dtype=float).reshape(3)
        self.I_body = np.diag(np.asarray(cfg.I_body_diag, dtype=float).reshape(3))

        # frames: base (z up) vs delta/vicon (z down)
        self.robot2vicon = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)

        # Leg kinematics backend selection
        self._leg_model = str(getattr(cfg, "leg_model", "delta")).strip().lower()
        if self._leg_model not in ("delta", "serial"):
            print(f"[modee] WARN: unknown leg_model='{self._leg_model}', falling back to 'delta'")
            self._leg_model = "delta"

        # delta kinematics (real robot)
        self.fk = ForwardKinematics() if self._leg_model == "delta" else None
        self.kin = InverseJacobian(use_simulink_filter=False, forgetting_factor=0.95, dt=float(self.dt)) if self._leg_model == "delta" else None

        # For serial MuJoCo leg, override l0 to match the model geometry (so TD/LO detection works).
        # When roll=pitch=0 and shift=0 (joint lower limit), the foot is at:
        #   z = -(serial_hip_z_off_m + serial_foot_z_m) in base frame, so ||foot|| ≈ serial_hip_z_off_m + serial_foot_z_m.
        if self._leg_model == "serial":
            try:
                l0_ser = float(abs(float(self.cfg.serial_hip_z_off_m)) + abs(float(self.cfg.serial_foot_z_m)))
                if l0_ser > 1e-6:
                    self.cfg.leg_l0_m = l0_ser
            except Exception:
                pass
            # Serial plant uses a PRISMATIC "shift" joint, so the 3rd actuator command is a generalized FORCE (N),
            # not a torque (Nm). The default (20Nm) is far too small to support the robot weight in MuJoCo.
            # We therefore boost ONLY the 3rd limit when it looks like a real-robot torque tuple was provided.
            try:
                tmax = np.asarray(self.cfg.tau_cmd_max_nm, dtype=float).reshape(3)
                if float(abs(tmax[2])) < 200.0:
                    self.cfg.tau_cmd_max_nm = (float(tmax[0]), float(tmax[1]), 2500.0)
            except Exception:
                pass
        
        # Foot velocity LPF state (flight phase only)
        self._foot_vrel_lpf = np.zeros(3, dtype=float)
        self._foot_vrel_lpf_init = False

        # Joint velocity LPF (used by kinematics to reduce Jacobian/qd jitter)
        self._joint_vel_lpf = np.zeros(3, dtype=float)
        self._joint_vel_lpf_init = False

        # Shift-coordinate LPF + debounce (phase robustness)
        self._q_shift_lpf: float = 0.0
        self._qd_shift_lpf: float = 0.0
        self._shift_lpf_init: bool = False
        self._td_debounce_count: int = 0
        self._lo_debounce_count: int = 0

        # MPC + WBC
        self.mpc = _make_mpc(cfg)
        self.wbc = WBCQP(
            WBCQPConfig(
                mu=float(cfg.mu),
                fz_min=0.0,
                fz_max=220.0,
                thrust_total_ratio_max=float(cfg.thrust_total_ratio_max),
                thrust_min_each=float(max(0.0, float(cfg.wbc_thrust_min_each_n))),
                w_f=float(max(0.0, float(getattr(cfg, "wbc_w_f", 1e-4)))),
                w_t=float(max(0.0, float(getattr(cfg, "wbc_w_t", 1e-4)))),
                w_f_ref=float(max(0.0, float(cfg.wbc_w_f_ref))),
                w_t_ref=float(max(0.0, float(cfg.wbc_w_t_ref))),
                w_tsum_ref=float(max(0.0, float(cfg.wbc_w_tsum_ref))),
                # ATTITUDE FIRST, but allow leg to also contribute: Balance between fxy tracking and attitude stability.
                # Increased w_slack_Fxy relative to w_slack_tau_xy to let leg deviate from f_ref for attitude torque.
                # Robot flipped when attitude priority was too low (5e4).
                w_slack_Fxy=8e4,  # increased from 2e4: allow leg fxy to deviate more for attitude balance
                w_slack_Fz=8e4,
                # Attitude must have higher priority to prevent flip.
                w_slack_tau_xy=8e4,  # same as Fxy now: leg and props share attitude control
                w_slack_tau_z=1e3,
                w_slack_Fxy_flight=2e3,
                w_slack_Fz_flight=6e3,
                w_slack_tau_flight_xy=2e3,
                w_slack_tau_flight_z=8e2,
                enable_tau_vars=True,
                w_tau=0.0,
                w_tau_ref=float(cfg.wbc_w_tau_ref_flight),
            )
        )

        # motor PWM map (thrust->PWM)
        self.use_hopper4_pwm = bool(cfg.use_hopper4_pwm_mapping)
        self.prop_k_thrust = float(cfg.prop_k_thrust)
        if not bool(self.use_hopper4_pwm):
            # Use lookup table (MotorTableModel) when Hopper4 mapping is disabled
            self.motor_table = MotorTableModel.default_from_table()
            # Clamp to FC configured range if needed
            self.motor_table.pwm_min_us = float(cfg.pwm_min_us)
            self.motor_table.pwm_max_us = float(cfg.pwm_max_us)
        else:
            self.motor_table = None

        # attitude estimator
        self.att = SimpleIMUAttitudeEstimator(kp_acc=0.6, acc_g_min=0.90, acc_g_max=1.10, acc_lpf_tau=0.25)

        # estimator state
        self._v_hat_w = np.zeros(3, dtype=float)
        self._p_hat_w = np.array([0.0, 0.0, float(cfg.hop_z0)], dtype=float)
        self._v_hat_inited = False
        # MPC velocity filter state (paper-grade adaptive filter)
        self._v_mpc_filtered = np.zeros(3, dtype=float)  # Filtered velocity for MPC
        self._v_mpc_filter_inited = False
        self._v_mpc_prev = np.zeros(3, dtype=float)  # Previous filtered value (for outlier detection)
        self._z_hat_contact_filt: float | None = None
        # Velocity fusion LPF: larger tau = smoother estimate, less noise
        # 0.05 was too noisy (velocity oscillated 0.5m/s per 20ms)
        self._v_hat_lpf_tau = 0.15  # increased for stability
        self._v_int_xy = np.zeros(2, dtype=float)
        # User override: freeze internal velocity estimate to zero (used to stop drift on demand).
        self._user_zero_vel_hold: bool = False

        # phase state
        self.sim_time = 0.0
        self._stance = False
        self._td_t: float | None = None
        self._lo_t: float | None = None
        self._q_shift_prev: float | None = None
        self._qd_shift_prev: float | None = None
        self._q_shift_td: float | None = None
        self._prev_vz: float | None = None

        # stance reference profile (unified stance: soft landing + push-off, no discrete COMP/PUSH switching)
        self._stance_prof_inited = False
        self._stance_t_comp: float | None = None
        self._stance_depth_tgt_m: float = 0.0
        self._stance_com_off_z: float = 0.0
        # Cached stance reference endpoints for event-based retiming (COM/world-z).
        self._stance_z_end: float | None = None
        # If True, we have retimed the stance profile once this stance (to avoid repeated edits).
        self._stance_retimed: bool = False
        # Quintic (minimum-jerk) z(t) coefficients in COM-z (world), used by stance MPC reference generation.
        # poly: z(t) = c0 + c1 t + ... + c5 t^5
        self._stance_poly1: np.ndarray | None = None  # [0, t_comp]
        self._stance_poly2: np.ndarray | None = None  # [t_comp, stance_T]
        self._stance_T1: float = 0.0
        self._stance_T2: float = 0.0
        self._v_to_cmd = float(cfg.v_to_min)

        # apex + swing gating
        self._apex_reached = False
        # Apex height convergence (per-hop outer loop): record liftoff/apex states for feedback.
        self._z_lo: float | None = None  # liftoff height (base z, world frame)
        self._vz_lo: float | None = None  # liftoff vertical velocity (world frame)
        self._z_apex_actual: float | None = None  # actual apex height from previous hop (base z, world frame)
        self._apex_err_int: float = 0.0  # integral of apex height error (for PI control)

        # last solution hold (robustness)
        self._wbc_last_t = np.zeros(3, dtype=float)
        self._wbc_last_f = np.zeros(3, dtype=float)
        self._tau_cmd_prev = np.zeros(3, dtype=float)

        # precompute prop positions in base frame (GREEN on +X)
        L = float(cfg.prop_arm_len_m)
        # order: [RED, GREEN, BLUE] (visual naming; GREEN forward)
        self.prop_positions_b = np.array(
            [
                [-0.5 * L, +math.sqrt(3) * 0.5 * L, 0.0],
                [+1.0 * L, 0.0, 0.0],
                [-0.5 * L, -math.sqrt(3) * 0.5 * L, 0.0],
            ],
            dtype=float,
        )

        # Validate prop PWM mapping (avoid silent duplicates / out-of-range indices)
        try:
            groups = tuple(tuple(int(x) for x in g) for g in cfg.prop_pwm_idx_per_arm)
            flat = [i for g in groups for i in g]
            if (len(groups) != 3) or any(len(g) < 1 for g in groups):
                raise ValueError("prop_pwm_idx_per_arm must be 3 groups, each with >= 1 index")
            if any((i < 0) or (i > 5) for i in flat):
                raise ValueError(f"prop_pwm_idx_per_arm out of range: {groups}")
            if len(set(flat)) != len(flat):
                raise ValueError(f"prop_pwm_idx_per_arm has duplicate indices: {groups}")
            self._prop_pwm_groups = groups
        except Exception as e:
            # Fallback to 3 motors on indices 0/1/2 (safe-ish default)
            print(f"[modee] WARN: invalid prop_pwm_idx_per_arm ({e}); falling back to ((0,),(1,),(2,))")
            self._prop_pwm_groups = ((0,), (1,), (2,))

        # 3-RSR torque map workspace clamp (same as MuJoCo demo)
        self._delta_ws = dict(xy=0.27, z_min=0.22, z_max=0.49, z_off=0.03)

    def user_reset(self) -> None:
        """
        User-requested reset (triggered by gamepad Y on the PC side).

        Purpose:
        - Zero drifting estimator/integrator states so a new experiment/log segment starts clean.
        - Keep the controller running; do NOT change driver mode here.
        """
        # Estimator/integrator states
        self._v_hat_w[:] = 0.0
        self._foot_vrel_lpf[:] = 0.0
        self._foot_vrel_lpf_init = False
        self._joint_vel_lpf[:] = 0.0
        self._joint_vel_lpf_init = False
        self._q_shift_lpf = 0.0
        self._qd_shift_lpf = 0.0
        self._shift_lpf_init = False
        self._td_debounce_count = 0
        self._lo_debounce_count = 0
        self._v_hat_inited = False
        self._v_int_xy[:] = 0.0
        self._z_hat_contact_filt = None
        self._prev_vz = None
        # Reset MPC velocity filter
        self._v_mpc_filtered[:] = 0.0
        self._v_mpc_filter_inited = False
        self._v_mpc_prev[:] = 0.0

        # Reset stance reference profile (re-initialize on next touchdown)
        self._stance_prof_inited = False
        self._stance_t_comp = None
        self._stance_depth_tgt_m = 0.0
        self._stance_com_off_z = 0.0
        self._stance_z_end = None
        self._stance_retimed = False
        self._stance_poly1 = None
        self._stance_poly2 = None
        self._stance_T1 = 0.0
        self._stance_T2 = 0.0

        # Rebase XY position for nicer logs (doesn't materially change the control because references are relative)
        self._p_hat_w[0] = 0.0
        self._p_hat_w[1] = 0.0

        # Reset swing placement memory/gating
        self._apex_reached = False
        
        # Reset apex feedback state (optional: keep integral for persistent learning)
        # User can choose: reset on user_reset() vs. keep integral across experiments
        # For now, reset to allow clean experiments:
        self._z_lo = None
        self._vz_lo = None
        self._z_apex_actual = None
        # NOTE: we keep _apex_err_int to allow persistent learning across resets if desired.
        # To reset it too, uncomment: self._apex_err_int = 0.0

        # Reset attitude estimator state (only used if use_fc_quat=False)
        try:
            self.att.reset()
        except Exception:
            pass

    def user_zero_velocity_hold(self, enable: bool) -> None:
        """
        User-requested "HARD STOP" of the internal velocity estimate.

        When enabled:
        - v_hat is forced to 0 every control step (no IMU integration drift)
        - integrators are kept at 0
        - flight foot placement stops drifting when desired_v==0

        This is a debugging / operator convenience feature (not physically enforcing real velocity to 0).
        """
        self._user_zero_vel_hold = bool(enable)
        if bool(self._user_zero_vel_hold):
            # Make the state look like a fresh start immediately.
            self.user_reset()

    @staticmethod
    def _pinv_ridge(A: np.ndarray, lambda_rel: float) -> np.ndarray:
        """
        Damped least-squares (ridge) pseudo-inverse:
          A^+ = (A^T A + λ^2 I)^(-1) A^T
        with λ = lambda_rel * ||A||_F.

        This is a small, dependency-free way to prevent Jacobian inversions from exploding when A
        becomes ill-conditioned (common for delta/3-RSR near workspace edges).
        """
        A = np.asarray(A, dtype=float)
        if A.shape != (3, 3):
            A = A.reshape(3, 3)
        lam_rel = float(max(0.0, float(lambda_rel)))
        if lam_rel <= 0.0:
            # Least-squares pseudo-inverse (still better than hard crash if singular)
            return np.linalg.pinv(A)
        fro = float(np.linalg.norm(A, ord="fro"))
        lam = float(lam_rel * max(1e-12, fro))
        M = (A.T @ A + (lam * lam) * np.eye(3, dtype=float)).astype(float)
        try:
            return np.linalg.solve(M, A.T).astype(float)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A)

    def _stable_inv3(self, A: np.ndarray) -> np.ndarray:
        """
        Robust inverse for 3x3 matrices used in delta kinematics:
        - If DLS is enabled, return ridge pseudo-inverse (stable near singularities).
        - Else, try exact inverse and fall back to pinv.
        """
        A = np.asarray(A, dtype=float).reshape(3, 3)
        if bool(getattr(self.cfg, "delta_jacobian_dls_enable", True)):
            lam_rel = float(getattr(self.cfg, "delta_jacobian_dls_lambda_rel", 0.0))
            return self._pinv_ridge(A, lam_rel)
        try:
            return np.linalg.inv(A).astype(float)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A)

    def compute_tau_from_force_base(
        self,
        *,
        joint_pos: np.ndarray,
        f_base: np.ndarray,
        use_contact_site_map: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Map a desired 3D force in BODY/BASE frame (+X forward, +Y left, +Z up)
        to delta motor torques using the same Jacobian convention as ModeE.

        Conventions:
        - FK/Jacobian are in delta/vicon frame (+Z down).
        - `robot2vicon` converts BASE(+Z up) -> DELTA(+Z down).
        - Mapping uses: tau = inv(J_inv^T) * f_delta

        Args:
          joint_pos: (3,) motor angles [0,1,2] in physical motor order.
          f_base: (3,) force in BASE frame (+Z up).
          use_contact_site_map: if True, apply the same 3cm contact-site offset + workspace clamp
            used by the stance mapping (`A_tau_f`). This makes stand-alone force tests match stance.

        Returns:
          tau: (3,) motor torques (Nm) in physical motor order [0,1,2]
          foot_vicon: (3,) FK foot position in delta/vicon frame (+Z down)
        """
        joint_pos = np.asarray(joint_pos, dtype=float).reshape(3)
        f_base = np.asarray(f_base, dtype=float).reshape(3)

        if self._leg_model == "serial":
            foot_b, J_body = self._serial_leg_fk_jac(
                q_roll=float(joint_pos[0]),
                q_pitch=float(joint_pos[1]),
                q_shift=float(joint_pos[2]),
            )
            foot_vicon = (self.robot2vicon @ np.asarray(foot_b, dtype=float).reshape(3)).reshape(3)
            tau = (np.asarray(J_body, dtype=float).reshape(3, 3).T @ f_base.reshape(3)).reshape(3)
            tau_sign = np.asarray(self.cfg.tau_cmd_sign, dtype=float).reshape(3)
            tau = (tau_sign.reshape(3) * tau.reshape(3)).reshape(3).astype(float)
            return tau, foot_vicon

        # delta (real robot)
        if self.fk is None or self.kin is None:
            raise RuntimeError("delta leg model requested but kinematics is not initialized")

        foot_vicon, _ = self.fk.forward_kinematics(joint_pos)
        foot_vicon = np.asarray(foot_vicon, dtype=float).reshape(3)

        x3 = foot_vicon.copy()
        if bool(use_contact_site_map):
            x3[2] = float(x3[2] + float(self._delta_ws["z_off"]))
            x3[0] = float(np.clip(x3[0], -float(self._delta_ws["xy"]), +float(self._delta_ws["xy"])))
            x3[1] = float(np.clip(x3[1], -float(self._delta_ws["xy"]), +float(self._delta_ws["xy"])))
            x3[2] = float(np.clip(x3[2], float(self._delta_ws["z_min"]), float(self._delta_ws["z_max"])))

        # Compute inverse Jacobian at x3 (delta/vicon frame)
        J_inv_map, _ = self.kin.inverse_jacobian(x3, np.zeros(3, dtype=float), theta=None)
        J_inv_map = np.asarray(J_inv_map, dtype=float).reshape(3, 3)
        inv_Jt = self._stable_inv3(J_inv_map.T)

        # base(+Z up) -> delta(+Z down)
        f_delta = (self.robot2vicon @ f_base.reshape(3)).reshape(3)

        # tau = inv(J_inv^T) * f_delta
        tau = (inv_Jt @ f_delta.reshape(3)).reshape(3)

        # Motor wiring/driver sign override
        tau_sign = np.asarray(self.cfg.tau_cmd_sign, dtype=float).reshape(3)
        tau = (tau_sign.reshape(3) * tau.reshape(3)).reshape(3).astype(float)
        return tau, foot_vicon

    def _serial_leg_fk_jac(self, *, q_roll: float, q_pitch: float, q_shift: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Serial-equivalent leg kinematics for MuJoCo `hopper_serial.xml`.

        Joint order:
          q = [roll, pitch, shift]
        Base frame:
          +X forward, +Y left, +Z up

        Returns:
          foot_b: (3,) foot origin position in base frame
          J_body: (3,3) Jacobian mapping qdot -> foot_vrel_b in base frame
        """
        # Geometry from MJCF:
        # base_link -> hip origin offset: z = -serial_hip_z_off_m
        p0 = np.array([0.0, 0.0, -float(self.cfg.serial_hip_z_off_m)], dtype=float)
        foot_z = float(self.cfg.serial_foot_z_m)

        # Rotation: roll about +X, pitch about +Y
        cr = float(np.cos(q_roll)); sr = float(np.sin(q_roll))
        cp = float(np.cos(q_pitch)); sp = float(np.sin(q_pitch))
        Rr = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
        Rp = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
        R = (Rr @ Rp).astype(float)

        # Prismatic axis is +Z in the roll/pitch frame; q_shift increases SHORTENING the leg.
        v = np.array([0.0, 0.0, float(q_shift) - float(foot_z)], dtype=float)
        foot_rel = (R @ v.reshape(3)).reshape(3)
        foot_b = (p0 + foot_rel).reshape(3)

        # Jacobian columns:
        axis_roll = np.array([1.0, 0.0, 0.0], dtype=float)
        axis_pitch = (Rr @ np.array([0.0, 1.0, 0.0], dtype=float).reshape(3)).reshape(3)
        axis_shift = R[:, 2].reshape(3)
        J0 = np.cross(axis_roll, foot_rel)
        J1 = np.cross(axis_pitch, foot_rel)
        J2 = axis_shift
        J_body = np.stack([J0, J1, J2], axis=1).astype(float)
        return foot_b, J_body

    def _touchdown_ok(self) -> bool:
        # placeholder for future gating; keep always true in this minimal core.
        return True

    def _liftoff_ok(self) -> bool:
        if not bool(self._stance):
            return False
        t_td = float(self._td_t) if self._td_t is not None else float(self.sim_time)
        return (float(self.sim_time) - t_td) >= float(self.cfg.stance_lo_min_T)

    @staticmethod
    def _quintic_coeff(p0: float, v0: float, a0: float, p1: float, v1: float, a1: float, T: float) -> np.ndarray:
        """
        Quintic polynomial coefficients for minimum-jerk interpolation:
          p(t) = c0 + c1 t + c2 t^2 + c3 t^3 + c4 t^4 + c5 t^5
        satisfying (p,v,a) at t=0 and t=T.
        """
        T = float(max(1e-6, float(T)))
        p0 = float(p0); v0 = float(v0); a0 = float(a0)
        p1 = float(p1); v1 = float(v1); a1 = float(a1)
        c0 = p0
        c1 = v0
        c2 = 0.5 * a0
        M = np.array(
            [
                [T**3, T**4, T**5],
                [3 * T**2, 4 * T**3, 5 * T**4],
                [6 * T, 12 * T**2, 20 * T**3],
            ],
            dtype=float,
        )
        b = np.array(
            [
                p1 - (c0 + c1 * T + c2 * T**2),
                v1 - (c1 + 2 * c2 * T),
                a1 - (2 * c2),
            ],
            dtype=float,
        )
        c3, c4, c5 = [float(x) for x in np.linalg.solve(M, b)]
        return np.array([c0, c1, c2, c3, c4, c5], dtype=float)

    @staticmethod
    def _quintic_eval(c: np.ndarray, t: float) -> tuple[float, float, float]:
        """Evaluate quintic polynomial (pos, vel, acc) at time t."""
        c = np.asarray(c, dtype=float).reshape(6)
        t = float(max(0.0, float(t)))
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        c0, c1, c2, c3, c4, c5 = [float(x) for x in c]
        p = c0 + c1 * t + c2 * t2 + c3 * t3 + c4 * t4 + c5 * t5
        v = c1 + 2 * c2 * t + 3 * c3 * t2 + 4 * c4 * t3 + 5 * c5 * t4
        a = 2 * c2 + 6 * c3 * t + 12 * c4 * t2 + 20 * c5 * t3
        return float(p), float(v), float(a)

    @staticmethod
    def _smoothstep01(x: float) -> float:
        """C1 smooth step from 0->1 for x in [0,1]."""
        x = float(np.clip(float(x), 0.0, 1.0))
        return float(x * x * (3.0 - 2.0 * x))

    def _init_unified_stance_profile(
        self,
        *,
        R_wb: np.ndarray,
        z_td_base: float,
        vz_td: float,
        q_shift_td: float,
    ) -> None:
        """
        Initialize a single, smooth stance COM-z reference curve:
          (z_td, vz_td) -> (z_min, 0) -> (z_end, v_to_cmd)
        """
        cfg = self.cfg
        R_wb = np.asarray(R_wb, dtype=float).reshape(3, 3)
        z_td_base = float(z_td_base)
        vz_td = float(vz_td)
        q_shift_td = float(q_shift_td)

        # Time budget
        T = float(max(float(cfg.stance_min_T), float(cfg.stance_T)))

        # COM offset in world-z (approx constant over the stance reference generation)
        com_off_z = float((R_wb @ self.com_b.reshape(3))[2])
        self._stance_com_off_z = float(com_off_z)

        # Touchdown COM-z reference origin
        z0 = float(z_td_base + com_off_z)

        # Estimate COM-z at "nominal leg length" (q_shift=0) for the end of stance reference.
        # We treat this as the nominal liftoff height. (The robot may liftoff earlier in practice.)
        z_end = float((z_td_base - q_shift_td) + com_off_z)

        # Desired takeoff speed (already computed at touchdown; clamp for safety)
        # NOTE: v_to_min is a legacy guard; we will still allow v_to to be reduced for feasibility
        # if the compression/extension distance is insufficient.
        v_to = float(np.clip(float(self._v_to_cmd), float(cfg.v_to_min), float(cfg.v_to_max)))

        # Adaptive compression depth/time from touchdown vertical speed (soft landing),
        # BUT also ensure the push segment is not forced to command an extra downward motion.
        #
        # Key insight:
        #   If the stance reference constrains BOTH end position z_end (near nominal leg length)
        #   AND a large end velocity v_to over a long push duration, then a smooth polynomial
        #   will inevitably go DOWN first (negative vz) to satisfy the boundary conditions.
        #   That makes MPC/WBC reduce vertical support (< mg), so the robot collapses instead of pushing off.
        #
        # We avoid this by choosing t_comp long enough (and thus depth large enough) that the remaining
        # extension distance dz = z_end - z_min can support a non-negative velocity profile up to v_to.
        v_in = float(max(0.0, -vz_td))
        a_max = float(max(1e-3, float(cfg.soft_land_a_max)))
        # Minimum time to brake the measured inbound speed under decel limit.
        t_comp_decel = float(v_in / a_max) if v_in > 1e-9 else 0.0
        # Pre-compression at touchdown (q_shift<0 means already shorter than nominal).
        precomp = float(max(0.0, -q_shift_td))
        # Additional requirement so push can be "mostly upward" (no extra downward motion):
        # Use a simple displacement lower bound for non-negative velocity:
        #   dz >= 0.5 * v_to * T2  where T2 = T - t_comp
        # with dz = (z_end - z_min) = depth + precomp (COM offset cancels).
        # and depth ≈ 0.5 * v_in * t_comp (area under braking from v_in to 0).
        t_comp_push = 0.0
        denom = float(v_in + v_to)
        numer = float(v_to * T - 2.0 * precomp)
        if (denom > 1e-6) and (numer > 0.0):
            t_comp_push = float(numer / denom)
        # Pick the larger of the two requirements, then clamp to stable bounds.
        t_comp = float(max(float(cfg.soft_land_tc_min), t_comp_decel, t_comp_push))
        t_comp = float(np.clip(t_comp, float(cfg.soft_land_tc_min), float(cfg.soft_land_tc_max_ratio) * T))
        t_comp = float(min(t_comp, max(1e-3, T - 1e-3)))

        depth = float(0.5 * v_in * t_comp)
        depth = float(np.clip(depth, float(cfg.soft_land_depth_min_m), float(cfg.soft_land_depth_max_m)))

        # Optional base-z guard (convert to COM-z)
        if float(cfg.z_guard) > 0.0:
            z_min_base = float(z_td_base - depth)
            if z_min_base < float(cfg.z_guard):
                z_min_base = float(cfg.z_guard)
                depth = float(max(0.0, z_td_base - z_min_base))
        z_min = float((z_td_base - depth) + com_off_z)

        # Build two quintic segments in COM-z.
        # If depth was clipped (or inbound speed estimate is small), the remaining dz may be too small
        # to reach the requested v_to without going DOWN first. Reduce v_to for feasibility.
        T1 = float(max(1e-3, t_comp))
        T2 = float(max(1e-3, T - T1))
        dz = float(z_end - z_min)
        if (dz > 1e-6) and (T2 > 1e-6):
            v_to_feas = float(2.0 * dz / T2)
            # keep a small margin to avoid numerical overshoot
            v_to = float(min(v_to, 0.98 * v_to_feas))
        poly1 = self._quintic_coeff(z0, vz_td, 0.0, z_min, 0.0, 0.0, T1)
        poly2 = self._quintic_coeff(z_min, 0.0, 0.0, z_end, v_to, 0.0, T2)

        self._stance_prof_inited = True
        self._stance_t_comp = float(T1)
        self._stance_depth_tgt_m = float(depth)
        self._stance_poly1 = poly1
        self._stance_poly2 = poly2
        self._stance_T1 = float(T1)
        self._stance_T2 = float(T2)
        self._stance_z_end = float(z_end)
        self._stance_retimed = False

    def _unified_stance_ref(self, t_in_stance: float) -> tuple[float, float, float]:
        """
        Return (z_ref, vz_ref, az_ref) in COM/world-z at time since touchdown.
        If profile is not initialized, falls back to holding current estimate.
        """
        t = float(max(0.0, float(t_in_stance)))
        if (not bool(self._stance_prof_inited)) or (self._stance_poly1 is None) or (self._stance_poly2 is None):
            # Fallback: hold current COM z and use current vz_hat (best effort)
            return float(self._p_hat_w[2] + float(self._stance_com_off_z)), float(self._v_hat_w[2]), 0.0

        T1 = float(max(1e-3, float(self._stance_T1)))
        T2 = float(max(1e-3, float(self._stance_T2)))
        # NOTE: use the second segment at the exact boundary (t==T1) to make retiming safe and
        # avoid any edge-case discontinuity.
        if t < T1:
            return self._quintic_eval(self._stance_poly1, min(t, T1))
        else:
            return self._quintic_eval(self._stance_poly2, min(t - T1, T2))

    def _retime_unified_stance_to_push(self, *, t_in_stance: float, z_now: float, vz_now: float) -> None:
        """
        Retiming hook for real-robot robustness:

        If the leg stops compressing (qd_shift crosses >=0) earlier than the planned t_comp,
        the original stance reference may still be in its 'braking/compression' segment while the
        physical leg is already extending. That mismatch often makes MPC choose fz near fz_min
        (sometimes < mg), preventing liftoff.

        This function rewrites the *push* segment (poly2) to start at the current state (z_now, vz_now)
        and reach the precomputed stance endpoint z_end with takeoff speed v_to over the remaining time.
        """
        cfg = self.cfg
        if not bool(self._stance_prof_inited):
            return
        if self._stance_z_end is None:
            return
        if self._stance_poly2 is None:
            return
        # Total stance time budget
        T = float(max(float(cfg.stance_min_T), float(cfg.stance_T)))
        t = float(np.clip(float(t_in_stance), 0.0, max(1e-3, T - 1e-3)))
        T_rem = float(max(1e-3, T - t))

        z_now = float(z_now)
        vz_now = float(vz_now)
        z_end = float(self._stance_z_end)

        # Target takeoff speed (feasibility-clamped)
        v_to = float(np.clip(float(self._v_to_cmd), float(cfg.v_to_min), float(cfg.v_to_max)))
        dz = float(z_end - z_now)
        if dz > 1e-6:
            v_to_feas = float(2.0 * dz / max(1e-6, T_rem))
            v_to = float(min(v_to, 0.98 * v_to_feas))
        else:
            # already above endpoint; don't demand more upward speed
            v_to = float(min(v_to, 0.0))

        # New push segment from current state to endpoint
        poly2 = self._quintic_coeff(z_now, vz_now, 0.0, z_end, v_to, 0.0, T_rem)

        # Update internal timing so the push segment starts now.
        self._stance_poly2 = poly2
        self._stance_T1 = float(t)
        self._stance_T2 = float(T_rem)
        self._stance_t_comp = float(t)
        self._stance_retimed = True

    def step(
        self,
        *,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        imu_gyro_b: np.ndarray,
        imu_acc_b: np.ndarray,
        imu_quat_wxyz: np.ndarray | None,
        desired_v_xy_w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        One control step.

        Returns:
          tau_cmd (3,), pwm_us (6,), info dict
        """
        self.sim_time = float(self.sim_time + self.dt)

        # Copy to avoid mutating caller buffers (LCM controller may reuse arrays).
        desired_v_xy_w = np.asarray(desired_v_xy_w, dtype=float).reshape(2).copy()
        joint_pos = np.asarray(joint_pos, dtype=float).reshape(3)
        joint_vel = np.asarray(joint_vel, dtype=float).reshape(3)
        imu_gyro_b = np.asarray(imu_gyro_b, dtype=float).reshape(3)
        imu_acc_b = np.asarray(imu_acc_b, dtype=float).reshape(3)

        # --- Signal conditioning: joint velocity LPF (used by kinematics) ---
        joint_vel_kin = joint_vel.copy()
        try:
            tau = float(getattr(self.cfg, "joint_vel_lpf_tau", 0.0))
        except Exception:
            tau = 0.0
        if float(tau) > 0.0:
            if not bool(self._joint_vel_lpf_init):
                self._joint_vel_lpf = joint_vel.copy()
                self._joint_vel_lpf_init = True
            else:
                a = float(np.clip(float(self.dt) / (float(tau) + float(self.dt)), 0.0, 1.0))
                self._joint_vel_lpf = (1.0 - a) * self._joint_vel_lpf + a * joint_vel
            joint_vel_kin = np.asarray(self._joint_vel_lpf, dtype=float).reshape(3).copy()

        # --- Foot kinematics ---
        # Variables are kept in the same naming convention as the delta version for compatibility with the rest
        # of the controller and debug logs:
        # - foot_vicon: delta/vicon frame (+Z down)
        # - foot_b:     base/body frame (+Z up)
        # - foot_vdot_vicon: foot velocity in delta/vicon frame
        # - foot_vrel_b:     foot velocity in base/body frame
        J_inv: np.ndarray | None = None
        J_body: np.ndarray | None = None

        if self._leg_model == "serial":
            # MuJoCo serial equivalent plant: q = [roll, pitch, shift]
            foot_b, J_body = self._serial_leg_fk_jac(
                q_roll=float(joint_pos[0]),
                q_pitch=float(joint_pos[1]),
                q_shift=float(joint_pos[2]),
            )
            foot_b = np.asarray(foot_b, dtype=float).reshape(3)
            J_body = np.asarray(J_body, dtype=float).reshape(3, 3)
            foot_vrel_b = (J_body @ joint_vel_kin.reshape(3)).reshape(3)
            foot_vicon = (self.robot2vicon @ foot_b.reshape(3)).reshape(3)
            foot_vdot_vicon = (self.robot2vicon @ foot_vrel_b.reshape(3)).reshape(3)
        else:
            # delta/vicon frame (+Z down)
            if self.fk is None or self.kin is None:
                raise RuntimeError("delta leg model requested but kinematics is not initialized")
            foot_vicon, _ = self.fk.forward_kinematics(joint_pos)
            foot_vicon = np.asarray(foot_vicon, dtype=float).reshape(3)
            # NOTE: we do NOT trust the raw xdot solve inside `inverse_jacobian` near singularities.
            # Always compute xdot from the returned J_inv with a stable inverse (optionally DLS).
            J_inv_raw, _ = self.kin.inverse_jacobian(foot_vicon, joint_vel_kin, theta=None)
            J_inv = np.asarray(J_inv_raw, dtype=float).reshape(3, 3)
            foot_vdot_vicon = (self._stable_inv3(J_inv) @ joint_vel_kin.reshape(3)).reshape(3).astype(float)

        # Convert to base frame (z up)
        foot_b = (self.robot2vicon @ foot_vicon.reshape(3)).reshape(3)
        foot_vrel_b = (self.robot2vicon @ foot_vdot_vicon.reshape(3)).reshape(3)

        # ===== Equivalent shift coordinate for phase detection =====
        # delta mode: use leg-length shift q_shift = ||foot|| - l0 (negative when compressed)
        # serial mode: use the prismatic shift joint directly (q_shift_joint >= 0 means compression),
        # and map it into the SAME "delta-style" convention via: q_shift = -q_shift_joint.
        leg_length = float(np.linalg.norm(foot_vicon))
        
        # Cache g_eff and dz_tgt for logging (computed at touchdown)
        g_eff_log = float("nan")
        dz_tgt_log = float("nan")
        if self._leg_model == "serial":
            # In hopper_serial.xml, shift joint increases with COMPRESSION.
            q_shift = -float(joint_pos[2])
            qd_shift = -float(joint_vel_kin[2])
        else:
            q_shift = float(leg_length - self.cfg.leg_l0_m)
            # qd_shift: rate of change of leg length (positive = extending, negative = compressing)
            # foot_vdot_vicon is in delta/vicon frame, we need the component along the leg direction
            if leg_length > 1e-6:
                leg_dir = foot_vicon / leg_length  # unit vector from base to foot
                leg_extension_vel = float(np.dot(foot_vdot_vicon, leg_dir))  # positive = extending
                qd_shift = float(leg_extension_vel)  # positive = extending, negative = compressing
            else:
                qd_shift = 0.0

        # --- LPF q_shift / qd_shift for phase detection robustness ---
        q_shift_raw = float(q_shift)
        qd_shift_raw = float(qd_shift)
        try:
            tau_q = float(getattr(self.cfg, "q_shift_lpf_tau", 0.0))
        except Exception:
            tau_q = 0.0
        try:
            tau_qd = float(getattr(self.cfg, "qd_shift_lpf_tau", 0.0))
        except Exception:
            tau_qd = 0.0
        if (float(tau_q) > 0.0) or (float(tau_qd) > 0.0):
            if not bool(self._shift_lpf_init):
                self._q_shift_lpf = q_shift_raw
                self._qd_shift_lpf = qd_shift_raw
                self._shift_lpf_init = True
            else:
                if float(tau_q) > 0.0:
                    a = float(np.clip(float(self.dt) / (float(tau_q) + float(self.dt)), 0.0, 1.0))
                    self._q_shift_lpf = float((1.0 - a) * float(self._q_shift_lpf) + a * q_shift_raw)
                else:
                    self._q_shift_lpf = q_shift_raw
                if float(tau_qd) > 0.0:
                    a = float(np.clip(float(self.dt) / (float(tau_qd) + float(self.dt)), 0.0, 1.0))
                    self._qd_shift_lpf = float((1.0 - a) * float(self._qd_shift_lpf) + a * qd_shift_raw)
                else:
                    self._qd_shift_lpf = qd_shift_raw
            # overwrite for downstream phase machine
            q_shift = float(self._q_shift_lpf) if float(tau_q) > 0.0 else q_shift_raw
            qd_shift = float(self._qd_shift_lpf) if float(tau_qd) > 0.0 else qd_shift_raw

        # --- Attitude estimate (body->world) ---
        if bool(self.cfg.use_fc_quat) and (imu_quat_wxyz is not None):
            q_hat = _quat_normalize_wxyz(np.asarray(imu_quat_wxyz, dtype=float).reshape(4))
        else:
            q_hat = self.att.update(omega_b=imu_gyro_b, acc_b=imu_acc_b, dt=float(self.dt))
        R_wb_hat = _quat_to_R_wb(q_hat)
        rpy_hat = _R_to_rpy_xyz(R_wb_hat)
        z_w = np.asarray(R_wb_hat[:, 2], dtype=float).reshape(3)

        # ===== Falling cat gating (tilt/inversion recovery) =====
        # Decide whether quaternion-mapped foot targeting is allowed, and optionally freeze vxy.
        falling_cat_enabled = bool(getattr(self.cfg, "falling_cat_enable", False))
        falling_cat_upright = True
        falling_cat_quat_ok = True
        falling_cat_active = False
        falling_cat_rp_max_deg = float(
            max(
                abs(float(np.rad2deg(float(rpy_hat[0])))),
                abs(float(np.rad2deg(float(rpy_hat[1])))),
            )
        )
        if bool(falling_cat_enabled):
            rp_lim_deg = float(max(0.0, float(getattr(self.cfg, "falling_cat_quat_map_rp_deg", 45.0))))
            rp_lim_rad = float(np.deg2rad(rp_lim_deg))
            if bool(getattr(self.cfg, "falling_cat_require_upright", True)):
                zmin = float(getattr(self.cfg, "falling_cat_upright_z_w_min", 0.0))
                falling_cat_upright = float(z_w[2]) > float(zmin)
            else:
                falling_cat_upright = True
            falling_cat_quat_ok = bool(falling_cat_upright) and (abs(float(rpy_hat[0])) <= rp_lim_rad) and (abs(float(rpy_hat[1])) <= rp_lim_rad)
            falling_cat_active = not bool(falling_cat_quat_ok)

            if bool(falling_cat_active) and bool(getattr(self.cfg, "falling_cat_zero_desired_vxy", True)):
                desired_v_xy_w[:] = 0.0

        # --- Touchdown velocity estimate from leg kinematics (best-effort) ---
        # At the instant of touchdown, the foot is approximately stationary in WORLD.
        # Then base linear velocity (in body) is:
        #   v_base_b ≈ -(v_foot_rel_b + ω×r_foot_b)
        # This is often much more informative than IMU-integrated vertical velocity for "soft landing"
        # because our FC accel convention is a gravity/down vector (not specific force).
        v_base_from_foot_b = -(foot_vrel_b + np.cross(imu_gyro_b, foot_b)).astype(float)
        v_base_from_foot_w = (R_wb_hat @ v_base_from_foot_b.reshape(3)).reshape(3)

        # --- IMU propagation for base velocity ---
        # Convention (matches our SimpleIMUAttitudeEstimator and current hopper_driver):
        # - `imu_acc_b` is a *gravity / down* vector in BODY frame (units: m/s^2),
        #   i.e. when the robot is level + stationary: imu_acc_b ≈ [0, 0, -9.81] (because +Z is UP).
        # - Then linear acceleration is: a_w = g_w - R_wb * imu_acc_b
        #   (stationary: R_wb*imu_acc_b == g_w  => a_w == 0).
        g_w = np.array([0.0, 0.0, -float(self.gravity)], dtype=float)
        a_w = (g_w - (R_wb_hat @ imu_acc_b.reshape(3))).reshape(3)
        # Hopper4-like XY velocity estimator:
        # - STANCE: estimate XY from leg kinematics (foot velocity + omega×r), fused below
        # - FLIGHT: HOLD XY (do not integrate IMU accel bias)
        #
        # Default is True to match Hopper4 behavior (even if the config field is absent).
        use_h4_vxy = bool(getattr(self.cfg, "use_hopper4_vxy_estimator", True))
        # User request: allow a "hard stop" mode that keeps internal velocity estimate exactly zero
        # (prevents IMU integration drift from moving foot targets when desired_v == 0).
        if bool(getattr(self, "_user_zero_vel_hold", False)):
            self._v_hat_w[:] = 0.0
            self._v_hat_inited = True
            self._v_int_xy[:] = 0.0
            v_pred = np.zeros(3, dtype=float)
        else:
            if not bool(self._v_hat_inited):
                self._v_hat_w = np.zeros(3, dtype=float)
                self._v_hat_inited = True
            v_pred = (np.asarray(self._v_hat_w, dtype=float).reshape(3) + a_w * float(self.dt)).reshape(3)
            # Hopper4-like behavior: do NOT integrate IMU accel for horizontal velocity (XY).
            # - STANCE: XY will be corrected by leg-kinematics fusion below
            # - FLIGHT: hold XY to avoid drift
            if bool(use_h4_vxy):
                v_pred[0] = float(self._v_hat_w[0])
                v_pred[1] = float(self._v_hat_w[1])

        # cache previous shift/qd (used for debounce / retiming)
        q_shift_prev = self._q_shift_prev
        qd_shift_prev = self._qd_shift_prev

        touchdown_evt = False
        liftoff_evt = False
        apex_evt = False

        # ===== Phase machine (engineering mode): FLIGHT -> STANCE -> FLIGHT =====
        # Goal: keep transitions simple and only use leg-length signals.
        #
        # Definitions:
        #   q_shift = leg_length - leg_l0_m
        #     <0 : compressed (leg shorter than nominal)
        #     >0 : extended  (leg longer than nominal)
        #   qd_shift < 0 means compressing, > 0 means extending.
        #
        # Transitions:
        #   FLIGHT -> STANCE : (q_shift <= td_q_shift_gate) AND (qd_shift < td_qd_shift_gate)
        #   STANCE -> FLIGHT : q_shift >= 0 (leg extended to nominal or longer)
        if (not bool(self._stance)) and self._touchdown_ok():
            # Debounce touchdown to avoid chattering on noisy delta Jacobian / qd.
            cond_td = (float(q_shift) <= float(self.cfg.td_q_shift_gate)) and (float(qd_shift) < float(self.cfg.td_qd_shift_gate))
            if (q_shift_prev is not None) and np.isfinite(float(q_shift_prev)):
                dq = float(q_shift) - float(q_shift_prev)
                if np.isfinite(dq):
                    cond_td = bool(cond_td) and (float(dq) <= float(self.cfg.td_dq_shift_gate))
            if bool(cond_td):
                self._td_debounce_count = int(self._td_debounce_count) + 1
            else:
                self._td_debounce_count = 0
            td_n = int(max(1, int(getattr(self.cfg, "td_debounce_steps", 1))))
            if int(self._td_debounce_count) >= td_n:
                touchdown_evt = True
                self._stance = True
                self._td_t = float(self.sim_time)
                self._apex_reached = False
                self._td_debounce_count = 0
                self._lo_debounce_count = 0

                # latch TD shift for compression measurement
                self._q_shift_td = float(q_shift)

                # touchdown z estimate from kinematics (assume foot at ground z=0)
                z_td_est = -float((R_wb_hat @ foot_b.reshape(3))[2])
                self._z_hat_contact_filt = float(z_td_est)
                self._p_hat_w[2] = float(z_td_est)

                # takeoff speed target for desired apex (ballistic, with prop assist)
                # NOTE: hop_peak_z is an ABSOLUTE height (world z), not relative to touchdown.
                # We compute dz from touchdown base height to apex target.
                g_eff = float(self.gravity - (float(z_w[2]) * float(self.mass) * float(self.gravity) * float(self.cfg.prop_base_thrust_ratio)) / max(1e-6, float(self.mass)))
                g_eff = float(max(1e-3, g_eff))
                # Ensure minimum dz to avoid tiny hops (safety: at least 0.15m from TD to apex)
                dz_tgt = float(max(0.15, float(self.cfg.hop_peak_z) - float(z_td_est)))
                g_eff_log = float(g_eff)  # cache for logging
                dz_tgt_log = float(dz_tgt)  # cache for logging
                v_to_nominal = float(np.sqrt(2.0 * g_eff * dz_tgt))
                
                # ===== Apex height feedback (per-hop outer loop) =====
                # Use actual apex from previous hop to correct v_to_cmd (PI control).
                # This is the "robust/optimal" way: each hop learns from the previous one.
                if bool(getattr(self.cfg, "apex_use_feedback", True)) and (self._z_apex_actual is not None) and (self._z_lo is not None) and (self._vz_lo is not None):
                    # Predict apex from previous liftoff (using same g_eff as current hop)
                    if float(self._vz_lo) > 0.0:
                        z_apex_pred = float(self._z_lo) + (float(self._vz_lo) * float(self._vz_lo)) / (2.0 * g_eff)
                    else:
                        z_apex_pred = float(self._z_lo)
                    
                    # Apex height error (desired - actual)
                    apex_err = float(self.cfg.hop_peak_z) - float(self._z_apex_actual)
                    
                    # PI control: v_to_cmd = v_to_nominal + kp*err + ki*integral(err)
                    kp = float(max(0.0, float(getattr(self.cfg, "apex_kp", 0.1))))
                    ki = float(max(0.0, float(getattr(self.cfg, "apex_ki", 0.02))))
                    int_max = float(max(0.0, float(getattr(self.cfg, "apex_int_max", 0.5))))
                    
                    # Update integral (clamp to prevent windup)
                    self._apex_err_int = float(self._apex_err_int) + float(apex_err)
                    self._apex_err_int = float(np.clip(self._apex_err_int, -int_max, +int_max))
                    
                    # PI correction
                    v_to_correction = float(kp * apex_err + ki * self._apex_err_int)
                    self._v_to_cmd = float(np.clip(v_to_nominal + v_to_correction, float(self.cfg.v_to_min), float(self.cfg.v_to_max)))
                else:
                    # No feedback yet (first hop or feedback disabled): use nominal calculation
                    self._v_to_cmd = float(np.clip(v_to_nominal, float(self.cfg.v_to_min), float(self.cfg.v_to_max)))

                # Initialize a single smooth stance profile (COM-z) so the stance does not require
                # a discrete COMP/PUSH mode switch.
                try:
                    self._init_unified_stance_profile(
                        R_wb=R_wb_hat,
                        z_td_base=float(z_td_est),
                        vz_td=float(v_base_from_foot_w[2]) if np.isfinite(float(v_base_from_foot_w[2])) else float(v_pred[2]),
                        q_shift_td=float(self._q_shift_td) if self._q_shift_td is not None else float(q_shift),
                    )
                except Exception:
                    # If profile init fails, keep going; MPC will fall back to a conservative reference.
                    self._stance_prof_inited = False

                # Force an MPC refresh on touchdown (we no longer reset based on COMP/PUSH mode switching)
                if hasattr(self, "_mpc_last_t"):
                    try:
                        self._mpc_last_t = -1e9
                    except Exception:
                        pass
        else:
            # reset in all other cases
            self._td_debounce_count = 0

        # track previous shift for debounce
        if np.isfinite(q_shift):
            self._q_shift_prev = float(q_shift)
        if np.isfinite(qd_shift):
            self._qd_shift_prev = float(qd_shift)

        # ===== liftoff detection (debounced + minimum stance time) =====
        if bool(self._stance) and np.isfinite(q_shift):
            cond_lo = (float(q_shift) >= 0.0) and bool(self._liftoff_ok())
            if bool(cond_lo):
                self._lo_debounce_count = int(self._lo_debounce_count) + 1
            else:
                self._lo_debounce_count = 0
            lo_n = int(max(1, int(getattr(self.cfg, "lo_debounce_steps", 1))))
            if int(self._lo_debounce_count) >= lo_n:
                liftoff_evt = True
                self._stance = False
                self._lo_t = float(self.sim_time)
                self._lo_debounce_count = 0
                # Record liftoff state for apex prediction (used by apex feedback loop)
                if bool(getattr(self.cfg, "apex_use_feedback", True)):
                    self._z_lo = float(self._p_hat_w[2])
                    self._vz_lo = float(self._v_hat_w[2])
        else:
            self._lo_debounce_count = 0

        # ===== estimator fusion (stance: no-slip / leg-kinematics) =====
        # Initialize v_meas_w for MPC filter (scope outside if block)
        v_meas_w = None
        if bool(self._stance):
            v_meas_b = -(foot_vrel_b + np.cross(imu_gyro_b, foot_b))
            v_meas_w = (R_wb_hat @ v_meas_b.reshape(3)).reshape(3)
            # Numerical guard: if kinematics produces NaNs, skip fusion and fall back to prediction.
            if not (np.all(np.isfinite(v_meas_w)) and np.all(np.isfinite(v_pred))):
                self._v_hat_w = np.asarray(v_pred, dtype=float).reshape(3).copy()
                # Decay integrator in this case (prevents windup on NaNs)
                self._v_int_xy = (0.995 * self._v_int_xy).astype(float)
                v_meas_w = None  # Mark as invalid for MPC filter
            else:

                # Slip detection (optional, disabled by default to match Hopper4 behavior)
                if bool(getattr(self.cfg, "v_use_slip_detection", False)):
                    omega_w_hat = (R_wb_hat @ imu_gyro_b.reshape(3)).reshape(3)
                    omega_noyaw = omega_w_hat.copy()
                    omega_noyaw[2] = 0.0
                    r_foot_w_rel = (R_wb_hat @ foot_b.reshape(3)).reshape(3)
                    v_foot_w_pred = (v_pred + np.cross(omega_noyaw, r_foot_w_rel) + (R_wb_hat @ foot_vrel_b.reshape(3))).reshape(3)
                    slip_speed = float(np.linalg.norm(v_foot_w_pred))
                    slip_ref = float(max(0.1, float(self.cfg.v_slip_ref)))
                    gate = float(np.clip(np.exp(-slip_speed / max(1e-6, slip_ref)), 0.20, 1.0))
                else:
                    # No slip detection: always trust leg kinematics (like Hopper4)
                    gate = 1.0

                tau = float(self._v_hat_lpf_tau)
                a = float(np.clip(float(self.dt) / (tau + float(self.dt)), 0.0, 1.0)) if tau > 1e-9 else 1.0
                a_eff = float(a * gate)
                # User request: forbid XY asymmetry in velocity fusion.
                # Keep the legacy knob name `v_fuse_vx_scale` but apply it to BOTH X and Y.
                vxy_scale = float(np.clip(float(self.cfg.v_fuse_vx_scale), 0.0, 1.0))
                a_eff_xy = float(a_eff * vxy_scale)
                v_new = np.asarray(v_pred, dtype=float).reshape(3).copy()
                v_new[0] = float((1.0 - a_eff_xy) * float(v_pred[0]) + a_eff_xy * float(v_meas_w[0]))
                v_new[1] = float((1.0 - a_eff_xy) * float(v_pred[1]) + a_eff_xy * float(v_meas_w[1]))
                v_new[2] = float((1.0 - a_eff) * float(v_pred[2]) + a_eff * float(v_meas_w[2]))
                self._v_hat_w = v_new.reshape(3)

                # PI integrator (VXY, slip-gated; user request: forbid XY asymmetry)
                try:
                    err_xy = (desired_v_xy_w - np.asarray(self._v_hat_w[0:2], dtype=float).reshape(2)).reshape(2)
                    v_int_max = float(max(0.0, float(self.cfg.v_int_max)))
                    self._v_int_xy[0] = float(self._v_int_xy[0] + float(err_xy[0]) * float(self.dt) * float(gate))
                    self._v_int_xy[1] = float(self._v_int_xy[1] + float(err_xy[1]) * float(self.dt) * float(gate))
                    self._v_int_xy[0] = float(np.clip(self._v_int_xy[0], -v_int_max, +v_int_max))
                    self._v_int_xy[1] = float(np.clip(self._v_int_xy[1], -v_int_max, +v_int_max))
                except Exception:
                    pass
        else:
            self._v_hat_w = v_pred.copy()
            self._v_int_xy = (0.995 * self._v_int_xy).astype(float)

        # integrate position + stance z correction
        self._p_hat_w = self._p_hat_w + self._v_hat_w * float(self.dt)
        if bool(self._stance):
            z_meas = -float((R_wb_hat @ foot_b.reshape(3))[2])
            if self._z_hat_contact_filt is None:
                self._z_hat_contact_filt = float(z_meas)
            z_tau = 0.05
            az = float(np.clip(float(self.dt) / (z_tau + float(self.dt)), 0.0, 1.0))
            self._z_hat_contact_filt = (1.0 - az) * float(self._z_hat_contact_filt) + az * float(z_meas)
            self._p_hat_w[2] = float(self._z_hat_contact_filt)

        # ===== MPC Velocity Filter (Paper-Grade Adaptive Filter) =====
        # Advanced velocity filtering for MPC input baseline.
        # Implements adaptive filtering with phase-aware tuning, outlier rejection, and confidence-based fusion.
        if bool(getattr(self.cfg, "use_mpc_velocity_filter", True)):
            v_raw = np.asarray(self._v_hat_w, dtype=float).reshape(3)
            
            # Initialize filter on first call
            if not self._v_mpc_filter_inited:
                self._v_mpc_filtered = v_raw.copy()
                self._v_mpc_prev = v_raw.copy()
                self._v_mpc_filter_inited = True
            
            # Phase-aware time constant selection
            tau_stance = float(getattr(self.cfg, "mpc_v_filter_tau_stance", 0.08))
            tau_flight = float(getattr(self.cfg, "mpc_v_filter_tau_flight", 0.20))
            tau = float(tau_stance if bool(self._stance) else tau_flight)
            
            # Compute base filter coefficient
            alpha_base = float(np.clip(float(self.dt) / max(1e-9, tau + float(self.dt)), 0.0, 1.0))
            
            # Outlier detection and adaptive filtering
            # If velocity change is large, reduce filter coefficient (reject outlier)
            outlier_th = float(getattr(self.cfg, "mpc_v_filter_outlier_th", 2.0))
            alpha_min = float(getattr(self.cfg, "mpc_v_filter_alpha_min", 0.10))
            alpha_max = float(getattr(self.cfg, "mpc_v_filter_alpha_max", 0.95))
            
            v_diff = np.abs(v_raw - self._v_mpc_filtered)
            v_diff_norm = float(np.linalg.norm(v_diff))
            
            # Adaptive coefficient: reduce alpha for large changes (outlier rejection)
            if v_diff_norm > outlier_th:
                # Heavy filtering for outliers: reduce alpha proportionally
                outlier_factor = float(outlier_th / max(1e-9, v_diff_norm))
                alpha_adaptive = float(alpha_base * outlier_factor * 0.5)  # Extra 50% reduction for outliers
                alpha_adaptive = float(np.clip(alpha_adaptive, alpha_min, alpha_base))
            else:
                alpha_adaptive = float(np.clip(alpha_base, alpha_min, alpha_max))
            
            # In STANCE: blend with leg kinematics measurement (if available)
            # Confidence-based fusion: trust kinematics more in stance
            if bool(self._stance) and (v_meas_w is not None) and np.all(np.isfinite(v_meas_w)) and np.all(np.isfinite(v_raw)):
                kinematics_weight = float(getattr(self.cfg, "mpc_v_filter_kinematics_weight", 0.85))
                # Blend raw estimate (which includes kinematics fusion) with kinematics directly
                # This double-fusion improves robustness
                v_for_filter = (kinematics_weight * v_meas_w + (1.0 - kinematics_weight) * v_raw).reshape(3)
            else:
                v_for_filter = v_raw.copy()
            
            # Apply adaptive low-pass filter
            self._v_mpc_filtered = ((1.0 - alpha_adaptive) * self._v_mpc_filtered + alpha_adaptive * v_for_filter).reshape(3)
            self._v_mpc_prev = self._v_mpc_filtered.copy()
        else:
            # Filter disabled: use raw estimate
            self._v_mpc_filtered = np.asarray(self._v_hat_w, dtype=float).reshape(3).copy()

        # apex detection (flight): vz_hat sign change
        vz_hat = float(self._v_hat_w[2])
        if self._prev_vz is None:
            self._prev_vz = float(vz_hat)
        if (not bool(self._stance)) and (float(self._prev_vz) > 0.0) and (float(vz_hat) <= 0.0):
            apex_evt = True
            self._apex_reached = True
            # Record actual apex height for feedback loop (used on next touchdown)
            if bool(getattr(self.cfg, "apex_use_feedback", True)):
                self._z_apex_actual = float(self._p_hat_w[2])
        self._prev_vz = float(vz_hat)

        # ===== stance: unified reference (no discrete COMP/PUSH switching) =====
        # We keep "compress_active" as a debug label only (pre/post max-compression time),
        # but the controller itself uses a single smooth stance reference curve.
        az_des = -float(self.gravity)  # default (flight)
        compress_active = False
        depth_now = 0.0
        depth_tgt = 0.0
        depth_tgt_act = 0.0
        z_now = float(self._p_hat_w[2])
        s = 0.0
        t_in_stance = 0.0

        if bool(self._stance):
            t_td = float(self._td_t) if (self._td_t is not None) else float(self.sim_time)
            t_in_stance = float(float(self.sim_time) - t_td)
            s = float(np.clip(t_in_stance / max(1e-6, float(self.cfg.stance_T)), 0.0, 1.0))

            # Actual compression depth for logging (from leg length shift)
            q_shift_td = self._q_shift_td
            if (q_shift_td is None) or (not np.isfinite(float(q_shift_td))) or (not np.isfinite(q_shift)):
                depth_now = 0.0
            else:
                depth_now = float(max(0.0, float(q_shift_td) - float(q_shift)))

            z_now = float(self._p_hat_w[2])

            # Desired vertical acceleration for fallback when MPC is infeasible:
            # use the unified stance reference curve (COM-z), which is smooth by construction.
            if bool(self.cfg.use_unified_stance) and bool(self._stance_prof_inited):
                # Event-based retiming:
                # If the leg stops compressing (qd_shift crosses from negative -> positive) significantly
                # earlier than the planned t_comp, retime the stance reference to start the push segment now.
                # This avoids MPC choosing fz near fz_min (< mg) to "brake" an early rebound.
                if bool(getattr(self.cfg, "stance_retime_on_qd_cross", True)) and (not bool(self._stance_retimed)):
                    # Robust trigger (real robot):
                    # In logs we often see the leg begin extending (qd_shift>0) *much earlier*
                    # than the planned t_comp (time-based). When that happens, the stance reference
                    # still behaves like a braking/compression profile and MPC may hug fz_min,
                    # producing a tiny hop.
                    #
                    # We therefore retime as soon as we detect "clear early extension" while we
                    # are still within the planned compression window.
                    try:
                        qd_eps = float(max(0.0, float(getattr(self.cfg, "stance_retime_qd_eps", 0.01))))
                        t_comp_plan = float(self._stance_t_comp) if self._stance_t_comp is not None else 0.0
                        margin = float(max(0.0, float(getattr(self.cfg, "stance_retime_early_margin_s", 0.02))))

                        # Require some meaningful compression before retiming (avoid noise at touchdown).
                        depth_gate = float(max(0.01, 0.5 * float(self._stance_depth_tgt_m)))

                        if np.isfinite(qd_shift) and (float(qd_shift) >= +qd_eps):
                            if (t_comp_plan > 1e-6) and ((t_in_stance + margin) < t_comp_plan) and (float(depth_now) >= depth_gate):
                                z_com_now = float(self._p_hat_w[2] + float(self._stance_com_off_z))
                                vz_now = float(self._v_hat_w[2])
                                self._retime_unified_stance_to_push(t_in_stance=t_in_stance, z_now=z_com_now, vz_now=vz_now)
                    except Exception:
                        pass
                _, _, az_ref = self._unified_stance_ref(t_in_stance)
                az_des = float(az_ref)
                depth_tgt = float(self._stance_depth_tgt_m)
                depth_tgt_act = float(depth_tgt)
                t_comp = float(self._stance_t_comp) if self._stance_t_comp is not None else 0.0
                compress_active = bool(t_in_stance < float(t_comp))
            else:
                # If the stance profile isn't initialized, keep a conservative default.
                az_des = 0.0
                depth_tgt = 0.0
                depth_tgt_act = 0.0
                compress_active = False

        # ===== User requested: STANCE output kill-switch (hardcode outputs to zero) =====
        # Keep phase machine + estimators running, but command:
        #   - tau_cmd = 0
        #   - thrusts = 0  -> pwm_us = pwm_min
        # and expose zeros in debug force fields.
        if bool(self._stance) and bool(self.cfg.stance_zero_output):
            tau_cmd = np.zeros(3, dtype=float)
            thrusts = np.zeros(3, dtype=float)
            f_contact_w = np.zeros(3, dtype=float)
            f_tau_b = np.zeros(3, dtype=float)
            f_tau_delta = np.zeros(3, dtype=float)
            slack = np.zeros(6, dtype=float)
            status = "stance_zero_output"

            # ensure internal holds don't re-introduce previous outputs
            self._tau_cmd_prev = tau_cmd.copy()
            self._wbc_last_f = f_contact_w.copy()
            self._wbc_last_t = thrusts.copy()

            pwm_us = np.ones(6, dtype=float) * float(self.cfg.pwm_min_us)

            # Debug targets for logging/printing (in stance these should be NaNs / inactive)
            foot_des_b_dbg = np.full(3, np.nan, dtype=float)
            foot_des_w_dbg = np.full(3, np.nan, dtype=float)
            p_foot_des_w_dbg = np.full(3, np.nan, dtype=float)
            s2s_active_dbg = 0

            info = {
                "t": float(self.sim_time),
                "stance": int(bool(self._stance)),
                "touchdown": int(touchdown_evt),
                "liftoff": int(liftoff_evt),
                "apex": int(apex_evt),
                "compress": int(bool(compress_active)),
                "push": int(bool(self._stance) and (not bool(compress_active))),
                "desired_v_xy_w": np.asarray(desired_v_xy_w, dtype=float).reshape(2).copy(),
                "q_hat_wxyz": q_hat.copy(),
                "rpy_hat": rpy_hat.copy(),
                "p_hat_w": np.asarray(self._p_hat_w, dtype=float).reshape(3).copy(),
                "v_hat_w": np.asarray(self._v_hat_w, dtype=float).reshape(3).copy(),
                # Debug: base velocity measured from leg kinematics (foot assumed stationary in WORLD).
                # v_base_b ≈ -(v_foot_rel_b + ω×r_foot_b), then rotate to WORLD via R_wb.
                "v_meas_foot_w": np.asarray(v_base_from_foot_w, dtype=float).reshape(3).copy(),
                "foot_vicon": foot_vicon.copy(),
                "foot_b": foot_b.copy(),
                "foot_vdot_vicon": foot_vdot_vicon.copy(),
                "foot_vrel_b": foot_vrel_b.copy(),
                "s2s_active": int(s2s_active_dbg),
                "foot_des_b": foot_des_b_dbg.copy(),
                "foot_des_w": foot_des_w_dbg.copy(),
                "p_foot_des_w": p_foot_des_w_dbg.copy(),
                "q_shift_equiv": float(q_shift),
                "qd_shift_equiv": float(qd_shift),
                "comp_m": float(depth_now),
                "comp_tgt_m": float(depth_tgt),
                "comp_tgt_act_m": float(depth_tgt_act),
                "z_now_m": float(z_now) if bool(self._stance) else 0.0,
                "s_stance": float(s) if bool(self._stance) else 0.0,
                "compress_active": int(bool(compress_active)),
                "push_started": int(bool(self._stance) and (not bool(compress_active))),
                "mpc_used": 0,
                "mpc_fx_cmd": 0.0,
                "mpc_fy_cmd": 0.0,
                "f_contact_w": f_contact_w.copy(),
                "f_tau_b": f_tau_b.copy(),
                "f_tau_delta": f_tau_delta.copy(),
                "thrusts_arm": thrusts.copy(),
                "tau_cmd": tau_cmd.copy(),
                "pwm_us": pwm_us.copy(),
                "slack": slack.copy(),
                "status": status,
                # Falling cat debug
                "falling_cat_enable": int(bool(falling_cat_enabled)),
                "falling_cat_active": int(bool(falling_cat_active)),
                "falling_cat_quat_ok": int(bool(falling_cat_quat_ok)),
                "falling_cat_upright": int(bool(falling_cat_upright)),
                "falling_cat_rp_max_deg": float(falling_cat_rp_max_deg),
            }
            return tau_cmd, pwm_us, info

        # ===== Build lever arms about COM (world) =====
        r_foot_w = (R_wb_hat @ (foot_b - self.com_b).reshape(3)).reshape(3)
        prop_r_w = (R_wb_hat @ (self.prop_positions_b - self.com_b.reshape(1, 3)).T).T.copy()

        # ===== Torque mapping A_tau_f (world GRF -> delta motor torques) =====
        A_tau_f_3rsr = None
        if self._leg_model == "serial":
            try:
                if J_body is not None:
                    # For serial model, J_body is in BASE frame (+Z up):
                    #   tau = J^T * f_leg_b,   f_leg_b = -R^T * f_grf_w
                    # => tau = -(J^T * R^T) * f_grf_w
                    A_tau_f_3rsr = (-(np.asarray(J_body, dtype=float).reshape(3, 3).T @ np.asarray(R_wb_hat, dtype=float).reshape(3, 3).T)).astype(float)
                    tau_sign = np.asarray(self.cfg.tau_cmd_sign, dtype=float).reshape(3)
                    A_tau_f_3rsr = (np.diag(tau_sign) @ A_tau_f_3rsr).astype(float)
            except Exception:
                A_tau_f_3rsr = None
        else:
            try:
                if self.kin is None:
                    raise RuntimeError("delta kinematics not initialized")
                # Use same foot offset convention as the MuJoCo demo (contact site 3cm below link origin)
                # Here x3 is already in delta/vicon frame (z positive)
                x3 = foot_vicon.copy()
                x3[2] = float(x3[2] + float(self._delta_ws["z_off"]))
                x3[0] = float(np.clip(x3[0], -float(self._delta_ws["xy"]), +float(self._delta_ws["xy"])))
                x3[1] = float(np.clip(x3[1], -float(self._delta_ws["xy"]), +float(self._delta_ws["xy"])))
                x3[2] = float(np.clip(x3[2], float(self._delta_ws["z_min"]), float(self._delta_ws["z_max"])))

                # Recompute inverse Jacobian at the clamped workspace point for numerical robustness.
                # Hopper4 returns J_inv such that: thetadot = J_inv * xdot  (delta/vicon frame)
                # Torque mapping: tau = inv(J_inv^T) * f_delta
                J_inv_map, _ = self.kin.inverse_jacobian(x3, np.zeros(3, dtype=float), theta=None)
                J_inv_map = np.asarray(J_inv_map, dtype=float).reshape(3, 3)
                inv_Jt = self._stable_inv3(J_inv_map.T)

                # f_w is the GRF (ground -> robot) in WORLD frame (+Z up).
                # Our torque convention matches Hopper4/ModeE: tau maps to the force the LEG applies on the ground
                # (robot -> ground), which is the opposite of GRF. Therefore the stance torque map has a leading '-'.
                #
                # World GRF -> BODY -> DELTA (+Z down):
                #   f_delta_grf = robot2vicon * (R_wb^T * f_w)
                # Desired leg force for torque mapping (robot->ground) is: f_delta_leg = -f_delta_grf
                # tau = inv(J_inv^T) * f_delta_leg
                A_tau_f_3rsr = (-(inv_Jt @ self.robot2vicon @ np.asarray(R_wb_hat, dtype=float).reshape(3, 3).T)).astype(float)
                # Motor torque sign convention (real robot)
                tau_sign = np.asarray(self.cfg.tau_cmd_sign, dtype=float).reshape(3)
                A_tau_f_3rsr = (np.diag(tau_sign) @ A_tau_f_3rsr).astype(float)
            except Exception:
                A_tau_f_3rsr = None

        tau_cmd_max = np.asarray(self.cfg.tau_cmd_max_nm, dtype=float).reshape(3)

        # ===== MPC references (stance) =====
        thrust_sum_ref = float(self.mass * self.gravity * float(self.cfg.prop_base_thrust_ratio))
        # Minimum stance thrust sum bound (policy): allows "legs do most of the stance work" when set to 0.
        thrust_sum_min_stance = float(self.mass * self.gravity * float(getattr(self.cfg, "stance_thrust_sum_min_ratio", 0.0)))
        # User request: leg-only stance attitude by default (disable props in stance).
        thrust_max_each_qp = float(self.cfg.thrust_max_each_n)
        if bool(self._stance) and (not bool(self.cfg.stance_use_props)):
            thrust_sum_ref = 0.0
            thrust_sum_min_stance = 0.0
            thrust_max_each_qp = 0.0
        f_ref = np.zeros(3, dtype=float)
        thrust_ref = None
        mpc_used = False
        fx_cmd_dbg = 0.0
        fy_cmd_dbg = 0.0

        if bool(self._stance):
            # Debug: allow disabling MPC to use fallback PD for staged tuning
            use_mpc = bool(getattr(self.cfg, "use_mpc_in_stance", True))
            if bool(falling_cat_active) and bool(getattr(self.cfg, "falling_cat_disable_mpc_in_stance", True)):
                use_mpc = False
            
            if bool(use_mpc):
                # recompute MPC at its own dt
                if not hasattr(self, "_mpc_last_t"):
                    self._mpc_last_t = -1e9
                    self._mpc_status = "init"
                    self._mpc_f0 = np.zeros(3, dtype=float)
                    self._mpc_t0 = np.ones(3, dtype=float) * (thrust_sum_ref / 3.0)

                if (float(self.sim_time) - float(self._mpc_last_t)) >= float(self.mpc.cfg.dt) - 1e-9:
                    pos_com_hat = (np.asarray(self._p_hat_w, dtype=float).reshape(3) + (R_wb_hat @ self.com_b.reshape(3))).reshape(3)
                    # Use filtered velocity for MPC (paper-grade baseline)
                    vel_hat = np.asarray(self._v_mpc_filtered, dtype=float).reshape(3) if bool(getattr(self.cfg, "use_mpc_velocity_filter", True)) else np.asarray(self._v_hat_w, dtype=float).reshape(3)
                    omega_w_hat = (R_wb_hat @ imu_gyro_b.reshape(3)).reshape(3)
                    yaw = float(rpy_hat[2])
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
                    sched = np.ones(N, dtype=int)
                    x_ref_seq = np.zeros((N, self.mpc.nx), dtype=float)

                    # Unified stance reference:
                    # - pz_ref/vz_ref come from a single smooth stance profile (soft-landing -> push-off)
                    # - vx/vy are blended: early stance keeps current velocity (soft landing), late stance tracks command.
                    t0 = float(t_in_stance)
                    t_comp = float(self._stance_t_comp) if (self._stance_t_comp is not None) else 0.0
                    denom_vxy = float(max(1e-6, float(self.cfg.stance_T) - t_comp))
                    for k in range(N):
                        tk = float(t0 + float((k + 1) * dtm))
                        if bool(self.cfg.use_unified_stance) and bool(self._stance_prof_inited):
                            pz_ref, vz_ref, _ = self._unified_stance_ref(tk)
                        else:
                            # Conservative fallback: hold height, keep current vz.
                            pz_ref = float(pos_com_hat[2])
                            vz_ref = float(vel_hat[2])

                        # Blend horizontal velocity reference:
                        # - During compression: keep current velocity to avoid large horizontal impulses that can
                        #   generate pitch/roll moments via the single-point contact.
                        # - During push: smoothly transition to desired velocity for convergence.
                        alpha = 0.0
                        try:
                            if float(denom_vxy) > 1e-6:
                                alpha = float((tk - float(t_comp)) / float(denom_vxy))
                        except Exception:
                            alpha = 0.0
                        alpha = float(np.clip(alpha, 0.0, 1.0))
                        vx_ref = float((1.0 - alpha) * float(vel_hat[0]) + alpha * float(desired_v_xy_w[0]))
                        vy_ref = float((1.0 - alpha) * float(vel_hat[1]) + alpha * float(desired_v_xy_w[1]))

                        x_ref_seq[k, :] = np.array(
                            [
                                pos_com_hat[0],
                                pos_com_hat[1],
                                float(pz_ref),
                                vx_ref,
                                vy_ref,
                                float(vz_ref),
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

                    sol_mpc = self.mpc.solve(
                        x0=x0,
                        x_ref_seq=x_ref_seq,
                        contact_schedule=sched,
                        m=float(self.mass),
                        g=float(self.gravity),
                        I_body=self.I_body,
                        r_foot_w=r_foot_w,
                        prop_r_w=prop_r_w,
                        z_w=z_w,
                        thrust_max_each=float(thrust_max_each_qp),
                        yaw_rate_ref=0.0,
                        thrust_sum_ref=float(thrust_sum_ref),
                        thrust_sum_target=float(thrust_sum_ref),
                        A_tau_f=A_tau_f_3rsr,
                        tau_cmd_max=tau_cmd_max,
                    )
                    self._mpc_status = str(sol_mpc.get("status", ""))
                    if self._mpc_status in ("solved", "solved_inaccurate"):
                        self._mpc_f0 = np.asarray(sol_mpc.get("f0", np.zeros(3)), dtype=float).reshape(3).copy()
                        self._mpc_t0 = np.asarray(sol_mpc.get("t0", np.ones(3) * (thrust_sum_ref / 3.0)), dtype=float).reshape(3).copy()
                    self._mpc_last_t = float(self.sim_time)
            else:
                # MPC disabled: force fallback PD
                self._mpc_status = "disabled"

            if str(getattr(self, "_mpc_status", "")) in ("solved", "solved_inaccurate"):
                f_ref = np.asarray(self._mpc_f0, dtype=float).reshape(3).copy()
                thrust_ref = np.asarray(self._mpc_t0, dtype=float).reshape(3).copy()

                mpc_used = True
                fx_cmd_dbg = float(f_ref[0])
                fy_cmd_dbg = float(f_ref[1])
            else:
                # fallback
                vx_used = float(self._v_hat_w[0])
                vy_used = float(self._v_hat_w[1])
                # User request: forbid XY asymmetry in horizontal velocity convergence.
                ki = float(self.cfg.ki_xy)
                fx = float(self.mass * float(self.cfg.axy_damp) * (float(desired_v_xy_w[0]) - vx_used) + self.mass * ki * float(self._v_int_xy[0]))
                fy = float(self.mass * float(self.cfg.axy_damp) * (float(desired_v_xy_w[1]) - vy_used) + self.mass * ki * float(self._v_int_xy[1]))
                fx_cmd_dbg = float(fx)
                fy_cmd_dbg = float(fy)
                F_des_z = float(self.mass * (float(az_des) + float(self.gravity)))
                fz = float(F_des_z - float(z_w[2]) * float(thrust_sum_ref))
                fz = float(np.clip(fz, 0.0, 220.0))
                mu = float(self.cfg.mu)
                fxy_lim = float(mu * fz)
                fx = float(np.clip(fx, -fxy_lim, fxy_lim))
                fy = float(np.clip(fy, -fxy_lim, fxy_lim))
                f_ref = np.array([fx, fy, fz], dtype=float)
                mpc_used = False

        else:
            f_ref[:] = 0.0

        # ===== PUSH takeoff-velocity catch (height convergence / leg-only hop) =====
        # Requirement: by the time q_shift reaches 0 (nominal leg length, liftoff trigger), the robot must reach
        # the takeoff speed needed to catch the apex. If MPC is allowed to sit at fz_min in PUSH, hop height will
        # be only a few cm. Here we compute a required acceleration from remaining vertical displacement and impose
        # a vertical force floor in PUSH.
        try:
            stance_push_catch = bool(getattr(self.cfg, "stance_push_v_to_catch", True))
            if bool(falling_cat_active) and bool(getattr(self.cfg, "falling_cat_disable_push_catch", True)):
                stance_push_catch = False
            if bool(self._stance) and (not bool(compress_active)) and bool(stance_push_catch):
                if (self._stance_z_end is not None) and np.isfinite(float(self._stance_z_end)):
                    # Current COM position (world)
                    pos_com_hat = (np.asarray(self._p_hat_w, dtype=float).reshape(3) + (R_wb_hat @ self.com_b.reshape(3))).reshape(3)
                    z_end = float(self._stance_z_end)
                    dz_rem = float(z_end - float(pos_com_hat[2]))
                    dz_min = float(max(1e-6, float(getattr(self.cfg, "stance_push_dz_min_m", 0.02))))
                    dz_use = float(max(dz_min, dz_rem))

                    # Desired takeoff speed (clip for safety)
                    v_goal = float(np.clip(float(self._v_to_cmd), float(self.cfg.v_to_min), float(self.cfg.v_to_max)))
                    vz_now = float(self._v_hat_w[2])

                    # Only accelerate upward if we're below target.
                    if np.isfinite(v_goal) and np.isfinite(vz_now) and (v_goal > vz_now + 1e-3):
                        # Required constant acceleration to reach v_goal over dz_use:
                        #   v_goal^2 = vz^2 + 2 a dz
                        a_req = float((v_goal * v_goal - vz_now * vz_now) / (2.0 * dz_use))
                        a_req = float(max(0.0, a_req))

                        # Saturate by MPC/QP force cap (convert to accel cap)
                        fz_cap = float(getattr(self.mpc.cfg, "fz_max", 220.0))
                        a_cap = float(max(0.0, (fz_cap / max(1e-6, float(self.mass))) - float(self.gravity)))
                        a_req = float(min(a_req, a_cap))

                        # Convert to required contact force (world z). If thrust_sum_ref is used, subtract its z contribution.
                        fz_req = float(float(self.mass) * (float(self.gravity) + a_req) - float(z_w[2]) * float(thrust_sum_ref))
                        fz_req = float(np.clip(fz_req, float(self.cfg.mpc_fz_min), fz_cap))

                        # Smooth ramp-in at PUSH start to avoid a hard step.
                        t_ramp = float(max(0.0, float(getattr(self.cfg, "stance_push_v_to_catch_ramp_s", 0.02))))
                        if t_ramp > 1e-6:
                            # push time since PUSH start ~ max(0, t_in_stance - t_comp)
                            t_comp_now = float(self._stance_t_comp) if self._stance_t_comp is not None else 0.0
                            t_push = float(max(0.0, float(t_in_stance) - t_comp_now))
                            w = float(self._smoothstep01(t_push / t_ramp))
                        else:
                            w = 1.0
                        fz_floor = float((1.0 - w) * float(self.cfg.mpc_fz_min) + w * float(fz_req))

                        # Enforce floor (do not reduce if MPC wants more)
                        f_ref[2] = float(max(float(f_ref[2]), float(fz_floor)))

                        # Keep fx/fy within friction cone after modifying fz.
                        mu = float(self.cfg.mu)
                        fxy_lim = float(mu * float(f_ref[2]))
                        f_ref[0] = float(np.clip(float(f_ref[0]), -fxy_lim, +fxy_lim))
                        f_ref[1] = float(np.clip(float(f_ref[1]), -fxy_lim, +fxy_lim))
        except Exception:
            pass

        # ===== SO(3) attitude torque (yaw free, no Euler PD) =====
        if bool(self._stance) and bool(compress_active):
            tau_rp_max = 80.0  # increased from 50.0 for better attitude control
            # kW was accidentally too low; low damping lets roll rate build during PUSH and causes flip.
            kR, kW = 55.0, 20.0
        elif bool(self._stance):
            tau_rp_max = 80.0  # increased from 50.0
            kR, kW = 55.0, 20.0
        else:
            # Flight phase: use separate roll/pitch gains
            tau_rp_max = float(self.cfg.flight_tau_rp_max)
            kR_roll = float(self.cfg.flight_kR_roll)
            kW_roll = float(self.cfg.flight_kW_roll)
            kR_pitch = float(self.cfg.flight_kR_pitch)
            kW_pitch = float(self.cfg.flight_kW_pitch)

        yaw = float(rpy_hat[2])
        R_des = _Rz(yaw)
        E = (R_des.T @ R_wb_hat) - (R_wb_hat.T @ R_des)
        e_R = 0.5 * _vee_so3(E)
        
        if bool(self._stance):
            # Stance phase: use unified kR/kW for both axes (yaw free)
            omega_b = np.asarray(imu_gyro_b, dtype=float).reshape(3)
            tau_b = (-float(kR) * e_R) - (float(kW) * omega_b)
            tau_b[2] = 0.0  # yaw free (consistent with flight phase)
        else:
            # Flight phase: use separate roll/pitch gains
            omega_b = np.asarray(imu_gyro_b, dtype=float).reshape(3)
            tau_b = np.zeros(3, dtype=float)
            tau_b[0] = (-float(kR_roll) * float(e_R[0])) - (float(kW_roll) * float(omega_b[0]))  # roll (x)
            tau_b[1] = (-float(kR_pitch) * float(e_R[1])) - (float(kW_pitch) * float(omega_b[1]))  # pitch (y)
            tau_b[2] = 0.0  # yaw free
        tau_w = (R_wb_hat @ tau_b.reshape(3)).reshape(3)
        Tau_des = np.array([float(tau_w[0]), float(tau_w[1]), 0.0], dtype=float)
        Tau_des[0] = float(np.clip(Tau_des[0], -tau_rp_max, +tau_rp_max))
        Tau_des[1] = float(np.clip(Tau_des[1], -tau_rp_max, +tau_rp_max))

        # ===== Flight swing torque reference (only after apex) =====
        tau_ref = None
        # Debug: force that is fed into the Jacobian->torque mapping.
        # - f_tau_b:     BODY frame (+Z up)
        # - f_tau_delta: delta/vicon frame (+Z down)
        # In flight: this comes from swing foot-space PD.
        # In stance: we derive it from the solved contact force (GRF) so it matches the stance A_tau_f mapping.
        f_tau_b = np.zeros(3, dtype=float)
        f_tau_delta = np.zeros(3, dtype=float)
        # Debug targets for logging/printing (always populate with finite shape)
        foot_des_b_dbg = np.full(3, np.nan, dtype=float)
        foot_des_w_dbg = np.full(3, np.nan, dtype=float)       # world-frame vector (base->foot)
        p_foot_des_w_dbg = np.full(3, np.nan, dtype=float)     # world-frame point (absolute)
        s2s_active_dbg = 0

        # ===== Foot velocity filtering (flight only, for high kd noise rejection) =====
        xdot_for_pd = foot_vrel_b.copy()
        if (not bool(self._stance)) and bool(self.cfg.use_foot_vel_lpf):
            tau_lpf = float(max(1e-6, float(self.cfg.foot_vel_lpf_tau)))
            a = float(np.clip(self.dt / (tau_lpf + self.dt), 0.0, 1.0))
            if not bool(self._foot_vrel_lpf_init):
                self._foot_vrel_lpf = foot_vrel_b.copy()
                self._foot_vrel_lpf_init = True
            self._foot_vrel_lpf = ((1.0 - a) * self._foot_vrel_lpf + a * foot_vrel_b).astype(float)
            xdot_for_pd = self._foot_vrel_lpf.copy()
        
        if not bool(self._stance):
            # Swing target convention (match Hopper4 style):
            # - Compute desired touchdown in a *heading* frame H:
            #     H has +Z aligned with WORLD (gravity up), and yaw aligned with body yaw.
            #   This ensures that when XY target is (0,0), the leg target is always vertical in WORLD,
            #   even if the body is rolled/pitched.
            # - Convert that target back into BODY frame for the foot-space PD.
            yaw_h = float(rpy_hat[2])
            R_wh = _Rz(yaw_h)  # heading->world (yaw only)

            # default: keep leg near nominal extension (world-vertical) and optionally place XY in heading frame.
            # IMPORTANT (match Hopper4):
            # - XY comes from S2S (in heading frame)
            # - Z is computed so that ||foot_des_h|| == l0 (constant leg length in flight)
            l0 = float(self.cfg.leg_l0_m)
            foot_des_h = np.array([0.0, 0.0, -l0], dtype=float)
            s2s_active = False

            # Falling-cat logic: only enable quaternion-mapped flight foot target when:
            #   - PWM mode (props enabled) AND angle < 45deg AND upright
            # Otherwise keep "factory/original" target (0,0,-l0) in BODY frame (no Raibert XY, no quat mapping).
            use_quat_mapping = True
            if bool(falling_cat_enabled):
                require_pwm_mode = bool(getattr(self.cfg, "falling_cat_require_pwm_mode", True))
                if require_pwm_mode:
                    # Check if props are enabled (PWM mode): stance_use_props indicates props can be used
                    props_enabled = bool(self.cfg.stance_use_props)
                    # Also check if base thrust ratio > 0 (props are active)
                    base_thrust_active = float(self.cfg.prop_base_thrust_ratio) > 1e-6
                    in_pwm_mode = props_enabled or base_thrust_active
                    # Only enable quaternion mapping if: PWM mode AND angle < 45deg AND upright
                    use_quat_mapping = bool(in_pwm_mode) and bool(falling_cat_quat_ok)
                else:
                    # If not requiring PWM mode, use original logic: just check angle < 45deg
                    use_quat_mapping = bool(falling_cat_quat_ok)
            
            if not use_quat_mapping:
                foot_des_b = np.array([0.0, 0.0, -l0], dtype=float)
                foot_des_w = (R_wb_hat @ foot_des_b.reshape(3)).reshape(3)
                s2s_active = False
            else:
                # Compute heading-frame velocity (yaw-only) for foot placement.
                v_h = (R_wh.T @ np.asarray(self._v_hat_w, dtype=float).reshape(3)).reshape(3)
                v_xy_h = np.asarray(v_h[0:2], dtype=float).reshape(2)
                v_des_h = (R_wh.T @ np.array([float(desired_v_xy_w[0]), float(desired_v_xy_w[1]), 0.0], dtype=float).reshape(3)).reshape(3)
                v_des_xy_h = np.asarray(v_des_h[0:2], dtype=float).reshape(2)

                # Hopper4 Raibert (Kv/Kr) in heading XY:
                #   target_xy = Kv * v_xy + Kr * v_des_xy
                kv = float(self.cfg.flight_kv)
                kr = float(self.cfg.flight_kr)
                target_xy = (kv * v_xy_h + kr * v_des_xy_h).astype(float)

                # Hopper4: clamp XY magnitude by stepperLim
                step_lim = float(abs(float(self.cfg.flight_stepper_lim_m)))
                nxy = float(np.linalg.norm(target_xy))
                if (step_lim > 1e-9) and (nxy > step_lim):
                    target_xy = (target_xy * (step_lim / max(1e-12, nxy))).astype(float)

                foot_des_h[0] = float(target_xy[0])
                foot_des_h[1] = float(target_xy[1])
                s2s_active = True

                # Enforce constant leg length in flight: ||foot_des_h|| == l0
                xy = np.asarray(foot_des_h[0:2], dtype=float).reshape(2).copy()
                l0_sq = float(l0 * l0)
                xy_norm2 = float(xy[0] * xy[0] + xy[1] * xy[1])
                if xy_norm2 > (l0_sq * 0.999999):
                    scale = float(math.sqrt((l0_sq * 0.999999) / max(1e-12, xy_norm2)))
                    xy = (xy * scale).astype(float)
                    xy_norm2 = float(xy[0] * xy[0] + xy[1] * xy[1])
                foot_des_h[0] = float(xy[0])
                foot_des_h[1] = float(xy[1])
                foot_des_h[2] = -float(math.sqrt(max(0.0, l0_sq - xy_norm2)))

                # Heading -> world -> body (for PD)
                foot_des_w = (R_wh @ foot_des_h.reshape(3)).reshape(3)
                foot_des_b = (R_wb_hat.T @ foot_des_w.reshape(3)).reshape(3)

            # Expose flight target for debug:
            # - foot_des_b is in BODY frame (+Z up) and is what the PD uses.
            # - Convert to WORLD for logging/plotting convenience.
            foot_des_b_dbg = np.asarray(foot_des_b, dtype=float).reshape(3).copy()
            foot_des_w_dbg = (np.asarray(R_wb_hat, dtype=float).reshape(3, 3) @ foot_des_b_dbg.reshape(3)).reshape(3)
            p_foot_des_w_dbg = (np.asarray(self._p_hat_w, dtype=float).reshape(3) + foot_des_w_dbg.reshape(3)).reshape(3)
            s2s_active_dbg = int(bool(s2s_active))

            # ===== Hopper4 flight leg force (sideForce + springForce), in BODY frame =====
            # Match Hopper4 lines:
            #   sideForce = Khp*(targetFootPos - x) - Khd*(xdot - omega×x)
            #   sideForce -= dot(sideForce, unitSpring)*unitSpring
            #   force = -k*(l - l0)
            #   springForce = force*unitSpring - b*springVel
            #   footForce = sideForce + springForce
            x = np.asarray(foot_b, dtype=float).reshape(3)
            targetFootPos = np.asarray(foot_des_b, dtype=float).reshape(3)
            xdot = np.asarray(xdot_for_pd, dtype=float).reshape(3)  # leg-induced foot velocity (base frame)

            leg_length = float(np.linalg.norm(x))
            if leg_length < 1e-6:
                unitSpring = np.array([0.0, 0.0, -1.0], dtype=float)
                leg_length = 0.0
            else:
                unitSpring = (x / leg_length).astype(float)

            springVel = (float(np.dot(xdot, unitSpring)) * unitSpring).astype(float)

            omega_b = np.asarray(imu_gyro_b, dtype=float).reshape(3)
            # No special gain for falling cat initial phase (user request: remove special parameters before first touchdown)
            Khp = float(self.cfg.swing_kp_xy)
            Khd = float(self.cfg.swing_kd_xy)
            sideForce = (Khp * (targetFootPos - x) - Khd * (xdot - np.cross(omega_b, x))).astype(float)
            sideForce = (sideForce - float(np.dot(sideForce, unitSpring)) * unitSpring).astype(float)

            k = float(self.cfg.swing_kp_z)
            b = float(self.cfg.swing_kd_z)
            force_scalar = -k * (leg_length - float(l0))  # == k*(l0 - l)
            springForce = (force_scalar * unitSpring - b * springVel).astype(float)

            footForce = (sideForce + springForce).astype(float)
            f_b_cmd = footForce.copy()

            if self._leg_model == "serial":
                # serial plant: use J_body (BASE frame) directly
                f_tau_b = f_b_cmd.copy()
                f_tau_delta = (self.robot2vicon @ f_tau_b.reshape(3)).reshape(3)
                try:
                    if J_body is None:
                        raise RuntimeError("serial Jacobian missing")
                    tau_ref = (np.asarray(J_body, dtype=float).reshape(3, 3).T @ f_b_cmd.reshape(3)).reshape(3)
                    tau_sign = np.asarray(self.cfg.tau_cmd_sign, dtype=float).reshape(3)
                    tau_ref = (tau_sign.reshape(3) * tau_ref.reshape(3)).reshape(3)
                    tau_ref = np.clip(tau_ref, -tau_cmd_max, +tau_cmd_max).astype(float)
                except Exception:
                    tau_ref = None
            else:
                # base->delta
                f_delta_cmd = (self.robot2vicon @ f_b_cmd.reshape(3)).reshape(3)
                f_tau_b = f_b_cmd.copy()
                f_tau_delta = f_delta_cmd.copy()
                try:
                    if J_inv is None:
                        raise RuntimeError("delta inverse Jacobian missing")
                    inv_Jt = self._stable_inv3(np.asarray(J_inv, dtype=float).reshape(3, 3).T)
                    tau_ref = (inv_Jt @ f_delta_cmd.reshape(3)).reshape(3)
                    # Motor torque sign convention (real robot wiring/driver)
                    tau_sign = np.asarray(self.cfg.tau_cmd_sign, dtype=float).reshape(3)
                    tau_ref = (tau_sign.reshape(3) * tau_ref.reshape(3)).reshape(3)
                    # clip to limits
                    tau_ref = np.clip(tau_ref, -tau_cmd_max, +tau_cmd_max).astype(float)
                except Exception:
                    tau_ref = None

        # ===== SRB-QP solve =====
        F_des = f_ref + z_w * float(thrust_sum_ref)
        # Allow total thrust to rise significantly above baseline for attitude control if needed.
        # Use configured thrust_total_ratio_max (e.g. 0.35*mg) as the upper bound.
        thrust_sum_max = float(self.mass * self.gravity * float(self.cfg.thrust_total_ratio_max))
        sol = self.wbc.update_and_solve(
            m=float(self.mass),
            g=float(self.gravity),
            z_w=z_w,
            r_foot_w=r_foot_w,
            prop_r_w=prop_r_w,
            F_des=F_des,
            Tau_des=Tau_des,
            in_stance=bool(self._stance),
            thrust_sum_target=None,
            # Stance: enforce an optional minimum thrust sum (often 0.0) so the leg does most work,
            # while keeping a large upper bound so props can still intervene strongly under disturbances.
            # Flight: keep the previous behavior (allow some baseline).
            thrust_sum_bounds=((float(thrust_sum_min_stance), thrust_sum_max) if bool(self._stance) else (0.5 * float(thrust_sum_ref), thrust_sum_max)),
            thrust_sum_ref=float(thrust_sum_ref),
            thrust_max_each=float(thrust_max_each_qp),
            f_ref=f_ref,
            thrust_ref=thrust_ref,
            A_tau_f=A_tau_f_3rsr if bool(self._stance) else None,
            tau_cmd_max=tau_cmd_max,
            tau_ref=tau_ref,
        )

        status = str(sol.get("status", ""))
        f_contact_w = np.asarray(sol.get("f_foot_w", np.zeros(3)), dtype=float).reshape(3)
        thrusts = np.asarray(sol.get("thrusts", np.zeros(3)), dtype=float).reshape(3)
        tau_qp = np.asarray(sol.get("tau_cmd", np.zeros(3)), dtype=float).reshape(3)
        slack = np.asarray(sol.get("slack", np.zeros(6)), dtype=float).reshape(6)

        ok_status = str(status) in ("solved", "solved inaccurate", "solved_inaccurate")
        ok = bool(ok_status) and np.all(np.isfinite(f_contact_w)) and np.all(np.isfinite(thrusts)) and np.all(np.isfinite(tau_qp))
        if ok:
            self._wbc_last_t = thrusts.copy()
            if bool(self._stance):
                self._wbc_last_f = f_contact_w.copy()
        else:
            thrusts = self._wbc_last_t.copy()
            if bool(self._stance):
                f_contact_w = self._wbc_last_f.copy()
            else:
                f_contact_w[:] = 0.0
            tau_qp = self._tau_cmd_prev.copy()
            slack[:] = 0.0
            status = f"fallback({status})"

        # Debug: in stance, derive the force used for Jacobian->torque mapping from the solved GRF.
        # QP variable `f_contact_w` is the GRF (ground -> robot) in WORLD frame (+Z up).
        # The torque mapping uses the LEG force convention (robot -> ground), which is the opposite of GRF.
        # We expose that force in both BODY (+Z up) and DELTA (+Z down) frames for debugging.
        if bool(self._stance):
            # Apply negative sign to x and y components of contact force for torque mapping
            # (z component keeps the original negative sign from -f_contact_w)
            f_contact_w_for_tau = -f_contact_w.copy()
            f_contact_w_for_tau[0] = -float(f_contact_w_for_tau[0])
            f_contact_w_for_tau[1] = -float(f_contact_w_for_tau[1])
            f_tau_b = (R_wb_hat.T @ f_contact_w_for_tau.reshape(3)).reshape(3)
            f_tau_delta = (self.robot2vicon @ f_tau_b.reshape(3)).reshape(3)

        # final motor torques: scale proportionally to keep direction if any exceeds limit
        tau_qp = np.asarray(tau_qp, dtype=float).reshape(3)
        tau_cmd_max = np.asarray(tau_cmd_max, dtype=float).reshape(3)
        # Find scaling factor: scale = min(1.0, min(tau_cmd_max[i] / abs(tau_qp[i])) for all i)
        scale = 1.0
        for i in range(3):
            if abs(tau_qp[i]) > 1e-9:
                scale_i = float(tau_cmd_max[i]) / abs(float(tau_qp[i]))
                scale = float(min(scale, scale_i))
        tau_cmd = (tau_qp * float(scale)).astype(float)
        self._tau_cmd_prev = tau_cmd.copy()

        # thrust (3 arms) -> 6 PWM (map via prop_pwm_idx_per_arm)
        thrust_motor = np.zeros(6, dtype=float)
        for arm_i in range(3):
            idxs = self._prop_pwm_groups[arm_i]
            t_each = float(thrusts[arm_i]) / float(len(idxs))
            for k in idxs:
                thrust_motor[int(k)] = t_each
        
        # Convert thrust to PWM using selected method
        if bool(self.use_hopper4_pwm):
            # Hopper4-style: pwm = 1000 + sqrt(thrust / k_thrust)
            pwm_us = np.zeros(6, dtype=float)
            for i in range(6):
                thrust_i = float(thrust_motor[i])
                if thrust_i <= 0.0:
                    pwm_us[i] = float(self.cfg.pwm_min_us)
                else:
                    k = float(self.prop_k_thrust)
                    if k > 1e-12:
                        pwm_delta = float(math.sqrt(thrust_i / k))
                        pwm_us[i] = float(self.cfg.pwm_min_us) + pwm_delta
                    else:
                        pwm_us[i] = float(self.cfg.pwm_min_us)
                # Clamp to limits
                pwm_us[i] = float(np.clip(pwm_us[i], float(self.cfg.pwm_min_us), float(self.cfg.pwm_max_us)))
        else:
            # MotorTableModel lookup table
            if self.motor_table is None:
                # Fallback: use Hopper4 method if table not initialized
                pwm_us = np.zeros(6, dtype=float)
                for i in range(6):
                    thrust_i = float(thrust_motor[i])
                    if thrust_i <= 0.0:
                        pwm_us[i] = float(self.cfg.pwm_min_us)
                    else:
                        k = float(self.prop_k_thrust)
                        if k > 1e-12:
                            pwm_delta = float(math.sqrt(thrust_i / k))
                            pwm_us[i] = float(self.cfg.pwm_min_us) + pwm_delta
                        else:
                            pwm_us[i] = float(self.cfg.pwm_min_us)
                    pwm_us[i] = float(np.clip(pwm_us[i], float(self.cfg.pwm_min_us), float(self.cfg.pwm_max_us)))
            else:
                pwm_us = self.motor_table.pwm_from_thrust(thrust_motor).astype(float).reshape(6)

        info = {
            "t": float(self.sim_time),
            "stance": int(bool(self._stance)),
            "touchdown": int(touchdown_evt),
            "liftoff": int(liftoff_evt),
            "apex": int(apex_evt),
            "compress": int(bool(compress_active)),
            "push": int(bool(self._stance) and (not bool(compress_active))),
            "desired_v_xy_w": np.asarray(desired_v_xy_w, dtype=float).reshape(2).copy(),
            "q_hat_wxyz": q_hat.copy(),
            "rpy_hat": rpy_hat.copy(),
            "p_hat_w": np.asarray(self._p_hat_w, dtype=float).reshape(3).copy(),
            "v_hat_w": np.asarray(self._v_hat_w, dtype=float).reshape(3).copy(),
            # Debug: base velocity measured from leg kinematics (foot assumed stationary in WORLD).
            "v_meas_foot_w": np.asarray(v_base_from_foot_w, dtype=float).reshape(3).copy(),
            # Foot kinematics:
            # - foot_vicon: delta/vicon frame (+Z DOWN)
            # - foot_b:     body frame (+Z UP)
            "foot_vicon": foot_vicon.copy(),
            "foot_b": foot_b.copy(),
            "foot_vdot_vicon": foot_vdot_vicon.copy(),
            "foot_vrel_b": foot_vrel_b.copy(),
            # Flight S2S/swing target (for debugging). In stance these will be NaNs.
            "s2s_active": int(s2s_active_dbg),
            "foot_des_b": foot_des_b_dbg.copy(),
            "foot_des_w": foot_des_w_dbg.copy(),
            "p_foot_des_w": p_foot_des_w_dbg.copy(),
            "q_shift_equiv": float(q_shift),
            "qd_shift_equiv": float(qd_shift),
            "comp_m": float(depth_now),
            "comp_tgt_m": float(depth_tgt),
            "comp_tgt_act_m": float(depth_tgt_act),
            "z_now_m": float(z_now) if bool(self._stance) else 0.0,
            "s_stance": float(s) if bool(self._stance) else 0.0,
            "compress_active": int(bool(compress_active)),
            "push_started": int(bool(self._stance) and (not bool(compress_active))),
            "mpc_used": int(bool(mpc_used)),
            "mpc_fx_cmd": float(fx_cmd_dbg),
            "mpc_fy_cmd": float(fy_cmd_dbg),
            "f_contact_w": f_contact_w.copy(),
            # Debug: force that is fed into the Jacobian->torque mapping.
            "f_tau_b": f_tau_b.copy(),
            "f_tau_delta": f_tau_delta.copy(),
            "thrusts_arm": thrusts.copy(),
            "tau_cmd": tau_cmd.copy(),
            "pwm_us": pwm_us.copy(),
            "slack": slack.copy(),
            "status": status,
            # Apex height feedback (for debugging/convergence analysis)
            "z_lo_m": float(self._z_lo) if self._z_lo is not None else float("nan"),
            "vz_lo_m_s": float(self._vz_lo) if self._vz_lo is not None else float("nan"),
            "z_apex_actual_m": float(self._z_apex_actual) if self._z_apex_actual is not None else float("nan"),
            "apex_err_int": float(self._apex_err_int),
            "v_to_cmd_m_s": float(self._v_to_cmd),
            "desired_vz_from_apex_m_s": float(np.sqrt(2.0 * g_eff_log * dz_tgt_log)) if np.isfinite(g_eff_log) and np.isfinite(dz_tgt_log) else float("nan"),
            "hop_peak_z_m": float(self.cfg.hop_peak_z),
            # Falling cat debug (recovery gating)
            "falling_cat_enable": int(bool(falling_cat_enabled)),
            "falling_cat_active": int(bool(falling_cat_active)),
            "falling_cat_quat_ok": int(bool(falling_cat_quat_ok)),
            "falling_cat_upright": int(bool(falling_cat_upright)),
            "falling_cat_rp_max_deg": float(falling_cat_rp_max_deg),
        }

        return tau_cmd, pwm_us, info


