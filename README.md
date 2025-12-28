# Hopper_e

This is an **isolated, paper-style** controller stack for the MuJoCo hopper (serial-equivalent `roll/pitch/shift` leg)
that implements:

- **Event-based hop phase machine**: touchdown / compression / push / flight / apex (no fixed clock for switching)
- **SRB condensed Wrench MPC** (MIT-style): optimizes **GRF + tri-rotor thrusts** over a horizon, with contact schedule
- **SE3 attitude (roll/pitch only)**: yaw is **free** (no yaw tracking, no yaw torque)
- **Unified WBC-QP (OSQP)**: solves **GRF + tri-rotor thrusts + joint torques** with actuator constraints
- **3RSR feasibility in MPC (10 Nm)**: MPC enforces **real delta/3RSR motor torque limits** using Hopper4’s inverse Jacobian,
  while the MuJoCo simulation still executes the serial-equivalent leg torques.

Notes:
- Generated artifacts under `videos/` are ignored by `.gitignore`.

## How to run

### In-place / constant-velocity record

```bash
python scripts/record_modee.py --vx 0.6 --duration_s 20 --tag demo_vx06
```

### Task4 record (fwd → zero → lateral impulse)

```bash
python scripts/record_modee.py --task4 --vx 0.6 --task4_fwd_s 10 --task4_zero_s 10 --task4_push_s 5 --task4_dvy 0.25 --tag task4_vx06
```

Outputs:
- `videos/modee/*.mp4`
- `videos/modee/*_h264.mp4`
- `videos/modee/*.csv`

## LCM IO (MuJoCo “looks like hardware”)

This repo uses the **real MIT LCM** Python bindings (`import lcm`) and the matching generated message classes under `lcm_types/`.
This is useful when you want the MuJoCo side to publish the same messages as the real robot:

```bash
python scripts/mujoco_lcm_fake_robot.py --duration_s 20
```

Then, in another process, you can subscribe/publish on:
- `hopper_data_lcmt`, `hopper_imu_lcmt` (sensor)
- `hopper_cmd_lcmt`, `motor_pwm_lcmt` (command)

### Installing real `lcm` (if needed)

If `python -c "import lcm"` fails, you can build & install MIT LCM (with Python bindings) into your current conda prefix **without sudo**:

```bash
python -m pip install --upgrade cmake

mkdir -p ~/Hopper/_third_party && cd ~/Hopper/_third_party
rm -rf lcm && git clone --depth 1 https://github.com/lcm-proj/lcm.git
cd lcm && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DLCM_ENABLE_PYTHON=ON -DLCM_ENABLE_JAVA=OFF -DLCM_ENABLE_LUA=OFF
cmake --build . --parallel 4
cmake --install .
```


