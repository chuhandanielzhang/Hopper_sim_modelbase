#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

OUT_DIR="$(cd .. && pwd)/videos"
mkdir -p "$OUT_DIR"

OUT_MP4="$OUT_DIR/modee_serial_inplace.mp4"

echo "=== Hopper_sim / model_aero: record ModeE (serial) in-place hop ==="
echo "Output: $OUT_MP4"

# Clean up old processes (best-effort)
pkill -f mujoco_lcm_fake_robot.py 2>/dev/null || true
pkill -f run_modee.py 2>/dev/null || true
sleep 1

# Start MuJoCo fake robot (serial plant)
python3 mujoco_lcm_fake_robot.py \
  --arm \
  --realtime \
  --model "$(cd .. && pwd)/mjcf/hopper_serial.xml" \
  --q-sign 1 \
  --q-offset 0 \
  --hold-level-s 1.0 \
  --fake-gamepad \
  --fake-gamepad-y-once \
  --cmd-vx0 0.0 \
  --cmd-vy0 0.0 \
  --cmd-vx1 0.0 \
  --cmd-vy1 0.0 \
  --cmd-switch-after-s 9999 \
  --duration-s 10 \
  --record-mp4 "$OUT_MP4" \
  --hud \
  > /tmp/hopper_sim_modee_mj.log 2>&1 &
MJ_PID=$!

sleep 1

# Start ModeE controller (serial leg model; allow large shift force in MuJoCo)
python3 run_modee.py \
  --leg-model serial \
  --tau-out-max 2500 \
  --print-hz 50 \
  > /tmp/hopper_sim_modee_ctl.log 2>&1 &
CTL_PID=$!

echo "Running... (MuJoCo PID=$MJ_PID, controller PID=$CTL_PID)"
wait "$MJ_PID" || true

echo "Stopping controller..."
kill "$CTL_PID" 2>/dev/null || true
wait "$CTL_PID" 2>/dev/null || true

if [ -f "$OUT_MP4" ]; then
  echo "✅ Done: $OUT_MP4"
  ls -lh "$OUT_MP4"
else
  echo "❌ Video not found: $OUT_MP4"
  echo "--- tail mujoco log ---"
  tail -80 /tmp/hopper_sim_modee_mj.log 2>/dev/null || true
  echo "--- tail controller log ---"
  tail -80 /tmp/hopper_sim_modee_ctl.log 2>/dev/null || true
  exit 1
fi


