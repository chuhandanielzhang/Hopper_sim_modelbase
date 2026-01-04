#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

OUT_DIR="$(cd .. && pwd)/videos"
mkdir -p "$OUT_DIR"

OUT_MP4="$OUT_DIR/hopper4_leg_inplace.mp4"

echo "=== Hopper_sim / model_spring: record Hopper4 LEG-only in-place hop ==="
echo "Output: $OUT_MP4"

# Clean up old processes (best-effort)
pkill -f mujoco_lcm_fake_robot.py 2>/dev/null || true
pkill -f run_hopper4_leg_sim.py 2>/dev/null || true
pkill -f Hopper4.py 2>/dev/null || true
sleep 1

# Start MuJoCo fake robot (3RSR plant)
python3 ../model_aero/mujoco_lcm_fake_robot.py \
  --arm \
  --realtime \
  --model "$(cd .. && pwd)/mjcf/hopper_3rsr_parallel.xml" \
  --duration-s 10 \
  --record-mp4 "$OUT_MP4" \
  --hud \
  > /tmp/hopper_sim_hopper4_leg_mj.log 2>&1 &
MJ_PID=$!

sleep 1

# Start Hopper4 controller (leg only)
python3 run_hopper4_leg_sim.py --duration-s 10 > /tmp/hopper_sim_hopper4_leg_ctl.log 2>&1 &
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
  tail -80 /tmp/hopper_sim_hopper4_leg_mj.log 2>/dev/null || true
  echo "--- tail controller log ---"
  tail -80 /tmp/hopper_sim_hopper4_leg_ctl.log 2>/dev/null || true
  exit 1
fi


