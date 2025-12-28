## LCM types (generated python)

These message classes mirror the real-robot message definitions under `Hopper-aero/hopper_lcm_types/`.

They are used with the **real MIT LCM** Python bindings (`import lcm`) so that MuJoCo IO can look like:
- `hopper_data_lcmt` (joint pos/vel/torque)
- `hopper_imu_lcmt` (quat/gyro/acc/rpy)
- `hopper_cmd_lcmt` (torque command)
- `motor_pwm_lcmt` (6-channel PWM + arm flag)
- `gamepad_lcmt` (optional)

Import pattern (same as `Hopper-aero`):

```python
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lcm_types"))
from python.hopper_data_lcmt import hopper_data_lcmt
```


