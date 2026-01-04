
import numpy as np

class HopperConfig:
    
    def __init__(self):

        self.D = 0.188
        self.d = 0.398
        self.r = 0.02424


        self.m = 3.23


        self.l0 = 0.464
        self.k = 1000
        self.b = 20
        self.h = 0.15


        self.Kv = 0.1
        self.Kr = 0.09
        #40 2.9
        self.Khp = 50
        self.Khd = 1


        self.Kp = 7
        
        # Hip torque独立增益控制（X和Y方向）
        self.Kpp_x = 31  # Hip torque X方向位置增益（与MATLAB一致）
        self.Kpp_y = 31   # Hip torque Y方向位置增益（与MATLAB一致）
        self.Kpd_x = 3.3  # Hip torque X方向速度增益（与MATLAB一致）
        self.Kpd_y = 3.3  # Hip torque Y方向速度增益（与MATLAB一致）


        self.Kpj = 2
        self.Kdj = 1
        self.Kpos = 0.0


        self.stepperLim = 0.12
        self.hipTorqueLim = 7  # 30d
        self.posVelLim = 0.8  # 提高速度限制以支持0.8m/s的期望速度
        self.upLim = 1.38
        self.lowLim = -1.04
        self.max_joint_torque = 15



        self.touchdown_threshold = 0.02
        self.liftoff_threshold = 0.00


        self.propeller_arm_length = 0.57
        self.base_throttle = 1200
        self.pwm_min = 1000
        self.pwm_max = 1700
        self.target_roll = 0.0
        self.target_pitch = 0.0
        
        # ========== PWM调试参数 ==========
        # Stance Phase PWM设置
        self.stance_pwm = 1050  # 🔧 stance phase固定PWM值
        
        # Flight Phase PWM设置
        self.flight_base_pwm = 1050  # 🔧 flight phase基础PWM值
        
        # PWM滤波参数
        self.pwm_filter_alpha = 0.3  # 🔧 PWM低通滤波系数 (0=无滤波, 1=完全滤波)
        
        # ========== 推力模型参数 ==========
        self.Omega_nom = 2000  # 🔧 额定转速 (RPM)
        self.k_thrust = 1.47e-4  # 🔧 推力系数 (提高此值可增加推力)
        self.max_thrust_per_motor = 1500000.0  # 🔧 单电机最大推力 (N)
        
        # ========== 平衡控制增益 ==========
        # Roll轴PD增益
        self.Kp_roll = 30.0  # 🔧 Roll轴位置增益
        self.Kd_roll = 30.0  # 🔧 Roll轴速度增益
        
        # Pitch轴PD增益
        self.Kp_pitch = 30.0  # 🔧 Pitch轴位置增益
        self.Kd_pitch = 30.0  # 🔧 Pitch轴速度增益
        
        # ========== 矢量推进模式参数 ==========
        # 速度-姿态映射参数（用于velocity_to_attitude_command）
        self.max_propeller_vel = 0.8  # 最大速度 (m/s)，用于归一化
        self.max_tilt_angle = 15.0    # 最大倾角 (度)
        
        
    def print_config(self):
        print("=== Hopper Delta Robot Configuration (Clean) ===")
        print(f"Robot mass: {self.m:.2f} kg")
        print(f"Spring rest length: {self.l0:.4f} m")
        print(f"Spring stiffness: {self.k:.0f} N/m")
        print(f"Stepper limit: {self.stepperLim:.3f} m")
        print(f"Joint limits: [{self.lowLim:.3f}, {self.upLim:.3f}] rad")
        print("\n=== PWM & Propeller Control Parameters ===")
        print(f"PWM min/max: [{self.pwm_min}, {self.pwm_max}]")
        print(f"Stance PWM: {self.stance_pwm}")
        print(f"Flight base PWM: {self.flight_base_pwm}")
        print(f"PWM filter alpha: {self.pwm_filter_alpha}")
        print(f"k_thrust: {self.k_thrust:.2e}")
        print(f"Omega_nom: {self.Omega_nom} RPM")
        print("\n=== Balance Control Gains ===")
        print(f"Roll PD: Kp={self.Kp_roll:.1f}, Kd={self.Kd_roll:.1f}")
        print(f"Pitch PD: Kp={self.Kp_pitch:.1f}, Kd={self.Kd_pitch:.1f}")
        print("===============================================")


config = HopperConfig()

if __name__ == "__main__":
    config.print_config()
