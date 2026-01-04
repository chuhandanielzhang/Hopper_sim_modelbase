


import numpy as np
import time
import lcm
import threading
import sys
import os
import subprocess
from numpy._typing import _256Bit
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from collections import deque


current_dir = os.path.dirname(os.path.abspath(__file__))
lcm_types_dir = os.path.join(current_dir, '..', 'hopper_lcm_types', 'lcm_types')
sys.path.append(lcm_types_dir)


from python.hopper_data_lcmt import hopper_data_lcmt
from python.hopper_cmd_lcmt import hopper_cmd_lcmt
from python.hopper_imu_lcmt import hopper_imu_lcmt
from python.gamepad_lcmt import gamepad_lcmt
from python.motor_pwm_lcmt import motor_pwm_lcmt


from hopper_config import HopperConfig
from forward_kinematics import ForwardKinematics, InverseJacobian
from com_filter import ComplementaryFilter

def velocity_to_attitude_command(desired_vel, max_vel=0.8, max_tilt_angle=15.0):
    """
    将期望速度转换为期望姿态（矢量推进模式）
    
    Args:
        desired_vel: 期望速度向量 [vx, vy, vz] (m/s)
        max_vel: 最大速度 (m/s)
        max_tilt_angle: 最大倾角 (度)
    
    Returns:
        desired_rpy: 期望姿态 [roll, pitch, yaw] (rad)
    """
    max_tilt = np.radians(max_tilt_angle)
    
    # 线性映射：速度 → 倾角
    # 向前(+X) → 前倾(+Pitch)
    # 向左(+Y) → 左倾(+Roll)
    desired_pitch = np.clip(
        desired_vel[0] / max_vel * max_tilt,
        -max_tilt, max_tilt
    )
    
    desired_roll = np.clip(
        desired_vel[1] / max_vel * max_tilt,
        -max_tilt, max_tilt
    )
    
    return np.array([desired_roll, desired_pitch, 0.0])

class VirtualSpringController:
    
    def __init__(self):
        config = HopperConfig()
        

        self.l0 = config.l0
        self.k = config.k
        self.b = config.b
        self.m = config.m
        self.h = config.h
        

        self.Kv = config.Kv
        self.Kr = config.Kr
        self.Khp = config.Khp
        self.Khd = config.Khd
        

        self.Kp = config.Kp
        
        # Hip torque独立增益控制（X和Y方向）
        self.Kpp_x = config.Kpp_x
        self.Kpp_y = config.Kpp_y
        self.Kpd_x = config.Kpd_x
        self.Kpd_y = config.Kpd_y

        self.stepperLim = config.stepperLim
        self.hipTorqueLim = config.hipTorqueLim
        self.posVelLim = config.posVelLim
        self.upLim = config.upLim
        self.lowLim = config.lowLim
        self.max_torque = config.max_joint_torque
        

        self.flight_to_stance_threshold = config.touchdown_threshold
        self.stance_to_flight_threshold = config.liftoff_threshold
        self.Kpos = config.Kpos
        self.Kpj = config.Kpj
        self.Kdj = config.Kdj
        

        self.state = 1
        self.state_safety = 0
        self.time = time.time()
        

        # 禁用SimulinkVelocityFilter，直接使用C++底层的qd
        self.kinematics = InverseJacobian(
            use_simulink_filter=False,
            forgetting_factor=0.95,
            dt=0.001
        )


        self.forcemap = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        

        self.footForce = np.zeros(3)
        self.target_landing_point = np.zeros(2)
        self.springForce_scalar_copy = 0
        self.energy_compensation_copy = 0
        self.leg_velocity_copy = 0
        self.hip_torque = np.zeros(3)  # Hip torque for stance phase
        self.X_debug = np.zeros(3)
        self.P_rotated_debug = np.zeros(3)
        self.l_debug = 0


        self.current_robot_tilt = np.zeros(3)
        self.flight_sideforce = np.zeros(3)
        self.stance_sideforce = np.zeros(3)
        self.flight_footforce = np.zeros(3)
        self.stance_footforce = np.zeros(3)
        self.flight_springforce = np.zeros(3)
        self.stance_springforce = np.zeros(3)
        self.flight_x = np.zeros(3)
        self.stance_x = np.zeros(3)
        self.flight_target_pos = np.zeros(3)
        self.stance_target_pos = np.zeros(3)
        self.flight_torque = np.zeros(3)
        self.stance_torque = np.zeros(3)
        
        # 原始值（未经过四元数转换）
        self.flight_x_raw = np.zeros(3)
        self.stance_x_raw = np.zeros(3)
        self.flight_target_pos_raw = np.zeros(3)
        self.stance_target_pos_raw = np.zeros(3)
        
        # 世界坐标系值（经过四元数转换）
        self.flight_x_world = np.zeros(3)
        self.stance_x_world = np.zeros(3)
        
        self.print_ready = False
    
    def virtual_spring_control(self, Zoffset, X, xdot, joint, jointVel, vel, quat, angVel, robotPos, 
                              desiredPos, flip=0, rpy=None, gamepad_data=None, imu_acc=None, foot_vel_filtered=None,
                              propeller_mode=False, propeller_desired_vel=None):
        current_time = time.time()
        

        


        quat_copy = quat.copy()
        
        from scipy.spatial.transform import Rotation
        
        # 四元数格式转换：输入quat是[w,x,y,z]，scipy需要[x,y,z,w]
        quat_scipy = [quat[1], quat[2], quat[3], quat[0]]  # [x, y, z, w]
        vicon2world = Rotation.from_quat(quat_scipy).as_matrix()
        

        robot2vicon = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, -1]])
        

        x = X + np.array([0, 0, 0.03])
        l = np.linalg.norm(x)
        


        

      
        

        

        


        


        rAngVel =  angVel
        

        if desiredPos[2] == 0:
            desiredVel = self.Kpos * (desiredPos - robotPos)
        else:
            # 速度控制模式：直接使用desiredPos的X和Y作为期望速度
            desiredVel = np.array([desiredPos[0], desiredPos[1], 0])
            

        if np.linalg.norm(desiredVel) > self.posVelLim:
            desiredVel = desiredVel / np.linalg.norm(desiredVel) * self.posVelLim
            


        unitSpring = x / l
        
            
        springVel = np.dot(xdot, unitSpring) * unitSpring
        

        leg_velocity = np.dot(xdot, unitSpring)
        
        groundHeight = np.dot(vicon2world @ x, np.array([0, 0, -1]))
        

        energy = 0.5 * self.m * np.dot(springVel, springVel) + 0.5 * self.k * (self.l0 - l)**2 + self.m * 9.81 * (-1)*groundHeight
        target = self.m * 9.81 * (self.l0 + self.h) + 0.5 * self.m * np.dot(desiredVel, desiredVel)
        error = target - energy
        force = -self.k * (l - self.l0)

        self.energy_error = error
        self.energy_target = target
        self.energy_current = energy
        self.ground_height_copy = groundHeight
        



        worldUp = robot2vicon.T @ vicon2world.T @ np.array([0, 0, 1])
        robotTilt = np.cross(worldUp, np.array([0, 0, -1]))
        
        # 角度归一化（与MATLAB一致）
        if np.linalg.norm(robotTilt) > 0.01:
            robotTilt = robotTilt / np.linalg.norm(robotTilt) * np.arccos(np.clip(np.dot(worldUp, np.array([0, 0, -1])), -1.0, 1.0))
        
        self.robotTilt_debug = robotTilt.copy()
        self.worldUp_debug = worldUp.copy()

        robotTilt_copy = robotTilt.copy()

        self.current_robot_tilt = robotTilt.copy()
        

        if self.state < 3 or self.state > 7:
            # 安全检查：检测机器人是否倾倒过大
            from scipy.spatial.transform import Rotation
            # 四元数格式转换：输入quat是[w,x,y,z]，scipy需要[x,y,z,w]
            quat_scipy = [quat[1], quat[2], quat[3], quat[0]]  # [x, y, z, w]
            euler = Rotation.from_quat(quat_scipy).as_euler('xyz', degrees=False)
            roll_rad = abs(euler[0])
            pitch_rad = abs(euler[1])
            


            robotTilt_deg = np.degrees(robotTilt)
            if roll_rad > np.radians(100) or pitch_rad > np.radians(100):
                self.state = 1001





            elif abs(l - self.l0) > 1.15 and flip != 1:
                self.state = 1003

                
        footForce = np.zeros(3)
        torque = np.zeros(3)
        balanceSafety = 1
        desirePropCurr = 0
        flipping = 0
        hipTorque = np.zeros(3)
        hipTorque_copy = np.zeros(3)
        

        self.X_debug = x.copy()
        self.P_rotated_debug = x.copy()
        self.l_debug = l
        
        if self.state == 1:
            ground = 0
            

            # Raibert足端放置控制：targetFootPos = Kv * v_current + Kr * v_desired
            # Kv: 当前速度反馈增益（预测飞行距离）
            # Kr: 期望速度增益（速度跟踪）
            targetFootPos = self.Kv * np.array([vel[0], vel[1], 0]) + self.Kr * np.array([desiredVel[0], desiredVel[1], 0])
            

            normTarget = np.linalg.norm(targetFootPos)
            if normTarget > self.stepperLim:
                targetFootPos = targetFootPos / normTarget * self.stepperLim
                normTarget = np.linalg.norm(targetFootPos)

            targetFootPos[2] = -np.sqrt(self.l0**2 - normTarget**2)
            
            # 🔧 保存原始targetFootPos（未经过四元数转换）
            targetFootPos_raw = targetFootPos.copy()

            targetFootPos = robot2vicon.T @ vicon2world.T @ targetFootPos
            

                
            # 记录控制数据 (只进行Kv Kr后的targetFootPos)
            targetFootPos_after_kvkr = targetFootPos.copy()

            sideForce = self.Khp * (targetFootPos - x) - self.Khd * (xdot - np.cross(rAngVel, x))
            

            sideForce = sideForce - np.dot(sideForce, unitSpring) * unitSpring

            springForce = force * unitSpring  - self.b * springVel
            


            footForce =  sideForce + springForce
            ff = footForce

            
            J, _ = self.kinematics.inverse_jacobian(x, np.zeros(3))
            torque = np.linalg.inv(J.T) @ footForce

            

            self.state_safety = self.state_safety + 1
            

            self.footForce = ff.copy()
            self.sideForce = sideForce.copy()
            self.springForce_vec = springForce.copy()
            self.target_landing_point = targetFootPos[:2].copy()
            self.springForce_scalar_copy = 0
            self.energy_compensation_copy = 0
            self.leg_velocity_copy = np.dot(xdot, unitSpring)
            

            self.flight_sideforce = sideForce.copy()
            self.flight_footforce = footForce.copy()
            self.flight_springforce = springForce.copy()
            self.flight_x = x.copy()
            self.flight_torque = torque.copy()


            self.flight_target_pos = targetFootPos.copy()
            self.flight_target_pos_raw = targetFootPos_raw.copy()  # 原始targetFootPos（未转换）
            self.flight_x_raw = X.copy()  # 原始X（未加offset）
            # 转换foot_pos到世界坐标系
            self.flight_x_world = robot2vicon.T @ vicon2world.T @ x
            
            if l < self.l0 - self.flight_to_stance_threshold:
                self.state = 2
                self.state_safety = 0
                
        elif self.state == 2:
            ground = 10
            

            springForce_scalar = -self.k * (l - self.l0)

            leg_velocity = np.dot(xdot, unitSpring)

            energy_compensation = 0
            
            if leg_velocity > 0.1:
                energy_compensation = np.sign(leg_velocity) * self.Kp * error
                springForce_scalar = springForce_scalar + energy_compensation
            if springForce_scalar < 0:
                springForce_scalar = 0

            springForce = springForce_scalar * unitSpring
            

          



            # ========== 计算期望姿态 ==========
            if propeller_mode and propeller_desired_vel is not None:
                # 矢量推进模式：期望姿态由Propeller期望速度决定
                config = HopperConfig()
                desired_rpy = velocity_to_attitude_command(
                    propeller_desired_vel, 
                    config.max_propeller_vel,
                    config.max_tilt_angle
                )
            else:
                # 传统模式：期望姿态为0（保持水平）
                desired_rpy = np.array([0.0, 0.0, 0.0])
            
            # ========== Hip Torque姿态跟踪（统一期望） ==========
            # 计算姿态误差
            rpy_error = desired_rpy - rpy if rpy is not None else np.zeros(3)
            
            # X和Y方向独立增益控制
            hipTorque = np.zeros(3)
            hipTorque[0] = self.Kpp_x * (robotTilt[0] + rpy_error[0]) - self.Kpd_x * rAngVel[0]
            hipTorque[1] = self.Kpp_y * (robotTilt[1] + rpy_error[1]) - self.Kpd_y * rAngVel[1]
            hipTorque[2] = 0  # Z方向不控制
            
            hipTorque = hipTorque - np.dot(hipTorque, unitSpring) * unitSpring
                
             # 按照MATLAB逻辑：限制hipTorque大小
            if np.linalg.norm(hipTorque) > self.hipTorqueLim:
                hipTorque = hipTorque / np.linalg.norm(hipTorque) * self.hipTorqueLim
                

            sideForce = np.cross(hipTorque, x) / np.dot(x, x)
            footForce = springForce + sideForce

            J, _ = self.kinematics.inverse_jacobian(x, np.zeros(3))
            torque = np.linalg.inv(J.T) @ footForce
            

            

            self.state_safety = self.state_safety + 1
            

            self.footForce = footForce.copy()
            self.sideForce = sideForce.copy()
            self.springForce_vec = springForce.copy()
            self.springForce_scalar_copy = springForce_scalar
            self.energy_compensation_copy = energy_compensation
            self.leg_velocity_copy = leg_velocity
            self.hip_torque = hipTorque.copy()  # 保存hip torque


            self.stance_sideforce = sideForce.copy()
            self.stance_footforce = footForce.copy()
            self.stance_springforce = springForce.copy()
            self.stance_x = x.copy()
            self.stance_torque = torque.copy()


            self.stance_target_pos = np.array([0.0, 0.0, 0.0])
            self.stance_target_pos_raw = np.array([0.0, 0.0, 0.0])  # Stance phase无目标位置
            self.stance_x_raw = X.copy()  # 原始X（未加offset）
            # 转换foot_pos到世界坐标系
            self.stance_x_world = robot2vicon.T @ vicon2world.T @ x
            
            if l > self.l0 + self.stance_to_flight_threshold:
                self.state = 1
                self.state_safety = 0
                
        else:
            ground = 0

            safe_position = np.array([-20, -20, -20]) * np.pi / 180
            torque =0
            

            self.state_safety = self.state_safety + 1
            if self.state_safety > 250:
                self.state = 1
                self.state_safety = 0
                balanceSafety = 0
                

        if any(joint < self.lowLim) or any(joint > self.upLim):
            torque = np.zeros(3)
        

        max_abs_torque = np.max(np.abs(torque))
        if max_abs_torque > self.max_torque:
            scale_factor = self.max_torque / max_abs_torque
            torque = torque * scale_factor
        
        torque = -torque
        

        debug = np.array([l, desiredVel[0], desiredVel[1], 
                         rAngVel[0], rAngVel[1], rAngVel[2],
                         vel[0], vel[1], self.state,
                         np.sqrt(vel[0]**2 + vel[1]**2)])
        
        return torque, balanceSafety, debug, desirePropCurr, flipping

class HopperLCMController:
    
    def __init__(self):
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        

        self.robot_state = {
            'q': np.zeros(3),
            'qd': np.zeros(3),
            'tau': np.zeros(3),
            'imu_quat': np.zeros(4),
            'imu_gyro': np.zeros(3),
            'imu_acc': np.zeros(3),
            'imu_rpy': np.zeros(3),
            'gamepad_data': None
        }
        

        self.controller = VirtualSpringController()
        

        self.com_filter = ComplementaryFilter()
        self.controller.com_filter = self.com_filter
        

        self.desired_position = np.array([0.0, 0.0, 0.0])
        self.desired_velocity = np.array([0.0, 0.0, 0.0])  # 期望速度（由手柄右摇杆控制）
        self.flip_command = 0

        self.robot_position = np.zeros(3)
        self.robot_velocity = np.zeros(3)
        
        # ========== 矢量推进模式控制 ==========
        self.propeller_vector_mode = False  # A键切换
        self.desired_velocity_propeller = np.array([0.0, 0.0, 0.0])  # Propeller期望速度
        self.a_pressed = False  # A键状态
        
        # 数据记录（用于程序结束时绘图）
        self.data_log = {
            'time': [],
            'hip_torque_x': [],
            'hip_torque_y': [],
            'rpy_roll': [],
            'rpy_pitch': [],
            'rpy_yaw': [],
            'robot_vel_x': [],
            'robot_vel_y': [],
            'robot_vel_z': [],
            'foot_pos_x': [],
            'foot_pos_y': [],
            'foot_pos_z': [],
            'target_pos_x': [],
            'target_pos_y': [],
            'target_pos_z': [],
            'robot_pos_x': [],  # 机器人X位置
            'robot_pos_y': [],  # 机器人Y位置
            'dq0': [],  # 关节速度0
            'dq1': [],  # 关节速度1
            'dq2': [],  # 关节速度2
            'state': [],
            # 新增：姿态控制相关
            'desired_roll': [],  # 期望roll
            'desired_pitch': [],  # 期望pitch
            'error_roll': [],  # roll误差
            'error_pitch': [],  # pitch误差
            'propeller_L': [],  # Roll力矩
            'propeller_M': [],  # Pitch力矩
            'desired_vel_x': [],  # 期望速度X
            'desired_vel_y': [],  # 期望速度Y
        }
        self.log_start_time = None  # 按Y键时才设置
        self.enable_data_logging = False  # 控制是否记录数据
        
        # 临时变量用于数据记录
        self.current_desired_rpy = np.array([0.0, 0.0, 0.0])
        self.current_error_rpy = np.array([0.0, 0.0, 0.0])
        self.current_propeller_LM = np.array([0.0, 0.0])
        

        config = HopperConfig()
        

        self.m = 3.23
        self.g = 9.81
        

        self.Ix = 0.2
        self.Iy = 0.2
        self.Iz = 0.3
        self.Ixz = 0.0
        

        self.propeller_l1 = config.propeller_arm_length



        
        # 平衡控制增益（从config读取）
        self.Kp_roll = config.Kp_roll
        self.Kd_roll = config.Kd_roll
        self.Kp_pitch = config.Kp_pitch
        self.Kd_pitch = config.Kd_pitch
        
        
        # 螺旋桨PWM参数（从config读取）
        self.base_throttle = config.base_throttle
        self.pwm_min = config.pwm_min
        self.pwm_max = config.pwm_max
        self.pwm_filter_alpha = config.pwm_filter_alpha
        self.prev_pwms = [self.pwm_min, self.pwm_min, self.pwm_min]
        
        # 推力模型参数（从config读取）
        self.Omega_nom = config.Omega_nom
        self.k_thrust = config.k_thrust
        self.max_thrust_per_motor = config.max_thrust_per_motor
        
        # Phase相关PWM参数（从config读取）
        self.stance_pwm = config.stance_pwm
        self.flight_base_pwm = config.flight_base_pwm
        

        self.target_roll = config.target_roll
        self.target_pitch = config.target_pitch
        


        L = self.propeller_l1
        sqrt3_2 = np.sqrt(3) / 2











        self.A = np.array([
            [1, 1, 1],
            [L/2, -L, L/2],
            [L*sqrt3_2, 0, -L*sqrt3_2]
        ])
        self.A_inv = np.linalg.inv(self.A)


        self.propeller_armed = False
        
        # LCM Logger控制
        self.lcm_logger_process = None
        self.lcm_logging_active = False
        self.log_directory = os.path.expanduser("~/hopper_logs")
        os.makedirs(self.log_directory, exist_ok=True)
        

        self.robot_position = np.zeros(3)
        self.robot_velocity = np.zeros(3)
        self.foot_position = np.zeros(3)
        self.foot_velocity = np.zeros(3)
        

        self.running = True
        self.lock = threading.Lock()
        

        self.lc.subscribe("hopper_data_lcmt", self._handle_robot_data)
        self.lc.subscribe("hopper_imu_lcmt", self._handle_imu_data)
        self.lc.subscribe("gamepad_lcmt", self._handle_gamepad_data)
        
    def _handle_robot_data(self, channel, data):
        msg = hopper_data_lcmt.decode(data)
        with self.lock:
            self.robot_state['q'] = np.array(msg.q)
            self.robot_state['qd'] = np.array(msg.qd)
            self.robot_state['tau'] = np.array(msg.tauIq)
            
    def _handle_imu_data(self, channel, data):
        msg = hopper_imu_lcmt.decode(data)
        with self.lock:

            raw_quat = np.array(msg.quat)
            raw_gyro = np.array(msg.gyro)
            raw_acc = np.array(msg.acc)
            raw_rpy = np.array(msg.rpy)
            





            

            self.robot_state['imu_quat'] = raw_quat
            


            self.robot_state['imu_gyro'] = np.array([
                raw_gyro[0],
                raw_gyro[1],
                raw_gyro[2]
            ])
            


            self.robot_state['imu_acc'] = raw_acc
            

            self.robot_state['imu_rpy'] = raw_rpy
    
    def _handle_gamepad_data(self, channel, data):
        try:
            msg = gamepad_lcmt.decode(data)
            with self.lock:
                self.robot_state['gamepad_data'] = msg
                
                # 🎮 手柄右摇杆实时控制期望速度
                # rightStickAnalog[0] = X方向 (左右)
                # rightStickAnalog[1] = Y方向 (前后)
                if hasattr(msg, 'rightStickAnalog') and len(msg.rightStickAnalog) >= 2:
                    max_vel = 0.8  # 最大期望速度 (m/s)
                    dead_zone = 0.1  # 死区，避免漂移
                    
                    stick_x = msg.rightStickAnalog[0]
                    stick_y = msg.rightStickAnalog[1]
                    
                    # 应用死区
                    if abs(stick_x) < dead_zone:
                        stick_x = 0.0
                    if abs(stick_y) < dead_zone:
                        stick_y = 0.0
                    
                    if self.propeller_vector_mode:
                        # 矢量推进模式：右摇杆控制Propeller期望速度
                        self.desired_velocity_propeller[0] = stick_x * max_vel
                        self.desired_velocity_propeller[1] = stick_y * max_vel
                        # Raibert期望速度设为0（在run_controller中会被使用）
                    else:
                        # 传统模式：右摇杆控制Raibert期望速度
                        self.desired_velocity[0] = stick_x * max_vel
                        self.desired_velocity[1] = stick_y * max_vel
                        self.desired_velocity[2] = 1.0  # 模式标志

                static_vars = getattr(self._handle_gamepad_data, 'static_vars', {'last_a': False, 'last_y': False, 'last_b': False})
                
                # A键：切换矢量推进模式
                if msg.a and not static_vars['last_a']:
                    self.propeller_armed = not self.propeller_armed
                    self.propeller_vector_mode = self.propeller_armed  # 同步切换矢量推进模式
                static_vars['last_a'] = msg.a
                
                # Y键：启动LCM logger + 清空所有变量（重置状态） + 开始数据记录
                if msg.y and not static_vars['last_y']:
                    # 1. 清空所有累积状态（就像刚启动一样）
                    self.reset_com_position()
                    # 2. 启动LCM logger
                    if not self.lcm_logging_active:
                        self._start_lcm_logger()
                    # 3. 开始数据记录（用于绘图）
                    self.enable_data_logging = True
                    self.log_start_time = time.time()
                    print("📊 按Y键: 开始数据记录...")
                    # 清空之前的数据
                    for key in self.data_log.keys():
                        self.data_log[key].clear()
                static_vars['last_y'] = msg.y
                
                # B键：停止数据记录并立即绘图
                if msg.b and not static_vars['last_b']:
                    if self.enable_data_logging:
                        print("📊 按B键: 停止数据记录，开始绘图...")
                        self.enable_data_logging = False
                        # 立即绘制并保存数据
                        self.plot_data()
                        print("✅ 绘图完成！可以继续运行或按Ctrl+C退出")
                    else:
                        print("⚠️ 数据记录未启动，请先按Y键开始记录")
                static_vars['last_b'] = msg.b
                
                self._handle_gamepad_data.static_vars = static_vars
        except:
            pass
            
    
    def _start_lcm_logger(self):
        """启动LCM logger进程"""
        if self.lcm_logger_process is not None:
            return
        
        # 生成日志文件名
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_directory, f"hopper_{timestamp}.log")
        
        try:
            # 启动lcm-logger作为子进程
            self.lcm_logger_process = subprocess.Popen(
                ['lcm-logger', log_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.lcm_logging_active = True
            self.current_log_file = log_file
        except Exception as e:
            self.lcm_logger_process = None
    
    def _stop_lcm_logger(self):
        """停止LCM logger进程"""
        if self.lcm_logger_process is None:
            return
        
        try:
            # 发送SIGINT信号（Ctrl+C）
            self.lcm_logger_process.terminate()
            self.lcm_logger_process.wait(timeout=5)
            self.lcm_logging_active = False
            self.lcm_logger_process = None
        except Exception as e:
            pass
    
    def omega_to_pwm(self, omega_values):
        pwms = []
        for omega in omega_values:

            omega = max(0, min(omega, 2000))






            if omega <= 0:
                pwm = self.pwm_min
            else:

                pwm = 1000 + (omega / 2000) * 400


            pwm = max(self.pwm_min, min(pwm, self.pwm_max))

            if pwm > self.pwm_max:
                pwm = self.pwm_max

            pwms.append(pwm)
        return pwms
    


    def basic_pd_control(self, roll_rad, pitch_rad, p, q, dt):

        phi = roll_rad
        theta = pitch_rad
        

        e_phi = phi - self.target_roll
        e_theta = theta - self.target_pitch


        e_phi_dot = p
        e_theta_dot = q


        L_desired = (self.Kp_roll * e_phi + self.Kd_roll * e_phi_dot)
        M_desired = (self.Kp_pitch * e_theta + self.Kd_pitch * e_theta_dot)
        

        L_desired = np.clip(L_desired, -1000.0, 1000.0)
        M_desired = np.clip(M_desired, -1000.0, 1000.0)

        return L_desired, M_desired

    def control_allocation(self, L, M, total_thrust):


        control_vector = np.array([total_thrust, L, M])
        motor_thrusts = self.A_inv @ control_vector


        min_thrust = np.min(motor_thrusts)
        if min_thrust < 0:

            compensation = -min_thrust + 1.0
            motor_thrusts += compensation



        omega_values = []
        for thrust in motor_thrusts:
            if thrust <= 0:
                omega = 0
            else:

                pwm_equiv = 1000 + np.sqrt(thrust / self.k_thrust)

                omega = max(0, (pwm_equiv - 1000) * 2000 / 400)
            omega_values.append(omega)
        
        omega0, omega1, omega2 = omega_values

        return np.array([omega0, omega1, omega2])
    
    def send_motor_command(self, m0, m1, m2, armed):
        """
        发送6个螺旋桨PWM命令（测试阶段：只启用Motor 2和Motor 4）
        
        当前测试配置:
        - Motor 2 (Ch3, 索引2): 使用m1的PWM值
        - Motor 4 (Ch5, 索引4): 使用m2的PWM值
        - 其他通道: 关闭 (pwm_min=1000)
        
        通道索引: 0, 1, 2, 3, 4, 5
        对应电机: M0, M1, M2, M3, M4, M5
        """
        m0 = max(self.pwm_min, min(m0, self.pwm_max))
        m1 = max(self.pwm_min, min(m1, self.pwm_max))
        m2 = max(self.pwm_min, min(m2, self.pwm_max))
        
        msg = motor_pwm_lcmt()
        msg.timestamp = int(time.time() * 1000000)
        
        # 测试阶段：只启用Motor 2(Ch3)和Motor 4(Ch5)
        # Ch3(索引2)使用m1的值，Ch5(索引4)使用m2的值
        pwm_values_send = [
            float(self.pwm_min),  # Ch1 (索引0, M0): 关闭
            float(m1),  # Ch2 (索引1, M1): 关闭
            float(m2),            # Ch3 (索引2, M2): Motor 2 - 使用m1
            float(self.pwm_min),  # Ch4 (索引3, M3): 关闭
            float(m0),            # Ch5 (索引4, M4): Motor 4 - 使用m2
            float(self.pwm_min)   # Ch6 (索引5, M5): 关闭
        ]
        msg.pwm_values = pwm_values_send
        
        msg.control_mode = 1 if armed else 0
        self.lc.publish("motor_pwm_lcmt", msg.encode())
    
    

    
    def _estimate_state(self):
        with self.lock:
            joint_pos = self.robot_state['q'].copy()
            joint_vel = self.robot_state['qd'].copy()
            imu_quat = self.robot_state['imu_quat'].copy()
            imu_gyro = self.robot_state['imu_gyro'].copy()
            imu_acc = self.robot_state['imu_acc'].copy()
            imu_rpy = self.robot_state['imu_rpy'].copy()
            gamepad_data = self.robot_state['gamepad_data']
            

        raw_foot_pos, _ = self.controller.kinematics.forward_kinematics(joint_pos)
        true_foot_pos = raw_foot_pos
        
        # ========== 直接使用C++底层qd ==========
        # 计算foot_vel（使用C++底层EMA滤波的joint_vel，λ=0.4）
        J, foot_vel = self.controller.kinematics.inverse_jacobian(true_foot_pos, joint_vel, theta=None)

        controller_state = self.controller.state
        

        # 调用com_filter
        estimate_vel, estimate_pos = self.com_filter.process(
            imu_accel=imu_acc,
            imu_gyro=imu_gyro,
            imu_quat=imu_quat,
            foot_pos=true_foot_pos,
            foot_vel=foot_vel,  # 直接使用C++底层qd计算的foot_vel
            state=controller_state,
            accel_weight=0.0,
            orient_reset=0,
            arm_length=0.5,
            bound=np.array([45, 65]),
            vicon_flag=0,
            orient_lambda=0.03,
            vel_lambda=0.1,
            vicon_pos=None,
            vicon_orient=None,
            vicon_time=0
        )
        
        quat_orient = imu_quat
        ang_vel = imu_gyro
        
        self.robot_position = estimate_pos
        self.robot_velocity = estimate_vel
        self.foot_position = true_foot_pos
        self.foot_velocity = foot_vel  # 统一使用C++底层qd
            
        return {
            'robot_pos': self.robot_position,
            'robot_vel': self.robot_velocity,
            'foot_pos': self.foot_position,
            'foot_vel': self.foot_velocity,  # 统一使用C++底层qd
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,  # C++底层EMA滤波的qd
            'quat': quat_orient,
            'ang_vel': ang_vel,
            'imu_acc': imu_acc,
            'imu_rpy': imu_rpy,
            'gamepad_data': gamepad_data
        }

    def _send_command(self, torque):
        cmd_msg = hopper_cmd_lcmt()
        cmd_msg.tau_ff = torque.tolist()
        cmd_msg.q_des = [0.0, 0.0, 0.0]
        cmd_msg.qd_des = [0.0, 0.0, 0.0]
        cmd_msg.kp_joint = [0.0, 0.0, 0.0]
        cmd_msg.kd_joint = [0.0, 0.0, 0.0]
        self.lc.publish("hopper_cmd_lcmt", cmd_msg.encode())
    
    def run_controller(self):
        dt = 0.00075
        next_time = time.time()
        

        freq_samples = []
        last_freq_report = time.time()
        

        delay_samples = []
        cycle_count = 0
        
        last_debug_time = 0
        





        
        while self.running:
            try:
                current_time = time.time()
                cycle_start_time = current_time
                

                state = self._estimate_state()
                

                vel_real = state['robot_vel']
                robotPos_real = state['robot_pos']
                # ========== 根据模式设置Raibert期望速度 ==========
                if self.propeller_vector_mode:
                    # 矢量推进模式：Raibert期望速度为0（原地跳）
                    desiredPos_real = np.array([0.0, 0.0, 1.0])  # Z=1.0表示速度控制模式
                else:
                    # 传统模式：Raibert期望速度来自手柄
                    desiredPos_real = self.desired_velocity  # 🎮 使用手柄右摇杆控制的期望速度
                
                quat_real = state['quat']
                angVel_real = state['ang_vel']
                imu_acc_real = state['imu_acc']
                
                torque, balance_safety, debug, prop_curr, flipping = self.controller.virtual_spring_control(
                    Zoffset=0.0,
                    X=state['foot_pos'],
                    xdot=state['foot_vel'],  # 统一使用C++底层qd
                    joint=state['joint_pos'],
                    jointVel=state['joint_vel'],
                    vel=vel_real,
                    quat=quat_real,
                    angVel=angVel_real,
                    robotPos=robotPos_real,
                    desiredPos=desiredPos_real,
                    flip=0,
                    rpy=state['imu_rpy'],
                    gamepad_data=state.get('gamepad_data', None),
                    imu_acc=imu_acc_real,
                    foot_vel_filtered=state['foot_vel'],  # 统一使用C++底层qd
                    propeller_mode=self.propeller_vector_mode,  # 矢量推进模式标志
                    propeller_desired_vel=self.desired_velocity_propeller  # Propeller期望速度
                )
                

                self._send_command(torque) 
                
                # 记录数据（用于程序结束时绘图）- 只在按Y键后才记录
                if self.enable_data_logging and self.log_start_time is not None:
                    current_log_time = time.time() - self.log_start_time
                    self.data_log['time'].append(current_log_time)
                    self.data_log['hip_torque_x'].append(self.controller.hip_torque[0])
                    self.data_log['hip_torque_y'].append(self.controller.hip_torque[1])
                    self.data_log['rpy_roll'].append(state['imu_rpy'][0])
                    self.data_log['rpy_pitch'].append(state['imu_rpy'][1])
                    self.data_log['rpy_yaw'].append(state['imu_rpy'][2])
                    self.data_log['robot_vel_x'].append(vel_real[0])
                    self.data_log['robot_vel_y'].append(vel_real[1])
                    self.data_log['robot_vel_z'].append(vel_real[2])
                    self.data_log['robot_pos_x'].append(robotPos_real[0])
                    self.data_log['robot_pos_y'].append(robotPos_real[1])
                    # 记录关节速度dq
                    self.data_log['dq0'].append(state['joint_vel'][0])
                    self.data_log['dq1'].append(state['joint_vel'][1])
                    self.data_log['dq2'].append(state['joint_vel'][2])
                    # 记录世界坐标系下的foot_pos和target_pos（经过四元数转换）
                    if self.controller.state == 1:
                        # Flight phase: 使用世界坐标系
                        self.data_log['foot_pos_x'].append(self.controller.flight_x_world[0])
                        self.data_log['foot_pos_y'].append(self.controller.flight_x_world[1])
                        self.data_log['foot_pos_z'].append(self.controller.flight_x_world[2])
                        self.data_log['target_pos_x'].append(self.controller.flight_target_pos[0])
                        self.data_log['target_pos_y'].append(self.controller.flight_target_pos[1])
                        self.data_log['target_pos_z'].append(self.controller.flight_target_pos[2])
                    else:
                        # Stance phase: 使用世界坐标系
                        self.data_log['foot_pos_x'].append(self.controller.stance_x_world[0])
                        self.data_log['foot_pos_y'].append(self.controller.stance_x_world[1])
                        self.data_log['foot_pos_z'].append(self.controller.stance_x_world[2])
                        self.data_log['target_pos_x'].append(0.0)
                        self.data_log['target_pos_y'].append(0.0)
                        self.data_log['target_pos_z'].append(0.0)
                    self.data_log['state'].append(self.controller.state)
                    
                    # 记录姿态控制相关数据
                    self.data_log['desired_roll'].append(self.current_desired_rpy[0])
                    self.data_log['desired_pitch'].append(self.current_desired_rpy[1])
                    self.data_log['error_roll'].append(self.current_error_rpy[0])
                    self.data_log['error_pitch'].append(self.current_error_rpy[1])
                    self.data_log['propeller_L'].append(self.current_propeller_LM[0])
                    self.data_log['propeller_M'].append(self.current_propeller_LM[1])
                    self.data_log['desired_vel_x'].append(self.desired_velocity[0])
                    self.data_log['desired_vel_y'].append(self.desired_velocity[1])


                control_end_time = time.time()
                control_delay = (control_end_time - cycle_start_time) * 1000
                delay_samples.append(control_delay)
                cycle_count += 1
                



                



                if self.propeller_armed:
                    # 🔧 根据 phase 决定控制策略
                    current_state = self.controller.state
                    
                    if current_state == 2:
                        # ===== Stance Phase: 所有PWM输出固定值（从config读取） =====
                        m0 = self.stance_pwm
                        m1 = self.stance_pwm
                        m2 = self.stance_pwm
                        self.send_motor_command(m0, m1, m2, True)
                        
                        # Stance phase数据记录（姿态控制不活跃）
                        self.current_desired_rpy = np.array([0.0, 0.0, 0.0])
                        self.current_error_rpy = np.array([0.0, 0.0, 0.0])
                        self.current_propeller_LM = np.array([0.0, 0.0])
                        
                    elif current_state == 1:
                        # ===== Flight Phase: 平衡控制（期望姿态根据速度决定） =====
                        imu_rpy = state['imu_rpy']
                        imu_gyro = state['ang_vel']
                        

                        roll_rad = imu_rpy[0]
                        pitch_rad = imu_rpy[1]
                        

                        p, q, r = imu_gyro
                        
                        # ========== 计算期望姿态（基于期望速度） ==========
                        # 如果有期望速度，计算对应的倾斜角度
                        # 否则保持水平（期望姿态为0）
                        if self.propeller_vector_mode and np.linalg.norm(self.desired_velocity_propeller[:2]) > 0.01:
                            # 矢量推进模式：使用Propeller期望速度
                            desired_rpy = velocity_to_attitude_command(
                                self.desired_velocity_propeller,
                                self.config.max_propeller_vel,
                                self.config.max_tilt_angle
                            )
                        elif np.linalg.norm(self.desired_velocity[:2]) > 0.01:
                            # 传统模式：使用Raibert期望速度
                            desired_rpy = velocity_to_attitude_command(
                                self.desired_velocity,
                                self.config.max_propeller_vel,
                                self.config.max_tilt_angle
                            )
                        else:
                            # 无期望速度：保持水平
                            desired_rpy = np.array([0.0, 0.0, 0.0])
                        
                        # ========== Flight Phase平衡控制（统一使用PD控制） ==========
                        # 计算姿态误差（期望 - 当前）
                        e_roll = desired_rpy[0] - roll_rad
                        e_pitch = desired_rpy[1] - pitch_rad
                        
                        # PD控制计算力矩
                        L_desired = self.Kp_roll * e_roll + self.Kd_roll * (-p)
                        M_desired = self.Kp_pitch * e_pitch + self.Kd_pitch * (-q)
                        
                        # 限制力矩
                        L_desired = np.clip(L_desired, -1000.0, 1000.0)
                        M_desired = np.clip(M_desired, -1000.0, 1000.0)
                        
                        # 保存用于数据记录
                        self.current_desired_rpy = desired_rpy.copy()
                        self.current_error_rpy = np.array([e_roll, e_pitch, 0])
                        self.current_propeller_LM = np.array([L_desired, M_desired])
                        
                        # 计算基础推力
                        current_pwm = self.flight_base_pwm
                        base_pwm_delta = current_pwm - 1000
                        base_thrust_per_motor = self.k_thrust * base_pwm_delta**2
                        total_thrust = base_thrust_per_motor * 3


                        omega_values = self.control_allocation(L_desired, M_desired, total_thrust)
                        

                        m0, m1, m2 = self.omega_to_pwm(omega_values)
                        

                        smoothed_pwms = []
                        for i, pwm in enumerate([m0, m1, m2]):
                            smoothed = self.pwm_filter_alpha * pwm + (1 - self.pwm_filter_alpha) * self.prev_pwms[i]
                            smoothed_pwms.append(smoothed)
                        self.prev_pwms = smoothed_pwms
                        

                        m0, m1, m2 = [max(self.pwm_min, min(pwm, self.pwm_max)) for pwm in smoothed_pwms]

                        # 🔧 测试：只启用Motor 2(Ch3, 索引2)和Motor 4(Ch5, 索引4)
                        # m1 → Ch3 (索引2, Motor 2)
                        # m2 → Ch5 (索引4, Motor 4)
                        
                        self.send_motor_command(m0, m1, m2, True)
                    else:
                        # ===== 其他状态（错误、初始化等）: 关闭螺旋桨 =====
                        self.send_motor_command(self.pwm_min, self.pwm_min, self.pwm_min, False)
                else:
                    # ===== 未ARM: 关闭螺旋桨 =====
                    self.send_motor_command(self.pwm_min, self.pwm_min, self.pwm_min, False)
                



















                pass
                

                actual_dt = current_time - (next_time - dt)
                if actual_dt > 0:
                    freq = 1.0 / actual_dt
                    freq_samples.append(freq)
                

                if current_time - last_freq_report >= 5.0:
                    if len(freq_samples) > 0:
                        avg_freq = np.mean(freq_samples)
                        min_freq = np.min(freq_samples)
                        max_freq = np.max(freq_samples)

                        freq_samples = []
                    
                    if len(delay_samples) > 0:
                        avg_delay = np.mean(delay_samples)
                        min_delay = np.min(delay_samples)
                        max_delay = np.max(delay_samples)

                        delay_samples = []
                    
                    last_freq_report = current_time
                

                next_time += dt
                sleep_time = next_time - time.time()
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:

                    next_time = time.time()
                    
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:

                pass
        


        for _ in range(10):
            self.send_motor_command(self.pwm_min, self.pwm_min, self.pwm_min, False)
            time.sleep(0.01)
        
    def run_lcm_handler(self):
        while self.running:
            try:
                self.lc.handle()
            except Exception as e:
                time.sleep(0.01)
                
    def set_desired_position(self, pos):
        
        self.desired_position = np.array(pos)
        
    def set_flip_command(self, flip):
        self.flip_command = flip
        
    def reset_com_position(self):
        """重置控制器累积状态（不清空传感器数据）"""
        # 重置com_filter的累积状态
        if hasattr(self, 'com_filter'):
            # 位置和速度相关
            self.com_filter.last_pos = np.zeros(3)
            self.com_filter.flight_vel = np.zeros(3)
            self.com_filter.last_orient = np.array([1.0, 0.0, 0.0, 0.0])
            
            # 角度和状态相关
            self.com_filter.pAng = 0.0
            self.com_filter.state_count = 0
            self.com_filter.stance_flag = 0
            self.com_filter.last_state = 1
            
            # 足端数据历史缓存
            self.com_filter.last_foot_vel = np.zeros((10, 3))
            self.com_filter.last_foot_pos = np.zeros((10, 3))
            
            # 滤波状态
            self.com_filter.filtered_accel = np.zeros(3)
            self.com_filter.ema_accel = np.zeros(3)
            self.com_filter.ema_gyro = np.zeros(3)
            
            # Vicon相关状态
            self.com_filter.last_vicon_time = 0.0
            self.com_filter.last_vicon_quat = np.array([1.0, 0.0, 0.0, 0.0])
            self.com_filter.last_vicon_pos = np.zeros(3)
            self.com_filter.last_correcting_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # 重置控制器计算的状态（不是传感器数据）
        self.robot_position = np.zeros(3)
        self.robot_velocity = np.zeros(3)
        self.foot_position = np.zeros(3)
        self.foot_velocity = np.zeros(3)
        
        # 重置控制器状态
        if hasattr(self, 'controller'):
            self.controller.state = 1  # 重置为flight phase
            self.controller.state_safety = 0
            
        # 重置PWM滤波器
        self.prev_pwms = [self.pwm_min, self.pwm_min, self.pwm_min]
        
        # 重置Sigmoid增益控制的历史状态
        if hasattr(self.controller, 'last_gains'):
            self.controller.last_gains = {'kpp': self.controller.Kpp, 'kpd': self.controller.Kpd}
        
        # 重置SimulinkVelocityFilter的累积状态
        if hasattr(self.controller, 'kinematics') and hasattr(self.controller.kinematics, 'simulink_filter'):
            if self.controller.kinematics.simulink_filter is not None:
                self.controller.kinematics.simulink_filter.reset()
        
        # 注意：不清空 robot_state（包含IMU、关节数据等传感器原始数据）
        
    def plot_data(self):
        """程序结束时绘制数据图表"""
        if len(self.data_log['time']) < 10:
            return
        
        time_data = np.array(self.data_log['time'])
        states = np.array(self.data_log['state'])
        
        # 创建3x2的子图（一个窗口）
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Hopper Controller Data', fontsize=16)
        
        # 为每个子图添加背景色（区分Flight/Stance phase）
        def add_phase_background(ax, time_data, states):
            """添加phase背景色：Flight=浅蓝色，Stance=浅绿色"""
            # 找到phase切换点
            phase_changes = np.where(np.diff(states) != 0)[0]
            
            # 添加第一段
            start_idx = 0
            for change_idx in phase_changes:
                end_idx = change_idx + 1
                phase = states[start_idx]
                color = 'lightblue' if phase == 1 else 'lightgreen'
                ax.axvspan(time_data[start_idx], time_data[end_idx], 
                          alpha=0.2, color=color, zorder=0)
                start_idx = end_idx
            
            # 添加最后一段
            if start_idx < len(states):
                phase = states[start_idx]
                color = 'lightblue' if phase == 1 else 'lightgreen'
                ax.axvspan(time_data[start_idx], time_data[-1], 
                          alpha=0.2, color=color, zorder=0)
        
        # 1. Hip Torque X vs Roll
        ax = axes[0, 0]
        add_phase_background(ax, time_data, states)
        ax1 = ax
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(time_data, self.data_log['hip_torque_x'], 
                        'r-', label='Hip Torque X', linewidth=1.5)
        line2 = ax2.plot(time_data, np.degrees(self.data_log['rpy_roll']), 
                        'r--', label='Roll', linewidth=1.5, alpha=0.7)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Hip Torque X (N·m)', color='r')
        ax2.set_ylabel('Roll (deg)', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax1.set_title('Hip Torque X vs Roll')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Hip Torque Y vs Pitch
        ax = axes[0, 1]
        add_phase_background(ax, time_data, states)
        ax1 = ax
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(time_data, self.data_log['hip_torque_y'], 
                        'y-', label='Hip Torque Y', linewidth=1.5)
        line2 = ax2.plot(time_data, np.degrees(self.data_log['rpy_pitch']), 
                        'y--', label='Pitch', linewidth=1.5, alpha=0.7)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Hip Torque Y (N·m)', color='y')
        ax2.set_ylabel('Pitch (deg)', color='y')
        ax1.tick_params(axis='y', labelcolor='y')
        ax2.tick_params(axis='y', labelcolor='y')
        ax1.set_title('Hip Torque Y vs Pitch')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 3. Robot Velocity
        ax = axes[1, 0]
        add_phase_background(ax, time_data, states)
        ax.plot(time_data, self.data_log['robot_vel_x'], 'r-', label='Vel X', linewidth=1.5)
        ax.plot(time_data, self.data_log['robot_vel_y'], 'y-', label='Vel Y', linewidth=1.5)
        ax.plot(time_data, self.data_log['robot_vel_z'], 'b-', label='Vel Z', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Robot Velocity')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 4. Foot Position vs Target Position
        ax = axes[1, 1]
        add_phase_background(ax, time_data, states)
        ax.plot(time_data, self.data_log['foot_pos_x'], 'r-', label='Foot X', linewidth=1.5)
        ax.plot(time_data, self.data_log['foot_pos_y'], 'y-', label='Foot Y', linewidth=1.5)
        ax.plot(time_data, self.data_log['foot_pos_z'], 'b-', label='Foot Z', linewidth=1.5)
        ax.plot(time_data, self.data_log['target_pos_x'], 'r--', label='Target X', linewidth=1.5, alpha=0.7)
        ax.plot(time_data, self.data_log['target_pos_y'], 'y--', label='Target Y', linewidth=1.5, alpha=0.7)
        ax.plot(time_data, self.data_log['target_pos_z'], 'b--', label='Target Z', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        ax.set_title('Foot Position vs Target Position (World Frame)')
        ax.legend(loc='upper left', ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 5. Robot X vs Y Position (轨迹图)
        ax = axes[2, 0]
        # 根据phase着色轨迹
        robot_pos_x = np.array(self.data_log['robot_pos_x'])
        robot_pos_y = np.array(self.data_log['robot_pos_y'])
        
        # 分段绘制（根据phase）
        for i in range(len(robot_pos_x) - 1):
            phase = states[i]
            color = 'blue' if phase == 1 else 'green'
            alpha = 0.6 if phase == 1 else 0.8
            ax.plot(robot_pos_x[i:i+2], robot_pos_y[i:i+2], 
                   color=color, alpha=alpha, linewidth=1.5)
        
        # 标记起点和终点
        ax.plot(robot_pos_x[0], robot_pos_y[0], 'go', markersize=10, label='Start', zorder=5)
        ax.plot(robot_pos_x[-1], robot_pos_y[-1], 'ro', markersize=10, label='End', zorder=5)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Robot Trajectory (X vs Y)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')  # 保持X和Y轴比例一致
        
        # 6. Joint Velocities dq0, dq1, dq2 (合并在一个图)
        ax = axes[2, 1]
        add_phase_background(ax, time_data, states)
        ax.plot(time_data, self.data_log['dq0'], 'r-', label='dq0', linewidth=1.5)
        ax.plot(time_data, self.data_log['dq1'], 'y-', label='dq1', linewidth=1.5)
        ax.plot(time_data, self.data_log['dq2'], 'b-', label='dq2', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Joint Velocity (rad/s)')
        ax.set_title('Joint Velocities (dq0, dq1, dq2)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 添加图例说明
        fig.text(0.5, 0.02, 'Background: Light Blue = Flight Phase, Light Green = Stance Phase', 
                ha='center', fontsize=12, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        
        # 保存图片和数据
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename_png = f"hopper_data_{timestamp}.png"
        filename_npz = f"hopper_data_{timestamp}.npz"
        
        # 保存PNG图片
        plt.savefig(filename_png, dpi=150, bbox_inches='tight')
        print(f"✅ 图片已保存: {filename_png}")
        
        # 保存NPZ数据（用于analyze_pid.py分析）
        np.savez(filename_npz, **self.data_log)
        print(f"✅ 数据已保存: {filename_npz}")
        print(f"📊 分析数据: python3 analyze_pid.py {filename_npz}")
        
        # 显示图表
        plt.show()
        
    def start(self):
        try:
            lcm_thread = threading.Thread(target=self.run_lcm_handler)
            lcm_thread.daemon = True
            lcm_thread.start()
            time.sleep(1.0)
            
            self.run_controller()
        except KeyboardInterrupt:
            self.running = False
        finally:
            self.running = False
            
            # 停止LCM logger
            if self.lcm_logging_active:
                self._stop_lcm_logger()
            
            # 确保电机关闭
            for _ in range(5):
                self.send_motor_command(self.pwm_min, self.pwm_min, self.pwm_min, False)
                time.sleep(0.01)
            
            # 绘制数据图表
            self.plot_data()
            
def main():
    controller = HopperLCMController()
    controller.start()

if __name__ == "__main__":
    main()
