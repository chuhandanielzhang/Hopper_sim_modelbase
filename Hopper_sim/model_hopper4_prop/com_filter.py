

import numpy as np
from scipy.spatial.transform import Rotation as R

class ComplementaryFilter:
    
    def __init__(self):

        self.pAng = 0.0
        self.state_count = 0
        self.last_state = 1
        self.stance_flag = 0
        self.last_orient = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_foot_vel = np.zeros((10, 3))
        self.last_foot_pos = np.zeros((10, 3))
        self.flight_vel = np.zeros(3)
        self.last_pos = np.zeros(3)
        
        self.filtered_accel = np.zeros(3)
        

        self.last_vicon_time = 0.0
        self.last_vicon_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_vicon_pos = np.zeros(3)
        self.last_correcting_quat = np.array([1.0, 0.0, 0.0, 0.0])
        

        # 与MATLAB版本一致的坐标转换矩阵
        self.imu2vicon = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        self.robot2vicon = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        

        self.time_step = 0.00075
        
        self.low_pass_tau = 0.1
        
        self.accel_bias = np.array([-0.0175, -0.0067, -0.0122])
        self.accel_gain_matrix = np.array([
            [1.0061, -0.0076, 0.0018],
            [0.0045, 0.9996, 0.0047],
            [-0.0038, 0.0031, 1.0009]
        ])
        
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.gyro_gain_matrix = np.eye(3)
        
        # 添加必要的属性
        self.bound = np.array([90*0.5, 130*0.5])
        self.accel_weight = 0.01
        self.orient_lambda = 0.03
        self.vel_lambda = 0.1
        
        # Simulink风格的EMA滤波参数
        self.forgetting_factor = 0.4  # 与SimulinkVelocityFilter一致
        
        # EMA状态变量
        self.ema_accel = np.zeros(3)
        self.ema_gyro = np.zeros(3)
        
    def low_pass_filter(self, input_signal, prev_output, tau, dt):
        alpha = dt / (tau + dt)
        return alpha * input_signal + (1 - alpha) * prev_output
    
    def ema_filter(self, input_signal, prev_output, forgetting_factor):
        """Simulink风格的指数加权移动平均滤波"""
        return forgetting_factor * prev_output + (1 - forgetting_factor) * input_signal
    
    def calibrate_accelerometer(self, raw_accel):
        accel_bias_corrected = raw_accel + self.accel_bias
        accel_calibrated = self.accel_gain_matrix @ accel_bias_corrected
        return accel_calibrated
    
    def calibrate_gyroscope(self, raw_gyro):
        gyro_bias_corrected = raw_gyro + self.gyro_bias
        gyro_calibrated = self.gyro_gain_matrix @ gyro_bias_corrected
        return gyro_calibrated
        
    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def quaternion_conjugate(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def quaternion_normalize(self, q):
        norm = np.linalg.norm(q)
        if norm > 0:
            return q / norm
        return q
    
    def quaternion_from_rotvec(self, rotvec):
        angle = np.linalg.norm(rotvec)
        if angle < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        axis = rotvec / angle
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = np.sin(half_angle) * axis
        
        q = np.array([w, xyz[0], xyz[1], xyz[2]])
        if q[0] < 0:
            q = -q
        return q
    
    def quat2rotm(self, q):
        w, x, y, z = q
        
        rotm = np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
        
        return rotm
    
    def process(self, imu_accel, imu_gyro, imu_quat, foot_pos, foot_vel, state, 
                accel_weight=None, orient_reset=0, arm_length=0.5, 
                bound=None, vicon_flag=0, orient_lambda=None, vel_lambda=None,
                vicon_pos=None, vicon_orient=None, vicon_time=0):
        

        if accel_weight is not None:
            self.accel_weight = accel_weight
        if bound is not None:
            self.bound = bound
        if orient_lambda is not None:
            self.orient_lambda = orient_lambda
        if vel_lambda is not None:
            self.vel_lambda = vel_lambda
            

        # 确保所有输入数据为numpy数组
        imu_accel = np.array(imu_accel)
        imu_gyro = np.array(imu_gyro)
        imu_quat = np.array(imu_quat)
        foot_pos = np.array(foot_pos)
        foot_vel = np.array(foot_vel)
        
        # 使用原始数据，不进行硬编码
        
        # 使用Simulink风格的EMA滤波处理IMU数据
        if np.isnan(imu_accel).any():
            self.last_pos = np.array([100, 100, 100])
            imu_accel = np.nan_to_num(imu_accel)
        if np.isnan(imu_gyro).any():
            self.last_pos = np.array([100, 100, 100])
            imu_gyro = np.nan_to_num(imu_gyro)
        
        # 直接使用LCM原始IMU数据，不进行EMA滤波
        gyro_inc = self.imu2vicon @ imu_gyro * self.time_step
        gyro_pAngInc = gyro_inc[0]
        
        # 直接使用LCM原始加速度数据
        imu_accel = self.imu2vicon @ imu_accel
        imu_accel_global = imu_accel.copy()
        

        accel_norm = np.linalg.norm(imu_accel)
        if accel_norm > 0:
            imu_accel_normalized = imu_accel / accel_norm
        else:
            imu_accel_normalized = np.array([0, 0, 1])
        

        accel_pAng = np.arctan(imu_accel_normalized[1] / imu_accel_normalized[2])
        self.pAng = self.pAng + gyro_pAngInc
        

        # 按照MATLAB版本逻辑进行四元数更新
        quat_gyro_inc = self.quaternion_from_rotvec(gyro_inc)
        if quat_gyro_inc[0] < 0:
            quat_gyro_inc = -quat_gyro_inc
        quat_gyro_inc = self.quaternion_normalize(quat_gyro_inc)
        
        temp = self.quaternion_multiply(self.last_orient, quat_gyro_inc)
        if temp[0] < 0:
            temp = -temp
        self.last_orient = self.quaternion_normalize(temp)
        

        if state == 2 and self.last_state == 1:
            self.stance_flag = 1
            self.state_count = 0
        elif state == 1 and self.last_state == 2:
            self.stance_flag = 0
            self.state_count = 0
        

        if self.stance_flag == 1:
            if self.state_count > self.bound[0] and self.state_count < self.bound[1]:
                self.pAng = (self.pAng + gyro_pAngInc) * (1 - self.accel_weight) + accel_pAng * self.accel_weight
                
                correcting_accel = np.cross([0, 0, 1], -imu_accel_normalized)
                correcting_accel_ang = self.accel_weight * np.arcsin(np.linalg.norm(correcting_accel))
                temp = np.sin(correcting_accel_ang / 2) * correcting_accel
                correcting_quat = np.array([np.cos(correcting_accel_ang / 2), temp[0], temp[1], temp[2]])
                if correcting_quat[0] < 0:
                    correcting_quat = -correcting_quat
                correcting_quat = self.quaternion_normalize(correcting_quat)
                
                temp = self.quaternion_multiply(self.last_orient, correcting_quat)
                if temp[0] < 0:
                    temp = -temp
                self.last_orient = self.quaternion_normalize(temp)
            
            self.state_count += 1
        

        hover_flag = 0
        if state == 99 and orient_reset == 0:
            if self.state_count > 2*500 and abs(np.linalg.norm(imu_accel_global) - 1) < 0.1:
                hover_flag = 1
                self.pAng = (self.pAng + gyro_pAngInc) * (1 - self.accel_weight) + accel_pAng * self.accel_weight
                
                correcting_accel = np.cross([0, 0, 1], -imu_accel_normalized)
                correcting_accel_ang = self.accel_weight * np.arcsin(np.linalg.norm(correcting_accel))
                temp = np.sin(correcting_accel_ang / 2) * correcting_accel
                correcting_quat = np.array([np.cos(correcting_accel_ang / 2), temp[0], temp[1], temp[2]])
                if correcting_quat[0] < 0:
                    correcting_quat = -correcting_quat
                correcting_quat = self.quaternion_normalize(correcting_quat)
                
                temp = self.quaternion_multiply(self.last_orient, correcting_quat)
                if temp[0] < 0:
                    temp = -temp
                self.last_orient = self.quaternion_normalize(temp)
            
            self.state_count += 1
        

        # 直接使用LCM原始IMU数据计算角速度
        robot_rotm = self.quat2rotm(imu_quat)
        ang_vel = robot_rotm @ self.imu2vicon @ (imu_gyro*0.1)
        


        # 按照MATLAB版本逻辑处理足端数据
        for i in range(len(self.last_foot_vel)):
            if i < len(self.last_foot_vel) - 1:
                self.last_foot_vel[i] = self.last_foot_vel[i + 1]
                self.last_foot_pos[i] = self.last_foot_pos[i + 1]
            elif i == len(self.last_foot_vel) - 1:
                self.last_foot_vel[i] = foot_vel
                self.last_foot_pos[i] = foot_pos
        
        avg_foot_vel = np.mean(self.last_foot_vel, axis=0)
        avg_foot_pos = np.mean(self.last_foot_pos, axis=0)
        
        # Debug: 检查滑动窗口是否填满
        non_zero_count = np.count_nonzero(np.sum(np.abs(self.last_foot_vel), axis=1))
        

        if self.last_state == 2 and state == 1:
            # 检查滑动窗口是否足够可靠（方案1：修复第一跳vel异常）
            if non_zero_count < 10:
                # 窗口未填满，使用当前值而不是平均值，避免初始化异常影响
                use_foot_vel = foot_vel
                use_foot_pos = foot_pos
            else:
                # 窗口已填满，使用平均值（正常情况）
                use_foot_vel = avg_foot_vel
                use_foot_pos = avg_foot_pos
            
            # 按照MATLAB版本逻辑计算飞行速度
            self.flight_vel = (robot_rotm @ (self.robot2vicon @ (-use_foot_vel)) + 
                              robot_rotm @ np.cross(ang_vel, self.robot2vicon @ (-use_foot_pos)))


        if self.last_state == 1:
            estimate_vel = self.flight_vel
        elif self.last_state == 2:
            # 按照MATLAB版本逻辑计算支撑速度
            estimate_vel = (robot_rotm @ (self.robot2vicon @ (-foot_vel)) + 
                           robot_rotm @ np.cross(ang_vel, self.robot2vicon @ (-foot_pos)))
        else:
            estimate_vel = np.zeros(3)
        

        estimate_vel[2] = 0
        

        if orient_reset == 1:
            self.pAng = accel_pAng
            rotv_accel = np.cross(-imu_accel_normalized, [0, 0, 1])
            rotv_accel_ang = np.arcsin(np.linalg.norm(rotv_accel) / np.linalg.norm(imu_accel))
            temp = np.sin(rotv_accel_ang / 2) * rotv_accel
            quat_accel = np.array([np.cos(rotv_accel_ang / 2), temp[0], temp[1], temp[2]])
            if quat_accel[0] < 0:
                quat_accel = -quat_accel
            self.last_orient = self.quaternion_normalize(quat_accel)
            
            self.flight_vel = np.zeros(3)
            estimate_vel = np.zeros(3)
            self.last_pos = np.zeros(3)
            self.state_count = 0
        

        estimate_pos = self.last_pos + estimate_vel * self.time_step
        
        # 按照MATLAB版本逻辑进行vicon位置融合
        if abs(vicon_time - self.last_vicon_time) >= 0.01 and vicon_flag == 1:
            pos = vicon_pos + self.quat2rotm(self.last_vicon_quat) @ np.array([-0.01388, 0.02222, -0.1656])
            estimate_pos = estimate_pos * (1 - self.vel_lambda) + self.last_vicon_pos * self.vel_lambda
            self.last_vicon_time = vicon_time
            self.last_vicon_pos = pos
        
        mapped_vicon_pos = self.last_vicon_pos
        


        debug = np.concatenate([imu_accel_normalized, [hover_flag]])
        

        self.last_state = state
        self.last_pos = estimate_pos
        
        # 只返回vel和pos，不返回滤波相关的IMU数据
        return (estimate_vel, estimate_pos)
