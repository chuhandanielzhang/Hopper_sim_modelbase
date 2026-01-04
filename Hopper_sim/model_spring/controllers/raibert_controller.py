"""
Raibert + 虚拟弹簧控制器（MuJoCo 版本）

完全复用 Hopper4.py 的 VirtualSpringController 逻辑
只调整参数以适应 MuJoCo 串联腿模型

关键差异：
1. MuJoCo 使用简化串联腿（Roll-Pitch-Shift）
2. 3-RSR 使用并联机构（三个旋转关节）
3. 控制逻辑完全相同，只是运动学不同

坐标系约定（与 Hopper4.py 一致）：
- robot2vicon = [[1,0,0], [0,1,0], [0,0,-1]]
- 足端在机体下方时 Z 为正
"""

import numpy as np
from scipy.spatial.transform import Rotation


class RaibertController:
    """
    Raibert + 虚拟弹簧控制器
    
    完全按照 Hopper4.py VirtualSpringController 实现
    参数针对 MuJoCo 串联腿调整
    """
    
    def __init__(self):
        # ========== 虚拟弹簧参数（Task1 优化后）==========
        self.l0 = 0.464      # 腿自然长度
        # Higher k for stable hopping (original Hopper4 used 1000)
        self.k = 2500        # spring stiffness (N/m)
        self.b = 25          # damping (N/(m/s))
        self.m = 3.23        # 机器人质量
        self.h = 0.12        # target hop height (m)
        
        # MuJoCo 串联腿在 shift=0 时（足端 contact site），|X| ≈ 0.5953m。
        # 控制里还会做 x = X + [0,0,0.03] 的 3cm 偏移，所以用于能量/弹簧计算的 l_serial ≈ 0.6253m。
        # 为了让串联腿等效到真实腿 l0=0.464，需要减去固定偏移（0.6253 - 0.464 ≈ 0.1613）
        self.leg_offset = 0.1613  # serial → physical 长度偏移（包含 3cm 偏移）
        
        # ========== Raibert 足端放置增益（原始 Hopper4 值）==========
        self.Kv = 0.10     # 速度前馈
        self.Kr = 0.09     # 速度校正
        self.Khp = 50        # 飞行阶段足端位置增益
        self.Khd = 1.0       # flight foot velocity gain (damping)
        
        # ========== 能量环增益 ==========
        # Task1 优化：Kp=5.0 提供更好的速度响应
        self.Kp = 5.0      # energy injection gain
        
        # ========== Hip torque 姿态控制增益（优化后）==========
        # 姿态优化：Kpp=100, Kpd=10 → 前进时pitch≈14°
        self.Kpp_x = 100     # 姿态比例增益
        self.Kpp_y = 100     # 姿态比例增益
        self.Kpd_x = 10.0    # 姿态微分增益（Task1用16.0）
        self.Kpd_y = 10.0    # 姿态微分增益（Task1用16.0）
        
        # ========== 限位（优化后）==========
        self.stepperLim = 0.1      # 足端位置限制
        self.hipTorqueLim = 20     # Hip torque 限制（优化：15→20）
        self.posVelLim = 0.8        # 速度限制
        self.upLim = 1.38           # 关节上限
        self.lowLim = -1.04         # 关节下限
        # 力矩限制
        self.max_torque = 30        # Roll/Pitch 关节力矩限制 (增加以提供更强姿态控制)
        self.max_shift_torque = 600 # Shift 关节力矩限制 (increase for stronger push)
        
        # ========== 相位切换阈值 ==========
        # Phase switch thresholds (serial MuJoCo plant):
        # The previous Task1 values were tuned for a different length mapping and were too conservative here,
        # preventing touchdown detection -> no stance spring support -> the robot collapses.
        self.flight_to_stance_threshold = 0.02  # touchdown: l < l0 - thr
        self.stance_to_flight_threshold = 0.01  # liftoff:  l > l0 + thr
        self.Kpos = 0.0
        
        # ========== 相位切换去抖动 ==========
        # 最小保持时间（防止在阈值附近快速切换）
        self.min_phase_duration = 10  # ~20ms @ 500Hz (LCM demos)
        self.phase_duration_count = 0  # 当前相位持续计数
        
        # ========== 状态机 ==========
        self.state = 1  # 1=flight, 2=stance
        self.state_safety = 0
        
        # ========== 坐标变换矩阵（与 Hopper4.py 一致）==========
        self.robot2vicon = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, -1]])
        
        # ========== 调试变量 ==========
        self.footForce = np.zeros(3)
        self.sideForce = np.zeros(3)
        self.springForce_vec = np.zeros(3)
        self.hip_torque = np.zeros(3)
        self.robotTilt_debug = np.zeros(3)
    
    def reset(self):
        """重置控制器状态"""
        self.state = 1  # 1=flight, 2=stance
        self.state_safety = 0
        self.phase_duration_count = 0  # 重置相位持续计数
        self.springForce_scalar_copy = 0
        self.energy_compensation_copy = 0
        self.leg_velocity_copy = 0
        
        # 调试变量
        self.footForce = np.zeros(3)
        self.sideForce = np.zeros(3)
        self.springForce_vec = np.zeros(3)
        self.hip_torque = np.zeros(3)
        self.flight_target_pos = np.zeros(3)  # 初始化目标落点
        self.flight_target_pos_raw = np.zeros(3)  # 未旋转的目标落点（world）
        
    def update(self, state, desired_vel, dt=0.002):
        """
        更新控制器（适配 MuJoCo 接口）
        """
        # 适配 Hopper4.py 的接口参数
        torque, info_debug = self.compute_torque(
            X=state['foot_pos'],
            xdot=state['foot_vel'],
            joint=state['joint_pos'],
            jointVel=state['joint_vel'],
            vel=state['body_vel'],
            quat=state['body_quat'],
            angVel=state['body_ang_vel'],
            robotPos=state['body_pos'],
            desiredPos=np.array([desired_vel[0], desired_vel[1], 1.0]), # Z=1.0 表示速度控制模式
            rpy=state['body_rpy']
        )
        
        # 构建 info 字典
        info = {
            'torque': torque,
            'state': self.state,
            'debug': info_debug
        }
        return info

    def _serial_jacobian(self, roll, pitch, shift):
        """
        串联腿 Jacobian（与 Hopper_rl_t-master/hopper.py jacobian_tensor 完全一致）
        
        参数定义（与 Hopper_rl_t 一致）：
        - th1 = roll
        - th2 = pitch  
        - d = shift - CONST_OFFSET (CONST_OFFSET = 0.4)
        - n1 = [1, 0, 0] (固定的 X 轴单位向量)
        
        Jacobian 列向量：
        - col1 = cross(n1, r) 
        - col2 = cross(n2, r)
        - col3 = n3
        
        其中：
        - n2 = [0, cos(th1), sin(th1)]
        - n3 = [sin(th2), -sin(th1)*cos(th2), cos(th1)*cos(th2)]
        - r = d * n3
        """
        CONST_OFFSET = 0.4
        d = shift - CONST_OFFSET  # 与 Hopper_rl_t 一致：d = dof_pos[2] - CONST_OFFSET
        
        c1, s1 = np.cos(roll), np.sin(roll)
        c2, s2 = np.cos(pitch), np.sin(pitch)
        
        # n1 = [1, 0, 0]
        # n2 = [0, c1, s1]
        # n3 = [s2, -s1*c2, c1*c2]
        # r = d * n3 = [d*s2, -d*s1*c2, d*c1*c2]
        
        # col1 = cross(n1, r) = cross([1,0,0], [d*s2, -d*s1*c2, d*c1*c2])
        #      = [0*d*c1*c2 - 0*(-d*s1*c2), 0*d*s2 - 1*d*c1*c2, 1*(-d*s1*c2) - 0*d*s2]
        #      = [0, -d*c1*c2, -d*s1*c2]
        # 注意：cross([1,0,0], [a,b,c]) = [0*c - 0*b, 0*a - 1*c, 1*b - 0*a] = [0, -c, b]
        col1 = np.array([0, -d * c1 * c2, -d * s1 * c2])
        
        # col2 = cross(n2, r) = cross([0, c1, s1], [d*s2, -d*s1*c2, d*c1*c2])
        #      = [c1*d*c1*c2 - s1*(-d*s1*c2), s1*d*s2 - 0*d*c1*c2, 0*(-d*s1*c2) - c1*d*s2]
        #      = [d*c2*(c1^2 + s1^2), d*s1*s2, -d*c1*s2]
        #      = [d*c2, d*s1*s2, -d*c1*s2]
        col2 = np.array([d * c2, d * s1 * s2, -d * c1 * s2])
        
        # col3 = n3 = [s2, -s1*c2, c1*c2]
        col3 = np.array([s2, -s1 * c2, c1 * c2])
        
        # J_v = [col1, col2, col3] 作为列向量堆叠
        J = np.column_stack([col1, col2, col3])
        
        return J
    
    def compute_torque(self, X, xdot, joint, jointVel, vel, quat, angVel, robotPos, desiredPos, rpy=None):
        """
        计算关节扭矩（完全按照 Hopper4.py virtual_spring_control）
        
        Args:
            X: 足端位置（Hopper4.py 坐标系：Z 正向下）
            xdot: 足端速度（Hopper4.py 坐标系）
            joint: 关节角度 [roll, pitch, shift]
            jointVel: 关节速度
            vel: 机体速度（世界坐标系）
            quat: 机体四元数 [w, x, y, z]
            angVel: 机体角速度（机体坐标系）
            robotPos: 机体位置（世界坐标系）
            desiredPos: 期望位置/速度 [vx, vy, mode]
            rpy: 欧拉角（可选）
        
        Returns:
            torque: 关节扭矩 [τ_roll, τ_pitch, τ_shift]
            info: 调试信息
        """
        # ========== 四元数转旋转矩阵 ==========
        quat_scipy = [quat[1], quat[2], quat[3], quat[0]]  # scipy 格式
        vicon2world = Rotation.from_quat(quat_scipy).as_matrix()
        
        # ========== 足端位置处理（与 Hopper4.py 第 180 行一致）==========
        x = X + np.array([0, 0, 0.03])  # Hopper4.py 中的 3cm 偏移
        l_serial = np.linalg.norm(x)
        if l_serial < 1e-9:
            l_serial = 1e-9
        # 将串联腿长度映射到真实 3-RSR 的等效长度
        l = l_serial - self.leg_offset
        if l < 1e-6:
            l = 1e-6
        
        # ========== 角速度 ==========
        rAngVel = angVel
        
        # ========== 期望速度 ==========
        if desiredPos[2] == 0:
            desiredVel = self.Kpos * (desiredPos - robotPos)
        else:
            # 速度控制模式
            desiredVel = np.array([desiredPos[0], desiredPos[1], 0])
        
        if np.linalg.norm(desiredVel) > self.posVelLim:
            desiredVel = desiredVel / np.linalg.norm(desiredVel) * self.posVelLim
            
        # Debug: Check why TgtX is 0
        # if abs(desiredVel[0]) > 0.01 and self.state == 1:
        #     print(f"DEBUG: vel={vel[:2]}, dVel={desiredVel[:2]}, Kv={self.Kv}, Kr={self.Kr}")
        
        # ========== 弹簧相关计算 ==========
        unitSpring = x / l_serial
        springVel = np.dot(xdot, unitSpring) * unitSpring
        leg_velocity = np.dot(xdot, unitSpring)
        
        # 地面高度（用于能量计算）
        groundHeight = np.dot(vicon2world @ x, np.array([0, 0, -1]))
        
        # 能量计算
        energy = 0.5 * self.m * np.dot(springVel, springVel) + \
                 0.5 * self.k * (self.l0 - l)**2 + \
                 self.m * 9.81 * (-1) * groundHeight
        target = self.m * 9.81 * (self.l0 + self.h) + 0.5 * self.m * np.dot(desiredVel, desiredVel)
        error = target - energy
        force = -self.k * (l - self.l0)
        
        # 保存能量数据用于调试/绘图
        self.current_energy = energy
        self.target_energy = target
        self.energy_error = error
        
        # ========== 机器人倾斜角度 ==========
        worldUp = self.robot2vicon.T @ vicon2world.T @ np.array([0, 0, 1])
        robotTilt = np.cross(worldUp, np.array([0, 0, -1]))
        
        if np.linalg.norm(robotTilt) > 0.01:
            robotTilt = robotTilt / np.linalg.norm(robotTilt) * \
                       np.arccos(np.clip(np.dot(worldUp, np.array([0, 0, -1])), -1.0, 1.0))
        
        self.robotTilt_debug = robotTilt.copy()
        
        # ========== 安全检查（已禁用，依赖外部高度检测）==========
        # 之前的 state=1001/1003 会干扰正常控制，已移除
        # 摔倒检测由 run_raibert_mj.py 中的高度检查处理
        
        # ========== 初始化输出 ==========
        footForce = np.zeros(3)
        torque = np.zeros(3)
        hipTorque = np.zeros(3)
        
        # ========== Flight Phase（与 Hopper4.py 第 288-360 行一致）==========
        spring_force_along = 0.0
        
        if self.state == 1:
            # ========== Flight Phase (match Hopper4.py) ==========
            # Raibert foot placement (Hopper4.py):
            #   targetFootPos_xy = Kv * v_xy + Kr * v_des_xy
            targetFootPos = self.Kv * np.array([vel[0], vel[1], 0.0]) + self.Kr * np.array([desiredVel[0], desiredVel[1], 0.0])

            # Limit XY magnitude
            normTarget = np.linalg.norm(targetFootPos)
            if normTarget > self.stepperLim:
                targetFootPos = targetFootPos / normTarget * self.stepperLim
                normTarget = np.linalg.norm(targetFootPos)

            # Enforce ||targetFootPos|| == l0 by setting Z (Hopper4 convention: foot below => Z negative)
            targetFootPos[2] = -np.sqrt(max(0.0, self.l0**2 - normTarget**2))

            # Save raw (world) target before rotation, then rotate to BODY like Hopper4.py
            targetFootPos_raw = targetFootPos.copy()
            targetFootPos = self.robot2vicon.T @ vicon2world.T @ targetFootPos

            # Side force PD (remove component along leg axis)
            sideForce = self.Khp * (targetFootPos - x) - self.Khd * (xdot - np.cross(rAngVel, x))
            sideForce = sideForce - np.dot(sideForce, unitSpring) * unitSpring

            # Spring force along leg (same k/b structure as Hopper4.py)
            springForce = force * unitSpring - self.b * springVel
            spring_force_along = float(np.dot(springForce, unitSpring))

            footForce = sideForce + springForce

            # Map foot force -> joint torque (serial Jacobian)
            J = self._serial_jacobian(joint[0], joint[1], joint[2])
            try:
                torque = (np.linalg.inv(J.T) @ footForce).reshape(3)
            except Exception:
                torque = (np.linalg.pinv(J.T) @ footForce).reshape(3)

            # Debug/compat
            self.flight_target_pos = targetFootPos.copy()
            self.flight_target_pos_raw = targetFootPos_raw.copy()
            
            self.state_safety += 1
            
            # 保存调试信息
            self.footForce = footForce.copy()
            self.sideForce = sideForce.copy()
            self.springForce_vec = springForce.copy()
            
            # 相位切换：Flight → Stance（与 Hopper4.py 第 358-360 行一致）
            # 腿长小于 (l0 - threshold) 时触发，加入最小持续时间约束
            self.phase_duration_count += 1
            if l < self.l0 - self.flight_to_stance_threshold:
                if self.phase_duration_count >= self.min_phase_duration:
                    self.state = 2
                    self.state_safety = 0
                    self.phase_duration_count = 0  # 重置计数
        
        # ========== Stance Phase（与 Hopper4.py 第 362-451 行完全一致）==========
        elif self.state == 2:
            # 弹簧力（虚拟弹簧）- 与 Hopper4.py 第 366 行一致
            springForce_scalar = -self.k * (l - self.l0)
            leg_velocity = np.dot(xdot, unitSpring)
            
            # 能量补偿（与 Hopper4.py 第 370-376 行一致）
            # Hopper4.py 条件: leg_velocity > 0.1 (腿在伸展时补偿)
            # 但由于坐标系差异（Hopper4.py: Z负向下，MuJoCo: Z正向下）
            # leg_velocity 的符号相反，所以这里用 < -0.1
            energy_compensation = 0
            if leg_velocity < -0.1:  # 腿在伸展时（MuJoCo坐标系：负速度=向下=伸展）
                # error = target - energy（需要更多能量时 error > 0）
                # 但 leg_velocity < 0，所以 sign(leg_velocity) = -1
                # 为了得到正的补偿力，需要取反
                energy_compensation = -np.sign(leg_velocity) * self.Kp * error
                springForce_scalar = springForce_scalar + energy_compensation
            
            # 与 Hopper4.py 第 375-376 行一致：弹簧力不能为负
            if springForce_scalar < 0:
                springForce_scalar = 0
            
            # 保存能量信息用于调试（使用前面计算的 energy 和 target）
            # energy 和 target 已经在前面计算好了（第 241-244 行）
            self.energy_compensation_debug = energy_compensation
            
            springForce = springForce_scalar * unitSpring
            spring_force_along = springForce_scalar
            
            # Hip Torque 姿态控制
            # 注意：robotTilt 的符号和 rpy 相反！
            # 当机体前倾 (rpy[1] > 0) 时，robotTilt[1] < 0
            # 所以需要取负号来得到正确的控制方向
            # 或者直接使用 rpy 作为姿态误差
            hipTorque = np.zeros(3)
            # MuJoCo 力矩方向（通过测试验证 2024-11）：
            # - 正 Pitch 力矩 → 机体后仰（Pitch 减少）
            # - 负 Pitch 力矩 → 机体前倾（Pitch 增加）
            #
            # 控制目标：让 rpy 趋近于 0
            # - 当 rpy[1] > 0（前倾），需要正力矩来后仰
            # - 当 rpy[1] < 0（后仰），需要负力矩来前倾
            #
            # 公式：hipTorque[1] = +Kpp * rpy[1] + Kpd * rAngVel[1]
            # 位置项：当 rpy[1] > 0（前倾），产生正力矩来后仰
            # 阻尼项：当 rAngVel[1] > 0（正在前倾），产生正力矩来抵抗
            #        当 rAngVel[1] < 0（正在后仰），产生负力矩来抵抗
            if rpy is not None:
                hipTorque[0] = self.Kpp_x * rpy[0] + self.Kpd_x * rAngVel[0]
                hipTorque[1] = self.Kpp_y * rpy[1] + self.Kpd_y * rAngVel[1]
            else:
                # 如果没有 rpy，使用 robotTilt（符号和 rpy 相反）
                hipTorque[0] = -self.Kpp_x * robotTilt[0] + self.Kpd_x * rAngVel[0]
                hipTorque[1] = -self.Kpp_y * robotTilt[1] + self.Kpd_y * rAngVel[1]
            hipTorque[2] = 0
            
            # 去除弹簧方向分量
            hipTorque = hipTorque - np.dot(hipTorque, unitSpring) * unitSpring
            
            # 限制 hipTorque
            if np.linalg.norm(hipTorque) > self.hipTorqueLim:
                hipTorque = hipTorque / np.linalg.norm(hipTorque) * self.hipTorqueLim
            
            # 简化的力矩计算：
            # 1. Roll/Pitch 力矩直接使用 hipTorque（姿态控制）
            # 2. Shift 力矩通过 Jacobian 计算（弹簧力）
            #
            # 这样避免了 sideForce → Jacobian → torque 路径中的符号混乱
            
            # 计算 Shift 力矩（只来自弹簧力）
            J = self._serial_jacobian(joint[0], joint[1], joint[2])
            # 只使用弹簧力计算 Shift 力矩
            shift_torque_from_spring = J.T @ springForce
            
            # 组合力矩
            torque = np.zeros(3)
            torque[0] = hipTorque[0]  # Roll 力矩直接使用
            torque[1] = hipTorque[1]  # Pitch 力矩直接使用
            torque[2] = shift_torque_from_spring[2]  # Shift 力矩来自弹簧
            
            # 保存调试信息（保持兼容性）
            sideForce = np.cross(hipTorque, x) / np.dot(x, x)
            footForce = springForce + sideForce
            
            self.state_safety += 1
            
            # 保存调试信息
            self.footForce = footForce.copy()
            self.sideForce = sideForce.copy()
            self.springForce_vec = springForce.copy()
            self.hip_torque = hipTorque.copy()
            
            # 相位切换：Stance → Flight
            # 只有当腿伸展超过 l0 且足端离地时才切换
            # 相位切换：Stance → Flight（与 Hopper4.py 第 449-451 行一致）
            # 腿长大于 (l0 + threshold) 时触发，加入最小持续时间约束
            self.phase_duration_count += 1
            if l > self.l0 + self.stance_to_flight_threshold:
                if self.phase_duration_count >= self.min_phase_duration:
                    self.state = 1
                    self.state_safety = 0
                    self.phase_duration_count = 0  # 重置计数
        
        # ========== 错误状态 ==========
        else:
            torque = np.zeros(3)
            self.state_safety += 1
            if self.state_safety > 250:
                self.state = 1
                self.state_safety = 0
        
        # ========== 关节限位检查 ==========
        if any(joint < self.lowLim) or any(joint > self.upLim):
            torque = np.zeros(3)
        
        # ========== Shift 力矩符号修正 ==========
        # MuJoCo 串联腿模型需要反转 Shift 力矩符号
        # 验证：移除取反后机器人无法跳跃，说明这个取反是正确的
        torque[2] = -torque[2]
        
        # ========== 扭矩限制 ==========
        torque[0] = np.clip(torque[0], -self.max_torque, self.max_torque)
        torque[1] = np.clip(torque[1], -self.max_torque, self.max_torque)
        torque[2] = np.clip(torque[2], -self.max_shift_torque, self.max_shift_torque)
        
        # ========== 扭矩符号说明 ==========
        # MuJoCo 中的扭矩方向（通过测试验证 2024-11）：
        # - 正 Pitch 力矩 (+τ) → 机体后仰（Pitch 减少）
        # - 负 Pitch 力矩 (-τ) → 机体前倾（Pitch 增加）
        # - 正 Shift 力矩 (+τ) → 腿缩短 (Axis="0 0 1" up)
        # 
        # 现在 Roll/Pitch 力矩直接使用 hipTorque，不再经过 Jacobian 转换
        # 所以不需要符号反转
        
        # 调试信息
        info = {
            'state': self.state,
            'leg_length': l,
            'leg_velocity': leg_velocity,  # 添加腿速度
            'energy': self.current_energy,
            'target_energy': self.target_energy,
            'energy_error': error,
            'energy_compensation': getattr(self, 'energy_compensation_debug', 0),
            'spring_force': spring_force_along,
            'robotTilt': robotTilt,
            'desiredVel': desiredVel
        }
        
        return torque, info
