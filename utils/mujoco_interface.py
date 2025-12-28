"""
MuJoCo 状态接口

目标：输出与 Hopper4.py 控制器输入一致的状态

关键点：
1. Hopper4.py 的 X (foot_pos) 是在机体坐标系下，Z 正方向向上（足端在机体下方时 Z 为正）
2. MuJoCo 的足端在机体下方时 Z 为负
3. 需要进行坐标变换以匹配 Hopper4.py 的约定

坐标系约定（与 Hopper4.py 一致）：
- robot2vicon = [[1,0,0], [0,1,0], [0,0,-1]]
- 这意味着 Hopper4.py 的 Z 轴与 MuJoCo 的 Z 轴方向相反
"""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


class MuJoCoInterface:
    """
    MuJoCo 仿真接口
    
    输出与 Hopper4.py 控制器输入一致的状态
    """
    
    def __init__(self, model, data):
        """
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        
        # 关节名称到索引的映射
        self.joint_names = ['Leg_Joint_Roll', 'Leg_Joint_Pitch', 'Leg_Joint_Shift']
        self.actuator_names = ['roll_motor', 'pitch_motor', 'shift_motor']
        
        # 获取关节和执行器索引
        self.joint_ids = []
        for name in self.joint_names:
            joint_id = model.joint(name).id
            self.joint_ids.append(joint_id)
        # 对应的 dof 索引（用于 Jacobian/速度计算）
        self.joint_dof_ids = [int(model.jnt_dofadr[jid]) for jid in self.joint_ids]
        
        self.actuator_ids = []
        for name in self.actuator_names:
            actuator_id = model.actuator(name).id
            self.actuator_ids.append(actuator_id)
        
        # 足端 body 名称
        self.foot_body_name = 'Foot_Link'
        self.foot_body_id = model.body(self.foot_body_name).id
        
        # 基座 body 名称
        self.base_body_name = 'base_link'
        self.base_body_id = model.body(self.base_body_name).id

        # 触地传感器（如果 mjcf 里有 foot_touch）
        self.foot_touch_sensor_id = None
        self.foot_touch_adr = None
        self.foot_touch_dim = None
        try:
            sid = model.sensor('foot_touch').id
            self.foot_touch_sensor_id = int(sid)
            self.foot_touch_adr = int(model.sensor_adr[self.foot_touch_sensor_id])
            self.foot_touch_dim = int(model.sensor_dim[self.foot_touch_sensor_id])
        except Exception:
            # 兼容没有该 sensor 的模型
            self.foot_touch_sensor_id = None
            self.foot_touch_adr = None
            self.foot_touch_dim = None

        # Propeller 位置信息（机体坐标系），三臂布局（与 MJCF 里的 prop_site_* 一致）
        # Rotated +30deg to align green prop (Prop1) with +X axis (robot/IMU coordinate alignment)
        # 3 rotor centers (same as the visual arm tips)
        self.prop_positions_body = np.array([
            [-0.284451, 0.493317, 0.0],
            [0.569451, 0.000317, 0.0],
            [-0.285000, -0.493634, 0.0]
        ])
        # Legacy 3-rotor spin dirs (unused in the 6-motor coaxial setup below)
        self.prop_spin_dirs = np.array([1.0, -1.0, 1.0])

        # === 6-motor coaxial pairs (two props share each arm) ===
        # Each rotor center has an upper + lower motor with opposite spin to cancel reaction torque.
        # Motor positions are only used for applying external forces/torques on base_link.
        z_off = 0.03
        self.motor_positions_body = np.repeat(self.prop_positions_body, 2, axis=0).astype(float)
        self.motor_positions_body[0::2, 2] = +z_off
        self.motor_positions_body[1::2, 2] = -z_off
        self.motor_spin_dirs = np.tile(np.array([1.0, -1.0], dtype=float), 3)
        # 旋翼推力方向：+Z（产生向上的升力）
        self.prop_direction_body = np.array([0.0, 0.0, 1.0])
        
        # 坐标变换矩阵（与 Hopper4.py 一致）
        # robot2vicon 翻转 Z 轴
        self.robot2vicon = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, -1]])
    
    def get_state(self):
        """
        获取机器人状态（与 Hopper4.py 输入一致）
        
        Returns:
            state: 字典，包含：
                - body_pos: 机体位置（世界坐标系）
                - body_quat: 机体四元数 [w, x, y, z]
                - body_vel: 机体线速度（世界坐标系）
                - body_ang_vel: 机体角速度（机体坐标系）
                - body_rpy: 机体欧拉角
                - joint_pos: 关节角度 [roll, pitch, shift]
                - joint_vel: 关节速度
                - foot_pos: 足端位置（机体坐标系，Z 正向上，与 Hopper4.py 一致）
                - foot_vel: 足端速度（机体坐标系）
                - imu_acc: IMU 加速度（机体坐标系）
        """
        # ========== 1. 获取关节状态 ==========
        joint_pos = np.zeros(3)
        joint_vel = np.zeros(3)
        
        for i, jid in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jid]
            qvel_adr = self.model.jnt_dofadr[jid]
            joint_pos[i] = self.data.qpos[qpos_adr]
            joint_vel[i] = self.data.qvel[qvel_adr]
        
        # ========== 2. 获取机体状态 ==========
        # 位置（世界坐标系）
        body_pos = self.data.qpos[0:3].copy()
        
        # 四元数：MuJoCo 格式是 [w, x, y, z]
        body_quat = self.data.qpos[3:7].copy()
        
        # 速度（世界坐标系）
        body_vel_world = self.data.qvel[0:3].copy()
        
        # 角速度（世界坐标系 -> 机体坐标系）
        body_ang_vel_world = self.data.qvel[3:6].copy()
        quat_scipy = [body_quat[1], body_quat[2], body_quat[3], body_quat[0]]
        rot = Rotation.from_quat(quat_scipy)
        body_ang_vel = rot.inv().apply(body_ang_vel_world)
        
        # 欧拉角
        body_rpy = rot.as_euler('xyz')
        
        # ========== 3. 获取足端位置（机体坐标系）==========
        # 使用四元数进行坐标转换（与 IMU 一致）
        foot_pos_world = self.data.xpos[self.foot_body_id].copy()
        base_pos_world = body_pos  # 使用 qpos 中的位置，与四元数一致
        
        foot_pos_rel = foot_pos_world - base_pos_world
        # 使用四元数的逆将世界坐标系转换到机体坐标系
        foot_pos_mujoco = rot.inv().apply(foot_pos_rel)
        
        # 转换到 Hopper4.py 约定：Z 轴翻转
        # 在 Hopper4.py 中，足端在机体下方时 Z 为正
        # 在 MuJoCo 中，足端在机体下方时 Z 为负
        foot_pos = self.robot2vicon @ foot_pos_mujoco
        
        # ========== 4. 获取足端速度 ==========
        # 注意：历史上 controller 使用的是“足端绝对速度(表达在机体坐标系)”的版本。
        # 为了不破坏现有控制效果，我们同时输出：
        # - foot_vel_mj / foot_vel : 旧版（绝对足端速度）
        # - foot_vel_mj_rel / foot_vel_rel : 相对速度（足端相对机体），用于高层估计器
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, None, self.foot_body_id)
        foot_vel_world_abs = jacp @ self.data.qvel

        # 旧版：绝对速度 -> 机体坐标系
        foot_vel_mujoco = rot.inv().apply(foot_vel_world_abs)

        # 相对速度（更贴近真机做法）：
        # 只使用腿部关节的 Jacobian * 关节速度，得到足端相对机体的速度（不依赖 base 真值速度）
        dof_ids = self.joint_dof_ids
        foot_vel_world_rel = jacp[:, dof_ids] @ self.data.qvel[dof_ids]
        foot_vel_mujoco_rel = rot.inv().apply(foot_vel_world_rel)
        
        # 转换到 Hopper4.py 约定
        foot_vel = self.robot2vicon @ foot_vel_mujoco
        foot_vel_rel = self.robot2vicon @ foot_vel_mujoco_rel
        
        # ========== 5. IMU (gyro/acc) ==========
        # Gyro: body frame angular rate (rad/s)
        imu_gyro = body_ang_vel.copy()

        # Acc: specific force in body frame (m/s^2)
        # IMU accelerometer measures: f = R^T * (a_world - g_world)
        gravity_world = np.array([0.0, 0.0, -9.81], dtype=float)
        base_lin_acc_world = self.data.qacc[0:3].copy()  # free-joint translational acceleration in world
        specific_force_world = base_lin_acc_world - gravity_world
        imu_acc = rot.inv().apply(specific_force_world)

        # ========== 6. Foot touch sensor ==========
        foot_touch = 0.0
        if self.foot_touch_sensor_id is not None and self.foot_touch_adr is not None:
            # touch sensor dim=1
            foot_touch = float(self.data.sensordata[self.foot_touch_adr])
        
        return {
            'body_pos': body_pos,
            'body_quat': body_quat,
            'body_vel': body_vel_world,  # 世界坐标系，与 Hopper4.py 的 vel 一致
            'body_ang_vel': body_ang_vel,
            'body_rpy': body_rpy,
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'foot_pos': foot_pos,  # Hopper4.py 约定
            'foot_vel': foot_vel,  # Hopper4.py 约定
            # 原始 (MuJoCo body frame) 足端量：与 body_quat/imu 坐标一致，用于估计器
            'foot_pos_mj': foot_pos_mujoco,
            'foot_vel_mj': foot_vel_mujoco,
            # 相对足端速度（更符合接触约束/里程计语义）
            'foot_vel_mj_rel': foot_vel_mujoco_rel,
            'foot_vel_rel': foot_vel_rel,
            'imu_gyro': imu_gyro,
            'imu_acc': imu_acc,
            'foot_touch': foot_touch,
        }
    
    def set_torque(self, torque):
        """
        设置关节扭矩
        
        Args:
            torque: 关节扭矩 [τ_roll, τ_pitch, τ_shift]
        """
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = torque[i]
    
    def get_foot_contact(self):
        """
        检测足端是否接触地面
        
        Returns:
            contact: True 如果足端接触地面
        """
        # Prefer an explicit touch sensor if available (more reliable than geom-name heuristics).
        if self.foot_touch_sensor_id is not None and self.foot_touch_adr is not None:
            try:
                v = float(self.data.sensordata[self.foot_touch_adr])
                if v > 1e-6:
                    return True
            except Exception:
                pass

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = self.model.geom(contact.geom1).name
            geom2 = self.model.geom(contact.geom2).name
            
            if ('foot_collision' in [geom1, geom2] or 'foot_visual' in [geom1, geom2]) and 'ground' in [geom1, geom2]:
                return True
        
        return False
    
    def reset(self, init_height=0.8, init_shift=0.0):
        """
        重置机器人状态
        
        Args:
            init_height: 初始高度 (m)
            init_shift: 初始腿伸缩量 (m)
        """
        # 重置位置和姿态
        self.data.qpos[0:3] = [0, 0, init_height]
        self.data.qpos[3:7] = [1, 0, 0, 0]  # [w, x, y, z]
        
        # 重置速度
        self.data.qvel[:] = 0
        
        # 重置关节
        for i, jid in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jid]
            if i == 2:  # shift joint
                self.data.qpos[qpos_adr] = init_shift
            else:
                self.data.qpos[qpos_adr] = 0
        
        # 重置控制
        self.data.ctrl[:] = 0
        
        # 前向运动学更新
        mujoco.mj_forward(self.model, self.data)

    def apply_propeller_forces(self, thrusts, reaction_torques=None, attitude_only=False):
        """
        在 base_link 上施加三旋翼的等效合力/力矩
        
        Args:
            thrusts: 三个电机推力 (N)
            reaction_torques: 反扭力矩
            attitude_only: 如果 True，只保留力矩（姿态控制），不施加升力
        """
        thrusts = np.asarray(thrusts, dtype=float).reshape(-1)
        if thrusts.size not in (3, 6):
            raise ValueError("thrusts must be shape (3,) or (6,)")

        if reaction_torques is None:
            reaction_torques = np.zeros((thrusts.size,), dtype=float)
        else:
            reaction_torques = np.asarray(reaction_torques, dtype=float).reshape(-1)
            if reaction_torques.size != thrusts.size:
                raise ValueError("reaction_torques must match thrusts shape")

        base_rot = self.data.xmat[self.base_body_id].reshape(3, 3)
        total_force_world = np.zeros(3)
        total_torque_world = np.zeros(3)

        # Select rotor vs motor application points.
        if thrusts.size == 6:
            pos_body = self.motor_positions_body
        else:
            pos_body = self.prop_positions_body

        for thrust, torque_z, r_body in zip(thrusts, reaction_torques, pos_body):
            if thrust == 0.0 and torque_z == 0.0:
                continue
            force_body = self.prop_direction_body * thrust
            torque_body = np.cross(r_body, force_body) + np.array([0.0, 0.0, torque_z])
            total_force_world += base_rot @ force_body
            total_torque_world += base_rot @ torque_body

        # attitude_only 模式: 只保留力矩，不产生升力
        if attitude_only:
            total_force_world = np.zeros(3)

        self.data.xfrc_applied[self.base_body_id, 0:3] = total_force_world
        self.data.xfrc_applied[self.base_body_id, 3:6] = total_torque_world

    def clear_external_forces(self):
        """清除 base_link 上的外力（用于禁用 propeller）"""
        self.data.xfrc_applied[self.base_body_id, :] = 0.0


class KeyboardTeleop:
    """
    键盘遥控器
    """
    
    def __init__(self, step=0.1, max_speed=0.8):
        self.step = step
        self.max_speed = max_speed
        self.vx = 0.0
        self.vy = 0.0
        self.running = True
    
    def process_key(self, key):
        if key is None:
            return self.get_velocity()
        
        key = key.lower()
        
        if key == 'y':
            self.vx = min(self.vx + self.step, self.max_speed)
        elif key == 'h':
            self.vx = max(self.vx - self.step, -self.max_speed)
        elif key == 'j':
            self.vy = min(self.vy + self.step, self.max_speed)
        elif key == 'g':
            self.vy = max(self.vy - self.step, -self.max_speed)
        elif key == ' ':
            self.vx = 0.0
            self.vy = 0.0
        elif key == 'q':
            self.running = False
        
        return self.get_velocity()
    
    def get_velocity(self):
        return np.array([self.vx, self.vy, 1.0])
