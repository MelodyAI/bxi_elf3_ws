# 导入系统模块，用于修改Python路径
import sys
# 添加项目根目录到Python路径
sys.path.append('/home/deepcyber-mk/Documents/unitree_rl_gym')
# 添加公共模块目录到Python路径
sys.path.append('/home/deepcyber-mk/Documents/unitree_rl_gym/deploy/deploy_real/common')

# 导入项目根目录常量
from legged_gym import LEGGED_GYM_ROOT_DIR
# 导入类型提示
from typing import Union
# 导入NumPy数值计算库
import numpy as np
# 导入时间模块
import time
# 导入PyTorch深度学习框架
import torch
# 导入ONNX运行时
import onnxruntime
# 导入Unitree DDS通信库的发布器
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
# 导入Unitree DDS通信库的订阅器
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
# 导入Unitree DDS消息类型（HG版本）
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
# 导入Unitree DDS消息类型（GO版本）
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
# 导入HG版本的低级命令和状态消息
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
# 导入GO版本的低级命令和状态消息
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
# 导入CRC校验模块
from unitree_sdk2py.utils.crc import CRC
# 重新导入ONNX运行时（可能为了重命名）
import onnxruntime as ort
# 导入命令辅助函数
from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
# 导入旋转辅助函数
from common.rotation_helper import get_gravity_orientation, transform_imu_data, transform_pelvis_to_torso_complete
# 导入远程控制器模块
from common.remote_controller import RemoteController, KeyMap
# 导入配置类
from config import Config

# 定义策略模型使用的关节顺序列表（29个关节）
joint_seq =['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 
 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 
 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 
 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 
 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 
 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 
 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 
 'right_wrist_yaw_joint']

# 定义XML模型中的关节顺序列表（29个关节）
joint_xml = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint",  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",  "waist_roll_joint",     "waist_pitch_joint",
    "left_shoulder_pitch_joint",     "left_shoulder_roll_joint",     "left_shoulder_yaw_joint",
    "left_elbow_joint",     "left_wrist_roll_joint",    "left_wrist_pitch_joint",    "left_wrist_yaw_joint",    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",    "right_shoulder_yaw_joint",    "right_elbow_joint",    "right_wrist_roll_joint",    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"]

# 定义函数：计算四元数的共轭
def quaternion_conjugate(q):
    """四元数共轭: [w, x, y, z] -> [w, -x, -y, -z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

# 定义函数：计算两个四元数的乘积
def quaternion_multiply(q1, q2):
    """四元数乘法: q1 ⊗ q2"""
    # 提取四元数分量
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    # 计算乘积的四元数分量
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])

# 定义函数：将四元数转换为旋转矩阵
def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵

    参数:
        q (list 或 np.array): 四元数 [w, x, y, z]

    返回:
        np.array: 3x3 的旋转矩阵
    """
    # 确保输入是numpy数组并且是浮点数类型
    q = np.array(q, dtype=np.float64)
    
    # 归一化四元数，确保它是单位四元数
    q = q / np.linalg.norm(q)
    
    # 提取四元数分量
    w, x, y, z = q
    
    # 计算旋转矩阵的各个元素
    r00 = 1 - 2*y**2 - 2*z**2
    r01 = 2*x*y - 2*z*w
    r02 = 2*x*z + 2*y*w
    
    r10 = 2*x*y + 2*z*w
    r11 = 1 - 2*x**2 - 2*z**2
    r12 = 2*y*z - 2*x*w
    
    r20 = 2*x*z - 2*y*w
    r21 = 2*y*z + 2*x*w
    r22 = 1 - 2*x**2 - 2*y**2
    
    # 组合成3x3旋转矩阵
    rotation_matrix = np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ])
    
    return rotation_matrix

# 定义控制器类
class Controller:
    # 初始化方法
    def __init__(self, config: Config) -> None:
        # 保存配置对象
        self.config = config
        # 创建远程控制器对象
        self.remote_controller = RemoteController()
        
        # 初始化策略网络（使用ONNX格式）
        self.policy = onnxruntime.InferenceSession(config.policy_path)
        
        # 初始化过程变量
        # 关节位置
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        # 关节速度
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        # 动作值
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        # 目标关节位置（初始化为默认角度）
        self.target_dof_pos = config.default_angles.copy()
        # 观测向量
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        # 命令值（速度控制）
        self.cmd = np.array([0.0, 0, 0])
        # 计数器
        self.counter = 0
        # 时间步
        self.timestep = 0
        # 加载运动数据文件
        self.motion = np.load("/home/deepcyber-mk/Documents/unitree_rl_gym/deploy/deploy_real/bydmimic/dance_zui.npz")
        # 提取身体位置数据
        self.motionpos = self.motion['body_pos_w']
        # 提取身体四元数数据
        self.motionquat = self.motion['body_quat_w']
        # 提取关节位置数据
        self.motioninputpos = self.motion['joint_pos']
        # 提取关节速度数据
        self.motioninputvel = self.motion['joint_vel']
        # 动作缓冲区
        self.action_buffer = np.zeros((self.config.num_actions,), dtype=np.float32)
        # 初始化到世界坐标系的变换矩阵
        self.init_to_world = np.zeros((3,3), dtype=np.float32)
        # 关节索引映射（0-28）
        self.dof_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
                        12, 13, 14, 
                        15, 16, 17, 18, 19, 20, 21, 
                        22, 23, 24, 25, 26, 27, 28]
        
        # 根据配置选择消息类型
        if config.msg_type == "hg":
            # g1和h1_2机器人使用hg消息类型
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            # 电机模式：位置控制模式
            self.mode_pr_ = MotorMode.PR
            # 机器模式
            self.mode_machine_ = 0

            # 创建低级命令发布器
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            # 创建低级状态订阅器
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1机器人使用go消息类型
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            # 创建低级命令发布器
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            # 创建低级状态订阅器
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            # 无效消息类型时报错
            raise ValueError("Invalid msg_type")

        # 等待订阅器接收数据
        self.wait_for_low_state()

        # 初始化命令消息
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    # HG版本低级状态处理器
    def LowStateHgHandler(self, msg: LowStateHG):
        # 更新低级状态
        self.low_state = msg
        # 更新机器模式
        self.mode_machine_ = self.low_state.mode_machine
        # 更新远程控制器状态
        self.remote_controller.set(self.low_state.wireless_remote)

    # GO版本低级状态处理器
    def LowStateGoHandler(self, msg: LowStateGo):
        # 更新低级状态
        self.low_state = msg
        # 更新远程控制器状态
        self.remote_controller.set(self.low_state.wireless_remote)

    # 发送命令方法
    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        # 计算CRC校验码
        cmd.crc = CRC().Crc(cmd)
        # 发布命令
        self.lowcmd_publisher_.Write(cmd)

    # 等待低级状态方法
    def wait_for_low_state(self):
        # 循环等待直到收到状态数据
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    # 零力矩状态方法
    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        # 等待开始信号
        while self.remote_controller.button[KeyMap.start] != 1:
            # 创建零力矩命令
            create_zero_cmd(self.low_cmd)
            # 发送命令
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    # 移动到默认位置方法
    def move_to_default_pos(self):
        print("Moving to default pos.")
        # 移动总时间2秒
        total_time = 2
        # 计算步数
        num_step = int(total_time / self.config.control_dt)
        
        # 合并腿部和手臂腰部关节索引
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        # 获取刚度和阻尼参数
        kps = self.config.stiffness
        kds = self.config.damping
        # 默认位置
        default_pos = self.config.default_angles.copy()
        # 关节数量
        dof_size = len(dof_idx)
        
        # 记录当前位置
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size): 
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # 逐步移动到默认位置
        for i in range(num_step):
            # 计算插值系数
            alpha = i / num_step
            for j in range(dof_size):
                # 获取电机索引
                motor_idx = dof_idx[j]
                # 目标位置
                target_pos = default_pos[j]
                # 插值计算当前位置
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                # 速度设为0
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                # 设置刚度和阻尼
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                # 力矩设为0
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            # 发送命令
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    # 默认位置状态方法
    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        # 等待A按钮信号
        while self.remote_controller.button[KeyMap.A] != 1:
            # 设置腿部关节命令
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i]*5
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            # 设置手臂和腰部关节命令
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i+12]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i+12]*3
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i+12]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            # 发送命令
            self.send_cmd(self.low_cmd)
            # 获取四元数并打印
            quat = self.low_state.imu_state.quaternion
            print("quat",quat)
            time.sleep(self.config.control_dt)
    
    # 从四元数中提取偏航角的方法
    def yaw_quat(self,q):
        # 提取四元数分量
        w, x, y, z = q
        # 计算偏航角
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        # 返回仅包含偏航的四元数
        return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    
    # 矩阵转四元数方法
    def matrix_to_quaternion_simple(self, matrix):
        """
        简化的矩阵转四元数实现
        """
        # 转换为numpy数组
        matrix = np.array(matrix)
        # 提取矩阵元素
        m00, m01, m02 = matrix[0]
        m10, m11, m12 = matrix[1]
        m20, m21, m22 = matrix[2]
        
        # 计算迹
        trace = m00 + m11 + m22
        
        # 根据迹的大小选择不同的计算方式
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m21 - m12) * s
            y = (m02 - m20) * s
            z = (m10 - m01) * s
        elif m00 > m11 and m00 > m22:
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    # 主运行方法
    def run(self):
        # 计数器加1
        self.counter += 1
        # 获取当前关节位置和速度
        for i in range(len(self.dof_idx)):
            self.qj[i] = self.low_state.motor_state[self.dof_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.dof_idx[i]].dq

        # 获取IMU状态（四元数格式：w, x, y, z）
        quat = self.low_state.imu_state.quaternion
        # 获取角速度
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # 根据IMU类型进行坐标变换
        if self.config.imu_type == "torso":
            # h1和h1_2的IMU在躯干上
            # 需要将IMU数据转换到骨盆坐标系
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)
        
        if self.config.imu_type == "pelvis":
            # 骨盆IMU数据需要转换到躯干坐标系
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_roll = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[1]].q
            waist_pitch = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[2]].q
            # 转换到躯干坐标系
            quat_torso = transform_pelvis_to_torso_complete(waist_yaw, waist_roll, waist_pitch, quat)
        
        # 前两个时间步进行初始化
        if self.timestep < 2:
            # 获取参考运动的四元数
            ref_motion_quat = self.motionquat[self.timestep,9,:]
            # 提取偏航四元数
            yaw_motion_quat = self.yaw_quat(ref_motion_quat)
            # 将偏航四元数转换为旋转矩阵
            yaw_motion_matrix = np.zeros(9)
            yaw_motion_matrix = quaternion_to_rotation_matrix(yaw_motion_quat)
            yaw_motion_matrix = yaw_motion_matrix.reshape(3,3)
            
            # 获取机器人当前的四元数
            robot_quat = quat_torso
            # 提取机器人的偏航四元数
            yaw_robot_quat = self.yaw_quat(robot_quat)
            # 将机器人偏航四元数转换为旋转矩阵
            yaw_robot_matrix = np.zeros(9)
            yaw_robot_matrix = quaternion_to_rotation_matrix(yaw_robot_quat)
            yaw_robot_matrix = yaw_robot_matrix.reshape(3,3)
            # 计算初始到世界的变换矩阵
            self.init_to_world = yaw_robot_matrix @ yaw_motion_matrix.T
        
        # 打印躯干四元数
        print("quat_torso",quat_torso)
        # 打印原始四元数
        print("quat",quat)
        
        # 准备观测数据
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        # 计算关节位置误差
        qj_obs = qj_obs  # - self.config.default_angles
        
        # 获取当前时间步的运动输入
        motioninput = np.concatenate((self.motioninputpos[self.timestep,:],self.motioninputvel[self.timestep,:]), axis=0)
        # 获取当前时间步的运动身体位置
        motionposcurrent = self.motionpos[self.timestep,9,:]
        # 获取当前时间步的运动身体四元数
        motionquatcurrent = self.motionquat[self.timestep,9,:]

        # 计算相对四元数
        relquat = quaternion_multiply(self.matrix_to_quaternion_simple(self.init_to_world), motionquatcurrent)
        relquat = quaternion_multiply(quaternion_conjugate(quat_torso),relquat)
        # 归一化四元数
        relquat = relquat / np.linalg.norm(relquat)
        # 转换为旋转矩阵并取前两列展平
        relmatrix = quaternion_to_rotation_matrix(relquat)[:,:2].reshape(-1,)
        
        # 构建观测向量
        offset = 0
        # 添加运动输入
        self.obs[offset:offset+58] = motioninput
        offset += 58
        # 添加相对方向矩阵
        self.obs[offset:offset+6] = relmatrix
        offset += 6
        # 添加角速度
        self.obs[offset:offset+3] = ang_vel
        offset += 3
        
        # 处理关节位置观测
        qpos_urdf = qj_obs
        # 重新排序关节位置以匹配策略顺序
        qj_obs_seq = np.array([qpos_urdf[joint_xml.index(joint)] for joint in joint_seq])
        # 添加关节位置误差
        self.obs[offset:offset+29] = qj_obs_seq - self.config.default_angles_seq
        offset += 29
        
        # 处理关节速度观测
        qvel_urdf = dqj_obs
        # 重新排序关节速度以匹配策略顺序
        dqj_obs_seq = np.array([qvel_urdf[joint_xml.index(joint)] for joint in joint_seq])
        # 添加关节速度
        self.obs[offset:offset+29] = dqj_obs_seq
        offset += 29
        
        # 添加上一时刻的动作
        self.obs[offset:offset+29] = self.action_buffer
                
        # 从策略网络获取动作
        # 将观测转换为张量并添加批次维度
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        # 运行ONNX模型
        action = self.policy.run(['actions'], {'obs': obs_tensor.numpy(),'time_step':np.array([self.timestep], dtype=np.float32).reshape(1,1)})[0]
        
        # 处理动作输出
        action = np.asarray(action).reshape(-1)
        self.action = action.copy()
        self.action_buffer = action.copy()
        
        # 将动作转换为目标关节位置
        target_dof_pos = self.config.default_angles_seq + self.action * self.config.action_scale_seq
        target_dof_pos = target_dof_pos.reshape(-1,)
        # 重新排序目标关节位置以匹配XML顺序
        target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in joint_xml])
        
        # 时间步加1
        self.timestep += 1
        
        # 构建低级命令
        # 设置腿部关节命令
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # 设置手臂和腰部关节命令
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i+12]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i+12]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i+12]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # 发送命令
        self.send_cmd(self.low_cmd)

        # 等待控制周期
        time.sleep(self.config.control_dt)

# 主程序入口
if __name__ == "__main__":
    # 导入参数解析模块
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加网络接口参数
    parser.add_argument("net", type=str, help="network interface")
    # 添加配置文件参数
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1_for_bydmimic.yaml")
    # 解析参数
    args = parser.parse_args()

    # 加载配置文件
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # 初始化DDS通信
    ChannelFactoryInitialize(0, args.net)

    # 创建控制器实例
    controller = Controller(config)

    # 进入零力矩状态，按下开始键继续执行
    controller.zero_torque_state()

    # 移动到默认位置
    controller.move_to_default_pos()

    # 进入默认位置状态，按下A键继续执行
    controller.default_pos_state()

    # 主循环
    while True:
        try:
            # 运行控制器
            controller.run()
            # 按下选择键退出
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            # 捕获键盘中断
            break
    
    # 进入阻尼状态
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")

# 运行命令示例:
# python deploy_real4bydmimic.py enp4s0 g1_for_bydmimic.yaml