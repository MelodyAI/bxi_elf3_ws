import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
import communication.msg as bxiMsg
import communication.srv as bxiSrv
import nav_msgs.msg 
import sensor_msgs.msg
from threading import Lock
import numpy as np
# import torch
import time
import sys
import os
import math
from collections import deque
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

import onnxruntime as ort

robot_name = "elf3"

dof_num = 29

dof_use = 29#26

num_actions = 29 #
num_obs = 154 #154

# ankle_y_offset = 0.0

# joint_names: waist_y_joint,waist_x_joint,waist_z_joint,l_hip_y_joint,l_hip_x_joint,l_hip_z_joint,l_knee_y_joint,l_ankle_y_joint,l_ankle_x_joint,r_hip_y_joint,r_hip_x_joint,r_hip_z_joint,r_knee_y_joint,r_ankle_y_joint,r_ankle_x_joint,l_shoulder_y_joint,l_shoulder_x_joint,l_shoulder_z_joint,l_elbow_y_joint,l_wrist_x_joint,l_wrist_y_joint,l_wrist_z_joint,r_shoulder_y_joint,r_shoulder_x_joint,r_shoulder_z_joint,r_elbow_y_joint,r_wrist_x_joint,r_wrist_y_joint,r_wrist_z_joint
# joint_stiffness: 149.436,149.436,74.718,74.884,74.884,74.718,74.884,42.324,42.324,74.884,74.884,74.718,74.884,42.324,42.324,74.718,74.718,21.162,74.718,21.162,21.162,21.162,74.718,74.718,21.162,74.718,21.162,21.162,21.162
# joint_damping: 9.513,9.513,4.757,4.767,4.767,4.757,4.767,2.694,2.694,4.767,4.767,4.757,4.767,2.694,2.694,4.757,4.757,1.347,4.757,1.347,1.347,1.347,4.757,4.757,1.347,4.757,1.347,1.347,1.347
# default_joint_pos: 0.000,0.000,0.000,-0.100,0.000,0.000,0.300,-0.200,0.000,-0.100,0.000,0.000,0.300,-0.200,0.000,0.200,0.200,0.000,1.280,0.000,0.000,0.000,0.200,-0.200,0.000,1.280,0.000,0.000,0.000
# command_names: motion
# observation_names: command,motion_anchor_ori_b,base_ang_vel,joint_pos,joint_vel,actions
# action_scale: 0.167,0.167,0.167,0.501,0.501,0.167,0.501,0.295,0.295,0.501,0.501,0.167,0.501,0.295,0.295,0.167,0.167,0.295,0.167,0.295,0.295,0.295,0.167,0.167,0.295,0.167,0.295,0.295,0.295
# anchor_body_name: torso_link
# body_names: torso_link,l_hip_x_link,l_knee_y_link,l_ankle_x_link,r_hip_x_link,r_knee_y_link,r_ankle_x_link,waist_z_link,l_shoulder_x_link,l_elbow_y_link,l_wrist_z_link,r_shoulder_x_link,r_elbow_y_link,r_wrist_z_link

joint_name = (
    "waist_y_joint",
    "waist_x_joint",
    "waist_z_joint",
    
    "l_hip_y_joint",   # 左腿_髋关节_z轴
    "l_hip_x_joint",   # 左腿_髋关节_x轴
    "l_hip_z_joint",   # 左腿_髋关节_y轴
    "l_knee_y_joint",   # 左腿_膝关节_y轴
    "l_ankle_y_joint",   # 左腿_踝关节_y轴
    "l_ankle_x_joint",   # 左腿_踝关节_x轴

    "r_hip_y_joint",   # 右腿_髋关节_z轴    
    "r_hip_x_joint",   # 右腿_髋关节_x轴
    "r_hip_z_joint",   # 右腿_髋关节_y轴
    "r_knee_y_joint",   # 右腿_膝关节_y轴
    "r_ankle_y_joint",   # 右腿_踝关节_y轴
    "r_ankle_x_joint",   # 右腿_踝关节_x轴

    "l_shoulder_y_joint",   # 左臂_肩关节_y轴
    "l_shoulder_x_joint",   # 左臂_肩关节_x轴
    "l_shoulder_z_joint",   # 左臂_肩关节_z轴
    "l_elbow_y_joint",   # 左臂_肘关节_y轴
    "l_wrist_x_joint",
    "l_wrist_y_joint",
    "l_wrist_z_joint",
    
    "r_shoulder_y_joint",   # 右臂_肩关节_y轴   
    "r_shoulder_x_joint",   # 右臂_肩关节_x轴
    "r_shoulder_z_joint",   # 右臂_肩关节_z轴
    "r_elbow_y_joint",    # 右臂_肘关节_y轴
    "r_wrist_x_joint",
    "r_wrist_y_joint",
    "r_wrist_z_joint",
    )   

# joint_nominal_pos = np.array([   # 指定的固定关节角度
#     0.000,0.000,0.000,
#     -0.100,0.000,0.000,0.300,-0.200,0.000,
#     -0.100,0.000,0.000,0.300,-0.200,0.000,
#     0.200,0.200,0.000,1.280,0.000,0.000,0.000,
#     0.200,-0.200,0.000,1.280,0.000,0.000,0.000],    # 右臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
#     dtype=np.float32)

# joint_kp = np.array([     # 指定关节的kp，和joint_name顺序一一对应
#     149.436,149.436,74.718,
#     74.884,74.884,74.718,74.884,42.324,42.324,
#     74.884,74.884,74.718,74.884,42.324,42.324,
#     74.718,74.718,21.162,74.718,21.162,21.162,21.162,
#     74.718,74.718,21.162,74.718,21.162,21.162,21.162], 
#     dtype=np.float32)

# joint_kd = np.array([  # 指定关节的kd，和joint_name顺序一一对应
#     9.513,9.513,4.757,
#     4.767,4.767,4.757,4.767,2.694,2.694,
#     4.767,4.767,4.757,4.767,2.694,2.694,
#     4.757,4.757,1.347,4.757,1.347,1.347,1.347,
#     4.757,4.757,1.347,4.757,1.347,1.347,1.347], 
#     dtype=np.float32)


joint_nominal_pos = np.array([   # 指定的固定关节角度
    0.0, 0.0, 0.0,
    -0.4,0.0,0.0,0.8,-0.4,0.0,
    -0.4,0.0,0.0,0.8,-0.4,0.0,
    0.5,0.3,-0.1,-0.2, 0.0,0.0,0.0,     # 左臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
    0.5,-0.3,0.1,-0.2, 0.0,0.0,0.0],    # 右臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
    dtype=np.float32)

joint_kp = np.array([     # 指定关节的kp，和joint_name顺序一一对应
    300,300,300,
    150,100,100,200,50,20,
    150,100,100,200,50,20,
    80,80,80,60, 20,20,20,
    80,80,80,60, 20,20,20], 
    dtype=np.float32)

joint_kd = np.array([  # 指定关节的kd，和joint_name顺序一一对应
    3,3,3,
    2,2,2,2.5,1,1,
    2,2,2,2.5,1,1,
    2,2,2,2, 1,1,1,
    2,2,2,2, 1,1,1], 
    dtype=np.float32)


# 定义函数：提取四元数的偏航角分量
def yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

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

# 矩阵转四元数方法
def matrix_to_quaternion_simple(matrix):
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
    

class BxiExample(Node):

    def __init__(self):

        super().__init__('bxi_example_py')
        
        self.declare_parameter('/topic_prefix', 'default_value')
        self.topic_prefix = self.get_parameter('/topic_prefix').get_parameter_value().string_value
        print('topic_prefix:', self.topic_prefix)

        self.declare_parameter('/npz_file', 'default_value')
        self.npz_file = self.get_parameter('/npz_file').get_parameter_value().string_value
        print('npz_file:', self.npz_file)
        
        self.declare_parameter('/onnx_file', 'default_value')
        self.onnx_file = self.get_parameter('/onnx_file').get_parameter_value().string_value        
        print("onnx_file:", self.onnx_file)

        # 订阅和发布主题
        qos = QoSProfile(depth=1, durability=qos_profile_sensor_data.durability, reliability=qos_profile_sensor_data.reliability)
        
        self.act_pub = self.create_publisher(bxiMsg.ActuatorCmds, self.topic_prefix+'actuators_cmds', qos)  # CHANGE
        
        self.odom_sub = self.create_subscription(nav_msgs.msg.Odometry, self.topic_prefix+'odom', self.odom_callback, qos)
        self.joint_sub = self.create_subscription(sensor_msgs.msg.JointState, self.topic_prefix+'joint_states', self.joint_callback, qos)
        self.imu_sub = self.create_subscription(sensor_msgs.msg.Imu, self.topic_prefix+'imu_data', self.imu_callback, qos)
        self.touch_sub = self.create_subscription(bxiMsg.TouchSensor, self.topic_prefix+'touch_sensor', self.touch_callback, qos)
        self.joy_sub = self.create_subscription(bxiMsg.MotionCommands, 'motion_commands', self.joy_callback, qos)

        self.rest_srv = self.create_client(bxiSrv.RobotReset, self.topic_prefix+'robot_reset')
        self.sim_rest_srv = self.create_client(bxiSrv.SimulationReset, self.topic_prefix+'sim_reset')
        
        self.timer_callback_group_1 = MutuallyExclusiveCallbackGroup()

        self.lock_in = Lock()
        self.lock_ou = self.lock_in #Lock()

        self.qpos = np.zeros(num_actions,dtype=np.double)
        self.qvel = np.zeros(num_actions,dtype=np.double)
        self.omega = np.zeros(3,dtype=np.double)
        self.quat = np.zeros(4,dtype=np.double)
        
        self.motion =  np.load(self.npz_file)
        print("Load motion from:", self.npz_file)
        self.motionpos = self.motion["body_pos_w"]
        self.motionquat = self.motion["body_quat_w"]
        self.motioninputpos = self.motion["joint_pos"]
        self.motioninputvel = self.motion["joint_vel"]

        self.action = np.zeros(num_actions, dtype=np.float32)
        self.obs = np.zeros(num_obs, dtype=np.float32)
        
        
        print("policy test")
        self.timestep = 0
        self.obs_input = self.obs.reshape(1, -1).astype(np.float32) # 将obs从(154,)变成(1,154)并确保数据类型
        self.initialize_onnx(self.onnx_file)
        self.action[:] = self.inference_step(self.obs_input,self.timestep)
        
        self.action_buffer = np.zeros((num_actions,), dtype=np.float32)
        self.motioninput = np.concatenate((self.motioninputpos[self.timestep,:],self.motioninputvel[self.timestep,:]), axis=0)
        self.motionposcurrent = self.motionpos[self.timestep,0,:]
        self.motionquatcurrent = self.motionquat[self.timestep,0,:]
        self.target_dof_pos = self.joint_pos_array.copy()

        self.step = 0
        self.loop_count = 0
        # self.dt = 0.01  # loop @100Hz
        self.dt = 0.02  # loop @50Hz
        self.control_decimation = 2  # 控制降采样倍数（控制频率 = loop频率 / control_decimation）
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.timer_callback_group_1)

    # 初始化部分（完整版）
    def initialize_onnx(self, model_path):
        import onnx
        model = onnx.load(model_path)
        for prop in model.metadata_props:
            if prop.key == "joint_names":
                self.joint_seq = prop.value.split(",")
            if prop.key == "default_joint_pos":   
                self.joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
                self.joint_pos_array = np.array([self.joint_pos_array_seq[self.joint_seq.index(joint)] for joint in joint_name])
            if prop.key == "joint_stiffness":
                self.stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
                self.stiffness_array = np.array([self.stiffness_array_seq[self.joint_seq.index(joint)] for joint in joint_name])
                # stiffness_array = np.array([])
                
            if prop.key == "joint_damping":
                self.damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
                self.damping_array = np.array([self.damping_array_seq[self.joint_seq.index(joint)] for joint in joint_name])        
            
            if prop.key == "action_scale":
                self.action_scale = np.array([float(x) for x in prop.value.split(",")])
            print(f"{prop.key}: {prop.value}")
        
        # 配置执行提供者（根据硬件选择最优后端）
        providers = [
            'CUDAExecutionProvider',  # 优先使用GPU
            'CPUExecutionProvider'    # 回退到CPU
        ] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        
        # 启用线程优化配置
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # 设置计算线程数
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # 创建推理会话
        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=options
        )
        
        # 预存输入输出信息
        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        
        # 预分配输入内存（可选，适合固定输入尺寸）
        self.input_buffer = np.zeros(
            self.input_info.shape,
            dtype=np.float32
        )

    # 循环推理部分（极速版）
    def inference_step(self, obs_data, timestep):
        # 使用预分配内存（如果适用）
        np.copyto(self.input_buffer, obs_data)  # 比直接赋值更安全
        
        # 极简推理（比原版快5-15%）
        return self.session.run(['actions'], {'obs': obs_data, 'time_step':np.array([[timestep]], dtype=np.float32)})[0]

 
    def timer_callback(self):
        
        # ptyhon 与 rclpy 多线程不太友好，这里使用定时间+简易状态机运行a
        if self.step == 0:
            self.robot_reset(1, False) # first reset
            print('robot reset 1!')
            self.step = 1
            return
        elif self.step == 1 and self.loop_count >= (1./self.dt): # 延迟10s
            self.robot_reset(2, True) # first reset
            print('robot reset 2!')
            self.loop_count = 0
            self.step = 2
            return
        
        if self.step == 1:
            soft_start = self.loop_count/(10./self.dt) # 1秒关节缓启动
            # soft_start = self.loop_count/(5./self.dt) # 1秒关节缓启动
            if soft_start > 1:
                soft_start = 1
                
            soft_joint_kp = joint_kp * soft_start * 0.2
            soft_joint_kd = joint_kd * 0.2
                
            msg = bxiMsg.ActuatorCmds()
            msg.header.frame_id = robot_name
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.actuators_name = joint_name
            # msg.pos = joint_nominal_pos.tolist()
            
            # 设置初始位置
            qpos = self.target_dof_pos
            msg.pos = qpos.tolist()
            
            # msg.pos = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.kp = soft_joint_kp.tolist()
            msg.kd = soft_joint_kd.tolist()
            self.act_pub.publish(msg)
            
        elif self.step == 2:
            with self.lock_in:
                q = self.qpos
                dq = self.qvel
                quat = self.quat
                omega = self.omega
                
            # 前两个时间步进行初始化
            if self.timestep < 2:

                ref_motion_quat = self.motionquat[self.timestep,0,:]
                yaw_motion_quat = yaw_quat(ref_motion_quat)
                yaw_motion_matrix = np.zeros(9)
                # mujoco.mju_quat2Mat(yaw_motion_matrix, yaw_motion_quat)
                yaw_motion_matrix = quaternion_to_rotation_matrix(yaw_motion_quat).reshape(3,3)
                # yaw_motion_matrix = yaw_motion_matrix.reshape(3,3)
                
                robot_quat = quat
                yaw_robot_quat = yaw_quat(robot_quat)
                yaw_robot_matrix = np.zeros(9)
                # mujoco.mju_quat2Mat(yaw_robot_matrix, yaw_robot_quat)
                yaw_robot_matrix = quaternion_to_rotation_matrix(yaw_robot_quat).reshape(3,3)
                yaw_robot_matrix = yaw_robot_matrix.reshape(3,3)
                self.init_to_world =  yaw_robot_matrix @ yaw_motion_matrix.T
                # 打印躯干四元数
            # print("quat_torso",robot_quat)
            
            if self.timestep < self.motionpos.shape[0]:
                # print("Inference timestep:", self.motionpos.shape[0]) #5650
                # if self.loop_count % self.control_decimation == 0:
                self.position = q #23
                self.quaternion = dq
                # print("qpos:", len(q))
                # print("qvel:", len(dq))
                self.motioninput = np.concatenate((self.motioninputpos[self.timestep,:],self.motioninputvel[self.timestep,:]),axis=0)
                self.motionposcurrent = self.motionpos[self.timestep,0,:]
                self.motionquatcurrent = self.motionquat[self.timestep,0,:]
                               
                # 计算相对四元数
                self.relquat = quaternion_multiply(matrix_to_quaternion_simple(self.init_to_world), self.motionquatcurrent)
                self.relquat = quaternion_multiply(quaternion_conjugate(self.quat),self.relquat)
                # 归一化四元数
                self.relquat = self.relquat / np.linalg.norm(self.relquat)
                # 转换为旋转矩阵并取前两列展平
                self.relmatrix = quaternion_to_rotation_matrix(self.relquat)[:,:2].reshape(-1,) 
                    
                # create observation
                offset = 0
                self.obs[offset:offset + 58] = self.motioninput
                offset += 58
                self.obs[offset:offset + 6] = self.relmatrix  
                offset += 6
                self.obs[offset:offset + 3] = omega #self.dyaw
                offset += 3
                # qpos_xml = q  # joint positions
                # qpos_seq = np.array([qpos_xml[joint_name.index(joint)] for joint in self.joint_seq])
                self.obs[offset:offset + num_actions] = q - self.joint_pos_array_seq  # joint positions
                offset += num_actions
                self.obs[offset:offset + num_actions] = dq  # joint velocities
                offset += num_actions   
                self.obs[offset:offset + num_actions] = self.action_buffer
                
                self.obs_input = self.obs.reshape(1, -1).astype(np.float32) # 将obs从(154,)变成(1,154)并确保数据类型
                # self.action = policy.run(['actions'], {'obs': self.obs_input, 'time_step':np.array([[self.timestep]], dtype=np.float32)})[0]
                # self.action = self.session.run(['actions'], {'obs': self.obs_input, 'time_step':np.array([[self.timestep]], dtype=np.float32)})[0]
                self.action = self.inference_step(self.obs_input , self.timestep)
                self.action = np.asarray(self.action).reshape(-1)
                self.action_buffer = self.action.copy()
                
                self.target_dof_pos = self.action * self.action_scale + self.joint_pos_array
                self.target_dof_pos = self.target_dof_pos.reshape(-1,)
                self.target_dof_pos = np.array([self.target_dof_pos[self.joint_seq.index(joint)] for joint in joint_name])
                
                self.timestep += 1
                # print("timestep:", self.timestep)
                msg = bxiMsg.ActuatorCmds()
                msg.header.frame_id = robot_name
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.actuators_name = joint_name
                msg.pos = self.target_dof_pos.tolist()
                # msg.pos = qpos.tolist()
                msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
                msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
                
                msg.kp = (0.8*self.stiffness_array).tolist()   # 刚度0.8
                msg.kd = (0.2*self.damping_array).tolist()    # 阻尼0.2
                # print("kp:", msg.kp)
                # print("kd:", msg.kd)
                # msg.kp = (0.45*joint_kp).tolist()
                # msg.kd = (0.45*joint_kd).tolist()
                
                # msg.kp = (1*joint_kp).tolist()
                # msg.kd = (1*joint_kd).tolist()
                
                #发送指令
                self.act_pub.publish(msg)
                    # self.last_action=self.action.copy()
            
            if self.timestep >= self.motionpos.shape[0]:
                print("Motion replay finished, resetting simulation.")
                self.timestep = 0
        self.loop_count += 1
    
    def robot_reset(self, reset_step, release):
        req = bxiSrv.RobotReset.Request()
        req.reset_step = reset_step
        req.release = release
        req.header.frame_id = robot_name
    
        while not self.rest_srv.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
            
        self.rest_srv.call_async(req)
        
    def sim_robot_reset(self):        
        req = bxiSrv.SimulationReset.Request()
        req.header.frame_id = robot_name

        base_pose = Pose()
        base_pose.position.x = 0.0
        base_pose.position.y = 0.0
        base_pose.position.z = 1.0
        base_pose.orientation.x = 0.0
        base_pose.orientation.y = 0.0
        base_pose.orientation.z = 0.0
        base_pose.orientation.w = 1.0        

        joint_state = JointState()
        joint_state.name = joint_name
        joint_state.position = np.zeros(dof_num, dtype=np.float32).tolist()
        joint_state.velocity = np.zeros(dof_num, dtype=np.float32).tolist()
        joint_state.effort = np.zeros(dof_num, dtype=np.float32).tolist()
        
        req.base_pose = base_pose
        req.joint_state = joint_state
    
        while not self.sim_rest_srv.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
            
        self.sim_rest_srv.call_async(req)
    
    def joint_callback(self, msg):
        joint_pos = msg.position
        joint_vel = msg.velocity
        joint_tor = msg.effort
        # print(msg)
        with self.lock_in:
            # self.qpos[4] -= ankle_y_offset
            # self.qpos[10] -= ankle_y_offset
            
            # self.qpos[:(3+12+4)] = np.array(joint_pos[:(3+12+4)])
            # self.qpos[-4:] = np.array(joint_pos[-7:-3])
            self.qpos = np.array(joint_pos)
            
            # self.qvel[:(3+12+4)] = np.array(joint_vel[:(3+12+4)])
            # self.qvel[-4:] = np.array(joint_vel[-7:-3])
            self.qvel = np.array(joint_vel)

    def joy_callback(self, msg):
        with self.lock_in:
            self.vx = msg.vel_des.x * 2
            self.vx = np.clip(self.vx, -1.0, 2.0)
            self.vy = 0 #msg.vel_des.y
            self.dyaw = msg.yawdot_des
        
    def imu_callback(self, msg):
        quat = msg.orientation
        avel = msg.angular_velocity
        acc = msg.linear_acceleration

        # quat_tmp1 = np.array([quat.x, quat.y, quat.z, quat.w]).astype(np.double)
        quat_tmp1 = np.array([quat.w, quat.x, quat.y, quat.z]).astype(np.double)

        with self.lock_in:
            self.quat = quat_tmp1
            self.omega = np.array([avel.x, avel.y, avel.z])

    def touch_callback(self, msg):
        foot_force = msg.value
        
    def odom_callback(self, msg): # 全局里程计（上帝视角，仅限仿真使用）
        base_pose = msg.pose
        base_twist = msg.twist

def main(args=None):
   
    time.sleep(5)
    
    rclpy.init(args=args)
    node = BxiExample()
    
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        
    rclpy.shutdown()
        
if __name__ == '__main__':
    main()
    
