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
import json
from collections import deque
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

import onnxruntime as ort
import onnx

robot_name = "elf3"

dof_num = 29

dof_use = 29#26

num_actions = 29 #

num_obs = 154 #154

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

joint_nominal_pos_dance = np.array([  #第一帧舞蹈动作关节角度qpos
    -0.07626348,  0.14883403, -0.18536363,
    -0.0806383,   0.11905685, -0.15683933, 0.49942395, -0.24739617, -0.088495,
    -0.03730002, -0.02687607, -0.19142722, 0.43117728, -0.1995317,  0.0162572,
    0.04329295,  0.49213253, -0.42416662, 0.22063023,  0.08052273, -0.15644971, 0.32964338,
    0.12874779, -0.26010089, 0.51190308,  1.06329034,  0.21942973, -0.10307773, -0.17966156])

# ankle_y_offset = 0.0

joint_nominal_pos = np.array([   # 指定的固定关节角度
    0.0, 0.0, 0.0,
    -0.4,0.0,0.0,0.8,-0.4,0.0,
    -0.4,0.0,0.0,0.8,-0.4,0.0,
    0.5,0.3,-0.1,-0.2, 0.0,0.0,0.0,     # 左臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
    0.5,-0.3,0.1,-0.2, 0.0,0.0,0.0],    # 右臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
    dtype=np.float32)

joint_kp = np.array([     # 奔跑的关节kp，和joint_name顺序一一对应
    300,300,300,
    150,100,100,200,50,20,
    150,100,100,200,50,20,
    80,80,80,60, 20,20,20,
    80,80,80,60, 20,20,20], 
    dtype=np.float32)

joint_kd = np.array([  # 奔跑的关节kd，和joint_name顺序一一对应
    3,3,3,
    2,2,2,2.5,1,1,
    2,2,2,2.5,1,1,
    2,2,2,2, 1,1,1,
    2,2,2,2, 1,1,1], 
    dtype=np.float32)

class DanceMotionPolicy:
    """舞蹈动作策略管理类"""
    
    def __init__(self, motion_npz_path: str, model_onnx_path: str):
        """
        初始化舞蹈动作策略
        
        Args:
            motion_npz_path: 舞蹈动作数据文件路径(.npz)
            model_onnx_path: ONNX模型文件路径
            
        Use:
            ##1.初始化模型
            self.dance_policy = DanceMotionPolicy("path/to/motion.npz", "path/to/model.onnx")
            
            ##2.前两个时间步计算初始转换矩阵
            if self.dance_policy.timestep < 2:
                self.dance_policy.compute_init_to_world(quat, motion_quat)# robot_quat, motion_quat
                
            ##3.创建观测输入
            if self.dance_policy.timestep < self.dance_policy.motionpos.shape[0]:
                self.obs_input = self.dance_policy.create_obs_input(q, dq, quat, omega, motion_pos, motion_quat, motion_vel)
                
                ##4.推理动作
                self.target_dof_pos = self.dance_policy.inference_step(self.obs_input, self.dance_policy.timestep)
        """
        
        self.motion_npz_path = motion_npz_path
        
        self.model_onnx_path = model_onnx_path
        
        self.initialize_model(motion_npz_path, model_onnx_path)
        
    # 初始化部分（完整版）
    def initialize_model(self, npz_path, onnx_path):
        # 加载运动数据
        print("model init!!!")
        self.motion =  np.load(npz_path)
        self.motionpos = self.motion["body_pos_w"]
        self.motionquat = self.motion["body_quat_w"]
        self.motioninputpos = self.motion["joint_pos"]
        self.motioninputvel = self.motion["joint_vel"]
        print("Inference timestep:", self.motionpos.shape[0]) #总动作序列5650
        
        # 加载ONNX模型
        model = onnx.load(onnx_path)
        for prop in model.metadata_props:
            if prop.key == "joint_names":
                self.joint_seq = prop.value.split(",")
            if prop.key == "default_joint_pos":   
                self.joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
                self.joint_pos_array = np.array([self.joint_pos_array_seq[self.joint_seq.index(joint)] for joint in joint_name])

            if prop.key == "joint_stiffness":
                self.stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
                self.stiffness_array = np.array([self.stiffness_array_seq[self.joint_seq.index(joint)] for joint in joint_name])
                
            if prop.key == "joint_damping":
                self.damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
                self.damping_array = np.array([self.damping_array_seq[self.joint_seq.index(joint)] for joint in joint_name])        
            
            if prop.key == "action_scale":
                self.action_scale = np.array([float(x) for x in prop.value.split(",")])
            # print(f"{prop.key}: {prop.value}")#查看metadata_props内容
        
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
            onnx_path,
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
        
        #模型测试
        print("policy test start!!!")
        self.timestep = 0
        self.obs = np.zeros(num_obs, dtype=np.float32)
        self.obs_input = self.obs.reshape(1, -1).astype(np.float32) # 将obs从(154,)变成(1,154)并确保数据类型
        self.inference_step(self.obs_input , self.timestep)# 预推理一次
        self.action_buffer = np.zeros((num_actions,), dtype=np.float32)
        print("policy test finished!!!")

    # 循环推理部分（极速版）
    def inference_step(self, obs_data, timestep):
        # 使用预分配内存（如果适用）
        np.copyto(self.input_buffer, obs_data)  # 比直接赋值更安全
        self.action = self.session.run(['actions'], {'obs': obs_data, 'time_step':np.array([[timestep]], dtype=np.float32)})[0]
        
        self.action = np.asarray(self.action).reshape(-1)
        self.action_buffer = self.action.copy()
        
        self.target_dof_pos = self.action * self.action_scale + self.joint_pos_array_seq
        self.target_dof_pos = self.target_dof_pos.reshape(-1,)
        self.target_dof_pos = np.array([self.target_dof_pos[self.joint_seq.index(joint)] for joint in joint_name])
        # 极简推理（比原版快5-15%）
        return self.target_dof_pos

    # 计算初始到世界坐标系的转换矩阵
    def compute_init_to_world(self, robot_quat, motion_quat):
        yaw_motion_quat = self.yaw_quat(motion_quat)
        yaw_motion_matrix = np.zeros(9)
        yaw_motion_matrix = self.quaternion_to_rotation_matrix(yaw_motion_quat).reshape(3,3)
        
        yaw_robot_quat = self.yaw_quat(robot_quat)
        yaw_robot_matrix = np.zeros(9)
        yaw_robot_matrix = self.quaternion_to_rotation_matrix(yaw_robot_quat).reshape(3,3)
        yaw_robot_matrix = yaw_robot_matrix.reshape(3,3)
        self.init_to_world =  yaw_robot_matrix @ yaw_motion_matrix.T

    # 计算相对旋转矩阵    
    def compute_relmatrix(self, robot_quat, motion_quat):
        rel_quat = self.quaternion_multiply(self.matrix_to_quaternion_simple(self.init_to_world), motion_quat)
        rel_quat = self.quaternion_multiply(self.quaternion_conjugate(robot_quat),rel_quat)
        rel_quat = rel_quat / np.linalg.norm(rel_quat) # 归一化四元数
        rel_matrix = self.quaternion_to_rotation_matrix(rel_quat)[:,:2].reshape(-1,)  # 转换为旋转矩阵并取前两列展平
        return rel_matrix
 
    # 创建观测输入   
    def create_obs_input(self,q, dq, quat, omega, motion_pos, motion_quat, motion_vel):
        # create observation
        offset = 0
        motioninput = np.concatenate((motion_pos,motion_vel),axis=0)
        self.obs[offset:offset + 58] = motioninput
        
        offset += 58
        relmatrix = self.compute_relmatrix(quat, motion_quat)
        self.obs[offset:offset + 6] = relmatrix  
        
        offset += 6
        self.obs[offset:offset + 3] = omega 
        
        offset += 3
        qpos_seq = np.array([q[joint_name.index(joint)] for joint in self.joint_seq])
        self.obs[offset:offset + num_actions] = qpos_seq - self.joint_pos_array_seq  # joint positions
        
        offset += num_actions
        qvel_seq = np.array([dq[joint_name.index(joint)] for joint in self.joint_seq])
        self.obs[offset:offset + num_actions] = qvel_seq  # joint velocities
        
        offset += num_actions   
        self.obs[offset:offset + num_actions] = self.action_buffer
        
        self.obs_input = self.obs.reshape(1, -1).astype(np.float32) # 将obs从(154,)变成(1,154)并确保数据类型
        
        return self.obs_input
    
    # 定义函数：提取四元数的偏航角分量
    def yaw_quat(self, q):
        w, x, y, z = q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

    # 定义函数：计算四元数的共轭
    def quaternion_conjugate(self, q):
        """四元数共轭: [w, x, y, z] -> [w, -x, -y, -z]"""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    # 定义函数：计算两个四元数的乘积
    def quaternion_multiply(self, q1, q2):
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
    def quaternion_to_rotation_matrix(self, q):
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

class BxiExample(Node):
    
    def __init__(self):

        super().__init__('bxi_example_py')
        
        self.declare_parameter('/topic_prefix', 'default_value')
        self.topic_prefix = self.get_parameter('/topic_prefix').get_parameter_value().string_value
        print('topic_prefix:', self.topic_prefix)
        
        self.declare_parameter('/npz_file_dict', json.dumps({}))
        npz_file_json = self.get_parameter('/npz_file_dict').value
        self.npz_file_dict = json.loads(npz_file_json)
        print('npz_file:')
        for key,value in self.npz_file_dict.items():
            print("Load motion from ",key,": ",value)
            
        self.declare_parameter('/onnx_file_dict', json.dumps({}))
        onnx_file_json = self.get_parameter('/onnx_file_dict').value
        self.onnx_file_dict = json.loads(onnx_file_json)
        print('onnx_file:')
        for key,value in self.onnx_file_dict.items():
            print("Load model from ",key,": ",value)

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
        
        self.dance_jojo = DanceMotionPolicy(self.npz_file_dict["jojo"], self.onnx_file_dict["jojo"])

        self.step = 0
        self.loop_count = 0
        self.dt = 0.01  # loop @100Hz
        # self.dt = 0.02  # loop 模型时间1/dt=50Hz
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.timer_callback_group_1)
    
    def timer_callback(self):
        # ptyhon 与 rclpy 多线程不太友好，这里使用定时间+简易状态机运行a
        if self.step == 0:
            self.robot_reset(1, False) # first reset
            print('robot reset 1!')
            self.step = 1
            return
        elif self.step == 1 and self.loop_count >= (6./self.dt): # 6秒启动总时间
            self.robot_reset(2, True) # first reset
            print('robot reset 2!')
            self.loop_count = 0
            self.step = 2
            print("Dance motion start!")
            return
        
        if self.step == 1:
            soft_start = self.loop_count/(3./self.dt) # 3秒关节缓启动
            if soft_start > 1:
                soft_start = 1
                
            soft_joint_kp = joint_kp * soft_start #* 0.2
            soft_joint_kd = joint_kd #* 0.2
                
            msg = bxiMsg.ActuatorCmds()
            msg.header.frame_id = robot_name
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.actuators_name = joint_name
            
            # 设置初始位置
            msg.pos = self.dance_jojo.motioninputpos[0,:].tolist()#取第一帧动作作为初始位置
            # msg.pos = joint_nominal_pos_dance.tolist()#取预设的舞蹈初始位置
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
                
                motion_quat = self.dance_jojo.motionquat[self.dance_jojo.timestep,0,:]
                motion_pos = self.dance_jojo.motioninputpos[self.dance_jojo.timestep,:]
                motion_vel = self.dance_jojo.motioninputvel[self.dance_jojo.timestep,:]
                
            # 前两个时间步计算初始转换矩阵
            if self.dance_jojo.timestep < 2:
                self.dance_jojo.compute_init_to_world(quat, motion_quat)# robot_quat, motion_quat
            
            if self.dance_jojo.timestep < self.dance_jojo.motionpos.shape[0]:
                self.obs_input = self.dance_jojo.create_obs_input(q, dq, quat, omega, motion_pos, motion_quat, motion_vel)
                self.target_dof_pos = self.dance_jojo.inference_step(self.obs_input , self.dance_jojo.timestep)
                
                # 发布关节控制指令
                msg = bxiMsg.ActuatorCmds()
                msg.header.frame_id = robot_name
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.actuators_name = joint_name
                msg.pos = self.target_dof_pos.tolist()
                msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
                msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
                msg.kp = self.dance_jojo.stiffness_array.tolist()   # 刚度
                msg.kd = (0.2*self.dance_jojo.damping_array).tolist()    # 阻尼0.2,xml文件中额外添加了阻尼参数,需要调小
                
                #发送指令
                self.act_pub.publish(msg)
                
                # self.timestep =5600
                self.dance_jojo.timestep += 1
                # print("timestep:", self.timestep)
                
            if self.dance_jojo.timestep >= self.dance_jojo.motionpos.shape[0]:
                print("Motion replay finished, resetting simulation.")
                self.dance_jojo.timestep = 0
                
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
    
