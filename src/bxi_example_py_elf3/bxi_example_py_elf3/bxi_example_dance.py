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
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

import os
import sys
import math
import json
import time
import datetime
try:
    import pygame
    _pygame_available = True
except ImportError:
    _pygame_available = False
# import torch
import numpy as np
from threading import Lock
from collections import deque

from bxi_example_py_elf3.models.beyondmimic import DanceMotionPolicy, DanceMotionPolicyGravity, DanceMotionPolicyGravityIsaaclab, DanceMotionPolicyGravityIsaaclabV2, DanceMotionPolicyGravityIsaaclabV3
from bxi_example_py_elf3.models.host import TumbleRecoverPolicy
from bxi_example_py_elf3.models.amp import HumanoidGaitPolicy , HumanoidGaitPolicyLite
from bxi_example_py_elf3.utils.tfs import get_gravity_orientation, quaternion_to_euler_array

robot_name = "elf3"

dof_num = 29

dof_use = 29#26

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


class robotState:
    stand = 1
    stand_to_motion = 2
    
    motion = 3
    motion_to_stand = 4
    
    tumble = 5
    tumble_to_stand = 6
    
class motionType:
    walk = 1
    run = 2
    dance_jojo = 3
    dance_walk = 4
    dance_ydd = 5
    dance_d1s2 = 6
    amp_walk = 7
    amp_run = 8
    #dance_getup = 9
    dance_fall_getup = 10
    dance_lie_down = 11
    dance_goodtime = 12   
    dance_backflip = 13   
    dance_webster = 14   
    dance_shuishou = 15   
    dance_lichenxi = 16   
    dance_forwardflip = 17   
    dance_sideflip = 18   
    dance_balei = 19   
    dance_dingdongji = 20   
    dance_guofuchen = 21   
    dance_jinwumen = 22   
    dance_jixiewu = 23   
    dance_change_face = 24   
    dance_face1 = 25   
    dance_face2 = 26   
    dance_face3 = 27   
    dance_face4 = 28   
    dance_face5 = 29   
    dance_face6 = 30   

class BxiExample(Node):
    
    def __init__(self):

        super().__init__('bxi_example_py')
        
        self.load_files()
        
        # self.setup_logging()
        
        self.init_pub_sub()
        
        self.init_controller()

        # 机器人状态变量
        self.qpos = np.zeros(dof_num,dtype=np.double)
        self.qvel = np.zeros(dof_num,dtype=np.double)
        self.omega = np.zeros(3,dtype=np.double)
        self.quat = np.zeros(4,dtype=np.double)   
        
        # 状态机相关变量
        self.stand_to_motion_counter = None
        self.motion_to_stand_counter = None
        self.tumble_to_stand_counter = None
        self.state = robotState.stand
        # self.state = robotState.tumble
        # self.motion_type = None
        
        self.init_models()
        
        # 初始动作类型
        # self.motion_type = motionType.dance_jojo
        # self.motion_type = motionType.dance_walk
        # self.motion_type = motionType.dance_ydd
        # self.motion_type = motionType.dance_d1s2
        # self.motion_type = motionType.dance_fall_getup
        
        self.motion_type = motionType.amp_walk
        # self.motion_type = motionType.dance_backflip
        
        # 软启动参数
        self.start_frame_pos = self.amp_walk.default_dof_pos
        
        self.soft_start_kps = self.amp_walk.kps 
        self.soft_start_kds = self.amp_walk.kds
        
        # 定时器回调
        self.step = 0
        self.loop_count = 0
        # self.dt = 0.01  # loop @100Hz
        self.dt = 0.02  # loop 模型时间1/dt=50Hz
        # self.dt = 0.04  # loop 模型时间1/dt=50Hz
        # self.control_decimation = 2
        self.dance_flag = 1
        # self.stand_flag = 1
        # self.stand_flag = 0
        
        if self.use_hardware:
            self.keyboard_use = False
        else:
            # self.keyboard_use = True
            self.keyboard_use = False
        # 检查pygame可用性
        if not _pygame_available:
            print("[警告] 未检测到pygame，键盘控制功能不可用。请通过 pip install pygame 安装。")
            self.keyboard_use = False
        self.init_keyboard()
        
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.timer_callback_group_1)
        
    def load_files(self):
        self.declare_parameter('/use_hardware') # 声明 use_hardware 参数，默认 False
        self.use_hardware = self.get_parameter('/use_hardware').value
        
        self.declare_parameter('/topic_prefix', 'default_value')
        self.topic_prefix = self.get_parameter('/topic_prefix').get_parameter_value().string_value
        # print('topic_prefix:', self.topic_prefix)
        
        self.declare_parameter('/npz_file_dict', json.dumps({}))
        npz_file_json = self.get_parameter('/npz_file_dict').value
        self.npz_file_dict = json.loads(npz_file_json)
        # print('npz_file:')
        # for key,value in self.npz_file_dict.items():
            # print("Load motion from ",key,": ",value)
            
        self.declare_parameter('/onnx_file_dict', json.dumps({}))
        onnx_file_json = self.get_parameter('/onnx_file_dict').value
        self.onnx_file_dict = json.loads(onnx_file_json)

        # 模型切换过渡时长（秒），可在 launch 时配置；<=0 表示关闭混合
        # self.declare_parameter('/transition_time', 0.3)
        self.declare_parameter('/transition_time', 0.4)
        # self.declare_parameter('/transition_time', 0.5)
        # self.declare_parameter('/transition_time', 0.6)
        self._param_transition_time = float(self.get_parameter('/transition_time').value)
        # print('onnx_file:')
        # for key,value in self.onnx_file_dict.items():
            # print("Load model from ",key,": ",value)

    def setup_logging(self):
        """根据 use_hardware 参数将日志保存到对应目录"""
        if self.use_hardware:
            log_dir = "log/bxi_real_log"
        else:
            log_dir = "log/bxi_sim_log"
        
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"{timestamp}_elf_dance.log")
        
        print(f"Logging to: {log_file}")
        
        # 将 stdout 和 stderr 重定向到日志文件
        self.log_file_handle = open(log_file, 'w')
        sys.stdout = self.log_file_handle
        sys.stderr = self.log_file_handle
        
        print(f"Log started at {timestamp}, use_hardware={self.use_hardware}")

    def init_pub_sub(self):
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
        self.timer_callback_group_2 = MutuallyExclusiveCallbackGroup()

        self.lock_in = Lock()
        self.lock_ou = self.lock_in #Lock()
    
    def init_controller(self):
        self.vae_vel = np.zeros(3, dtype=np.float32)
        
        # 运动命令变量
        self.vx = 0.0
        self.vy = 0.0
        self.dyaw = 0.0
        self.stand_height = 1.0
        
        # 速度偏移变量
        self.vx_offset = 0.0
        self.vy_offset = 0.0
        self.dyaw_offset = 0.0
        # self.vy_offset = 0.1
        # self.dyaw_offset = 1.5
        
        # 遥控器相关变量
        self.motion_a_prev = False
        self.motion_x_prev = False
        self.motion_y_prev = False
        self.motion_b_prev = False
        self.motion_a_changed = False
        self.motion_x_changed = False
        self.motion_y_changed = False
        self.motion_b_changed = False

        # X 按键防抖：变化触发后，缓冲期内再次变化不更新 motion_x_changed
        # 缓冲时长 0.3s，可按需调整
        self._motion_x_debounce = 0.5  # 秒
        self._motion_x_debounce_until = -999.0

        # ABXY 按键统一防抖（与 X 共用同一缓冲时长）
        self._motion_a_debounce_until = -999.0
        self._motion_y_debounce_until = -999.0
        self._motion_b_debounce_until = -999.0

        # --- 模型切换平滑过渡状态 ---
        # 切换两个模型时，旧模型继续推理，与新模型按权重 alpha(0->1) 加权后发给电机。
        # transition_duration 单位为秒，可通过 ROS 参数 /transition_time 调整；
        # 若 <=0 则关闭过渡（保留旧版本即时切换行为）。
        self.transition_duration = float(getattr(self, '_param_transition_time', 0.4))
        self.transition_active = False
        self.transition_total_steps = 0
        self.transition_step_count = 0
        self.prev_motion_type = None
        self._capture_motor = False
        self._captured = None
        self._old_action = None
        self._blend_pending = False

    def init_models(self):
        #host模型
        self.recover = TumbleRecoverPolicy(self.onnx_file_dict["host"])
        
        # AMP模型
        self.amp_run = HumanoidGaitPolicyLite(self.onnx_file_dict["amp_run"])
        self.amp_walk = HumanoidGaitPolicyLite(self.onnx_file_dict["amp_walk"])
        
        # beyondmimic模型
        self.dance_walk = DanceMotionPolicy(self.npz_file_dict["walk1_subject1"], self.onnx_file_dict["walk1_subject1"], start_frame=150)#fixed policy
        self.dance_jojo = DanceMotionPolicy(self.npz_file_dict["jojo"], self.onnx_file_dict["jojo"], start_frame=150)#fixed policy
        self.dance_ydd = DanceMotionPolicy(self.npz_file_dict["ydd"], self.onnx_file_dict["ydd"], start_frame=100, fixed_pos=True)#fixed policy
        # self.dance_goodtime = DanceMotionPolicy(self.npz_file_dict["goodtime"], self.onnx_file_dict["goodtime"], start_frame=230, fixed_pos=True)#fixed policy
        self.dance_goodtime = DanceMotionPolicyGravityIsaaclabV3(self.npz_file_dict["goodtime"], self.onnx_file_dict["goodtime"], start_frame=230, fixed_pos=True)#fixed policy
        # self.dance_fall_getup = DanceMotionPolicy(self.npz_file_dict["fall_getup"], self.onnx_file_dict["fall_getup"], start_frame=600)#fixed policy
        self.dance_fall_getup = DanceMotionPolicyGravityIsaaclabV2(self.npz_file_dict["fall_getup"], self.onnx_file_dict["fall_getup"], start_frame=10, fixed_pos=False)#fixed policy
        self.dance_lie_down = DanceMotionPolicyGravityIsaaclabV2(self.npz_file_dict["lie_down"], self.onnx_file_dict["lie_down"], start_frame=100, fixed_pos=False)#fixed policy
        self.dance_webster = DanceMotionPolicy(self.npz_file_dict["webster"], self.onnx_file_dict["webster"], start_frame=70)#fixed policy
        # self.dance_backflip = DanceMotionPolicyGravityIsaaclab(self.npz_file_dict["backflip"], self.onnx_file_dict["backflip"], start_frame=40)#fixed policy
        self.dance_backflip = DanceMotionPolicyGravityIsaaclabV3(self.npz_file_dict["backflip"], self.onnx_file_dict["backflip"], start_frame=40)#fixed policy
        self.dance_forwardflip = DanceMotionPolicyGravityIsaaclab(self.npz_file_dict["forwardflip"], self.onnx_file_dict["forwardflip"], start_frame=150)#fixed policy
        self.dance_sideflip = DanceMotionPolicyGravityIsaaclab(self.npz_file_dict["sideflip"], self.onnx_file_dict["sideflip"], start_frame=150)#fixed policy
        self.dance_change_face = DanceMotionPolicyGravityIsaaclabV2(self.npz_file_dict["change_face"], self.onnx_file_dict["change_face"], start_frame=10, fixed_pos=True)#fixed policy
        self.dance_face3 = DanceMotionPolicyGravityIsaaclabV2(self.npz_file_dict["face3"], self.onnx_file_dict["face3"], start_frame=10, fixed_pos=True)#fixed policy
        # self.dance_change_face = DanceMotionPolicyGravityIsaaclabV3(self.npz_file_dict["change_face"], self.onnx_file_dict["change_face"], start_frame=10, fixed_pos=True)#fixed policy
        
        # self.dance_shuishou = DanceMotionPolicy(self.npz_file_dict["shuishou"], self.onnx_file_dict["shuishou"], start_frame=10, fixed_pos=True)#fixed policy
        self.dance_d1s2 = DanceMotionPolicyGravityIsaaclabV3(self.npz_file_dict["dance1_subject2"], self.onnx_file_dict["dance1_subject2"],start_frame=300, fixed_pos=True)#fixed policy
        self.dance_balei = DanceMotionPolicyGravityIsaaclabV3(self.npz_file_dict["balei"], self.onnx_file_dict["balei"], start_frame=10, fixed_pos=True)#fixed policy
        self.dance_guofuchen = DanceMotionPolicyGravityIsaaclabV3(self.npz_file_dict["guofuchen"], self.onnx_file_dict["guofuchen"], start_frame=50, fixed_pos=True)#fixed policy
        self.dance_jinwumen = DanceMotionPolicyGravityIsaaclabV3(self.npz_file_dict["jinwumen"], self.onnx_file_dict["jinwumen"], start_frame=10, fixed_pos=True)#fixed policy
        self.dance_shuishou = DanceMotionPolicyGravityIsaaclabV3(self.npz_file_dict["shuishou"], self.onnx_file_dict["shuishou"], start_frame=10, fixed_pos=True)#fixed policy
        self.dance_dingdongji = DanceMotionPolicyGravityIsaaclabV3(self.npz_file_dict["dingdongji"], self.onnx_file_dict["dingdongji"], start_frame=50, fixed_pos=True)#fixed policy
        # self.dance_jixiewu = DanceMotionPolicyGravityIsaaclabV3(self.npz_file_dict["jixiewu"], self.onnx_file_dict["jixiewu"], start_frame=150, fixed_pos=True)#fixed policy
        self.dance_jixiewu = DanceMotionPolicyGravityIsaaclabV2(self.npz_file_dict["jixiewu"], self.onnx_file_dict["jixiewu"], start_frame=150, fixed_pos=True)#fixed policy
        self.dance_lichenxi = DanceMotionPolicyGravityIsaaclabV3(self.npz_file_dict["lichenxi"], self.onnx_file_dict["lichenxi"], start_frame=400, fixed_pos=True)#fixed policy
        # self.dance_lichenxi = DanceMotionPolicyGravity(self.npz_file_dict["lichenxi"], self.onnx_file_dict["lichenxi"], start_frame=300, fixed_pos=True)#fixed policy
        # self.dance_lichenxi = DanceMotionPolicyGravity(self.npz_file_dict["lichenxi"], self.onnx_file_dict["lichenxi"], start_frame=400, fixed_pos=True)#fixed policy
        
        self.dance_webster.end_frame = self.dance_webster.end_frame - 40
        # self.dance_backflip.end_frame = self.dance_backflip.end_frame - 20
        self.dance_backflip.end_frame = self.dance_backflip.end_frame - 40
        
    def timer_callback(self):
        # ptyhon 与 rclpy 多线程不太友好，这里使用定时间+简易状态机运行a
        if self.step == 0:
            self.robot_reset(1, False) # first reset
            # self.sim_robot_reset()
            print('robot reset 1!')
            self.step = 1
            return
        # elif self.step == 1 and self.loop_count >= (6./self.dt): # 6秒启动总时间
        #     self.robot_reset(2, True) # first reset
        #     if not self.use_hardware:
        #         # self.sim_robot_reset()
        #         pass
        #     print('robot reset 2!')
        #     self.loop_count = 0
        #     if self.use_hardware:
        #         self.step = 2
        #     print("Dance motion start!")
        #     return
        if self.step == 1: #软启动
            soft_start = self.loop_count/(3./self.dt) # 3秒关节缓启动
            if soft_start > 1:
                soft_start = 1
            #软启动到舞蹈动作的第一帧    
            soft_joint_kp = self.soft_start_kps * soft_start
            soft_joint_kd = self.soft_start_kds
               
            self.send_to_motor(self.start_frame_pos, soft_joint_kp, soft_joint_kd)
                    
        elif self.step == 2:
            # 参数读取
            with self.lock_in:
                q = self.qpos
                dq = self.qvel
                quat = self.quat
                omega = self.omega
                cmd_vel = np.array([self.vx, self.vy, self.dyaw])

            #获取欧拉角
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
                            
            # 状态机
            if self.state==robotState.stand:
                self.state = robotState.stand_to_motion
                    # self.motion_type = motionType.dance_walk
                    # self.dance_agent.reset()
                    # self.dance_agent.motion_playing = False
                print("state: stand_to_motion [dance]")
                
            elif self.state==robotState.stand_to_motion:
                #动作过渡
                self.state=robotState.motion
                # pass
                # self.stand_to_motion_counter.step()
                # if self.stand_to_motion_counter.finished:
                #     self.state=robotState.motion
                #     self.stand_to_motion_counter = None
                #     if self.motion_type == motionType.dance:
                #         self.dance_jojo.motion_playing = True
                #         print("state: motion [dance]")

            elif self.state==robotState.motion:
                #跌到检测
                if (np.abs(eu_ang[1]) > (math.pi/2.5)) or (np.abs(eu_ang[2]) > (math.pi/2.5)):
                    # print("robot tumble!")
                    if self.use_hardware:
                        # os._exit() #急杀真机
                        pass
                    else:
                        # self.state = robotState.tumble
                        pass
                    
                # --- 模型切换平滑过渡（双模型加权） ---
                if self.transition_active and self.prev_motion_type is not None \
                        and self.prev_motion_type != self.motion_type:
                    self._capture_motor = True
                    self._captured = None
                    _saved_mt = self.motion_type
                    self.motion_type = self.prev_motion_type
                    try:
                        self._run_motion_dispatch(q, dq, quat, omega, cmd_vel)
                    finally:
                        self.motion_type = _saved_mt
                        self._capture_motor = False
                    self._old_action = self._captured
                    self._blend_pending = self._old_action is not None
                else:
                    self._blend_pending = False
                    self._old_action = None

                self._run_motion_dispatch(q, dq, quat, omega, cmd_vel)

                # 推进过渡进度
                if self.transition_active:
                    self.transition_step_count += 1
                    if self.transition_step_count >= self.transition_total_steps:
                        self.transition_active = False
                        self.prev_motion_type = None
                        self._old_action = None
                        print(f"motion transition finished -> {self.motion_type}")

            elif self.state==robotState.motion_to_stand:
                #站立过渡
                self.state=robotState.stand
                print("state: stand")
    
            elif self.state==robotState.tumble:
                #起身过渡
                # self.state=robotState.stand
                self.target_dof_pos = self.recover.inference_step(q, dq, quat, omega)
                self.send_to_motor(self.target_dof_pos, self.recover.kps, self.recover.kds)
                #check safe
                if self.loop_count%100==0:#2s判断一次起身
                    if (np.abs(eu_ang[1]) < (math.pi/5.0)) and (np.abs(eu_ang[2]) < (math.pi/5.0)):
                        print("robot recover!")
                        if self.loop_count%100==0:#2s稳定
                            # self.state = robotState.stand
                            self.state = robotState.tumble_to_stand
                            # self.dance_jojo.timestep = 700
                            self.loop_count = 0
            
            elif self.state==robotState.tumble_to_stand:          
                soft_start = self.loop_count/(0.2/self.dt) # 0.1秒过渡到站立动作
                if soft_start > 1:
                    soft_start = 1
                    self.state = robotState.stand
                #软启动到舞蹈动作的第一帧    
                soft_joint_kp = self.soft_start_kps * soft_start
                soft_joint_kd = self.soft_start_kds 
                
                self.send_to_motor(self.start_frame_pos, soft_joint_kp, soft_joint_kd)
            else:
                #其他状态机情况
                raise Exception   

        self.loop_count += 1
    
    def joy_callback(self, msg):
        with self.lock_in:
            if self.motion_type == motionType.amp_walk:
                self.vx = np.clip(msg.vel_des.x, -0.6, 1.0)
            else:
                # self.vx = np.clip(msg.vel_des.x, -1.0, 3.0)
                # self.vx = np.clip(msg.vel_des.x, -1.0, 4.0)
                # self.vx = np.clip(msg.vel_des.x, -1.0, 4.5)
                self.vx = np.clip(msg.vel_des.x, -1.0, 5.0)
                # self.vx = np.clip(msg.vel_des.x, -1.0, 5.5)
                # self.vx = np.clip(msg.vel_des.x, -1.0, 6.0)
   
            self.vy = msg.vel_des.y
            self.dyaw = msg.yawdot_des
            
            # stand_height = msg.height_des
            # stand_height = min(stand_height, 3.0)
            # stand_height = max(stand_height, 1.0)
            # self.stand_height = stand_height

            motion_a = msg.btn_5 # A
            motion_x = msg.btn_6 # X
            motion_y = msg.btn_7 # Y
            motion_b = msg.btn_10 # B

            #防止误触
            if self.step < 2:
                # self.motion_a_prev = motion_a
                # self.motion_x_prev = motion_x
                self.motion_y_prev = motion_y
                self.motion_b_prev = motion_b
            if self.step < 1:
                self.motion_a_prev = motion_a
                self.motion_x_prev = motion_x
                self.motion_y_prev = motion_y
                self.motion_b_prev = motion_b
                
            #按键状态变化检测
            _now = self.get_clock().now().nanoseconds * 1e-9
            def _debounced(changed, until_attr):
                if changed:
                    if _now >= getattr(self, until_attr):
                        setattr(self, until_attr, _now + self._motion_x_debounce)
                        return True
                    return False
                return False

            self.motion_a_changed = _debounced(motion_a != self.motion_a_prev, '_motion_a_debounce_until')
            self.motion_x_changed = _debounced(motion_x != self.motion_x_prev, '_motion_x_debounce_until')
            self.motion_y_changed = _debounced(motion_y != self.motion_y_prev, '_motion_y_debounce_until')
            self.motion_b_changed = _debounced(motion_b != self.motion_b_prev, '_motion_b_debounce_until')
            # print(f"Received motion command: A={motion_a} (changed: {self.motion_a_changed}), X={motion_x} (changed: {self.motion_x_changed}), Y={motion_y} (changed: {self.motion_y_changed}), B={motion_b} (changed: {self.motion_b_changed})")
            
            #按键状态保存
            self.motion_a_prev = motion_a
            self.motion_x_prev = motion_x
            self.motion_y_prev = motion_y
            self.motion_b_prev = motion_b
            
            use_button1 = True
            # use_button1 =  False
            
            if use_button1: 
                # 常用按键组合：A键切换amp_walk，X/Y/B键切换三个舞蹈动作
                if self.motion_a_changed == 1:
                    if self.step < 2:
                        self.robot_reset(2, True) # first reset
                        self.step = 2
                    if self.motion_type != motionType.amp_walk:
                        self.switch_to_motion(self.amp_walk, motionType.amp_walk, num=20, with_cmd_vel=True)
                    else:
                        self.motion_type = motionType.amp_walk

                elif self.motion_x_changed == 1:
                    # self.dance_flag += 1
                    # if self.dance_flag > 1:
                    #     self.dance_flag = 0
                    # if self.motion_type == motionType.amp_walk:
                    #     self.dance_flag = 1
                    #     self.dance_dingdongji.timestep = self.dance_dingdongji.start_frame
                    #     self.dance_dingdongji.timeinit = 0.0
                    #     self.switch_to_motion(self.dance_dingdongji, motionType.dance_dingdongji, num=20)
                        
                    # self.dance_flag += 1
                    # if self.dance_flag > 1:
                    #     self.dance_flag = 0
                    # if self.motion_type == motionType.amp_walk:
                    #     self.dance_flag = 1
                    #     self.dance_face3.timestep = self.dance_face3.start_frame
                    #     self.dance_face3.timeinit = 0.0
                    #     self.switch_to_motion(self.dance_face3, motionType.dance_face3, num=20)
                        
                    # self.dance_flag += 1
                    # if self.dance_flag > 1:
                    #     self.dance_flag = 0
                    # if self.motion_type == motionType.amp_walk:
                    #     self.dance_flag = 1
                    #     self.dance_shuishou.timestep = self.dance_shuishou.start_frame
                    #     self.dance_shuishou.timeinit = 0.0
                    #     self.switch_to_motion(self.dance_shuishou, motionType.dance_shuishou, num=20)
                        
                    self.dance_flag += 1
                    if self.dance_flag > 1:
                        self.dance_flag = 0
                    if self.motion_type == motionType.amp_walk:
                        self.dance_flag = 1
                        self.dance_jixiewu.timestep = self.dance_jixiewu.start_frame
                        self.dance_jixiewu.timeinit = 0.0
                        self.switch_to_motion(self.dance_jixiewu, motionType.dance_jixiewu, num=20)
                    
                    # self.dance_flag += 1
                    # if self.dance_flag > 1:
                    #     self.dance_flag = 0
                    # if self.motion_type == motionType.amp_walk:
                    #     self.dance_flag = 1
                    #     self.dance_goodtime.timestep = self.dance_goodtime.start_frame
                    #     self.dance_goodtime.timeinit = 0.0
                    #     self.switch_to_motion(self.dance_goodtime, motionType.dance_goodtime, num=20)
                    
                    # self.dance_flag += 1
                    # if self.dance_flag > 1:
                    #     self.dance_flag = 0
                    # if self.motion_type == motionType.amp_walk:
                    #     self.dance_flag = 1
                    #     self.dance_change_face.timestep = self.dance_change_face.start_frame
                    #     self.dance_change_face.timeinit = 0.0
                    #     self.switch_to_motion(self.dance_change_face, motionType.dance_change_face, num=20)
                        
                    # self.switch_to_motion(self.amp_run, motionType.amp_run, num=20, with_cmd_vel=True)
                       
                elif self.motion_y_changed == 1:
                    # self.dance_flag += 1
                    # if self.dance_flag > 1:
                    #     self.dance_flag = 0
                    # if self.motion_type == motionType.amp_walk:
                    #     self.dance_flag = 1
                    #     self.dance_lichenxi.timestep = self.dance_lichenxi.start_frame
                    #     self.dance_lichenxi.timeinit = 0.0
                    #     self.switch_to_motion(self.dance_lichenxi, motionType.dance_lichenxi, num=20)
                        
                    # self.dance_flag += 1
                    # if self.dance_flag > 1:
                    #     self.dance_flag = 0
                    # if self.motion_type == motionType.amp_walk:
                    #     self.dance_flag = 1
                    #     self.dance_jinwumen.timestep = self.dance_jinwumen.start_frame
                    #     self.dance_jinwumen.timeinit = 0.0
                    #     self.switch_to_motion(self.dance_jinwumen, motionType.dance_jinwumen, num=20)
                    
                    # self.dance_flag += 1
                    # if self.dance_flag > 1:
                    #     self.dance_flag = 0
                    # if self.motion_type == motionType.amp_walk:
                    #     self.dance_flag = 1
                    #     self.dance_balei.timestep = self.dance_balei.start_frame
                    #     self.dance_balei.timeinit = 0.0
                    #     self.switch_to_motion(self.dance_balei, motionType.dance_balei, num=20)
                        
                    # self.dance_flag += 1
                    # if self.dance_flag > 1:
                    #     self.dance_flag = 0
                    # if self.motion_type == motionType.amp_walk:
                    #     self.dance_flag = 1
                    #     self.dance_d1s2.timestep = self.dance_d1s2.start_frame
                    #     self.dance_d1s2.timeinit = 0.0
                    #     self.switch_to_motion(self.dance_d1s2, motionType.dance_d1s2, num=20)
                        
                    self.dance_flag += 1
                    if self.dance_flag > 1:
                        self.dance_flag = 0
                    if self.motion_type == motionType.amp_walk:
                        self.dance_flag = 1
                        self.dance_guofuchen.timestep = self.dance_guofuchen.start_frame
                        self.dance_guofuchen.timeinit = 0.0
                        self.switch_to_motion(self.dance_guofuchen, motionType.dance_guofuchen, num=20)

                # 切换到dance_backflip
                elif self.motion_b_changed == 1:
                    # self.dance_flag += 1
                    # if self.dance_flag > 1:
                    #     self.dance_flag = 0
                    # if self.motion_type == motionType.amp_walk:
                    
                    self.dance_flag = 1
                    self.dance_backflip.timestep = self.dance_backflip.start_frame
                    self.dance_backflip.timeinit = 0.0
                    self.switch_to_motion(self.dance_backflip, motionType.dance_backflip, num=20)
                    
                    # self.switch_to_motion(self.amp_run, motionType.amp_run, num=20, with_cmd_vel=True)
                    
                    
            
            else:
                # 起身按键组合        
                if self.motion_a_changed == 1:
                    if self.step < 2:
                        self.robot_reset(2, True) # first reset
                        self.step = 2
                    if self.motion_type != motionType.amp_walk:
                        self.switch_to_motion(self.amp_walk, motionType.amp_walk, num=20, with_cmd_vel=True)
                    else:
                        self.motion_type = motionType.amp_walk
                        
                elif self.motion_x_changed == 1:
                    if self.step < 2:
                        self.robot_reset(2, True) # first reset
                        self.step = 2
                    
                    self.select_fall_getup_motion()
                    
                    
                elif self.motion_y_changed == 1:
                    
                    self.dance_flag = 1
                    # self.dance_fall_getup.start_frame = 900
                    # self.dance_fall_getup.end_frame = 1350
                    # self.dance_fall_getup.start_frame = 850
                    # self.dance_fall_getup.end_frame = 1400
                    # self.dance_fall_getup.timestep = self.dance_fall_getup.start_frame
                    # self.preheat_model(self.dance_fall_getup, num=20)
                    # self.motion_type = motionType.dance_fall_getup
                    
                    # self.dance_lie_down.end_frame = 400
                    self.dance_lie_down.end_frame = 450
                    self.dance_lie_down.timestep = self.dance_lie_down.start_frame
                    self.preheat_model(self.dance_lie_down, num=20)
                    self.motion_type = motionType.dance_lie_down
                    
                    # self.dance_flag = 1
                    # self.dance_sideflip.timestep = self.dance_sideflip.start_frame
                    # self.switch_to_motion(self.dance_sideflip, motionType.dance_sideflip, num=20)
                    
                    
                elif self.motion_b_changed == 1:
                    # self.motion_type = motionType.amp_run
                    self.dance_flag = 1
                    self.dance_forwardflip.timestep = self.dance_forwardflip.start_frame
                    self.switch_to_motion(self.dance_forwardflip, motionType.dance_forwardflip, num=20)
                         
    def timer_callback2(self):
        """处理键盘输入的线程函数"""
        max_lin_vel_x = 0.3 + self.vx_offset  # 最大线速度 (m/s)
        max_lin_vel_y = 0.2 + self.vy_offset  # 最大线速度 (m/s)
        max_ang_vel = 0.4 + self.dyaw_offset  # 最大角速度 (rad/s)

        while not self.exit_flag and not self.shutdown_flag:
            try:
                keys = pygame.key.get_pressed()
                
                # 重置速度命令
                self.vx = self.vx_offset
                self.vy = self.vy_offset
                self.dyaw = self.dyaw_offset
                
                # 前进/后退 (W/S)
                if keys[pygame.K_w]:
                    self.vx = max_lin_vel_x + 0.1
                    print("前进:", self.vx)
                if keys[pygame.K_s]:
                    self.vx = -max_lin_vel_x - 0.1
                    print("后退:", self.vx)
                    
                # 左移/右移 (A/D)
                if keys[pygame.K_a]:
                    self.vy = max_lin_vel_y
                if keys[pygame.K_d]:
                    self.vy = -max_lin_vel_y
                    
                # 左转/右转 (Q/E)
                if keys[pygame.K_q]:
                    self.dyaw = max_ang_vel
                if keys[pygame.K_e]:
                    self.dyaw = -max_ang_vel
                        
                if keys[pygame.K_1]:
                    self.dance_flag = 1
                    time.sleep(0.2)

                if keys[pygame.K_2]:
                    self.dance_flag = 0
                    time.sleep(0.2)
                    
                if keys[pygame.K_3]:
                    self.motion_type = motionType.amp_walk
                    time.sleep(0.2)
                    
                if keys[pygame.K_4]:
                    self.motion_type = motionType.dance_ydd
                    self.dance_ydd.timestep = self.dance_ydd.start_frame
                    self.dance_flag = 1
                    time.sleep(0.2)
                    
                if keys[pygame.K_5]:
                    self.motion_type = motionType.dance_jojo
                    self.dance_jojo.timestep = self.dance_jojo.start_frame
                    self.dance_flag = 1
                    time.sleep(0.2)
                    
                if keys[pygame.K_6]:
                    self.motion_type = motionType.dance_d1s2
                    self.dance_d1s2.timestep = self.dance_d1s2.start_frame
                    self.dance_flag = 1
                    time.sleep(0.2)
        
                # 空格键停止所有运动
                if keys[pygame.K_SPACE]:
                    # self.dance_flag, self.stand_flag = 0, 0
                    self.dance_flag = 0
                    self.vx, self.vy, self.dyaw = 0.0, 0.0, 0.0
                    
                # 处理退出事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.exit_flag = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.exit_flag = True
                
                pygame.time.delay(50)
            except Exception as e:
                print(f"键盘处理错误: {e}")
                break

        # if self.keyboard_opened:
        #     pygame.quit()

    # 键盘控制初始化
    def init_keyboard(self):
        self.exit_flag = False
        self.shutdown_flag = False
        #pygame键盘处理线程
        if self.keyboard_use:
            self.timer2 = self.create_timer(self.dt, self.timer_callback2, callback_group=self.timer_callback_group_2)
            try:
                pygame.init()
                screen = pygame.display.set_mode((200, 100))
                pygame.display.set_caption("Keyboard Control")
                self.keyboard_opened = True
                print("键盘控制已启动。使用以下按键控制机器人：")
                print("W/S: 前进/后退")
                print("A/D: 左移/右移")
                print("Q/E: 左转/右转")
                #print("R/F: 升高/降低")
                print("空格键: 停止所有运动")
            except Exception as e:
                print(f"无法初始化键盘：{e}")
            
            self.exit_flag = False
    
    def send_to_motor(self, dof_pos_target, kps, kds):
        dof_pos_target = np.asarray(dof_pos_target, dtype=np.float32)
        kps = np.asarray(kps, dtype=np.float32)
        kds = np.asarray(kds, dtype=np.float32)

        # 切换过渡阶段：捕获旧模型输出，跳过实际发布
        if self._capture_motor:
            self._captured = (dof_pos_target.copy(), kps.copy(), kds.copy())
            return

        # 切换过渡阶段：把新模型输出与之前捕获的旧模型输出按权重混合
        if self._blend_pending and self._old_action is not None and self.transition_active:
            # t = min(1.0, max(0.0, (self.transition_step_count + 1) / max(1, self.transition_total_steps)))
            # alpha = t * t * (3.0 - 2.0 * t)  # smoothstep: slow start, slow end
            alpha = min(1.0, max(0.0, (self.transition_step_count + 1) / max(1, self.transition_total_steps)))
            old_pos, old_kps, old_kds = self._old_action
            if old_pos.shape == dof_pos_target.shape:
                dof_pos_target = (1.0 - alpha) * old_pos + alpha * dof_pos_target
            if old_kps.shape == kps.shape:
                kps = (1.0 - alpha) * old_kps + alpha * kps
            if old_kds.shape == kds.shape:
                kds = (1.0 - alpha) * old_kds + alpha * kds
            self._blend_pending = False  # 每个 tick 仅混合一次

        msg = bxiMsg.ActuatorCmds()
        msg.header.frame_id = robot_name
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.actuators_name = joint_name
        msg.pos = dof_pos_target.tolist()
        msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
        msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
        msg.kp = kps.tolist()
        msg.kd = kds.tolist()
        self.act_pub.publish(msg)   
    
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
        # base_pose.position.x = 0.0
        # base_pose.position.y = 0.0
        # base_pose.position.z = 1.0
        # base_pose.orientation.x = 0.0
        # base_pose.orientation.y = 0.0
        # base_pose.orientation.z = 0.0
        # base_pose.orientation.w = 1.0        
        
        #[0.707, 0.0, -0.707, 0.0]
        base_pose.position.x = 0.0
        base_pose.position.y = 0.0
        base_pose.position.z = 0.5
        base_pose.orientation.x = 0.0
        base_pose.orientation.y = -0.707
        base_pose.orientation.z = 0.0
        base_pose.orientation.w = 0.707  

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

    def _run_motion_dispatch(self, q, dq, quat, omega, cmd_vel):
        """运行当前 self.motion_type 对应的推理分支（输出会经 send_to_motor 发布/捕获）。"""
        if self.motion_type == motionType.dance_face3:
            if self.dance_face3.timestep <= self.dance_face3.end_frame:
                self.target_dof_pos = self.dance_face3.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_face3.kps, self.dance_face3.kds)

            # 动作管理    
            if self.dance_flag==1:
                # print("timestep:", self.dance_face3.timestep)
                self.dance_face3.timestep += 1
                
            # 动作结束检测    
            if self.dance_face3.timestep > self.dance_face3.end_frame:
                print("Motion replay finished, resetting simulation.")
                self.dance_face3.timestep = self.dance_face3.start_frame
                # self.motion_type = motionType.dance_walk
        
        if self.motion_type == motionType.dance_jojo:
            if self.dance_jojo.timestep <= self.dance_jojo.end_frame:
                self.target_dof_pos = self.dance_jojo.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_jojo.kps, self.dance_jojo.kds)

            # 动作管理    
            if self.dance_flag==1:
                # print("timestep:", self.dance_jojo.timestep)
                self.dance_jojo.timestep += 1
                
            # 动作结束检测    
            if self.dance_jojo.timestep > self.dance_jojo.end_frame:
                print("Motion replay finished, resetting simulation.")
                self.dance_jojo.timestep = self.dance_jojo.start_frame
                # self.motion_type = motionType.dance_walk
                
        if self.motion_type == motionType.dance_change_face:
            if self.dance_change_face.timestep <= self.dance_change_face.end_frame:
                self.target_dof_pos = self.dance_change_face.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_change_face.kps, self.dance_change_face.kds)

            # 动作管理    
            if self.dance_flag==1:
                # print("timestep:", self.dance_change_face.timestep)
                self.dance_change_face.timestep += 1
                
            # 动作结束检测    
            if self.dance_change_face.timestep > self.dance_change_face.end_frame:
                print("Motion replay finished, resetting simulation.")
                self.dance_change_face.timestep = self.dance_change_face.start_frame
                # self.motion_type = motionType.dance_walk
                
        if self.motion_type == motionType.dance_goodtime:
            if self.dance_goodtime.timestep <= self.dance_goodtime.end_frame:
                self.target_dof_pos = self.dance_goodtime.inference_step(q, dq, quat, omega)
                self.send_to_motor(self.target_dof_pos, self.dance_goodtime.kps, self.dance_goodtime.kds)
 
            if self.dance_flag==1:
                # print("timestep:", self.dance_goodtime.timestep)
                self.dance_goodtime.timestep += 1
                
            # 动作结束检测    
            if self.dance_goodtime.timestep > self.dance_goodtime.end_frame:
                # print("Motion replay finished, resetting simulation.")
                self.dance_goodtime.timestep = self.dance_goodtime.end_frame
                
        if self.motion_type == motionType.dance_backflip:
            if self.dance_backflip.timestep <= self.dance_backflip.end_frame:
                self.target_dof_pos = self.dance_backflip.inference_step(q, dq, quat, omega)
                self.send_to_motor(self.target_dof_pos, self.dance_backflip.kps, self.dance_backflip.kds)
 
            if self.dance_flag==1:
                # print("timestep:", self.dance_backflip.timestep)
                self.dance_backflip.timestep += 1
                
            # 动作结束检测    
            if self.dance_backflip.timestep > self.dance_backflip.end_frame:
                # print("Motion replay finished, resetting simulation.")
                # self.dance_backflip.timestep = self.dance_backflip.start_frame
                self.dance_backflip.timestep = self.dance_backflip.end_frame #停止动作
                # self.motion_type = motionType.amp_walk
                
        if self.motion_type == motionType.dance_sideflip:
            if self.dance_sideflip.timestep <= self.dance_sideflip.end_frame:
                self.target_dof_pos = self.dance_sideflip.inference_step(q, dq, quat, omega)
                self.send_to_motor(self.target_dof_pos, self.dance_sideflip.kps, self.dance_sideflip.kds)
 
            if self.dance_flag==1:
                # print("timestep:", self.dance_sideflip.timestep)
                self.dance_sideflip.timestep += 1
                
            # 动作结束检测    
            if self.dance_sideflip.timestep > self.dance_sideflip.end_frame:
                # print("Motion replay finished, resetting simulation.")
                # self.dance_sideflip.timestep = self.dance_sideflip.start_frame
                self.dance_sideflip.timestep = self.dance_sideflip.end_frame #停止动作
                # self.motion_type = motionType.amp_walk
                
        if self.motion_type == motionType.dance_forwardflip:
            if self.dance_forwardflip.timestep <= self.dance_forwardflip.end_frame:
                self.target_dof_pos = self.dance_forwardflip.inference_step(q, dq, quat, omega)
                self.send_to_motor(self.target_dof_pos, self.dance_forwardflip.kps, self.dance_forwardflip.kds)
 
            if self.dance_flag==1:
                # print("timestep:", self.dance_forwardflip.timestep)
                self.dance_forwardflip.timestep += 1
                
            # 动作结束检测    
            if self.dance_forwardflip.timestep > self.dance_forwardflip.end_frame:
                # print("Motion replay finished, resetting simulation.")
                # self.dance_forwardflip.timestep = self.dance_forwardflip.start_frame
                self.dance_forwardflip.timestep = self.dance_forwardflip.end_frame #停止动作
                # self.motion_type = motionType.amp_walk
                
        if self.motion_type == motionType.dance_webster:
            if self.dance_webster.timestep <= self.dance_webster.end_frame:
                self.target_dof_pos = self.dance_webster.inference_step(q, dq, quat, omega)
                self.send_to_motor(self.target_dof_pos, self.dance_webster.kps, self.dance_webster.kds)
 
            if self.dance_flag==1:
                # print("timestep:", self.dance_webster.timestep)
                self.dance_webster.timestep += 1
                
            # 动作结束检测    
            if self.dance_webster.timestep > self.dance_webster.end_frame:
                # print("Motion replay finished, resetting simulation.")
                # self.dance_webster.timestep = self.dance_webster.start_frame
                self.dance_webster.timestep = self.dance_webster.end_frame #停止动作
                # self.motion_type = motionType.amp_walk
        
        if self.motion_type == motionType.dance_shuishou:
            if self.dance_shuishou.timestep <= self.dance_shuishou.end_frame:
                self.target_dof_pos = self.dance_shuishou.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_shuishou.kps, self.dance_shuishou.kds)
                
            # 动作管理
            if self.dance_flag==1:
                # print("timestep:", self.dance_shuishou.timestep)
                self.dance_shuishou.timestep += 1
                
            # 动作结束检测    
            if self.dance_shuishou.timestep > self.dance_shuishou.end_frame:
                # print("Motion replay finished, resetting simulation.")
                # self.dance_shuishou.timestep = self.dance_shuishou.start_frame
                self.dance_shuishou.timestep = self.dance_shuishou.end_frame #停止动作
                # self.motion_type = motionType.dance_walk
                
        if self.motion_type == motionType.dance_jinwumen:
            if self.dance_jinwumen.timestep <= self.dance_jinwumen.end_frame:
                self.target_dof_pos = self.dance_jinwumen.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_jinwumen.kps, self.dance_jinwumen.kds)
                
            # 动作管理
            if self.dance_flag==1:
                # print("timestep:", self.dance_jinwumen.timestep)
                self.dance_jinwumen.timestep += 1
                
            # 动作结束检测    
            if self.dance_jinwumen.timestep > self.dance_jinwumen.end_frame:
                # print("Motion replay finished, resetting simulation.")
                # self.dance_jinwumen.timestep = self.dance_jinwumen.start_frame
                self.dance_jinwumen.timestep = self.dance_jinwumen.end_frame #停止动作
                # self.motion_type = motionType.dance_walk
                
        if self.motion_type == motionType.dance_jixiewu:
            if self.dance_jixiewu.timestep <= self.dance_jixiewu.end_frame:
                self.target_dof_pos = self.dance_jixiewu.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_jixiewu.kps, self.dance_jixiewu.kds)
                
            # 动作管理
            if self.dance_flag==1:
                # print("timestep:", self.dance_jixiewu.timestep)
                self.dance_jixiewu.timestep += 1
                
            # 动作结束检测    
            if self.dance_jixiewu.timestep > self.dance_jixiewu.end_frame:
                # print("Motion replay finished, resetting simulation.")
                # self.dance_jixiewu.timestep = self.dance_jixiewu.start_frame
                self.dance_jixiewu.timestep = self.dance_jixiewu.end_frame #停止动作
                # self.motion_type = motionType.dance_walk
                
        if self.motion_type == motionType.dance_guofuchen:
            if self.dance_guofuchen.timestep <= self.dance_guofuchen.end_frame:
                self.target_dof_pos = self.dance_guofuchen.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_guofuchen.kps, self.dance_guofuchen.kds)
                
            # 动作管理
            if self.dance_flag==1:
                # print("timestep:", self.dance_guofuchen.timestep)
                self.dance_guofuchen.timestep += 1
                
            # 动作结束检测    
            if self.dance_guofuchen.timestep > self.dance_guofuchen.end_frame:
                # print("Motion replay finished, resetting simulation.")
                # self.dance_guofuchen.timestep = self.dance_guofuchen.start_frame
                self.dance_guofuchen.timestep = self.dance_guofuchen.end_frame #停止动作
                # self.motion_type = motionType.dance_walk
        
        if self.motion_type == motionType.dance_balei:
            if self.dance_balei.timestep <= self.dance_balei.end_frame:
                self.target_dof_pos = self.dance_balei.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_balei.kps, self.dance_balei.kds)
                
            # 动作管理
            if self.dance_flag==1:
                # print("timestep:", self.dance_balei.timestep)
                self.dance_balei.timestep += 1
                
            # 动作结束检测    
            if self.dance_balei.timestep > self.dance_balei.end_frame:
                # print("Motion replay finished, resetting simulation.")
                # self.dance_balei.timestep = self.dance_balei.start_frame
                self.dance_balei.timestep = self.dance_balei.end_frame #停止动作
                # self.motion_type = motionType.dance_walk
        
        if self.motion_type == motionType.dance_dingdongji:
            if self.dance_dingdongji.timestep <= self.dance_dingdongji.end_frame:
                self.target_dof_pos = self.dance_dingdongji.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_dingdongji.kps, self.dance_dingdongji.kds)
                
            # 动作管理
            if self.dance_flag==1:
                # print("timestep:", self.dance_dingdongji.timestep)
                self.dance_dingdongji.timestep += 1
                
            # 动作结束检测    
            if self.dance_dingdongji.timestep > self.dance_dingdongji.end_frame:
                # print("Motion replay finished, resetting simulation.")
                # self.dance_dingdongji.timestep = self.dance_dingdongji.start_frame
                self.dance_dingdongji.timestep = self.dance_dingdongji.end_frame #停止动作
                # self.motion_type = motionType.dance_walk
        
        if self.motion_type == motionType.dance_lichenxi:
            if self.dance_lichenxi.timestep <= self.dance_lichenxi.end_frame:
                self.target_dof_pos = self.dance_lichenxi.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_lichenxi.kps, self.dance_lichenxi.kds)
                
            # 动作管理
            if self.dance_flag==1:
                # print("timestep:", self.dance_lichenxi.timestep)
                self.dance_lichenxi.timestep += 1
                
            # 动作结束检测    
            if self.dance_lichenxi.timestep > self.dance_lichenxi.end_frame:
                # print("Motion replay finished, resetting simulation.")
                # self.dance_lichenxi.timestep = self.dance_lichenxi.start_frame
                self.dance_lichenxi.timestep = self.dance_lichenxi.end_frame
                # self.motion_type = motionType.dance_walk
        
        if self.motion_type == motionType.dance_ydd:
            if self.dance_ydd.timestep <= self.dance_ydd.end_frame:
                self.target_dof_pos = self.dance_ydd.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_ydd.kps, self.dance_ydd.kds)
                
            # 动作管理
            if self.dance_flag==1:
                # print("timestep:", self.dance_ydd.timestep)
                self.dance_ydd.timestep += 1
                
            # 动作结束检测    
            if self.dance_ydd.timestep > self.dance_ydd.end_frame:
                # print("Motion replay finished, resetting simulation.")
                self.dance_ydd.timestep = self.dance_ydd.start_frame
                # self.motion_type = motionType.dance_walk
        
        if self.motion_type == motionType.dance_d1s2:
            if self.dance_d1s2.timestep <= self.dance_d1s2.end_frame:
                self.target_dof_pos = self.dance_d1s2.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_d1s2.kps, self.dance_d1s2.kds)
                
            # 动作管理
            if self.dance_flag==1:
                # print("timestep:", self.dance_d1s2.timestep)
                self.dance_d1s2.timestep += 1
                
            # 动作结束检测    
            if self.dance_d1s2.timestep > self.dance_d1s2.end_frame:
                self.dance_d1s2.timestep = self.dance_d1s2.start_frame
                # self.motion_type = motionType.dance_walk
                
        if self.motion_type == motionType.dance_walk:
            if self.dance_walk.timestep <= self.dance_walk.end_frame:
                self.target_dof_pos = self.dance_walk.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_walk.kps, self.dance_walk.kds)
  
            # 动作管理    
            if self.dance_flag==1:
                self.dance_walk.timestep += 1
                
            # 动作结束检测    
            # if self.dance_walk.timestep > self.dance_walk.end_frame:
            if self.dance_walk.timestep > 500:#500
                self.dance_walk.timestep = self.dance_walk.start_frame
                # self.motion_type = motionType.dance_jojo
          
        if self.motion_type == motionType.dance_fall_getup:
            if self.dance_fall_getup.timestep <= self.dance_fall_getup.end_frame:
                print(self.dance_fall_getup.end_frame)
                self.target_dof_pos = self.dance_fall_getup.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_fall_getup.kps, self.dance_fall_getup.kds)
                
                # 动作管理    
            if self.dance_flag==1:
                print("timestep:", self.dance_fall_getup.timestep)
                self.dance_fall_getup.timestep += 1
                
            # 动作结束检测    
            if self.dance_fall_getup.timestep > self.dance_fall_getup.end_frame:
                self.dance_fall_getup.timestep = self.dance_fall_getup.end_frame #停止动作
                # self.motion_type = motionType.amp_walk
                
        if self.motion_type == motionType.dance_lie_down:
            if self.dance_lie_down.timestep <= self.dance_lie_down.end_frame:
                print(self.dance_lie_down.end_frame)
                self.target_dof_pos = self.dance_lie_down.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_lie_down.kps, self.dance_lie_down.kds)
                
                # 动作管理    
            if self.dance_flag==1:
                print("timestep:", self.dance_lie_down.timestep)
                self.dance_lie_down.timestep += 1
                
            # 动作结束检测    
            if self.dance_lie_down.timestep > self.dance_lie_down.end_frame:
                self.dance_lie_down.timestep = self.dance_lie_down.end_frame #停止动作
                # self.motion_type = motionType.amp_walk
        
        if self.motion_type == motionType.amp_walk:
            # print("AMP walking...")
            self.target_dof_pos = self.amp_walk.inference_step(q, dq, quat, omega, cmd_vel)
            # print(self.target_dof_pos)
            # 发布关节控制指令
            # if self.stand_flag==1:
            self.send_to_motor(self.target_dof_pos, self.amp_walk.kps, self.amp_walk.kds)
            
        if self.motion_type == motionType.amp_run:
            # print("AMP running...")
            self.target_dof_pos = self.amp_run.inference_step(q, dq, quat, omega, cmd_vel)
            # print(self.target_dof_pos)
            # 发布关节控制指令
            self.send_to_motor(self.target_dof_pos, self.amp_run.kps, self.amp_run.kds)    
               
        if self.motion_type == motionType.walk:
            pass
        
        if self.motion_type == motionType.run:
            pass

    def start_motion_transition(self, prev_motion_type, transition_time=None):
        """启动从 prev_motion_type 到当前 self.motion_type 的混合过渡。"""
        if prev_motion_type is None or prev_motion_type == self.motion_type:
            return
        duration = self.transition_duration if transition_time is None else float(transition_time)
        if duration <= 0:
            return
        self.transition_total_steps = max(1, int(round(duration / self.dt)))
        self.transition_step_count = 0
        self.prev_motion_type = prev_motion_type
        self.transition_active = True
        self._old_action = None
        self._blend_pending = False
        print(f"motion transition start: {prev_motion_type} -> {self.motion_type}, "
              f"{self.transition_total_steps} steps ({duration:.2f}s)")

    def switch_to_motion(self, new_model, new_motion_type, num=20, with_cmd_vel=False, transition_time=None):
        """预热新模型并启动平滑过渡（旧模型推理 + 新模型推理按权重混合）。"""
        prev = self.motion_type
        self.preheat_model(new_model, num=num, with_cmd_vel=with_cmd_vel)
        self.motion_type = new_motion_type
        self.start_motion_transition(prev, transition_time=transition_time)

    # --- 模型切换过渡逻辑 ---
    def preheat_model(self, model, num=2, with_cmd_vel=False):
        # 用当前观测预推理 num 帧，不输出到电机
        q = self.qpos.copy()
        dq = self.qvel.copy()
        quat = self.quat.copy()
        omega = self.omega.copy()
        cmd_vel = np.array([self.vx, self.vy, self.dyaw], dtype=np.float32)
        for _ in range(num):
            if with_cmd_vel:
                model.inference_step(q, dq, quat, omega, cmd_vel)
            else:
                model.inference_step(q, dq, quat, omega)

    def select_fall_getup_motion(self):
        gravity_body = get_gravity_orientation(self.quat.copy())

        self.dance_flag = 1
        if gravity_body[0] < 0.0:
            # 面朝上起身
            # self.dance_fall_getup.start_frame = 600
            # self.dance_fall_getup.end_frame = 880  # 900
            self.dance_fall_getup.start_frame = 560
            self.dance_fall_getup.end_frame = 900  # 900
            getup_name = "face up"
        else:
            # 面朝下起身
            # self.dance_fall_getup.start_frame = 1350
            # self.dance_fall_getup.end_frame = 1690
            self.dance_fall_getup.start_frame = 1300
            self.dance_fall_getup.end_frame = 1700
            getup_name = "face down"

        self.dance_fall_getup.timestep = self.dance_fall_getup.start_frame
        self.switch_to_motion(self.dance_fall_getup, motionType.dance_fall_getup, num=20)
        print(
            f"fall getup: {getup_name}, gravity_body_x={gravity_body[0]:.3f}, "
            f"frames={self.dance_fall_getup.start_frame}-{self.dance_fall_getup.end_frame}"
        )        
        
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
