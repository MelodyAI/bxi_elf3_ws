import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, qos_profile_sensor_data
import communication.msg as bxiMsg
import communication.srv as bxiSrv
import nav_msgs.msg 
import sensor_msgs.msg
from threading import Lock
import numpy as np
import onnxruntime as ort
import onnx
import time
import json
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

robot_name = "elf3"
dof_num = 29
num_actions = 29
num_obs = 154

joint_name = (
    "waist_y_joint", "waist_x_joint", "waist_z_joint",
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint", 
    "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint", 
    "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint",
    "l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint",
    "l_elbow_y_joint", "l_wrist_x_joint", "l_wrist_y_joint", "l_wrist_z_joint",
    "r_shoulder_y_joint", "r_shoulder_x_joint", "r_shoulder_z_joint",
    "r_elbow_y_joint", "r_wrist_x_joint", "r_wrist_y_joint", "r_wrist_z_joint",
)

class DanceMotionPolicy:
    """舞蹈动作策略管理类"""
    
    def __init__(self, motion_npz_path: str, model_onnx_path: str):
        """
        初始化舞蹈动作策略
        
        Args:
            motion_npz_path: 舞蹈动作数据文件路径(.npz)
            model_onnx_path: ONNX模型文件路径
        """
        self.motion_npz_path = motion_npz_path
        self.model_onnx_path = model_onnx_path
        
        # 加载舞蹈动作数据
        self.load_motion_data()
        
        # 初始化ONNX模型
        self.initialize_onnx_model()
        
        # 初始化状态
        self.reset()
    
    def load_motion_data(self):
        """加载舞蹈动作数据"""
        print(f"Loading motion data from {self.motion_npz_path}")
        self.motion_data = np.load(self.motion_npz_path)
        
        # 提取动作数据
        self.motion_positions = self.motion_data["body_pos_w"]  # 身体位置序列
        self.motion_quaternions = self.motion_data["body_quat_w"]  # 身体旋转序列
        self.joint_positions = self.motion_data["joint_pos"]  # 关节位置序列
        self.joint_velocities = self.motion_data["joint_vel"]  # 关节速度序列
        
        # 计算动作总帧数
        self.total_frames = self.motion_positions.shape[0]
        print(f"Motion data loaded: {self.total_frames} frames")
    
    def initialize_onnx_model(self):
        """初始化ONNX模型"""
        print(f"Loading ONNX model from {self.model_onnx_path}")
        
        # 加载模型并解析元数据
        model = onnx.load(self.model_onnx_path)
        
        # 解析元数据
        metadata = {}
        for prop in model.metadata_props:
            metadata[prop.key] = prop.value
        
        # 存储元数据
        self.joint_seq = metadata.get("joint_names", "").split(",")
        self.default_joint_pos = np.array(
            [float(x) for x in metadata.get("default_joint_pos", "").split(",")]
        )
        self.joint_stiffness = np.array(
            [float(x) for x in metadata.get("joint_stiffness", "").split(",")]
        )
        self.joint_damping = np.array(
            [float(x) for x in metadata.get("joint_damping", "").split(",")]
        )
        self.action_scale = np.array(
            [float(x) for x in metadata.get("action_scale", "").split(",")]
        )
        
        # 配置ONNX Runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
            if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # 创建推理会话
        self.session = ort.InferenceSession(
            self.model_onnx_path,
            providers=providers,
            sess_options=options
        )
        
        # 获取输入输出信息
        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        
        print(f"Model loaded successfully")
        print(f"Joint sequence: {self.joint_seq}")
        print(f"Default joint pos shape: {self.default_joint_pos.shape}")
        print(f"Action scale shape: {self.action_scale.shape}")
    
    def reset(self):
        """重置策略状态"""
        self.current_frame = 0
        self.action_buffer = np.zeros(num_actions, dtype=np.float32)
        self.obs = np.zeros(num_obs, dtype=np.float32)
        self.last_action = np.zeros(num_actions, dtype=np.float32)
        self.init_to_world = None
        print(f"Policy reset to frame 0")
    
    def get_current_motion_frame(self):
        """获取当前帧的运动数据"""
        if self.current_frame >= self.total_frames:
            return None
            
        return {
            "joint_pos": self.joint_positions[self.current_frame, :],
            "joint_vel": self.joint_velocities[self.current_frame, :],
            "body_pos": self.motion_positions[self.current_frame, 0, :],
            "body_quat": self.motion_quaternions[self.current_frame, 0, :]
        }
    
    def compute_init_to_world(self, robot_quat, motion_quat):
        """计算初始世界坐标系变换"""
        from scipy.spatial.transform import Rotation as R
        
        # 提取偏航角
        yaw_robot = self.extract_yaw_from_quat(robot_quat)
        yaw_motion = self.extract_yaw_from_quat(motion_quat)
        
        # 计算旋转矩阵
        rot_robot = R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]]).as_matrix()
        rot_motion = R.from_quat([motion_quat[1], motion_quat[2], motion_quat[3], motion_quat[0]]).as_matrix()
        
        # 只保留偏航旋转
        yaw_rot_robot = R.from_euler('z', yaw_robot).as_matrix()
        yaw_rot_motion = R.from_euler('z', yaw_motion).as_matrix()
        
        # 计算初始变换
        self.init_to_world = yaw_rot_robot @ yaw_rot_motion.T
        return self.init_to_world
    
    def extract_yaw_from_quat(self, q):
        """从四元数中提取偏航角"""
        w, x, y, z = q
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    def quaternion_multiply(self, q1, q2):
        """四元数乘法"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([w, x, y, z])
    
    def quaternion_conjugate(self, q):
        """四元数共轭"""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def quaternion_to_rotation_matrix(self, q):
        """四元数转旋转矩阵"""
        from scipy.spatial.transform import Rotation as R
        return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    
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
        
    def compute_observation(self, robot_state, motion_frame):
        """
        计算观测向量
        
        Args:
            robot_state: 机器人状态字典，包含q, dq, quat, omega
            motion_frame: 当前帧运动数据
        """
        q = robot_state['q']
        dq = robot_state['dq']
        quat = robot_state['quat']
        omega = robot_state['omega']
        
        # 构建观测向量
        obs = np.zeros(num_obs, dtype=np.float32)
        offset = 0
        
        # 1. 参考运动数据 (关节位置 + 关节速度)
        motion_input = np.concatenate([motion_frame['joint_pos'], motion_frame['joint_vel']])
        obs[offset:offset + 58] = motion_input
        offset += 58
        
        # 2. 身体姿态相对旋转矩阵的前两列
        if self.init_to_world is None:
            self.compute_init_to_world(quat, motion_frame['body_quat'])
        
        # 计算相对四元数
        # rel_quat = self.quaternion_multiply(
        #     self.quaternion_conjugate(quat),
        #     self.quaternion_multiply(
        #         R.from_matrix(self.init_to_world).as_quat(),
        #         motion_frame['body_quat']
        #     )
        # )
        relquat = self.quaternion_multiply(self.matrix_to_quaternion_simple(self.init_to_world), motion_frame['body_quat'])
        rel_quat = rel_quat / np.linalg.norm(rel_quat)
        
        # 转换为旋转矩阵并取前两列
        rel_matrix = self.quaternion_to_rotation_matrix(rel_quat)[:, :2].flatten()
        obs[offset:offset + 6] = rel_matrix
        offset += 6
        
        # 3. 角速度
        obs[offset:offset + 3] = omega
        offset += 3
        
        # 4. 关节位置偏差
        obs[offset:offset + num_actions] = q - self.default_joint_pos
        offset += num_actions
        
        # 5. 关节速度
        obs[offset:offset + num_actions] = dq
        offset += num_actions
        
        # 6. 上一时刻动作
        obs[offset:offset + num_actions] = self.action_buffer
        
        self.obs = obs
        return obs
    
    def inference(self, robot_state, motion_frame):
        """
        执行推理
        
        Args:
            robot_state: 机器人状态
            motion_frame: 当前帧运动数据
            
        Returns:
            target_joint_positions: 目标关节位置（按joint_name顺序）
            stiffness: 关节刚度
            damping: 关节阻尼
        """
        # 计算观测
        obs = self.compute_observation(robot_state, motion_frame)
        
        # 准备输入数据
        obs_input = obs.reshape(1, -1).astype(np.float32)
        time_step_input = np.array([[self.current_frame]], dtype=np.float32)
        
        # 执行推理
        action = self.session.run(
            ['actions'], 
            {'obs': obs_input, 'time_step': time_step_input}
        )[0]
        
        # 处理输出
        action = action.reshape(-1)
        self.last_action = action.copy()
        self.action_buffer = action.copy()
        
        # 计算目标关节位置
        target_pos = action * self.action_scale + self.default_joint_pos
        
        # 将目标位置映射到joint_name顺序
        target_pos_mapped = np.array([
            target_pos[self.joint_seq.index(joint)] 
            for joint in joint_name
        ])
        
        # 映射刚度和阻尼
        stiffness_mapped = np.array([
            self.joint_stiffness[self.joint_seq.index(joint)] 
            for joint in joint_name
        ])
        
        damping_mapped = np.array([
            self.joint_damping[self.joint_seq.index(joint)] 
            for joint in joint_name
        ])
        
        # 递增帧数
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.current_frame = 0
            print("Motion replay finished, restarting...")
        
        return target_pos_mapped, stiffness_mapped, damping_mapped
    
    def get_info(self):
        """获取策略信息"""
        return {
            "motion_file": self.motion_npz_path,
            "model_file": self.model_onnx_path,
            "total_frames": self.total_frames,
            "current_frame": self.current_frame,
            "joint_sequence": self.joint_seq
        }

class DancePolicyManager:
    """舞蹈策略管理器"""
    
    def __init__(self, npz_file_dict, onnx_file_dict):
        """
        初始化策略管理器
        
        Args:
            npz_file_dict: NPZ文件字典 {舞蹈名称: 文件路径}
            onnx_file_dict: ONNX文件字典 {舞蹈名称: 文件路径}
        """
        self.policies = {}
        self.current_dance = None
        
        # 创建所有策略
        for dance_name in npz_file_dict:
            if dance_name in onnx_file_dict:
                npz_path = npz_file_dict[dance_name]
                onnx_path = onnx_file_dict[dance_name]
                
                try:
                    policy = DanceMotionPolicy(npz_path, onnx_path)
                    self.policies[dance_name] = policy
                    print(f"Created policy for dance: {dance_name}")
                except Exception as e:
                    print(f"Failed to create policy for {dance_name}: {e}")
        
        # 设置默认舞蹈
        if self.policies:
            self.current_dance = list(self.policies.keys())[0]
            print(f"Default dance set to: {self.current_dance}")
    
    def switch_dance(self, dance_name):
        """
        切换到指定舞蹈
        
        Args:
            dance_name: 舞蹈名称
            
        Returns:
            bool: 是否切换成功
        """
        if dance_name in self.policies:
            self.current_dance = dance_name
            self.policies[dance_name].reset()
            print(f"Switched to dance: {dance_name}")
            return True
        else:
            print(f"Dance {dance_name} not found")
            return False
    
    def get_current_policy(self):
        """获取当前策略"""
        if self.current_dance:
            return self.policies[self.current_dance]
        return None
    
    def list_dances(self):
        """列出所有可用的舞蹈"""
        return list(self.policies.keys())
    
    def get_dance_info(self, dance_name=None):
        """获取舞蹈信息"""
        if dance_name is None:
            dance_name = self.current_dance
        
        if dance_name in self.policies:
            return self.policies[dance_name].get_info()
        return None


class BxiExample(Node):
    
    def __init__(self):
        super().__init__('bxi_example_py')
        
        # 声明参数
        self.declare_parameter('/topic_prefix', 'default_value')
        self.topic_prefix = self.get_parameter('/topic_prefix').get_parameter_value().string_value
        
        self.declare_parameter('/npz_file_dict', json.dumps({}))
        npz_file_json = self.get_parameter('/npz_file_dict').value
        self.npz_file_dict = json.loads(npz_file_json)
        
        self.declare_parameter('/onnx_file_dict', json.dumps({}))
        onnx_file_json = self.get_parameter('/onnx_file_dict').value
        self.onnx_file_dict = json.loads(onnx_file_json)
        
        print('Loading dance configurations:')
        for key in self.npz_file_dict:
            print(f"  {key}: motion={self.npz_file_dict[key]}, model={self.onnx_file_dict.get(key, 'Not found')}")
        
        # 初始化舞蹈策略管理器
        self.policy_manager = DancePolicyManager(self.npz_file_dict, self.onnx_file_dict)
        
        # 订阅和发布
        qos = QoSProfile(depth=1, durability=qos_profile_sensor_data.durability, 
                        reliability=qos_profile_sensor_data.reliability)
        
        self.act_pub = self.create_publisher(bxiMsg.ActuatorCmds, self.topic_prefix+'actuators_cmds', qos)
        self.odom_sub = self.create_subscription(nav_msgs.msg.Odometry, self.topic_prefix+'odom', 
                                                self.odom_callback, qos)
        self.joint_sub = self.create_subscription(sensor_msgs.msg.JointState, self.topic_prefix+'joint_states', 
                                                 self.joint_callback, qos)
        self.imu_sub = self.create_subscription(sensor_msgs.msg.Imu, self.topic_prefix+'imu_data', 
                                               self.imu_callback, qos)
        self.touch_sub = self.create_subscription(bxiMsg.TouchSensor, self.topic_prefix+'touch_sensor', 
                                                 self.touch_callback, qos)
        self.joy_sub = self.create_subscription(bxiMsg.MotionCommands, 'motion_commands', 
                                               self.joy_callback, qos)
        
        # 添加舞蹈切换服务
        from std_srvs.srv import SetBool
        self.switch_dance_srv = self.create_service(
            SetBool, 
            'switch_dance',
            self.switch_dance_callback
        )
        
        self.rest_srv = self.create_client(bxiSrv.RobotReset, self.topic_prefix+'robot_reset')
        self.sim_rest_srv = self.create_client(bxiSrv.SimulationReset, self.topic_prefix+'sim_reset')
        
        # 初始化状态
        self.qpos = np.zeros(num_actions, dtype=np.double)
        self.qvel = np.zeros(num_actions, dtype=np.double)
        self.omega = np.zeros(3, dtype=np.double)
        self.quat = np.zeros(4, dtype=np.double)
        
        # 控制参数
        self.step = 0
        self.loop_count = 0
        self.dt = 0.02  # 50Hz
        
        # 线程安全锁
        self.lock_in = Lock()
        self.timer_callback_group_1 = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(self.dt, self.timer_callback, 
                                      callback_group=self.timer_callback_group_1)
    
    def switch_dance_callback(self, request, response):
        """处理舞蹈切换请求"""
        # 这里可以根据request.data来切换不同的舞蹈
        # 例如: request.data=True 切换到下一个舞蹈
        dances = self.policy_manager.list_dances()
        if dances:
            current_idx = dances.index(self.policy_manager.current_dance)
            next_idx = (current_idx + 1) % len(dances)
            success = self.policy_manager.switch_dance(dances[next_idx])
            response.success = success
            response.message = f"Switched to {dances[next_idx]}" if success else "Failed to switch dance"
        else:
            response.success = False
            response.message = "No dances available"
        return response
    
    def timer_callback(self):
        """定时器回调函数"""
        # 启动阶段
        if self.step == 0:
            self.robot_reset(1, False)
            print('Robot reset 1!')
            self.step = 1
            return
        elif self.step == 1 and self.loop_count >= (6./self.dt):
            self.robot_reset(2, True)
            print('Robot reset 2!')
            self.loop_count = 0
            self.step = 2
            return
        
        # 缓启动阶段
        if self.step == 1:
            soft_start = min(self.loop_count/(3./self.dt), 1.0)
            self.publish_soft_start(soft_start)
        
        # 舞蹈执行阶段
        elif self.step == 2:
            self.execute_dance()
        
        self.loop_count += 1
    
    def publish_soft_start(self, soft_start):
        """发布缓启动控制指令"""
        joint_kp = np.array([300,300,300,150,100,100,200,50,20,
                            150,100,100,200,50,20,80,80,80,60,
                            20,20,20,80,80,80,60,20,20,20])
        joint_kd = np.array([3,3,3,2,2,2,2.5,1,1,2,2,2,2.5,1,1,
                            2,2,2,2,1,1,1,2,2,2,2,1,1,1])
        
        soft_joint_kp = joint_kp * soft_start
        soft_joint_kd = joint_kd
        
        msg = bxiMsg.ActuatorCmds()
        msg.header.frame_id = robot_name
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.actuators_name = joint_name
        msg.pos = [0.0] * dof_num  # 初始位置
        msg.vel = [0.0] * dof_num
        msg.torque = [0.0] * dof_num
        msg.kp = soft_joint_kp.tolist()
        msg.kd = soft_joint_kd.tolist()
        self.act_pub.publish(msg)
    
    def execute_dance(self):
        """执行舞蹈动作"""
        # 获取当前策略
        current_policy = self.policy_manager.get_current_policy()
        if not current_policy:
            print("No active dance policy!")
            return
        
        # 获取当前帧运动数据
        motion_frame = current_policy.get_current_motion_frame()
        if motion_frame is None:
            print("No motion frame available!")
            return
        
        # 获取机器人状态
        with self.lock_in:
            robot_state = {
                'q': self.qpos.copy(),
                'dq': self.qvel.copy(),
                'quat': self.quat.copy(),
                'omega': self.omega.copy()
            }
        
        # 执行推理
        target_pos, stiffness, damping = current_policy.inference(robot_state, motion_frame)
        
        # 发布控制指令
        msg = bxiMsg.ActuatorCmds()
        msg.header.frame_id = robot_name
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.actuators_name = joint_name
        msg.pos = target_pos.tolist()
        msg.vel = [0.0] * dof_num
        msg.torque = [0.0] * dof_num
        msg.kp = (0.8 * stiffness).tolist()  # 刚度缩放
        msg.kd = (0.2 * damping).tolist()    # 阻尼缩放
        self.act_pub.publish(msg)
    
    def robot_reset(self, reset_step, release):
        """重置机器人"""
        req = bxiSrv.RobotReset.Request()
        req.reset_step = reset_step
        req.release = release
        req.header.frame_id = robot_name
        
        while not self.rest_srv.wait_for_service(timeout_sec=1.0):
            print('Service not available, waiting again...')
        
        self.rest_srv.call_async(req)
    
    def joint_callback(self, msg):
        """关节状态回调"""
        with self.lock_in:
            self.qpos = np.array(msg.position)
            self.qvel = np.array(msg.velocity)
    
    def imu_callback(self, msg):
        """IMU回调"""
        quat = msg.orientation
        avel = msg.angular_velocity
        
        with self.lock_in:
            self.quat = np.array([quat.w, quat.x, quat.y, quat.z], dtype=np.double)
            self.omega = np.array([avel.x, avel.y, avel.z], dtype=np.double)
    
    def joy_callback(self, msg):
        """手柄控制回调"""
        with self.lock_in:
            self.vx = msg.vel_des.x * 2
            self.vx = np.clip(self.vx, -1.0, 2.0)
            self.vy = 0
            self.dyaw = msg.yawdot_des
    
    def touch_callback(self, msg):
        """触觉传感器回调"""
        pass
    
    def odom_callback(self, msg):
        """里程计回调"""
        pass


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