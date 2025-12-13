import time

import mujoco.viewer
import mujoco
import numpy as np
#from legged_gym import LEGGED_GYM_ROOT_DIR
#import torch
#import yaml

import onnxruntime

import numpy as np
#import mujoco

xml_path = "./elf3_description/xml/elf3_bg.xml"
# xml_path:  "/home/ym/Whole_body_tracking/unitree_description/g1_xml.xml"

# Total simulation time
simulation_duration = 300.0
# Simulation time step
simulation_dt = 0.01
# Controller update frequency (meets the requirement of simulation_dt * controll_decimation=0.02; 50Hz)
control_decimation = 2
def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    # 沿着 q 和 v 的最后一个维度将向量旋转四元数的倒数（NumPy 版本）。

    # 参数：
    #     q：(w, x, y, z) 中的四元数。形状为(..., 4)。
    #     v：(x, y, z) 中的向量。形状为 (..., 3)。

    # 返回：
    #     (x, y, z) 中的旋转向量。形状为 (..., 3)。
    # 
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    
    # Component a: v * (2.0 * q_w^2 - 1.0)
    a = v * np.expand_dims(2.0 * q_w**2 - 1.0, axis=-1)
    
    # Component b: cross(q_vec, v) * q_w * 2.0
    b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0
    
    # Component c: q_vec * dot(q_vec, v) * 2.0
    # For efficient computation, handle different dimensionalities
    if q_vec.ndim == 2:
        # For 2D case: use matrix multiplication for better performance
        dot_product = np.sum(q_vec * v, axis=-1, keepdims=True)
        c = q_vec * dot_product * 2.0
    else:
        # For general case: use Einstein summation
        dot_product = np.expand_dims(np.einsum('...i,...i->...', q_vec, v), axis=-1)
        c = q_vec * dot_product * 2.0
    
    return a - b + c
#import numpy as np

def matrix_to_quaternion_simple(matrix):
    """
    简化的矩阵转四元数实现
    """
    matrix = np.array(matrix)
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    
    trace = m00 + m11 + m22
    
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
def subtract_frame_transforms_mujoco(pos_a, quat_a, pos_b, quat_b,init_to_world):
    """
    与IsaacLab中subtract_frame_transforms完全相同的实现（一维版本）
    计算从坐标系A到坐标系B的相对变换
    
    参数:
        pos_a: 坐标系A的位置 (3,)
        quat_a: 坐标系A的四元数 (4,) [w, x, y, z]格式
        pos_b: 坐标系B的位置 (3,)
        quat_b: 坐标系B的四元数 (4,) [w, x, y, z]格式
        
    返回:
        rel_pos: B相对于A的位置 (3,)
        rel_quat: B相对于A的旋转四元数 (4,) [w, x, y, z]格式
    """
    # 计算相对位置: pos_B_to_A = R_A^T * (pos_B - pos_A)
    rotm_a = np.zeros(9)
    mujoco.mju_quat2Mat(rotm_a, quat_a)
    rotm_a = rotm_a.reshape(3, 3)
    
    rel_pos = rotm_a.T @ (pos_b - pos_a)
    
    # 计算相对旋转: quat_B_to_A = quat_A^* ⊗ quat_B
    rel_quat = quaternion_multiply(matrix_to_quaternion_simple(init_to_world), quat_b)
    rel_quat = quaternion_multiply(quaternion_conjugate(quat_a), rel_quat)
    
    # 确保四元数归一化（与IsaacLab保持一致）
    rel_quat = rel_quat / np.linalg.norm(rel_quat)
    
    return rel_pos, rel_quat

def quaternion_conjugate(q):
    """四元数共轭: [w, x, y, z] -> [w, -x, -y, -z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply(q1, q2):
    """四元数乘法: q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])
def get_all_body_poses(d, m):
    """
    获取MuJoCo模型中所有连杆在世界坐标系下的位置和姿态
    
    参数:
        d: mujoco.MjData 对象
        m: mujoco.MjModel 对象
        
    返回:
        body_poses: 字典，键为连杆名称，值为包含位置、四元数、旋转矩阵等信息的字典
    """
    body_poses = {}
    
    # 遍历所有body（从1开始，跳过世界body，body_id=0）
    for body_id in range(1, m.nbody):
        # 获取连杆名称
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)
        
        if body_name:  # 确保名称不为空（有些body可能没有名称）
            # 获取世界坐标系下的位置和姿态
            position = d.body(body_id).xpos.copy()      # 位置 (3,)
            quaternion = d.body(body_id).xquat.copy()   # 四元数 (4,)
            rotation_matrix = d.body(body_id).xmat.reshape(3, 3).copy()  # 旋转矩阵 (3,3)
            
            body_poses[body_name] = {
                'body_id': body_id,
                'position': position,
                'quaternion': quaternion,
                'rotation_matrix': rotation_matrix,
                'xmat_flat': d.body(body_id).xmat.copy()  # 平坦化的旋转矩阵 (9,)
            }
    
    return body_poses

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


joint_xml = [
      "waist_y_joint",
      "waist_x_joint",
      "waist_z_joint",

      "l_hip_y_joint",
      "l_hip_x_joint",
      "l_hip_z_joint",
      "l_knee_y_joint",
      "l_ankle_y_joint",
      "l_ankle_x_joint",
      
      "r_hip_y_joint",
      "r_hip_x_joint",
      "r_hip_z_joint",
      "r_knee_y_joint",
      "r_ankle_y_joint",
      "r_ankle_x_joint",

      "l_shoulder_y_joint",
      "l_shoulder_x_joint",
      "l_shoulder_z_joint",
      "l_elbow_y_joint",
      "l_wrist_x_joint",
      "l_wrist_y_joint",
      "l_wrist_z_joint",
      
      "r_shoulder_y_joint",
      "r_shoulder_x_joint",
      "r_shoulder_z_joint",
      "r_elbow_y_joint",
      "r_wrist_x_joint",
      "r_wrist_y_joint",
      "r_wrist_z_joint",
]


def yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

if __name__ == "__main__":
    # get config file name from command line

    #import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("config_file", default=,type=str, help="config file name in the config folder")
    # args = parser.parse_args()
    # config_file = "/home/ym/Whole_body_tracking/configs/g1.yaml"
    motion_file = "policy/jojo.npz"
    motion =  np.load(motion_file)
    motionpos = motion["body_pos_w"]
    motionquat = motion["body_quat_w"]
    motioninputpos = motion["joint_pos"]
    motioninputvel = motion["joint_vel"]
    i = 0


    policy_path ="policy/jojo.onnx"


    num_actions = 29 #
    num_obs = 154 #154
    import onnx
    model = onnx.load(policy_path)
    for prop in model.metadata_props:
        if prop.key == "joint_names":
            joint_seq = prop.value.split(",")
        if prop.key == "default_joint_pos":   
            joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
            joint_pos_array = np.array([joint_pos_array_seq[joint_seq.index(joint)] for joint in joint_xml])
        if prop.key == "joint_stiffness":
            stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
            stiffness_array = np.array([stiffness_array_seq[joint_seq.index(joint)] for joint in joint_xml])
            # stiffness_array = np.array([])
            
        if prop.key == "joint_damping":
            damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
            damping_array = np.array([damping_array_seq[joint_seq.index(joint)] for joint in joint_xml])        
        
        if prop.key == "action_scale":
            action_scale = np.array([float(x) for x in prop.value.split(",")])
        print(f"{prop.key}: {prop.value}")
    action = np.zeros(num_actions, dtype=np.float32)
    # target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    # policy = torch.jit.load(policy_path)
                
    policy = onnxruntime.InferenceSession(policy_path)
    input_name = policy.get_inputs()[0].name
    output_name = policy.get_outputs()[0].name

    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    timestep = 0
    motioninput = np.concatenate((motioninputpos[timestep,:],motioninputvel[timestep,:]), axis=0)
    motionposcurrent = motionpos[timestep,0,:]
    motionquatcurrent = motionquat[timestep,0,:]
    target_dof_pos = joint_pos_array.copy()
    
    # 设置初始位置
    d.qpos[2] = 1.2
    d.qpos[7:] = target_dof_pos
    # target_dof_pos = joint_pos_array_seq
    body_name = "torso_link"  # robots_ref_body_index=3motion_ref_body_index=7
    # body_name = "pelvis"
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body {body_name} not found in model")
        
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            if timestep < 2:

                ref_motion_quat = motionquat[timestep,0,:]
                yaw_motion_quat = yaw_quat(ref_motion_quat)
                yaw_motion_matrix = np.zeros(9)
                mujoco.mju_quat2Mat(yaw_motion_matrix, yaw_motion_quat)
                yaw_motion_matrix = yaw_motion_matrix.reshape(3,3)
                
                robot_quat = d.xquat[body_id]
                yaw_robot_quat = yaw_quat(robot_quat)
                yaw_robot_matrix = np.zeros(9)
                mujoco.mju_quat2Mat(yaw_robot_matrix, yaw_robot_quat)
                yaw_robot_matrix = yaw_robot_matrix.reshape(3,3)
                init_to_world =  yaw_robot_matrix @ yaw_motion_matrix.T


            mujoco.mj_step(m, d)
            tau = pd_control(target_dof_pos, d.qpos[7:], stiffness_array, np.zeros_like(damping_array), d.qvel[6:], damping_array)# xml

            d.ctrl[:] = tau
            counter += 1
            if counter % control_decimation == 0:
                # 在此应用控制信号。
                position = d.xpos[body_id]
                quaternion = d.xquat[body_id]
                motioninput = np.concatenate((motioninputpos[timestep,:],motioninputvel[timestep,:]),axis=0)
                motionposcurrent = motionpos[timestep,0,:]
                motionquatcurrent = motionquat[timestep,0,:]
                
                anchor_pos,anchor_quat = subtract_frame_transforms_mujoco(position,quaternion,motionposcurrent,motionquatcurrent,init_to_world)
                anchor_ori = np.zeros(9)
                mujoco.mju_quat2Mat(anchor_ori, anchor_quat)
                
                anchor_ori = anchor_ori.reshape(3, 3)[:, :2]
                anchor_ori = anchor_ori.reshape(-1,)
                
                # create observation
                offset = 0
                obs[offset:offset + 58] = motioninput
                offset += 58
                obs[offset:offset + 6] = anchor_ori  
                offset += 6
                
                angvel = quat_rotate_inverse_np(d.qpos[3:7], d.qvel[3 : 6])
                #linvel = quat_rotate_inverse_np(d.qpos[3:7], d.qvel[0 : 3])
                obs[offset:offset + 3] = d.qvel[3 : 6]
                offset += 3
                qpos_xml = d.qpos[7 : 7 + num_actions]  # joint positions
                qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
                obs[offset:offset + num_actions] = qpos_seq - joint_pos_array_seq  # joint positions
                offset += num_actions
                qvel_xml = d.qvel[6 : 6 + num_actions]  # joint positions
                qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])
                obs[offset:offset + num_actions] = qvel_seq  # joint velocities
                offset += num_actions   
                obs[offset:offset + num_actions] = action_buffer

                # obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # action = policy.run(['actions'], {'obs': obs_tensor.numpy(),'time_step':np.array([timestep], dtype=np.float32).reshape(1,1)})[0]
                # 
                obs_input = obs.reshape(1, -1).astype(np.float32) # 将obs从(154,)变成(1,154)并确保数据类型
                action = policy.run(['actions'], {'obs': obs_input, 'time_step':np.array([[timestep]], dtype=np.float32)})[0]
                
                action = np.asarray(action).reshape(-1)
                action_buffer = action.copy()
                target_dof_pos = action * action_scale + joint_pos_array_seq
                target_dof_pos = target_dof_pos.reshape(-1,)
                target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in joint_xml])
                i += 1
                if i > 0:# 1000
                    timestep += 1

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
