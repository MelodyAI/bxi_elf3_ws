# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR


ELF3LITE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/elf3_lite/usd/elf3.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            "waist_y_joint": 0.0,
            "waist_x_joint": 0.0,
            "waist_z_joint": 0.0,

            "l_hip_y_joint": -0.3,   # 左腿_髋关节_z轴
            "l_hip_x_joint": 0.0,   # 左腿_髋关节_x轴
            "l_hip_z_joint": 0.0,   # 左腿_髋关节_y轴
            "l_knee_y_joint": 0.6,   # 左腿_膝关节_y轴
            "l_ankle_y_joint": -0.3,   # 左腿_踝关节_y轴
            "l_ankle_x_joint": 0.0,   # 左腿_踝关节_x轴

            "r_hip_y_joint": -0.3,   # 右腿_髋关节_z轴    
            "r_hip_x_joint": 0.0,   # 右腿_髋关节_x轴
            "r_hip_z_joint": 0.0,   # 右腿_髋关节_y轴
            "r_knee_y_joint": 0.6,   # 右腿_膝关节_y轴
            "r_ankle_y_joint": -0.3,   # 右腿_踝关节_y轴
            "r_ankle_x_joint": 0.0,   # 右腿_踝关节_x轴

            "l_shoulder_y_joint": 0.2,   # 左臂_肩关节_y轴
            "l_shoulder_x_joint": 0.2,   # 左臂_肩关节_x轴
            "l_shoulder_z_joint": 0.0,   # 左臂_肩关节_z轴
            "l_elbow_y_joint": 0.6,   # 左臂_肘关节_y轴
            "l_wrist_x_joint": 0.0,
            "l_wrist_y_joint": 0.0,
            "l_wrist_z_joint": 0.0,

            "r_shoulder_y_joint": 0.2,   # 右臂_肩关节_y轴   
            "r_shoulder_x_joint": -0.2,   # 右臂_肩关节_x轴
            "r_shoulder_z_joint": 0.0,   # 右臂_肩关节_z轴
            "r_elbow_y_joint": 0.6,    # 右臂_肘关节_y轴
            "r_wrist_x_joint": 0.0,
            "r_wrist_y_joint": 0.0,
            "r_wrist_z_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_y_joint",
                "waist_x_joint",
                "waist_z_joint",
            ],
            effort_limit_sim={
                "waist_y_joint": 100,
                "waist_x_joint": 100,
                "waist_z_joint": 100,
            },
            velocity_limit_sim={
                "waist_y_joint": 20,
                "waist_x_joint": 20,
                "waist_z_joint": 20,
            },
            stiffness={
                "waist_y_joint": 108.448,
                "waist_x_joint": 162.672,
                "waist_z_joint": 176.421,
            },
            damping={
                "waist_y_joint": 6.904,
                "waist_x_joint": 10.356,
                "waist_z_joint": 11.231,
            },
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_y_joint",   # 左腿_髋关节_z轴
                ".*_hip_x_joint",   # 左腿_髋关节_x轴
                ".*_hip_z_joint",   # 左腿_髋关节_y轴
                ".*_knee_y_joint",   # 左腿_膝关节_y轴
            ],
            effort_limit_sim={
                ".*_hip_y_joint": 100,
                ".*_hip_x_joint": 100,
                ".*_hip_z_joint": 50,
                ".*_knee_y_joint": 150,
            },
            velocity_limit_sim={
                ".*_hip_y_joint": 20,   # 左腿_髋关节_z轴
                ".*_hip_x_joint": 20,   # 左腿_髋关节_x轴
                ".*_hip_z_joint": 20,   # 左腿_髋关节_y轴
                ".*_knee_y_joint": 20,   # 左腿_膝关节_y轴
            },
            stiffness={
                ".*_hip_y_joint": 176.421,   # 左腿_髋关节_z轴
                ".*_hip_x_joint": 176.421,   # 左腿_髋关节_x轴
                ".*_hip_z_joint": 54.224,   # 左腿_髋关节_y轴
                ".*_knee_y_joint": 176.421,   # 左腿_膝关节_y轴
            },
            damping={
                ".*_hip_y_joint": 11.231,   # 左腿_髋关节_z轴
                ".*_hip_x_joint": 11.231,   # 左腿_髋关节_x轴
                ".*_hip_z_joint": 3.452,   # 左腿_髋关节_y轴
                ".*_knee_y_joint": 11.231,   # 左腿_膝关节_y轴
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_y_joint",   # 左腿_踝关节_y轴
                ".*_ankle_x_joint",   # 左腿_踝关节_x轴
            ],
            effort_limit_sim={
                ".*_ankle_y_joint": 50,
                ".*_ankle_x_joint": 20,
            },
            velocity_limit_sim={
                ".*_ankle_y_joint": 20,   # 左腿_踝关节_y轴
                ".*_ankle_x_joint": 20,   # 左腿_踝关节_x轴
            },
            stiffness={
                ".*_ankle_y_joint": 33.493,   # 左腿_踝关节_y轴
                ".*_ankle_x_joint": 21.771,   # 左腿_踝关节_x轴
            },
            damping={
                ".*_ankle_y_joint": 2.132,   # 左腿_踝关节_y轴
                ".*_ankle_x_joint": 1.386,   # 左腿_踝关节_x轴
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_y_joint",   # 左臂_肩关节_y轴
                ".*_shoulder_x_joint",   # 左臂_肩关节_x轴
                ".*_shoulder_z_joint",   # 左臂_肩关节_z轴
                ".*_elbow_y_joint",   # 左臂_肘关节_y轴
            ],
            effort_limit_sim={
                ".*_shoulder_y_joint": 50,   # 左臂_肩关节_y轴
                ".*_shoulder_x_joint": 50,   # 左臂_肩关节_x轴
                ".*_shoulder_z_joint": 25,   # 左臂_肩关节_z轴
                ".*_elbow_y_joint": 50,   # 左臂_肘关节_y轴
            },
            velocity_limit_sim={
                ".*_shoulder_y_joint": 20,   # 左臂_肩关节_y轴
                ".*_shoulder_x_joint": 20,   # 左臂_肩关节_x轴
                ".*_shoulder_z_joint": 20,   # 左臂_肩关节_z轴
                ".*_elbow_y_joint": 20,   # 左臂_肘关节_y轴
            },
            stiffness={
                ".*_shoulder_y_joint": 54.224,   # 左臂_肩关节_y轴
                ".*_shoulder_x_joint": 54.224,   # 左臂_肩关节_x轴
                ".*_shoulder_z_joint": 16.747,   # 左臂_肩关节_z轴
                ".*_elbow_y_joint": 54.224,   # 左臂_肘关节_y轴
            },
            damping={
                ".*_shoulder_y_joint": 3.452,   # 左臂_肩关节_y轴
                ".*_shoulder_x_joint": 3.452,   # 左臂_肩关节_x轴
                ".*_shoulder_z_joint": 1.066,   # 左臂_肩关节_z轴
                ".*_elbow_y_joint": 3.452,   # 左臂_肘关节_y轴
            },
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_x_joint",
                ".*_wrist_y_joint",
                ".*_wrist_z_joint",
            ],
            effort_limit_sim={
                ".*_wrist_x_joint": 25,
                ".*_wrist_y_joint": 25,
                ".*_wrist_z_joint": 25,
            },
            velocity_limit_sim={
                ".*_wrist_x_joint": 20,
                ".*_wrist_y_joint": 20,
                ".*_wrist_z_joint": 20,
            },
            stiffness={
                ".*_wrist_x_joint": 16.747,
                ".*_wrist_y_joint": 16.747,
                ".*_wrist_z_joint": 16.747,
            },
            damping={
                ".*_wrist_x_joint": 1.066,   # 左臂_腕关节_x轴
                ".*_wrist_y_joint": 1.066,   # 左臂_腕关节_y轴
                ".*_wrist_z_joint": 1.066,   # 左臂_腕关节_z轴
            },
        ),
    },
)


# self.action_scale = np.array([
        #     0.231, 0.154, 0.213,  # waist_y, waist_x, waist_z
        #     0.213, 0.213, 0.231,  # l_hip_y, l_hip_x, l_hip_z
        #     0.213, 0.373, 0.230,  # l_knee_y, l_ankle_y, l_ankle_x
        #     0.213, 0.213, 0.231,  # r_hip_y, r_hip_x, r_hip_z
        #     0.213, 0.373, 0.230,  # r_knee_y, r_ankle_y, r_ankle_x
        #     0.231, 0.231, 0.373,  # l_shoulder_y, l_shoulder_x, l_shoulder_z
        #     0.231, 0.373,         # l_elbow_y, l_wrist_x
        #     0.373, 0.373,         # l_wrist_y, l_wrist_z
        #     0.231, 0.231, 0.373,  # r_shoulder_y, r_shoulder_x, r_shoulder_z
        #     0.231, 0.373,         # r_elbow_y, r_wrist_x
        #     0.373, 0.373,         # r_wrist_y, r_wrist_z
        # ])

# 旧版本配置，
ELF3LITE_CFG_OLD = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/elf3_lite/usd/elf3.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.1),
        # joint_pos={
        #     "waist_y_joint": 0.0,
        #     "waist_x_joint": 0.0,
        #     "waist_z_joint": 0.0,

        #     "l_hip_y_joint": -0.4,   # 左腿_髋关节_z轴
        #     "l_hip_x_joint": 0.0,   # 左腿_髋关节_x轴
        #     "l_hip_z_joint": 0.0,   # 左腿_髋关节_y轴
        #     "l_knee_y_joint": 0.8,   # 左腿_膝关节_y轴
        #     "l_ankle_y_joint": -0.4,   # 左腿_踝关节_y轴
        #     "l_ankle_x_joint": 0.0,   # 左腿_踝关节_x轴

        #     "r_hip_y_joint": -0.4,   # 右腿_髋关节_z轴    
        #     "r_hip_x_joint": 0.0,   # 右腿_髋关节_x轴
        #     "r_hip_z_joint": 0.0,   # 右腿_髋关节_y轴
        #     "r_knee_y_joint": 0.8,   # 右腿_膝关节_y轴
        #     "r_ankle_y_joint": -0.4,   # 右腿_踝关节_y轴
        #     "r_ankle_x_joint": 0.0,   # 右腿_踝关节_x轴

        #     "l_shoulder_y_joint": 0.5,   # 左臂_肩关节_y轴
        #     "l_shoulder_x_joint": 0.3,   # 左臂_肩关节_x轴
        #     "l_shoulder_z_joint": -0.1,   # 左臂_肩关节_z轴
        #     "l_elbow_y_joint": -0.2,   # 左臂_肘关节_y轴
        #     "l_wrist_x_joint": 0.0,
        #     "l_wrist_y_joint": 0.0,
        #     "l_wrist_z_joint": 0.0,

        #     "r_shoulder_y_joint": 0.5,   # 右臂_肩关节_y轴   
        #     "r_shoulder_x_joint": -0.3,   # 右臂_肩关节_x轴
        #     "r_shoulder_z_joint": 0.1,   # 右臂_肩关节_z轴
        #     "r_elbow_y_joint": -0.2,    # 右臂_肘关节_y轴
        #     "r_wrist_x_joint": 0.0,
        #     "r_wrist_y_joint": 0.0,
        #     "r_wrist_z_joint": 0.0,
        # },
        joint_pos={
            "waist_y_joint": 0.0,
            "waist_x_joint": 0.0,
            "waist_z_joint": 0.0,

            "l_hip_y_joint": -0.3,   # 左腿_髋关节_z轴
            "l_hip_x_joint": 0.0,   # 左腿_髋关节_x轴
            "l_hip_z_joint": 0.0,   # 左腿_髋关节_y轴
            "l_knee_y_joint": 0.6,   # 左腿_膝关节_y轴
            "l_ankle_y_joint": -0.3,   # 左腿_踝关节_y轴
            "l_ankle_x_joint": 0.0,   # 左腿_踝关节_x轴

            "r_hip_y_joint": -0.3,   # 右腿_髋关节_z轴    
            "r_hip_x_joint": 0.0,   # 右腿_髋关节_x轴
            "r_hip_z_joint": 0.0,   # 右腿_髋关节_y轴
            "r_knee_y_joint": 0.6,   # 右腿_膝关节_y轴
            "r_ankle_y_joint": -0.3,   # 右腿_踝关节_y轴
            "r_ankle_x_joint": 0.0,   # 右腿_踝关节_x轴

            "l_shoulder_y_joint": 0.2,   # 左臂_肩关节_y轴
            "l_shoulder_x_joint": 0.2,   # 左臂_肩关节_x轴
            "l_shoulder_z_joint": 0.0,   # 左臂_肩关节_z轴
            "l_elbow_y_joint": 0.6,   # 左臂_肘关节_y轴
            "l_wrist_x_joint": 0.0,
            "l_wrist_y_joint": 0.0,
            "l_wrist_z_joint": 0.0,

            "r_shoulder_y_joint": 0.2,   # 右臂_肩关节_y轴   
            "r_shoulder_x_joint": -0.2,   # 右臂_肩关节_x轴
            "r_shoulder_z_joint": 0.0,   # 右臂_肩关节_z轴
            "r_elbow_y_joint": 0.6,    # 右臂_肘关节_y轴
            "r_wrist_x_joint": 0.0,
            "r_wrist_y_joint": 0.0,
            "r_wrist_z_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_y_joint",
                "waist_x_joint",
                "waist_z_joint",
            ],
            effort_limit_sim={
                "waist_y_joint": 100,
                "waist_x_joint": 100,
                "waist_z_joint": 100,
            },
            velocity_limit_sim={
                "waist_y_joint": 20,
                "waist_x_joint": 20,
                "waist_z_joint": 20,
            },
            stiffness={
                "waist_y_joint": 300,
                "waist_x_joint": 300,
                "waist_z_joint": 300,
            },
            damping={
                "waist_y_joint": 3,
                "waist_x_joint": 3,
                "waist_z_joint": 3,
            },
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_y_joint",   # 左腿_髋关节_z轴
                ".*_hip_x_joint",   # 左腿_髋关节_x轴
                ".*_hip_z_joint",   # 左腿_髋关节_y轴
                ".*_knee_y_joint",   # 左腿_膝关节_y轴
            ],
            effort_limit_sim={
                ".*_hip_y_joint": 100,
                ".*_hip_x_joint": 100,
                ".*_hip_z_joint": 50,
                ".*_knee_y_joint": 150,
            },
            velocity_limit_sim={
                ".*_hip_y_joint": 20,   # 左腿_髋关节_z轴
                ".*_hip_x_joint": 20,   # 左腿_髋关节_x轴
                ".*_hip_z_joint": 20,   # 左腿_髋关节_y轴
                ".*_knee_y_joint": 20,   # 左腿_膝关节_y轴
            },
            stiffness={
                ".*_hip_y_joint": 150,   # 左腿_髋关节_z轴
                ".*_hip_x_joint": 100,   # 左腿_髋关节_x轴
                ".*_hip_z_joint": 100,   # 左腿_髋关节_y轴
                ".*_knee_y_joint": 200,   # 左腿_膝关节_y轴
            },
            damping={
                ".*_hip_y_joint": 2,   # 左腿_髋关节_z轴
                ".*_hip_x_joint": 2,   # 左腿_髋关节_x轴
                ".*_hip_z_joint": 2,   # 左腿_髋关节_y轴
                ".*_knee_y_joint": 2.5,   # 左腿_膝关节_y轴
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_y_joint",   # 左腿_踝关节_y轴
                ".*_ankle_x_joint",   # 左腿_踝关节_x轴
            ],
            effort_limit_sim={
                ".*_ankle_y_joint": 50,
                ".*_ankle_x_joint": 20,
            },
            velocity_limit_sim={
                ".*_ankle_y_joint": 20,   # 左腿_踝关节_y轴
                ".*_ankle_x_joint": 20,   # 左腿_踝关节_x轴
            },
            stiffness={
                ".*_ankle_y_joint": 50,   # 左腿_踝关节_y轴
                ".*_ankle_x_joint": 20,   # 左腿_踝关节_x轴
            },
            damping={
                ".*_ankle_y_joint": 1,   # 左腿_踝关节_y轴
                ".*_ankle_x_joint": 1,   # 左腿_踝关节_x轴
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_y_joint",   # 左臂_肩关节_y轴
                ".*_shoulder_x_joint",   # 左臂_肩关节_x轴
                ".*_shoulder_z_joint",   # 左臂_肩关节_z轴
                ".*_elbow_y_joint",   # 左臂_肘关节_y轴
            ],
            effort_limit_sim={
                ".*_shoulder_y_joint": 50,   # 左臂_肩关节_y轴
                ".*_shoulder_x_joint": 50,   # 左臂_肩关节_x轴
                ".*_shoulder_z_joint": 25,   # 左臂_肩关节_z轴
                ".*_elbow_y_joint": 50,   # 左臂_肘关节_y轴
            },
            velocity_limit_sim={
                ".*_shoulder_y_joint": 20,   # 左臂_肩关节_y轴
                ".*_shoulder_x_joint": 20,   # 左臂_肩关节_x轴
                ".*_shoulder_z_joint": 20,   # 左臂_肩关节_z轴
                ".*_elbow_y_joint": 20,   # 左臂_肘关节_y轴
            },
            stiffness={
                ".*_shoulder_y_joint": 80,   # 左臂_肩关节_y轴
                ".*_shoulder_x_joint": 80,   # 左臂_肩关节_x轴
                ".*_shoulder_z_joint": 80,   # 左臂_肩关节_z轴
                ".*_elbow_y_joint": 60,   # 左臂_肘关节_y轴
            },
            damping={
                ".*_shoulder_y_joint": 2,   # 左臂_肩关节_y轴
                ".*_shoulder_x_joint": 2,   # 左臂_肩关节_x轴
                ".*_shoulder_z_joint": 2,   # 左臂_肩关节_z轴
                ".*_elbow_y_joint": 2,   # 左臂_肘关节_y轴
            },
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_x_joint",
                ".*_wrist_y_joint",
                ".*_wrist_z_joint",
            ],
            effort_limit_sim={
                ".*_wrist_x_joint": 25,
                ".*_wrist_y_joint": 25,
                ".*_wrist_z_joint": 25,
            },
            velocity_limit_sim={
                ".*_wrist_x_joint": 20,
                ".*_wrist_y_joint": 20,
                ".*_wrist_z_joint": 20,
            },
            stiffness={
                ".*_wrist_x_joint": 20,
                ".*_wrist_y_joint": 20,
                ".*_wrist_z_joint": 20,
            },
            damping={
                ".*_wrist_x_joint": 1,   # 左臂_腕关节_x轴
                ".*_wrist_y_joint": 1,   # 左臂_腕关节_y轴
                ".*_wrist_z_joint": 1,   # 左臂_腕关节_z轴
            },
        ),
    },
)
