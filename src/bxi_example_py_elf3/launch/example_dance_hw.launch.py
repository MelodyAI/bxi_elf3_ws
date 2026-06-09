import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import json

def generate_launch_description():

    npz_file_dict = {
        # mjlab
        "jojo": "policy/dance_mjlab/jojo.npz",
        "walk1_subject1": "policy/dance_mjlab/walk1_subject1.npz",
        "ydd": "policy/dance_mjlab/ydd.npz",
        "fall_getup": "policy/dance_mjlab/fall_getup.npz",
        "goodtime": "policy/dance_mjlab/goodtime.npz",
        "webster": "policy/dance_mjlab/webster.npz",
        "lichenxi": "policy/dance_mjlab/lichenxi.npz",
        
        # isaaclab
        # "shuishou": "policy/dance_isaaclab/shuishou.npz",
        "shuishou": "policy/dance_isaaclab/shuishou_clean_isaac.npz",
        "guofuchen": "policy/dance_isaaclab/guofuchen_clean_isaac.npz",
        "backflip": "policy/dance_isaaclab/backflip.npz",
        "forwardflip": "policy/dance_isaaclab/forwardflip.npz",
        "sideflip": "policy/dance_isaaclab/sideflip.npz",
        # "balei": "policy/dance_isaaclab/balei.npz",
        "balei": "policy/dance_isaaclab/balei_clean_isaac.npz",
        "dingdongji": "policy/dance_isaaclab/dingdongji.npz",
        "dance1_subject2": "policy/dance_isaaclab/dance1_subject2.npz",
        "jinwumen": "policy/dance_isaaclab/jinwumen.npz",
        "jixiewu": "policy/dance_isaaclab/jixiewu.npz",
        
    }  
    onnx_file_dict = {
        "amp_walk": "policy/amp_dwaq3.onnx",##symmetry
        # "amp_run": "policy/myrun6.onnx",##sim 5.5=6
        # "amp_run": "policy/myrun10.onnx",##hw 5=5.18
        "amp_run": "policy/myrun14.onnx",#run_dwaq
        # "amp_run": "policy/lyprun.onnx",
        # "amp_run": "policy/lyprun2.onnx",
        
        "host": "policy/elf3_ground.onnx",
        
        "jojo": "policy/dance_mjlab/jojo.onnx",
        "walk1_subject1": "policy/dance_mjlab/walk1_subject1.onnx",
        "ydd": "policy/dance_mjlab/ydd.onnx",
        "fall_getup": "policy/dance_mjlab/fall_getup.onnx",
        "goodtime": "policy/dance_mjlab/goodtime.onnx",
        "webster": "policy/dance_mjlab/webster.onnx",
        "lichenxi": "policy/dance_mjlab/lichenxi8.onnx",
        
        # isaaclab
        "backflip": "policy/dance_isaaclab/backflip.onnx",
        "forwardflip": "policy/dance_isaaclab/forwardflip.onnx",
        "sideflip": "policy/dance_isaaclab/sideflip.onnx",

        # isaaclab3
        # "shuishou": "policy/dance_isaaclab/shuishou_1w.onnx",
        "shuishou": "policy/dance_isaaclab/shuishou_16k.onnx",
        "guofuchen": "policy/dance_isaaclab/guofuchen_3w.onnx",
        "balei": "policy/dance_isaaclab/balei_17w.onnx",
        "dingdongji": "policy/dance_isaaclab/dingdongji2.onnx",
        "dance1_subject2": "policy/dance_isaaclab/dance1_subject2.onnx",
        # "jinwumen": "policy/dance_isaaclab/jinwumen_42k.onnx",
        "jinwumen": "policy/dance_isaaclab/jinwumen_60k.onnx",
        "jixiewu": "policy/dance_isaaclab/jixiewu_33k.onnx",
    }
    
    for key, value in npz_file_dict.items():
        npz_file_dict[key] = os.path.join(get_package_share_path("bxi_example_py_elf3"), value)
    for key, value in onnx_file_dict.items():
        onnx_file_dict[key] = os.path.join(get_package_share_path("bxi_example_py_elf3"), value)

    return LaunchDescription(
        [
            Node(
                package="hardware_elf3",
                executable="hardware_elf3",
                name="hardware_elf3",
                output="screen",
                parameters=[
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),

            Node(
                package="bxi_example_py_elf3",
                executable="bxi_example_py_elf3_dance",
                name="bxi_example_py_elf3_dance",
                output="screen",
                parameters=[
                    {"/topic_prefix": "hardware/"},
                    {"/use_hardware": True},
                    {"/npz_file_dict": json.dumps(npz_file_dict)},
                    {"/onnx_file_dict": json.dumps(onnx_file_dict)},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
