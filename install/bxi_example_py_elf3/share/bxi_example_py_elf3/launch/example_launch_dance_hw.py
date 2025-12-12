import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import json

def generate_launch_description():

    npz_file_dict = {
        "jojo": "policy/jojo.npz",
    }
    onnx_file_dict = {
        "jojo": "policy/jojo.onnx",
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
                    {"/npz_file": json.dumps(npz_file_dict)},
                    {"/onnx_file": json.dumps(onnx_file_dict)},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
