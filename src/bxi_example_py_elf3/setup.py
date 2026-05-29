from setuptools import setup
import os

package_name = 'bxi_example_py_elf3'

def get_policy_files():
    data_files = []
    source_dir = 'policy'  # 源目录，相对于setup.py的位置
    target_dir = os.path.join('share', package_name, 'policy')  # 目标目录

    # 遍历源目录下的所有文件和子目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # 计算相对于源目录的相对路径，以保持子目录结构
            relative_path = os.path.relpath(root, source_dir)
            install_dir = os.path.join(target_dir, relative_path)
            data_files.append((install_dir, [file_path]))
    
    return data_files

def get_robot_files():
    data_files = []
    source_dir = 'robot'  # 源目录，相对于setup.py的位置
    target_dir = os.path.join('share', package_name, 'robot')  # 目标目录

    # 遍历源目录下的所有文件和子目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # 计算相对于源目录的相对路径，以保持子目录结构
            relative_path = os.path.relpath(root, source_dir)
            install_dir = os.path.join(target_dir, relative_path)
            data_files.append((install_dir, [file_path]))
    
    return data_files

def get_launch_files():
    data_files = []
    source_dir = 'launch'  # 源目录，相对于setup.py的位置
    target_dir = os.path.join('share', package_name, 'launch')  # 目标目录

    # 遍历源目录下的所有文件和子目录#可能索引不到launch,故修改名称为.launch.py后缀
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # 计算相对于源目录的相对路径，以保持子目录结构
            relative_path = os.path.relpath(root, source_dir)
            install_dir = os.path.join(target_dir, relative_path)
            data_files.append((install_dir, [file_path]))
    
    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,
                f'{package_name}.models',
                f'{package_name}.utils',
                ],
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + get_policy_files() + get_robot_files() + get_launch_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='qiusuoxiaoshen',
    maintainer_email='1716114012@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bxi_example_py_elf3 = bxi_example_py_elf3.bxi_example:main',
            'bxi_example_py_elf3_run = bxi_example_py_elf3.bxi_example_run:main',
            'bxi_example_py_elf3_dance = bxi_example_py_elf3.bxi_example_dance:main',
        ],
    },
)
