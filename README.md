
## git
```bash
git clone https://github.com/MelodyAI/bxi_elf3_ws.git
```

## build
```bash
cd bxi_elf3_ws
```

```bash
rm -rf ./build/ ./install/ ./log/
```

```bash
colcon build
```

## sim2sim
```bash
ros2 launch bxi_example_py_elf3 example_launch_dance.py
```

## sim2real
```bash
ros2 launch bxi_example_py_elf3 example_launch_dance_hw.py
```
