#!/bin/bash
export ROS_DOMAIN_ID=66
SESSION="bxi_hardware"
LOG_DIR="./log/bxi_logs"
#检查是否生成
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi
# 检查 tmux
if ! command -v tmux &>/dev/null; then
    sudo apt update && sudo apt install tmux -y
fi

# 确保旧 Session 彻底关闭
tmux kill-session -t $SESSION 2>/dev/null

# 创建新 Session，并设置初始大小防止 split 失败
tmux new-session -d -s $SESSION -x "$(tput cols)" -y "$(tput lines)"

# 全局配置
tmux set-option -g mouse on
tmux set-option -g pane-border-status top
tmux set-option -g pane-border-format " #[fg=black,bg=green] #T #[default] "

# ---------------------------------------------------------------
# 核心布局划分
# ---------------------------------------------------------------

# 先水平切分：右侧占 45%
tmux split-window -h -p 45 -t $SESSION

# 命名窗格
tmux select-pane -t 0 -T "hardware"
tmux select-pane -t 1 -T "遥控器"

# 执行指令
# 使用 tee -a 保证你在 tmux 窗口能看到实时日志，同时后台保存
tmux send-keys -t $SESSION:0.0 "bash remote_kill.sh" C-m
tmux send-keys -t $SESSION:0.0 "source /opt/bxi/bxi_ros2_pkg/setup.bash && source install/setup.bash" C-m
tmux send-keys -t $SESSION:0.0 "ros2 launch bxi_example_py_elf3 example_dance_hw.launch.py 2>&1 | tee -a \"$LOG_DIR/hw_\$(date +%Y%m%d_%H%M%S).log\"" C-m

sleep 1

tmux send-keys -t $SESSION:0.1 "source /opt/bxi/bxi_ros2_pkg/setup.bash && source install/setup.bash" C-m
tmux send-keys -t $SESSION:0.1 "ros2 launch remote_controller remote_conroller_launch.py" C-m

# 保持在硬件窗口
tmux select-pane -t 0
tmux attach-session -t $SESSION