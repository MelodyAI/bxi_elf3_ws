#!/bin/bash

# 定义进程名称
PROCESS_NAME="remote_controller"

echo "正在尝试关闭进程: $PROCESS_NAME ..."

# 获取进程 PID
# pgrep -f 会匹配包含该字符串的完整命令行
PID=$(pgrep -f "$PROCESS_NAME")

if [ -z "$PID" ]; then
    echo "未找到正在运行的 $PROCESS_NAME 进程。"
else
    echo "发现 PID: $PID，正在发送终止信号..."
    
    # 先尝试温柔地终止 (SIGTERM)
    kill $PID
    
    # 等待一秒检查是否成功，如果没有则强制结束 (SIGKILL)
    sleep 1
    if ps -p $PID > /dev/null; then
        echo "进程未响应，正在强制关闭 (kill -9)..."
        kill -9 $PID
    fi
    
    echo "进程 $PROCESS_NAME 已关闭。"
fi