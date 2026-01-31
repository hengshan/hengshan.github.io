#!/bin/bash
# AI博客自动生成 - Cron运行脚本
# 每天中午12点自动运行

# 设置错误时退出
set -e

# 定义路径
PROJECT_DIR="/home/hank/projects/hengshan.github.io"
LOG_DIR="$PROJECT_DIR/.ai-agent/logs"
LOG_FILE="$LOG_DIR/cron_$(date +\%Y\%m\%d).log"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 记录开始时间
echo "========================================" >> "$LOG_FILE"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# 切换到项目目录
cd "$PROJECT_DIR"

# 初始化 conda（根据你的 conda 安装位置调整）
# 方法1: 如果使用 miniconda/anaconda
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 方法2: 如果上面不工作，取消注释下面这行
# eval "$(conda shell.bash hook)"

# 激活虚拟环境（替换成你的环境名）
conda activate blog-ai-agent

# 运行博客生成脚本
echo "运行博客生成脚本..." >> "$LOG_FILE"
python .ai-agent/main.py >> "$LOG_FILE" 2>&1

# 记录结束状态
EXIT_CODE=$?
echo "" >> "$LOG_FILE"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "退出码: $EXIT_CODE" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 可选：保留最近30天的日志
find "$LOG_DIR" -name "cron_*.log" -mtime +30 -delete

exit $EXIT_CODE
