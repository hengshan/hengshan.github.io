#!/bin/bash
# AI博客生成系统安装脚本

set -e

echo "🚀 开始安装AI博客生成系统..."

# 推荐使用虚拟环境
echo ""
echo "⚠️  建议使用虚拟环境（更好的依赖管理）"
echo ""
read -p "是否创建虚拟环境? (推荐) [Y/n]: " use_venv

if [[ ! "$use_venv" =~ ^[Nn]$ ]]; then
    echo "📦 运行虚拟环境设置脚本..."
    bash .ai-agent/setup-venv.sh
else
    echo "📌 使用系统Python环境"
    # 检查Python版本
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    echo "   Python版本: $python_version"

    # 安装依赖
    echo "📦 安装Python依赖..."
    pip3 install -r .ai-agent/requirements.txt
fi

# 创建环境变量文件
if [ ! -f .ai-agent/.env ]; then
    echo "📝 创建环境变量文件..."
    cp .ai-agent/.env.example .ai-agent/.env
    echo "   ⚠️  请编辑 .ai-agent/.env 填入你的API密钥和邮箱配置"
else
    echo "   ✓ .env 文件已存在"
fi

# 创建草稿目录
echo "📁 创建草稿目录..."
mkdir -p drafts

# 设置权限
echo "🔐 设置文件权限..."
chmod +x .ai-agent/main.py
chmod 600 .ai-agent/.env 2>/dev/null || true

# 测试运行
echo "🧪 运行测试..."
read -p "是否运行试运行测试? (y/n): " run_test

if [ "$run_test" = "y" ]; then
    echo "   运行试运行模式..."
    python3 .ai-agent/main.py --dry-run
fi

# 配置crontab
echo ""
echo "⏰ 定时任务设置"
echo "建议添加到crontab:"
echo "   0 17 * * * cd $(pwd) && /usr/bin/python3 .ai-agent/main.py"
echo ""
read -p "是否现在配置crontab? (y/n): " setup_cron

if [ "$setup_cron" = "y" ]; then
    # 备份当前crontab
    crontab -l > /tmp/crontab.backup 2>/dev/null || true

    # 添加新任务
    (crontab -l 2>/dev/null; echo "# AI博客生成系统"; echo "0 17 * * * cd $(pwd) && /usr/bin/python3 .ai-agent/main.py") | crontab -

    echo "   ✓ Crontab已配置（每天17:00运行）"
    echo "   查看: crontab -l"
fi

echo ""
echo "✅ 安装完成！"
echo ""
echo "下一步："
echo "  1. 编辑 .ai-agent/.env 填入API密钥"
echo "  2. 编辑 .ai-agent/config.yaml 调整配置"
echo "  3. 运行测试: python3 .ai-agent/main.py --dry-run"
echo "  4. 开始使用: python3 .ai-agent/main.py"
echo ""
echo "详细文档: cat .ai-agent/README.md"
