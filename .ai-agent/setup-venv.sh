#!/bin/bash
# 创建和配置虚拟环境

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

echo "🐍 AI博客生成系统 - 虚拟环境设置"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检测Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "❌ 未找到Python，请先安装Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "📌 使用Python: $PYTHON_CMD ($PYTHON_VERSION)"

# 选择虚拟环境方式
echo ""
echo "选择虚拟环境方式:"
echo "  1) venv (Python标准库，推荐)"
echo "  2) conda (如果你习惯用conda)"
echo ""
read -p "请选择 [1/2]: " choice

case $choice in
    1)
        echo "🔧 使用venv创建虚拟环境..."
        $PYTHON_CMD -m venv "$VENV_DIR"

        echo "✓ 虚拟环境已创建: $VENV_DIR"
        echo ""
        echo "📦 安装依赖..."
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip
        pip install -r "$PROJECT_ROOT/.ai-agent/requirements.txt"

        echo ""
        echo "✅ 设置完成！"
        echo ""
        echo "激活虚拟环境："
        echo "  source .venv/bin/activate"
        echo ""
        echo "运行AI Agent："
        echo "  python .ai-agent/main.py --dry-run"
        echo ""
        echo "退出虚拟环境："
        echo "  deactivate"
        ;;

    2)
        echo "🔧 使用conda创建虚拟环境..."

        if ! command -v conda &> /dev/null; then
            echo "❌ 未找到conda，请确保已安装miniconda/anaconda"
            exit 1
        fi

        ENV_NAME="blog-ai-agent"

        # 检查环境是否已存在
        if conda env list | grep -q "^$ENV_NAME "; then
            echo "⚠️  环境 $ENV_NAME 已存在"
            read -p "是否删除并重建? [y/N]: " rebuild
            if [[ "$rebuild" =~ ^[Yy]$ ]]; then
                conda env remove -n $ENV_NAME -y
            else
                echo "使用现有环境"
                conda activate $ENV_NAME
                pip install -r "$PROJECT_ROOT/.ai-agent/requirements.txt"
                exit 0
            fi
        fi

        conda create -n $ENV_NAME python=3.10 -y

        echo "✓ Conda环境已创建: $ENV_NAME"
        echo ""
        echo "📦 安装依赖..."

        # 激活并安装依赖
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $ENV_NAME
        pip install -r "$PROJECT_ROOT/.ai-agent/requirements.txt"

        echo ""
        echo "✅ 设置完成！"
        echo ""
        echo "激活conda环境："
        echo "  conda activate $ENV_NAME"
        echo ""
        echo "运行AI Agent："
        echo "  python .ai-agent/main.py --dry-run"
        echo ""
        echo "退出环境："
        echo "  conda deactivate"
        ;;

    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

# 创建激活脚本快捷方式
cat > "$PROJECT_ROOT/activate-ai-agent.sh" << 'EOF'
#!/bin/bash
# 快捷激活脚本

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "🐍 激活venv环境..."
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "✓ 虚拟环境已激活"
elif conda env list | grep -q "blog-ai-agent"; then
    echo "🐍 激活conda环境..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate blog-ai-agent
    echo "✓ Conda环境已激活"
else
    echo "❌ 未找到虚拟环境"
    echo "运行: bash .ai-agent/setup-venv.sh"
    exit 1
fi

echo ""
echo "当前Python: $(which python)"
echo "Python版本: $(python --version)"
echo ""
echo "运行AI Agent: python .ai-agent/main.py --dry-run"
EOF

chmod +x "$PROJECT_ROOT/activate-ai-agent.sh"

echo ""
echo "💡 已创建快捷激活脚本: ./activate-ai-agent.sh"
echo "   使用方法: source ./activate-ai-agent.sh"
