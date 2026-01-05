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
