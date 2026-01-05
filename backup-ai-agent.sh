#!/bin/bash
# AI博客生成系统备份脚本
# 备份敏感配置文件到本地

set -e

BACKUP_DIR=~/backups/blog-ai-agent
DATE=$(date +%Y%m%d-%H%M%S)

echo "🔒 开始备份AI博客生成系统..."

# 创建备份目录
mkdir -p "$BACKUP_DIR"

# 检查源文件是否存在
if [ ! -d ".ai-agent" ]; then
    echo "❌ .ai-agent 目录不存在"
    exit 1
fi

# 备份配置文件
echo "📦 备份配置文件..."
tar -czf "$BACKUP_DIR/ai-agent-config-$DATE.tar.gz" \
    .ai-agent/.env \
    .ai-agent/config.yaml \
    .ai-agent/prompts/ \
    .ai-agent/sources/ \
    2>/dev/null

if [ $? -eq 0 ]; then
    echo "  ✓ 配置文件已备份"
else
    echo "  ⚠️  部分配置文件备份失败（可能文件不存在）"
fi

# 备份草稿（如果存在）
if [ -d "drafts" ] && [ "$(ls -A drafts)" ]; then
    echo "📝 备份草稿文件..."
    tar -czf "$BACKUP_DIR/drafts-$DATE.tar.gz" drafts/
    echo "  ✓ 草稿已备份"
fi

# 清理旧备份（保留最近10个）
echo "🧹 清理旧备份（保留最近10个）..."
ls -t "$BACKUP_DIR"/ai-agent-config-*.tar.gz | tail -n +11 | xargs -r rm
ls -t "$BACKUP_DIR"/drafts-*.tar.gz 2>/dev/null | tail -n +11 | xargs -r rm

# 显示备份信息
echo ""
echo "✅ 备份完成！"
echo "📂 备份位置: $BACKUP_DIR"
echo "📊 备份列表:"
ls -lh "$BACKUP_DIR" | tail -n +2

echo ""
echo "💡 恢复备份："
echo "   tar -xzf $BACKUP_DIR/ai-agent-config-$DATE.tar.gz"
