# AI博客生成系统备份指南

## 重要说明

`.ai-agent/` 和 `drafts/` 文件夹已添加到 `.gitignore`，**不会**提交到GitHub public repo。

这确保了：
- ✅ API密钥和敏感信息不会泄露
- ✅ 草稿保持私密
- ✅ 工具代码不会出现在GitHub Pages

## 需要备份的文件

### 1. 敏感配置文件（必须备份）
```
.ai-agent/.env              # API密钥、邮箱密码
.ai-agent/config.yaml       # 个性化配置
```

### 2. 自定义内容（建议备份）
```
.ai-agent/prompts/*.txt     # 自定义提示词模板
.ai-agent/sources/tech_sources.yaml  # 技术信息源
```

### 3. 草稿和元数据（可选备份）
```
drafts/*.markdown           # 未发布的草稿
drafts/*.meta.yaml         # 生成元数据
```

## 备份方案

### 方案1: 本地备份（推荐）

创建备份脚本：

```bash
#!/bin/bash
# backup-ai-agent.sh

BACKUP_DIR=~/backups/blog-ai-agent
DATE=$(date +%Y%m%d)

mkdir -p $BACKUP_DIR

# 备份配置
tar -czf $BACKUP_DIR/ai-agent-config-$DATE.tar.gz \
  .ai-agent/.env \
  .ai-agent/config.yaml \
  .ai-agent/prompts/ \
  .ai-agent/sources/

# 备份草稿（可选）
tar -czf $BACKUP_DIR/drafts-$DATE.tar.gz drafts/

echo "✓ 备份完成: $BACKUP_DIR"
```

使用：
```bash
chmod +x backup-ai-agent.sh
./backup-ai-agent.sh
```

### 方案2: 私有Git仓库

创建一个**私有**GitHub仓库专门存储配置：

```bash
# 1. 在GitHub创建私有仓库: blog-ai-agent-private

# 2. 初始化本地仓库
cd ~/projects/hengshan.github.io/.ai-agent
git init
git add .
git commit -m "Initial commit of AI agent config"

# 3. 推送到私有仓库
git remote add origin git@github.com:your-username/blog-ai-agent-private.git
git push -u origin main
```

### 方案3: 加密备份到云盘

```bash
# 使用GPG加密
tar -czf - .ai-agent/ drafts/ | \
  gpg --symmetric --cipher-algo AES256 -o ~/Dropbox/blog-ai-backup.tar.gz.gpg

# 解密恢复
gpg -d ~/Dropbox/blog-ai-backup.tar.gz.gpg | tar -xzf -
```

## 在新机器上恢复

### 1. 克隆博客仓库
```bash
git clone https://github.com/your-username/hengshan.github.io.git
cd hengshan.github.io
```

### 2. 恢复AI Agent系统
```bash
# 如果使用私有Git仓库
cd .ai-agent
git clone https://github.com/your-username/blog-ai-agent-private.git tmp
cp -r tmp/* .
rm -rf tmp

# 或从备份恢复
tar -xzf ~/backups/blog-ai-agent/ai-agent-config-YYYYMMDD.tar.gz
```

### 3. 安装依赖
```bash
bash .ai-agent/setup.sh
```

### 4. 验证
```bash
python .ai-agent/main.py --dry-run
```

## 定期备份建议

添加到crontab（每周备份）：

```bash
# 每周日凌晨2点备份
0 2 * * 0 cd ~/projects/hengshan.github.io && ./backup-ai-agent.sh
```

## 安全检查清单

- [ ] `.ai-agent/` 在 `.gitignore` 中
- [ ] `drafts/` 在 `.gitignore` 中
- [ ] `.ai-agent/.env` 文件权限设置为 600
- [ ] 定期备份配置文件
- [ ] 私有仓库设置为 Private
- [ ] 不要在public repo提交 `.env.example` 的实际值

## 验证 .gitignore 是否生效

```bash
# 检查这两个文件夹不应出现在git status中
git status

# 强制检查是否被忽略
git check-ignore .ai-agent/
git check-ignore drafts/

# 应该返回这两个路径，表示已被忽略
```

## 紧急情况：已经commit了敏感信息

如果不小心提交了敏感信息：

```bash
# 1. 从历史中移除
git filter-branch --force --index-filter \
  "git rm -rf --cached --ignore-unmatch .ai-agent/" \
  --prune-empty --tag-name-filter cat -- --all

# 2. 强制推送
git push origin --force --all

# 3. 立即更换所有API密钥！
```

## 文件分离的好处

✅ **安全性**: API密钥不会泄露到public repo
✅ **隐私性**: 草稿和个人配置保持私密
✅ **灵活性**: 可以在不同机器使用不同配置
✅ **干净性**: GitHub Pages只显示博客内容，不包含工具代码

---

**重要**: 虽然这些文件夹不在GitHub上，但一定要做好本地备份！
