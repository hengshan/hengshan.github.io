# 快速入门指南

## 5分钟快速开始

### 步骤1: 安装依赖 (1分钟)

```bash
cd ~/projects/hengshan.github.io
bash .ai-agent/setup.sh
```

### 步骤2: 配置API密钥 (2分钟)

```bash
# 编辑环境变量文件
vim .ai-agent/.env
```

填入以下信息：
```bash
CLAUDE_API_KEY=sk-ant-...           # Claude API密钥
BLOG_EMAIL=your@gmail.com           # 你的Gmail
BLOG_EMAIL_PASSWORD=app-password    # Gmail应用专用密码
REVIEW_EMAIL=your@gmail.com         # 审阅邮箱（可相同）
```

**如何获取：**
- Claude API: https://console.anthropic.com/
- Gmail应用密码: https://myaccount.google.com/apppasswords

### 步骤3: 测试运行 (2分钟)

```bash
# 试运行（不发送邮件）
python .ai-agent/main.py --dry-run
```

如果成功，你会看到：
- ✓ 技术源监控完成
- ✓ 博客生成完成
- ✓ 代码验证通过
- ✓ 草稿已保存

### 步骤4: 正式运行

```bash
# 生成并发送审阅邮件
python .ai-agent/main.py

# 审阅后发布
python .ai-agent/main.py --publish

# 推送到GitHub
git add . && git commit -m "添加博客" && git push
```

## 设置定时任务

### 方法1: Crontab（推荐）

```bash
crontab -e
```

添加：
```
# 每天下午5点生成博客
0 17 * * * cd ~/projects/hengshan.github.io && /usr/bin/python3 .ai-agent/main.py
```

### 方法2: 手动运行

每天运行：
```bash
cd ~/projects/hengshan.github.io
python .ai-agent/main.py
```

## 工作流程

```
17:00 - 系统自动生成博客
     ↓
17:30 - 收到审阅邮件
     ↓
19:00 - 审阅草稿（建议时间）
     ↓
修改（如需要）
     ↓
发布: python .ai-agent/main.py --publish
     ↓
推送: git add . && git commit && git push
```

## 常用命令

```bash
# 生成博客
python .ai-agent/main.py

# 指定类别
python .ai-agent/main.py --category "CUDA/GPU编程"

# 试运行
python .ai-agent/main.py --dry-run

# 发布草稿
python .ai-agent/main.py --publish

# 发布指定草稿
python .ai-agent/main.py --publish --draft filename.markdown

# 查看草稿
ls -lh drafts/
cat drafts/最新草稿.markdown
```

## 自定义配置

编辑 `.ai-agent/config.yaml`:

```yaml
# 调整内容类别权重
content:
  categories:
    - name: "CUDA/GPU编程"
      weight: 0.3    # 增加或减少概率

# 修改生成时间（仅标记）
generation:
  schedule: "17:00"

# 调整AI参数
claude:
  temperature: 0.7   # 创造性 (0.0-1.0)
  max_tokens: 8000   # 最大长度
```

## 故障排查

### 问题1: API密钥错误
```bash
# 检查环境变量
cat .ai-agent/.env

# 确保密钥正确
export CLAUDE_API_KEY=your-key
python .ai-agent/main.py --dry-run
```

### 问题2: 邮件发送失败
```bash
# Gmail需要应用专用密码，不是常规密码
# 启用两步验证后生成：
# https://myaccount.google.com/apppasswords
```

### 问题3: 找不到模块
```bash
# 重新安装依赖
pip install -r .ai-agent/requirements.txt
```

### 问题4: 代码验证失败
```bash
# 关闭严格验证
vim .ai-agent/config.yaml

# 修改：
validation:
  python_syntax_check: false
  cuda_compile_check: false
```

## 下一步

1. 查看完整文档: `cat .ai-agent/README.md`
2. 自定义提示词: 编辑 `.ai-agent/prompts/`
3. 调整技术源: 编辑 `.ai-agent/sources/tech_sources.yaml`
4. 加入定时任务: `crontab -e`

## 获取帮助

```bash
python .ai-agent/main.py --help
```

## 成本估算

- Claude API: ~$2-5/月（每天1篇）
- 完全免费: 使用本地LLM（需要GPU）

---

**准备好了吗？运行你的第一个博客：**

```bash
python .ai-agent/main.py --dry-run
```
