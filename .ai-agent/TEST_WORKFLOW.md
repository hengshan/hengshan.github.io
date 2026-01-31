# 完整工作流程测试指南

## 前置准备

### 1. 环境变量检查

```bash
# 检查API密钥
echo $ANTHROPIC_API_KEY | head -c 20

# 检查邮件配置（需要设置这三个）
echo $BLOG_EMAIL
echo $BLOG_EMAIL_PASSWORD | head -c 10
echo $REVIEW_EMAIL
```

### 2. 如果邮件变量未设置

#### 方式A: 临时设置（本次有效）
```bash
export BLOG_EMAIL="your@gmail.com"
export BLOG_EMAIL_PASSWORD="your-app-password"
export REVIEW_EMAIL="your@gmail.com"
```

#### 方式B: 永久设置（推荐）
```bash
# 添加到 ~/.zshrc.local
echo 'export BLOG_EMAIL="your@gmail.com"' >> ~/.zshrc.local
echo 'export BLOG_EMAIL_PASSWORD="your-app-password"' >> ~/.zshrc.local
echo 'export REVIEW_EMAIL="your@gmail.com"' >> ~/.zshrc.local

# 重新加载
source ~/.zshrc.local
```

#### 方式C: 使用 .env 文件
```bash
# 创建 .env 文件
cp .ai-agent/.env.example .ai-agent/.env
vim .ai-agent/.env  # 填入真实配置

# 加载环境变量
set -a
source .ai-agent/.env
set +a
```

### 3. Gmail应用专用密码

如果使用Gmail，需要：
1. 启用两步验证
2. 生成应用专用密码: https://myaccount.google.com/apppasswords
3. 选择"邮件"和"其他(自定义名称)"
4. 使用生成的16位密码（去掉空格）

## 🧪 完整测试流程

### 步骤1: 生成博客并发送审阅邮件

```bash
# 进入项目目录
cd ~/projects/hengshan.github.io

# 激活虚拟环境（如果使用）
source ./activate-ai-agent.sh  # 或 conda activate blog-ai-agent

# 生成博客（会发送邮件）
python .ai-agent/main.py
```

**预期输出：**
```
🚀 AI博客生成系统启动中...
✓ 系统初始化完成

============================================================
  AI博客生成 - 2026年01月05日
============================================================
📡 正在监控技术信息源...
  ✓ 从arXiv获取了 5 篇论文
  ✓ 从GitHub获取了 3 个更新
  ✓ 从技术博客获取了 8 篇文章

🎯 根据你的偏好，推荐以下 10 个话题：
  1. [arxiv] Flash Attention 3: Fast and Accurate... (评分: 15.2)
  2. [github] pytorch/pytorch - Performance optimization... (评分: 12.8)
  ...

🎯 选定话题: Flash Attention 3: Fast and Accurate...
📂 分类: 推理优化
🔗 来源: https://arxiv.org/abs/...

🤖 正在生成博客: Flash Attention 3...
  ✓ 生成成功

🔍 正在验证代码...
  代码块 1/3 [python]:
    ✓ Python语法正确
  ...

📊 验证结果:
  总计: 3 个代码块
  通过: 3
  失败: 0
  ✓ 所有代码验证通过

📊 质量评分: 87/100

  ✓ 草稿已保存: drafts/2026-01-05-flash-attention-3-fast-and-accur.markdown

📧 正在发送审阅邮件到 your@gmail.com...
  ✓ 邮件发送成功

============================================================
✅ 博客生成完成!
📄 草稿位置: drafts/2026-01-05-flash-attention-3-fast-and-accur.markdown
📧 审阅邮件已发送 (审阅后运行: python .ai-agent/main.py --publish)
============================================================
```

### 步骤2: 查看审阅邮件

1. 打开邮箱，查看主题为 "📝 AI博客草稿审阅 - YYYY-MM-DD - {topic}" 的邮件
2. 邮件包含：
   - 博客元信息（分类、话题、来源）
   - 内容统计（字数、代码块数）
   - 内容预览（前500字）
   - 快速命令参考

### 步骤3: 审阅草稿

```bash
# 查看草稿
cat drafts/2026-01-05-*.markdown

# 或用编辑器打开
vim drafts/2026-01-05-*.markdown

# 如果需要修改，直接编辑保存即可
```

**检查要点：**
- ✅ 标题准确
- ✅ 代码完整可运行
- ✅ 注释详细清晰
- ✅ 技术解释正确
- ✅ 符合你的博客风格

### 步骤4: 发布博客

```bash
# 发布（会自动复制到正确的目录）
python .ai-agent/main.py --publish
```

**预期输出：**
```
📤 准备发布博客...

📄 草稿: 2026-01-05-flash-attention-3-fast-and-accur.markdown
✓ 博客已复制到: blog-ai-enterprise-data/_posts/2026-01-05-flash-attention-3-fast-and-accur.markdown

✅ 博客发布成功!
📁 位置: blog-ai-enterprise-data/_posts/2026-01-05-flash-attention-3-fast-and-accur.markdown

接下来:
  1. 运行 git status 查看更改
  2. 运行 git add . && git commit -m '添加博客' && git push
  3. GitHub Pages 将自动部署
```

### 步骤5: Git提交和推送

```bash
# 查看更改
git status

# 应该看到新增的博客文件
# blog-ai-enterprise-data/_posts/2026-01-05-*.markdown

# 查看文件内容
git diff blog-ai-enterprise-data/_posts/2026-01-05-*.markdown

# 提交
git add blog-ai-enterprise-data/_posts/2026-01-05-*.markdown
git commit -m "添加博客: Flash Attention 3教程

🤖 AI生成并人工审阅
📅 $(date +%Y-%m-%d)"

# 推送到GitHub
git push origin main
```

### 步骤6: 验证部署

```bash
# 等待1-2分钟，然后访问你的博客
open https://hengshan.github.io
```

## 🐛 常见问题排查

### 问题1: 邮件发送失败

```
✗ 邮件发送失败: Authentication failed
```

**解决方案：**
1. 确认使用的是Gmail应用专用密码，不是常规密码
2. 检查环境变量是否正确设置
3. 尝试手动测试SMTP连接

### 问题2: API密钥错误

```
Error: Invalid API key
```

**解决方案：**
```bash
# 检查环境变量
echo $ANTHROPIC_API_KEY

# 重新设置
export ANTHROPIC_API_KEY="sk-ant-..."

# 或检查 ~/.zshrc.local
grep ANTHROPIC_API_KEY ~/.zshrc.local
```

### 问题3: 代码验证失败

```
✗ Python语法错误: ...
```

**解决方案：**
1. 这是正常的，AI生成的代码可能有小错误
2. 在草稿审阅时修改代码
3. 可以关闭严格验证（编辑config.yaml）

### 问题4: 发布找不到草稿

```
❌ 未找到草稿文件
```

**解决方案：**
```bash
# 列出所有草稿
ls -lh drafts/

# 手动指定草稿
python .ai-agent/main.py --publish --draft "2026-01-05-*.markdown"
```

## 📊 完整测试清单

- [ ] API密钥设置正确
- [ ] 邮件配置完成
- [ ] 虚拟环境激活（如果使用）
- [ ] 生成博客成功
- [ ] 收到审阅邮件
- [ ] 草稿内容满意
- [ ] 发布到正确目录
- [ ] Git提交成功
- [ ] 推送到GitHub
- [ ] 网站部署更新

## 🎯 下一步

测试成功后，可以：
1. 设置crontab定时任务
2. 调整内容类别权重
3. 自定义提示词模板
4. 添加更多技术信息源

## 📝 测试记录模板

```markdown
## 测试日期: YYYY-MM-DD

### 生成阶段
- [ ] API调用成功
- [ ] 获取技术话题
- [ ] 生成博客内容
- [ ] 代码验证通过
- [ ] 质量评分: __/100

### 审阅阶段
- [ ] 收到邮件
- [ ] 内容质量: 满意/需修改
- [ ] 修改内容: __

### 发布阶段
- [ ] 复制到正确目录
- [ ] Git提交
- [ ] 推送成功
- [ ] 网站更新

### 总结
耗时: __ 分钟
问题: __
改进建议: __
```

---

**准备好了吗？开始测试！**

```bash
# 一键开始
python .ai-agent/main.py
```
