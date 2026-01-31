# 博客发布流程说明

## ✅ 确认：完全符合Jekyll规范

系统已经正确配置，会自动处理Jekyll的文件命名和目录结构。

## 📋 完整发布流程

### 1️⃣ 生成阶段 (drafts/)

```bash
python .ai-agent/main.py
```

**生成的文件：**
```
drafts/2026-01-05-flash-attention-3-fast-and-accur.markdown
```

**文件命名规则：**
- 格式：`YYYY-MM-DD-slug.markdown`
- 符合Jekyll规范 ✅
- slug从标题提取（前30字符，转小写，空格转横杠）

**Front Matter（自动添加）：**
```yaml
---
layout: post-wide
title: "Flash Attention 3: Fast and Accurate Attention"
date: 2026-01-05 17:23:45 +0800
category: AI                    # 根据内容自动确定
author: Hank Li
source_url: https://arxiv.org/abs/...
generated_by: AI Agent
---
```

### 2️⃣ 审阅阶段

```bash
# 查看草稿
cat drafts/2026-01-05-*.markdown

# 编辑草稿（如需修改）
vim drafts/2026-01-05-*.markdown
```

### 3️⃣ 发布阶段（自动复制到正确目录）

```bash
python .ai-agent/main.py --publish
```

**系统会自动：**

1. **读取草稿的category标签**
   ```yaml
   category: AI
   ```

2. **根据category映射到输出目录**（在config.yaml中配置）
   ```yaml
   categories:
     - name: "CUDA/GPU编程"
       category_tag: "Tools"
       output_dir: "blog-spatial-tool/_posts"     # → CUDA博客

     - name: "ML/DL算法实现"
       category_tag: "AI"
       output_dir: "blog-ai-enterprise-data/_posts"  # → AI博客

     - name: "强化学习"
       category_tag: "AI"
       output_dir: "blog-ai-enterprise-data/_posts"  # → AI博客

     - name: "推理优化"
       category_tag: "AI"
       output_dir: "blog-ai-enterprise-data/_posts"  # → AI博客

     - name: "优化与科学计算"
       category_tag: "Optimization"
       output_dir: "blog-spatial-optimization/_posts" # → 优化博客
   ```

3. **复制文件到正确目录**
   ```
   drafts/2026-01-05-flash-attention-3-fast-and-accur.markdown
       ↓ copy
   blog-ai-enterprise-data/_posts/2026-01-05-flash-attention-3-fast-and-accur.markdown
   ```

4. **保持原文件名不变** ✅
   - Jekyll要求的 `YYYY-MM-DD-*.markdown` 格式
   - 直接复制，不重命名

### 4️⃣ Git推送

```bash
git status  # 查看新增的博客文件

git add blog-ai-enterprise-data/_posts/2026-01-05-*.markdown
git commit -m "添加博客: Flash Attention 3教程"
git push
```

### 5️⃣ GitHub Pages自动部署 ✅

推送后GitHub Pages会自动：
1. 检测到 `_posts/` 目录下的新文件
2. Jekyll构建网站
3. 部署到 https://hengshan.github.io

## 📂 目录结构示例

```
hengshan.github.io/
├── blog-ai-enterprise-data/
│   └── _posts/
│       ├── 2024-10-17-learn-to-build-gpt-blog.markdown
│       └── 2026-01-05-flash-attention-3-fast-and-accur.markdown  ← 新博客
│
├── blog-spatial-tool/
│   └── _posts/
│       └── 2025-08-18-spatial-tool-blog-cuda13-01.markdown
│
├── blog-spatial-optimization/
│   └── _posts/
│       └── ...
│
└── drafts/  (不会提交到GitHub)
    └── 2026-01-05-flash-attention-3-fast-and-accur.markdown
```

## 🎯 分类映射逻辑

| 内容分类 | category标签 | 输出目录 |
|---------|-------------|---------|
| CUDA/GPU编程 | Tools | blog-spatial-tool/_posts |
| ML/DL算法实现 | AI | blog-ai-enterprise-data/_posts |
| 强化学习 | AI | blog-ai-enterprise-data/_posts |
| 推理优化 | AI | blog-ai-enterprise-data/_posts |
| 优化与科学计算 | Optimization | blog-spatial-optimization/_posts |

## ✅ Jekyll规范检查

### 文件命名 ✅
- ✓ 格式：`YYYY-MM-DD-title.markdown`
- ✓ 日期使用连字符
- ✓ 扩展名：`.markdown` 或 `.md`

### Front Matter ✅
- ✓ 使用YAML格式（三横杠包裹）
- ✓ 必需字段：`layout`, `title`, `date`
- ✓ 可选字段：`category`, `author`, `source_url`

### 目录结构 ✅
- ✓ 文章在 `_posts/` 子目录下
- ✓ 支持多个博客分类（blog-ai-enterprise-data, blog-spatial-tool等）

## 🔍 验证发布流程

```bash
# 1. 测试生成（不发送邮件）
python .ai-agent/main.py --dry-run

# 2. 检查草稿
ls -lh drafts/
cat drafts/$(ls -t drafts/ | head -1)

# 3. 测试发布（不会真正推送）
python .ai-agent/main.py --publish

# 4. 验证文件位置
ls -lh blog-ai-enterprise-data/_posts/

# 5. 检查front matter格式
head -15 blog-ai-enterprise-data/_posts/2026-01-05-*.markdown
```

## 常见问题

### Q1: 文件会被重命名吗？
**A:** 不会。文件名在生成时就符合Jekyll规范，发布时只是复制，保持原名。

### Q2: 如何手动指定输出目录？
**A:** 编辑草稿的front matter，修改 `category` 标签即可。系统会根据category映射到对应目录。

### Q3: 发布后草稿会被删除吗？
**A:** 不会。草稿会保留在 `drafts/` 文件夹（根据config.yaml的`keep_history`设置）。

### Q4: 如何修改分类映射？
**A:** 编辑 `.ai-agent/config.yaml` 中的 `content.categories` 部分。

### Q5: 支持自定义输出目录吗？
**A:** 支持。在 `config.yaml` 中添加新的category配置：
```yaml
- name: "新分类"
  weight: 0.1
  output_dir: "blog-new-category/_posts"
  category_tag: "NewCategory"
```

## 🎉 总结

✅ **文件命名**：自动符合Jekyll规范
✅ **目录映射**：根据category自动选择
✅ **Front Matter**：自动生成完整配置
✅ **发布流程**：一键复制到正确位置
✅ **Git集成**：可选自动commit（默认手动）

**你只需要：**
1. 审阅草稿内容
2. 运行 `python .ai-agent/main.py --publish`
3. `git add && git commit && git push`

其他的都由系统自动处理！
