# AI博客自动生成系统

每天自动生成高质量AI技术博客，聚焦CUDA/GPU编程、ML/DL算法、强化学习、推理优化等硬核技术。

## 功能特点

- 🔍 **智能技术监控**: 自动从arXiv、GitHub、技术博客获取最新技术动态
- 🤖 **AI内容生成**: 使用Claude Sonnet 4.5生成深度技术教程
- ✅ **代码验证**: 自动验证生成的Python/CUDA代码语法
- 📧 **邮件审阅**: 生成后发送邮件通知，人工审阅后发布
- 📊 **质量评估**: 自动评估内容质量，提供改进建议

## 系统架构

```
技术源监控 → 话题推荐 → AI生成 → 代码验证 → 邮件审阅 → 发布博客
```

## 快速开始

### 1. 环境设置

```bash
cd ~/projects/hengshan.github.io

# 安装Python依赖
pip install -r .ai-agent/requirements.txt

# 配置环境变量
cp .ai-agent/.env.example .ai-agent/.env
vim .ai-agent/.env  # 填入你的API密钥和邮箱配置
```

### 2. 配置个性化设置

编辑 `.ai-agent/config.yaml`，调整：
- 内容类别权重
- 生成时间
- 技术信息源

### 3. 运行

```bash
# 生成今日博客（会发送审阅邮件）
python .ai-agent/main.py

# 试运行（不发送邮件）
python .ai-agent/main.py --dry-run

# 指定类别生成
python .ai-agent/main.py --category "CUDA/GPU编程"

# 发布已审阅的草稿
python .ai-agent/main.py --publish
```

## 定时任务设置

### 方法1: 使用crontab（推荐）

```bash
# 编辑crontab
crontab -e

# 添加定时任务（每天下午5点生成）
0 17 * * * cd ~/projects/hengshan.github.io && /usr/bin/python3 .ai-agent/main.py

# 可选：添加提醒（每天晚上7点发送审阅提醒）
0 19 * * * echo "别忘了审阅今天的博客草稿！运行: cd ~/projects/hengshan.github.io && python .ai-agent/main.py --publish" | mail -s "博客审阅提醒" your-email@gmail.com
```

### 方法2: 使用Python schedule库

创建 `.ai-agent/scheduler.py`:

```python
import schedule
import time
from main import BlogGenerationSystem

def job():
    system = BlogGenerationSystem()
    system.generate_daily_blog()

# 每天下午5点执行
schedule.every().day.at("17:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

运行:
```bash
nohup python .ai-agent/scheduler.py &
```

## 工作流程

### 每日生成流程

1. **技术监控** (17:00)
   - 检查arXiv最新论文
   - 监控GitHub trending
   - 扫描技术博客RSS

2. **内容生成**
   - 根据你的知识体系分析话题
   - 使用Claude API生成教程
   - 包含完整代码实现

3. **质量检查**
   - Python语法验证
   - CUDA编译检查（可选）
   - 内容质量评分

4. **发送审阅** (17:30)
   - 保存草稿到 `drafts/`
   - 发送HTML邮件通知
   - 包含内容预览和统计

5. **人工审阅** (19:00建议)
   - 打开草稿文件审阅
   - 根据需要修改内容
   - 运行发布命令

6. **发布博客**
   ```bash
   python .ai-agent/main.py --publish
   git add . && git commit -m "添加博客" && git push
   ```

## 目录结构

```
.ai-agent/
├── config.yaml              # 主配置文件
├── main.py                  # 主程序
├── requirements.txt         # Python依赖
├── .env                     # 环境变量（需创建）
├── modules/
│   ├── tech_monitor.py      # 技术源监控
│   ├── content_generator.py # 内容生成
│   ├── email_sender.py      # 邮件发送
│   └── code_validator.py    # 代码验证
├── prompts/                 # 提示词模板
│   ├── cuda_tutorial.txt
│   ├── ml_algorithm.txt
│   ├── rl_tutorial.txt
│   └── optimization.txt
└── sources/
    └── tech_sources.yaml    # 技术信息源配置

drafts/                      # 草稿文件夹
└── YYYY-MM-DD-*.markdown    # 生成的草稿
```

## 配置说明

### 内容类别权重

在 `config.yaml` 中调整各类别权重：

```yaml
content:
  categories:
    - name: "CUDA/GPU编程"
      weight: 0.3           # 30%概率
    - name: "ML/DL算法实现"
      weight: 0.25          # 25%概率
    # ...
```

### 技术信息源

在 `sources/tech_sources.yaml` 中添加/删除信息源：

```yaml
arxiv:
  categories:
    - id: "cs.LG"
      keywords: ["transformer", "attention"]

github:
  repos:
    - owner: "pytorch"
      name: "pytorch"
```

## 常见问题

### Q1: 如何获取Claude API密钥？

访问 https://console.anthropic.com/ 注册并创建API密钥。

### Q2: 邮件发送失败怎么办？

如果使用Gmail：
1. 启用两步验证
2. 生成应用专用密码: https://myaccount.google.com/apppasswords
3. 使用应用专用密码而非常规密码

### Q3: 如何修改生成时间？

两个地方：
1. `config.yaml` 中的 `generation.schedule`（仅作记录）
2. crontab 或 scheduler.py 中的实际定时设置

### Q4: 生成的代码有错误怎么办？

1. 系统会自动验证Python语法
2. 人工审阅时仔细检查代码
3. 可以在草稿中直接修改
4. 如果经常出错，调整提示词模板

### Q5: 如何跳过某天的生成？

```bash
# 停止cron任务
crontab -e  # 注释掉相关行

# 或者手动控制
python .ai-agent/main.py --dry-run  # 测试但不发送邮件
```

## 高级功能

### 自定义提示词模板

编辑 `.ai-agent/prompts/` 下的模板文件，调整生成风格。

### 添加新的技术源

在 `sources/tech_sources.yaml` 中添加新的RSS源或GitHub仓库。

### 代码验证配置

在 `config.yaml` 中：

```yaml
validation:
  python_syntax_check: true
  cuda_compile_check: false  # 需要nvcc
  run_simple_tests: true
```

## 维护建议

1. **每周检查**
   - 查看草稿质量
   - 调整类别权重
   - 更新技术源

2. **每月更新**
   - 更新Python依赖
   - 检查API使用量
   - 归档旧草稿

3. **备份**
   - 定期备份配置文件
   - 保存高质量草稿

## 故障排查

### 日志查看

```bash
# 手动运行并查看详细输出
python .ai-agent/main.py --dry-run

# 查看cron日志
grep CRON /var/log/syslog
```

### 重置系统

```bash
# 清理草稿
rm -rf drafts/*.markdown

# 重新安装依赖
pip install -r .ai-agent/requirements.txt --upgrade
```

## 成本估算

Claude API费用（按Sonnet 4.5计费）：
- 每篇博客约 8000 tokens输出
- 每月30篇约 240K tokens
- 预估成本: ~$2-5/月

## 贡献

欢迎改进建议！关键改进方向：
- 更准确的技术话题识别
- 更好的代码生成质量
- 更智能的分类判断

## 许可

MIT License

---

**Made with ❤️ by Hank Li**
