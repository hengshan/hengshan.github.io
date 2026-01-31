# Python环境配置指南

## 两种方式对比

### 方式1: 虚拟环境（推荐） ⭐

**优点：**
- ✅ 依赖隔离，不污染系统环境
- ✅ 项目依赖版本明确可控
- ✅ 便于在其他机器复现
- ✅ 符合Python最佳实践
- ✅ 轻量级，启动快

**缺点：**
- 需要记得激活环境
- 需要额外的磁盘空间（~100MB）

### 方式2: 直接用Miniconda base

**优点：**
- ✅ 简单直接，无需配置
- ✅ 立即可用
- ✅ 不需要激活命令

**缺点：**
- ❌ 依赖会安装到base环境
- ❌ 可能与其他项目冲突
- ❌ 不符合最佳实践
- ❌ 难以追踪项目依赖

## 推荐：使用虚拟环境

对于这个项目，虽然依赖很少，但建议使用虚拟环境：

### 选项A: venv（Python标准库）

```bash
# 一键设置
bash .ai-agent/setup-venv.sh

# 选择 1) venv
```

**使用方法：**
```bash
# 激活环境
source .venv/bin/activate

# 运行AI Agent
python .ai-agent/main.py --dry-run

# 退出环境
deactivate
```

**快捷方式：**
```bash
# 使用快捷激活脚本
source ./activate-ai-agent.sh
```

### 选项B: Conda环境

```bash
# 一键设置
bash .ai-agent/setup-venv.sh

# 选择 2) conda
```

**使用方法：**
```bash
# 激活环境
conda activate blog-ai-agent

# 运行AI Agent
python .ai-agent/main.py --dry-run

# 退出环境
conda deactivate
```

## 如果选择直接用Miniconda base

如果你确定要用base环境：

```bash
# 确认当前Python
which python
# 应该输出: /home/hank/miniconda3/bin/python

# 安装依赖到base
pip install -r .ai-agent/requirements.txt

# 直接运行
/home/hank/miniconda3/bin/python .ai-agent/main.py --dry-run
```

## 定时任务配置

### 如果使用虚拟环境

**venv方式：**
```bash
crontab -e

# 添加
0 17 * * * cd ~/projects/hengshan.github.io && .venv/bin/python .ai-agent/main.py
```

**conda方式：**
```bash
crontab -e

# 添加（需要先找到conda的完整路径）
0 17 * * * cd ~/projects/hengshan.github.io && /home/hank/miniconda3/envs/blog-ai-agent/bin/python .ai-agent/main.py
```

### 如果使用Miniconda base

```bash
crontab -e

# 添加
0 17 * * * cd ~/projects/hengshan.github.io && /home/hank/miniconda3/bin/python .ai-agent/main.py
```

## 项目目录结构

```
hengshan.github.io/
├── .venv/                  # 虚拟环境（如果使用venv）
├── .ai-agent/             # AI Agent代码（被gitignore）
├── activate-ai-agent.sh   # 快捷激活脚本
└── ...
```

## 快速决策指南

**使用虚拟环境，如果：**
- ✅ 你想遵循Python最佳实践
- ✅ 你有多个Python项目
- ✅ 你想要清晰的依赖管理
- ✅ 你计划在其他机器复现环境

**使用Miniconda base，如果：**
- ✅ 这是你唯一的Python项目
- ✅ 你想要最简单的设置
- ✅ 你不关心依赖隔离
- ✅ 你的base环境很干净

## 常用命令速查

### venv环境

```bash
# 创建环境
python -m venv .venv

# 激活
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r .ai-agent/requirements.txt

# 查看已安装
pip list

# 退出
deactivate
```

### conda环境

```bash
# 创建环境
conda create -n blog-ai-agent python=3.10 -y

# 激活
conda activate blog-ai-agent

# 安装依赖
pip install -r .ai-agent/requirements.txt

# 查看环境列表
conda env list

# 退出
conda deactivate

# 删除环境
conda env remove -n blog-ai-agent
```

## 推荐的完整设置流程

```bash
# 1. 进入项目目录
cd ~/projects/hengshan.github.io

# 2. 一键设置虚拟环境
bash .ai-agent/setup-venv.sh
# 选择 1) venv（推荐）或 2) conda

# 3. 配置API密钥
cp .ai-agent/.env.example .ai-agent/.env
vim .ai-agent/.env  # 填入真实密钥

# 4. 测试运行
source ./activate-ai-agent.sh  # 激活环境
python .ai-agent/main.py --dry-run

# 5. 设置定时任务
crontab -e
# 添加: 0 17 * * * cd ~/projects/hengshan.github.io && /full/path/to/python .ai-agent/main.py
```

## 故障排查

### 问题1: 找不到模块

```bash
# 确认环境已激活
which python

# 应该输出虚拟环境的Python路径
# venv: /path/to/.venv/bin/python
# conda: /path/to/miniconda3/envs/blog-ai-agent/bin/python

# 重新安装依赖
pip install -r .ai-agent/requirements.txt
```

### 问题2: crontab没有运行

```bash
# 使用绝对路径
0 17 * * * cd /home/hank/projects/hengshan.github.io && /home/hank/projects/hengshan.github.io/.venv/bin/python .ai-agent/main.py

# 或者（如果用conda）
0 17 * * * cd /home/hank/projects/hengshan.github.io && /home/hank/miniconda3/envs/blog-ai-agent/bin/python .ai-agent/main.py
```

### 问题3: 环境冲突

```bash
# 删除并重建环境

# venv
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r .ai-agent/requirements.txt

# conda
conda env remove -n blog-ai-agent
conda create -n blog-ai-agent python=3.10 -y
conda activate blog-ai-agent
pip install -r .ai-agent/requirements.txt
```

## 我的建议

**推荐方案：venv虚拟环境**

原因：
1. Python标准库，无需额外安装
2. 轻量级，启动快
3. 项目依赖隔离
4. 符合Python社区最佳实践
5. 便于crontab配置（绝对路径简单）

**一键设置：**
```bash
bash .ai-agent/setup-venv.sh
```

选择选项1，然后按提示操作即可！
