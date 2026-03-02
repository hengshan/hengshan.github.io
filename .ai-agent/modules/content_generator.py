"""
内容生成模块
使用 Claude Code CLI 生成高质量技术博客
（不使用 API，使用 Claude Code 订阅）
"""

import os
import subprocess

# 确保 ~/.local/bin 在 PATH 中 (claude CLI 安装位置)
local_bin = os.path.expanduser("~/.local/bin")
if local_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH', '')}"
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional


class ContentGenerator:
    """博客内容生成器 - 使用 Claude Code CLI"""

    def __init__(self, config: Dict):
        # 检查 claude CLI 是否可用
        try:
            result = subprocess.run(
                ['claude', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("Claude CLI 返回错误")
            print(f"  ✓ 使用 Claude Code CLI: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError(
                "未找到 Claude CLI。请安装 Claude Code:\n"
                "  npm install -g @anthropic-ai/claude-code\n"
                "或参考: https://claude.ai/code"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude CLI 响应超时")

        self.model = config['claude'].get('model', 'sonnet')  # CLI 使用别名
        self.config = config

    def load_prompt_template(self, category: str) -> str:
        """加载对应类别的提示词模板（使用 v2 洞见优化版）"""
        prompt_map = {
            'CUDA/GPU编程': 'cuda_tutorial_v2.txt',
            'ML/DL算法实现': 'ml_algorithm_v2.txt',
            'Spatial Intelligence': 'spatial_intelligence_v2.txt',
            '强化学习': 'rl_tutorial_v2.txt',
            '推理优化': 'ml_algorithm_v2.txt',  # 复用ML模板
            '优化与科学计算': 'optimization_v2.txt'
        }

        prompt_file = prompt_map.get(category, 'ml_algorithm.txt')
        prompt_path = Path('.ai-agent/prompts') / prompt_file

        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def analyze_knowledge_base(self) -> str:
        """分析现有知识体系，生成上下文"""
        cuda_path = Path(self.config['knowledge_base']['cuda_tutorials']).expanduser()

        context = []
        context.append("# 用户现有知识体系\n")

        # 分析CUDA教程进度
        if cuda_path.exists():
            context.append("## CUDA学习进度")
            try:
                outline_file = cuda_path / "教程大纲.md"
                if outline_file.exists():
                    with open(outline_file, 'r', encoding='utf-8') as f:
                        context.append(f.read()[:2000])  # 前2000字符
            except Exception as e:
                print(f"Warning: 无法读取CUDA大纲: {e}")

        # 分析已有博客主题
        context.append("\n## 已发布博客主题")
        for ref_post in self.config['knowledge_base']['reference_posts']:
            try:
                with open(ref_post, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:50]  # 前50行
                    title = [l for l in lines if 'title:' in l]
                    if title:
                        context.append(f"- {title[0].strip()}")
            except:
                pass

        return '\n'.join(context)

    def generate_blog_post(self, tech_topic: Dict, category: str) -> Dict:
        """生成博客文章"""
        print(f"\n🤖 正在生成博客: {tech_topic['title'][:50]}...")

        # 加载提示词模板
        prompt_template = self.load_prompt_template(category)

        # 分析知识体系
        knowledge_context = self.analyze_knowledge_base()

        # 构建完整提示词
        full_prompt = f"""
{knowledge_context}

---

{prompt_template}

---

# 今日技术话题
来源: {tech_topic['source']}
标题: {tech_topic['title']}
摘要: {tech_topic.get('summary', tech_topic.get('description', ''))}
链接: {tech_topic['url']}

请基于以上信息，创建一篇深度技术教程博客。

要求：
1. 文章要有教育意义，不是简单的新闻摘要
2. 必须包含完整的代码实现（可运行）
3. 详细的中文注释
4. 从简单到复杂的渐进式讲解
5. 性能分析和优化建议
6. 适合中高级开发者

请直接输出Markdown格式的博客内容，不需要额外的解释。
"""

        # 使用 Claude Code CLI 生成
        try:
            # 将 prompt 写入临时文件（避免命令行长度限制）
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(full_prompt)
                prompt_file = f.name

            try:
                # 调用 Claude CLI
                # 使用 -p (print) 模式，禁用工具以纯文本生成
                # cwd=/tmp：避免加载项目 memory，防止 <function_calls> XML 污染输出
                result = subprocess.run(
                    [
                        'claude',
                        '-p',  # print mode (non-interactive)
                        '--model', 'sonnet',  # 使用 sonnet 模型
                        '--tools', '',  # 禁用工具，纯文本生成
                        '--no-session-persistence',  # 不读写 session，防止 memory 干扰
                        '--dangerously-skip-permissions',  # 跳过权限检查
                    ],
                    input=full_prompt,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10分钟超时
                    cwd='/tmp',  # 用 /tmp 避免加载项目 memory，防止 tool call XML 泄漏到输出
                )

                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout or "Unknown error"
                    raise RuntimeError(f"Claude CLI 错误: {error_msg}")

                blog_content = self._strip_tool_calls(result.stdout.strip())

            finally:
                # 清理临时文件
                try:
                    os.unlink(prompt_file)
                except:
                    pass

            if not blog_content:
                raise RuntimeError("Claude CLI 返回空内容")

            # 提取中文标题（第一个 # 标题）并移除
            chinese_title = None
            lines = blog_content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('# '):
                    chinese_title = line[2:].strip()
                    # 移除这个一级标题行（以及紧跟的空行）
                    lines = lines[:i] + lines[i+1:]
                    while lines and lines[0].strip() == '':
                        lines = lines[1:]
                    blog_content = '\n'.join(lines)
                    break

            # 如果没有提取到中文标题，使用原标题
            if not chinese_title:
                chinese_title = tech_topic['title']

            # 生成文件名和元数据
            date_str = datetime.now().strftime('%Y-%m-%d')
            # 从标题提取简短slug
            slug = tech_topic['title'][:50].lower()
            slug = ''.join(c if c.isalnum() or c == ' ' else '' for c in slug)
            slug = slug.strip().replace(' ', '-')
            # 去除连续的短横线
            while '--' in slug:
                slug = slug.replace('--', '-')
            slug = slug[:40]  # 限制长度

            # 检查同名文件是否已存在，如果是则追加序号避免覆盖
            base_filename = f"{date_str}-{slug}"
            filename = f"{base_filename}.markdown"
            drafts_dir = Path(self.config.get('drafts', {}).get('save_location', 'drafts'))
            counter = 2
            while (drafts_dir / filename).exists():
                filename = f"{base_filename}-{counter}.markdown"
                counter += 1

            # 添加Jekyll front matter（使用中文标题）
            # 处理标题中的引号，防止 YAML 解析失败
            safe_title = chinese_title.replace('"', '\\"').replace('\u201c', '\u201c').replace('\u201d', '\u201d')
            # 如果标题包含中文引号，用单引号包裹以避免 YAML 问题
            if '\u201c' in chinese_title or '\u201d' in chinese_title or '"' in chinese_title:
                title_line = f"title: '{chinese_title}'"
            else:
                title_line = f'title: "{chinese_title}"'
            front_matter = f"""---
layout: post-wide
{title_line}
date: {(datetime.now() - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S +0800')}
category: {self._get_category_tag(category)}
author: Hank Li
use_math: true
source_url: {tech_topic['url']}
generated_by: Claude Code CLI
---

"""

            full_content = front_matter + blog_content

            print("  ✓ 生成成功")

            return {
                'filename': filename,
                'content': full_content,
                'category': category,
                'tech_topic': tech_topic,
                'word_count': len(blog_content),
                'has_code': '```' in blog_content
            }

        except subprocess.TimeoutExpired:
            print("  ✗ 生成超时（10分钟）")
            raise RuntimeError("Claude CLI 生成超时")
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
            raise

    def _strip_tool_calls(self, text: str) -> str:
        """移除 Claude Code CLI 可能泄漏到输出的 <function_calls>...</function_calls> XML 块。
        即使 --tools '' 已禁用工具，某些版本的 CLI 仍会将 tool call XML 打印到 stdout。"""
        import re
        # 移除完整的 <function_calls> ... </function_calls> 块（含嵌套内容，跨行）
        text = re.sub(r'<function_calls>.*?</function_calls>\s*', '', text, flags=re.DOTALL)
        # 兜底：移除单独出现的 <invoke>/<parameter>/<function_calls> 残留标签
        text = re.sub(r'</?(?:function_calls|invoke|parameter)[^>]*>\s*', '', text, flags=re.DOTALL)
        return text.strip()

    def _get_category_tag(self, category: str) -> str:
        """获取分类标签"""
        tag_map = {
            'CUDA/GPU编程': 'Tools',
            'ML/DL算法实现': 'AI',
            'Spatial Intelligence': 'Spatial Intelligence',
            '强化学习': 'AI',
            '推理优化': 'AI',
            '优化与科学计算': 'Optimization'
        }
        return tag_map.get(category, 'AI')

    def save_draft(self, blog_data: Dict, draft_dir: str = 'drafts') -> str:
        """保存草稿"""
        draft_path = Path(draft_dir) / blog_data['filename']
        draft_path.parent.mkdir(parents=True, exist_ok=True)

        with open(draft_path, 'w', encoding='utf-8') as f:
            f.write(blog_data['content'])

        print(f"  ✓ 草稿已保存: {draft_path}")
        return str(draft_path)


if __name__ == "__main__":
    # 测试
    import yaml

    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    generator = ContentGenerator(config)

    # 测试话题
    test_topic = {
        'title': 'Flash Attention 3: Fast and Accurate Attention with Asynchrony and Low-precision',
        'summary': 'New advances in efficient attention mechanisms...',
        'url': 'https://arxiv.org/abs/2407.08608',
        'source': 'arxiv'
    }

    blog = generator.generate_blog_post(test_topic, 'CUDA/GPU编程')
    generator.save_draft(blog)
