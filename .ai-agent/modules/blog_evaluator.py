"""
博客质量评估模块
使用 Claude Code CLI 评估生成的博客质量
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Dict, Optional


class BlogEvaluator:
    """博客质量评估器 - 使用 Claude Code CLI"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        
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
        except FileNotFoundError:
            raise RuntimeError("未找到 Claude CLI")

    def evaluate_blog(self, blog_content: str, source_paper: Dict) -> Dict:
        """
        评估博客质量
        
        返回:
        {
            'overall_score': 1-10,
            'content_depth': {
                'score': 1-10,
                'comments': str
            },
            'code_quality': {
                'score': 1-10,
                'issues': [str],
                'runnable': bool
            },
            'structure': {
                'text_ratio': float,  # 文字占比
                'code_ratio': float,  # 代码占比
                'balanced': bool,
                'comments': str
            },
            'summary': str,
            'suggestions': [str]
        }
        """
        print("\n🔍 正在评估博客质量...")

        # 计算文字/代码比例
        text_code_ratio = self._calculate_ratio(blog_content)
        
        # 构建评估提示词
        evaluation_prompt = self._build_evaluation_prompt(
            blog_content, 
            source_paper,
            text_code_ratio
        )

        # 调用 Claude CLI 进行评估
        try:
            result = subprocess.run(
                [
                    'claude',
                    '-p',
                    '--model', 'sonnet',
                    '--tools', '',
                    '--dangerously-skip-permissions',
                ],
                input=evaluation_prompt,
                capture_output=True,
                text=True,
                timeout=180,  # 3分钟超时
            )

            if result.returncode != 0:
                print(f"  ⚠️ 评估失败: {result.stderr}")
                return self._default_evaluation(text_code_ratio)

            response = result.stdout.strip()
            evaluation = self._parse_evaluation(response, text_code_ratio)
            
            print(f"  ✓ 评估完成")
            return evaluation

        except subprocess.TimeoutExpired:
            print("  ⚠️ 评估超时")
            return self._default_evaluation(text_code_ratio)
        except Exception as e:
            print(f"  ⚠️ 评估错误: {e}")
            return self._default_evaluation(text_code_ratio)

    def _calculate_ratio(self, content: str) -> Dict:
        """计算文字和代码的比例"""
        lines = content.split('\n')
        
        in_code_block = False
        code_lines = 0
        text_lines = 0
        code_chars = 0
        text_chars = 0
        
        # 跳过 front matter
        content_start = 0
        if lines[0].strip() == '---':
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '---':
                    content_start = i + 1
                    break
        
        for line in lines[content_start:]:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                code_lines += 1
                code_chars += len(line)
            else:
                # 跳过空行
                if line.strip():
                    text_lines += 1
                    text_chars += len(line)
        
        total_chars = text_chars + code_chars
        if total_chars == 0:
            return {'text_ratio': 0.5, 'code_ratio': 0.5}
        
        text_ratio = text_chars / total_chars
        code_ratio = code_chars / total_chars
        
        return {
            'text_ratio': round(text_ratio, 2),
            'code_ratio': round(code_ratio, 2),
            'text_lines': text_lines,
            'code_lines': code_lines
        }

    def _build_evaluation_prompt(self, blog_content: str, source_paper: Dict, ratio: Dict) -> str:
        """构建评估提示词"""
        return f"""你是一位技术博客质量评估专家。请评估以下博客文章的质量。

## 原始论文信息
- 标题: {source_paper.get('title', 'Unknown')}
- 链接: {source_paper.get('url', 'Unknown')}
- 摘要: {source_paper.get('summary', source_paper.get('description', 'N/A'))}

## 当前文字/代码比例
- 文字占比: {ratio['text_ratio']:.0%}
- 代码占比: {ratio['code_ratio']:.0%}
- 理想比例: 文字 40-50%，代码 50-60%

## 博客内容
{blog_content[:15000]}  
{"[内容已截断...]" if len(blog_content) > 15000 else ""}

---

请从以下维度评估这篇博客，并给出 1-10 的评分（10分最高）：

### 1. 内容深度 (content_depth_score)
- 是否对论文有深入理解？
- 是否有洞见的评论和总结？
- 不仅仅是翻译或简单复述

### 2. 代码质量 (code_quality_score)  
- 代码是否完整、可运行？
- 有无语法错误或逻辑问题？
- 注释是否清晰？

### 3. 结构平衡 (structure_score)
- 文字和代码比例是否合理？（理想 4:6 或 5:5）
- 是否有足够的解释，而非堆砌代码？
- 整体结构是否清晰？

### 4. 总体评分 (overall_score)
- 综合以上各项，给出 1-10 的总分

请用以下 JSON 格式回复（只输出 JSON，不要其他内容）：

```json
{{
    "overall_score": <1-10>,
    "content_depth": {{
        "score": <1-10>,
        "comments": "<对内容深度的评价>"
    }},
    "code_quality": {{
        "score": <1-10>,
        "runnable": <true/false>,
        "issues": ["<问题1>", "<问题2>"]
    }},
    "structure": {{
        "score": <1-10>,
        "balanced": <true/false>,
        "comments": "<对结构的评价>"
    }},
    "summary": "<一句话总结>",
    "suggestions": ["<改进建议1>", "<改进建议2>"]
}}
```
"""

    def _parse_evaluation(self, response: str, ratio: Dict) -> Dict:
        """解析评估结果"""
        # 尝试提取 JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                # 添加比例信息
                result['structure']['text_ratio'] = ratio['text_ratio']
                result['structure']['code_ratio'] = ratio['code_ratio']
                return result
            except json.JSONDecodeError:
                pass
        
        # 尝试直接解析
        try:
            # 找到第一个 { 和最后一个 }
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
                result['structure']['text_ratio'] = ratio['text_ratio']
                result['structure']['code_ratio'] = ratio['code_ratio']
                return result
        except json.JSONDecodeError:
            pass
        
        # 解析失败，返回默认值
        print("  ⚠️ 无法解析评估结果，使用默认值")
        return self._default_evaluation(ratio)

    def _default_evaluation(self, ratio: Dict) -> Dict:
        """返回默认评估结果"""
        balanced = 0.35 <= ratio['text_ratio'] <= 0.55
        return {
            'overall_score': 7,
            'content_depth': {
                'score': 7,
                'comments': '无法自动评估，请人工审阅'
            },
            'code_quality': {
                'score': 7,
                'runnable': True,
                'issues': []
            },
            'structure': {
                'score': 7 if balanced else 5,
                'text_ratio': ratio['text_ratio'],
                'code_ratio': ratio['code_ratio'],
                'balanced': balanced,
                'comments': '文字/代码比例' + ('合理' if balanced else '需要调整')
            },
            'summary': '自动评估未能完成，请人工审阅',
            'suggestions': ['请人工审阅内容深度', '请检查代码可运行性']
        }

    def format_evaluation_report(self, evaluation: Dict) -> str:
        """格式化评估报告（用于邮件）"""
        report = []
        report.append("=" * 50)
        report.append("📊 AI 质量评估报告")
        report.append("=" * 50)
        report.append("")
        
        # 总分
        overall = evaluation.get('overall_score', 7)
        stars = '⭐' * min(overall, 10)
        report.append(f"🎯 总体评分: {overall}/10 {stars}")
        report.append("")
        
        # 内容深度
        depth = evaluation.get('content_depth', {})
        report.append(f"📚 内容深度: {depth.get('score', 7)}/10")
        report.append(f"   {depth.get('comments', 'N/A')}")
        report.append("")
        
        # 代码质量
        code = evaluation.get('code_quality', {})
        report.append(f"💻 代码质量: {code.get('score', 7)}/10")
        report.append(f"   可运行: {'✓ 是' if code.get('runnable', True) else '✗ 否'}")
        if code.get('issues'):
            report.append("   问题:")
            for issue in code['issues'][:3]:  # 最多显示3个
                report.append(f"   - {issue}")
        report.append("")
        
        # 结构平衡
        struct = evaluation.get('structure', {})
        report.append(f"📐 结构平衡: {struct.get('score', 7)}/10")
        report.append(f"   文字占比: {struct.get('text_ratio', 0.5):.0%}")
        report.append(f"   代码占比: {struct.get('code_ratio', 0.5):.0%}")
        report.append(f"   比例合理: {'✓ 是' if struct.get('balanced', True) else '✗ 需调整'}")
        report.append(f"   {struct.get('comments', '')}")
        report.append("")
        
        # 总结
        report.append(f"📝 总结: {evaluation.get('summary', 'N/A')}")
        report.append("")
        
        # 改进建议
        suggestions = evaluation.get('suggestions', [])
        if suggestions:
            report.append("💡 改进建议:")
            for s in suggestions[:5]:  # 最多显示5个
                report.append(f"   - {s}")
        
        report.append("")
        report.append("=" * 50)
        
        return '\n'.join(report)


if __name__ == "__main__":
    # 测试
    evaluator = BlogEvaluator()
    
    test_content = """---
layout: post
title: "Test Blog"
---

# Introduction

This is a test blog about machine learning.

## Code Example

```python
import numpy as np

def train_model(X, y):
    # Training logic here
    return model
```

## Conclusion

This was a brief overview.
"""
    
    test_paper = {
        'title': 'Test Paper',
        'url': 'https://arxiv.org/abs/1234.5678',
        'summary': 'A paper about ML'
    }
    
    result = evaluator.evaluate_blog(test_content, test_paper)
    print(evaluator.format_evaluation_report(result))
