"""
Blog Refiner 模块 (Phase 3)
根据 Blog Evaluator 的反馈迭代改进博客内容
"""

import subprocess
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RefinementResult:
    """改进结果"""
    refined_content: str
    changes_made: List[str]
    iteration: int
    score_before: float
    score_after: float
    converged: bool  # 是否达到目标


class BlogRefiner:
    """博客内容改进器"""
    
    REFINE_PROMPT = '''你是一位技术博客编辑，请根据评估反馈改进以下博客。

## 当前博客内容
{blog_content}

## 评估反馈
- 当前评分: {current_score}/10
- 目标评分: {target_score}/10
- 文字/代码比例: {text_ratio}% / {code_ratio}%
- 建议比例: 45% / 55%

## 具体改进建议
{suggestions}

## 改进要求

1. **提高文字比例**（如果代码过多）：
   - 精简冗长的代码块，保留核心逻辑
   - 用 `# ... (详细实现省略)` 替换非核心代码
   - 增加解释性文字

2. **增强洞见**（如果内容浅显）：
   - 添加"为什么这样设计"的分析
   - 增加与其他方法的对比
   - 指出方法的局限性

3. **改进结构**（如果逻辑不清）：
   - 确保有"一句话总结"
   - 确保有"什么时候用/不用"
   - 确保有批判性分析

4. **保持精华**：
   - 不要删除核心算法代码
   - 不要删除关键洞见
   - 保持技术准确性

## 输出格式

直接输出改进后的完整博客内容（Markdown格式）。
不要输出任何解释，只输出博客内容。
保留原有的 front matter（---开头的部分）。
'''

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.max_iterations = 2  # 最大迭代次数
        self.target_score = 8.0  # 目标评分
        self.target_text_ratio = (0.40, 0.60)  # 目标文字比例范围
        
    def refine(
        self,
        blog_content: str,
        evaluation: Dict,
        iteration: int = 1
    ) -> RefinementResult:
        """
        根据评估反馈改进博客
        
        Args:
            blog_content: 当前博客内容
            evaluation: Blog Evaluator 的评估结果
            iteration: 当前迭代次数
            
        Returns:
            RefinementResult: 改进结果
        """
        current_score = evaluation.get('overall_score', 7.0)
        structure = evaluation.get('structure', {})
        text_ratio = structure.get('text_ratio', 0.3) * 100
        code_ratio = structure.get('code_ratio', 0.7) * 100
        suggestions = evaluation.get('suggestions', [])
        
        # 检查是否已达标
        if self._is_acceptable(current_score, text_ratio / 100):
            return RefinementResult(
                refined_content=blog_content,
                changes_made=["已达到目标评分，无需改进"],
                iteration=iteration,
                score_before=current_score,
                score_after=current_score,
                converged=True
            )
        
        # 构建 prompt
        prompt = self.REFINE_PROMPT.format(
            blog_content=blog_content,
            current_score=current_score,
            target_score=self.target_score,
            text_ratio=f"{text_ratio:.0f}",
            code_ratio=f"{code_ratio:.0f}",
            suggestions="\n".join(f"- {s}" for s in suggestions) if suggestions else "无具体建议"
        )
        
        # 调用 Claude CLI 改进
        try:
            result = subprocess.run(
                ['claude', '-p', '--model', 'sonnet', '--output-format', 'text'],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=180  # 3分钟超时
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Claude CLI 错误: {result.stderr}")
            
            refined_content = result.stdout.strip()
            
            # 确保保留 front matter
            if not refined_content.startswith('---'):
                # 从原内容提取 front matter
                front_matter = self._extract_front_matter(blog_content)
                if front_matter:
                    refined_content = front_matter + "\n\n" + refined_content
            
            # 分析改动
            changes = self._analyze_changes(blog_content, refined_content)
            
            return RefinementResult(
                refined_content=refined_content,
                changes_made=changes,
                iteration=iteration,
                score_before=current_score,
                score_after=0,  # 需要重新评估
                converged=False
            )
            
        except subprocess.TimeoutExpired:
            print(f"  ⚠ 改进超时，保留原内容")
            return RefinementResult(
                refined_content=blog_content,
                changes_made=["改进超时，保留原内容"],
                iteration=iteration,
                score_before=current_score,
                score_after=current_score,
                converged=False
            )
        except Exception as e:
            print(f"  ⚠ 改进失败: {e}")
            return RefinementResult(
                refined_content=blog_content,
                changes_made=[f"改进失败: {e}"],
                iteration=iteration,
                score_before=current_score,
                score_after=current_score,
                converged=False
            )
    
    def _is_acceptable(self, score: float, text_ratio: float) -> bool:
        """检查是否达到目标"""
        return (
            score >= self.target_score and
            self.target_text_ratio[0] <= text_ratio <= self.target_text_ratio[1]
        )
    
    def _extract_front_matter(self, content: str) -> Optional[str]:
        """提取 front matter"""
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                return '---' + parts[1] + '---'
        return None
    
    def _analyze_changes(self, original: str, refined: str) -> List[str]:
        """分析改动"""
        changes = []
        
        # 统计行数变化
        orig_lines = len(original.split('\n'))
        new_lines = len(refined.split('\n'))
        if new_lines != orig_lines:
            changes.append(f"行数: {orig_lines} → {new_lines}")
        
        # 统计代码块变化
        orig_code_blocks = len(re.findall(r'```', original)) // 2
        new_code_blocks = len(re.findall(r'```', refined)) // 2
        if new_code_blocks != orig_code_blocks:
            changes.append(f"代码块: {orig_code_blocks} → {new_code_blocks}")
        
        # 统计字符数变化
        orig_chars = len(original)
        new_chars = len(refined)
        change_pct = (new_chars - orig_chars) / orig_chars * 100 if orig_chars > 0 else 0
        if abs(change_pct) > 5:
            changes.append(f"内容量: {change_pct:+.1f}%")
        
        if not changes:
            changes.append("微调优化")
        
        return changes
    
    def get_refine_summary(self, result: RefinementResult) -> str:
        """生成改进摘要"""
        lines = []
        lines.append(f"📝 博客改进结果 (迭代 {result.iteration}):")
        lines.append(f"   评分: {result.score_before} → {result.score_after if result.score_after > 0 else '待评估'}")
        lines.append(f"   收敛: {'✓ 已达标' if result.converged else '✗ 继续迭代'}")
        lines.append("   改动:")
        for change in result.changes_made:
            lines.append(f"     - {change}")
        return '\n'.join(lines)


if __name__ == "__main__":
    # 测试
    test_content = '''---
title: Test
---

# 测试博客

这是一篇测试博客。

```python
def test():
    pass
```
'''
    
    test_eval = {
        'overall_score': 6.0,
        'structure': {'text_ratio': 0.3, 'code_ratio': 0.7},
        'suggestions': ['增加更多解释', '精简代码']
    }
    
    refiner = BlogRefiner()
    result = refiner.refine(test_content, test_eval)
    print(refiner.get_refine_summary(result))
