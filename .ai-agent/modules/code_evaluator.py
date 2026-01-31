"""
Code Evaluator 模块
使用 LLM 评估代码块，分类并给出精简建议
"""

import subprocess
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

from .code_extractor import CodeBlock, ExtractionResult


@dataclass
class CodeEvaluation:
    """单个代码块的评估结果"""
    block_index: int
    classification: str  # "core" | "auxiliary" | "example"
    importance_score: float  # 0-10
    can_simplify: bool
    simplify_suggestion: str
    keep_in_blog: bool  # 是否保留在博客中
    reason: str


@dataclass
class CodeEvaluationResult:
    """整体代码评估结果"""
    evaluations: List[CodeEvaluation]
    overall_score: float
    needs_refactoring: bool
    summary: str
    

class CodeEvaluator:
    """使用 Claude CLI 评估代码块"""
    
    EVALUATION_PROMPT = '''你是一位代码审查专家。请评估以下从技术博客中提取的代码块。

## 博客主题
{blog_title}

## 代码块列表
{code_blocks_info}

## 评估任务

对每个代码块进行分类和评估：

1. **分类** (classification):
   - "core": 核心算法，必须保留在博客中展示
   - "auxiliary": 辅助代码（数据处理、工具函数），可以精简或移到代码库
   - "example": 使用示例，可以保留简化版本

2. **重要性评分** (importance_score): 0-10，对于理解文章核心内容的重要程度

3. **是否可精简** (can_simplify): true/false

4. **精简建议** (simplify_suggestion): 如果可以精简，具体怎么做

5. **是否保留在博客** (keep_in_blog): true/false
   - 核心代码（<50行）: 保留
   - 辅助代码: 移到代码库，博客中用注释说明
   - 过长的代码（>50行）: 精简后保留骨架

## 输出格式

请输出 JSON 格式：
```json
{{
  "evaluations": [
    {{
      "block_index": 0,
      "classification": "core",
      "importance_score": 9.0,
      "can_simplify": false,
      "simplify_suggestion": "",
      "keep_in_blog": true,
      "reason": "这是核心算法实现，对理解文章至关重要"
    }}
  ],
  "overall_score": 8.0,
  "needs_refactoring": true,
  "summary": "共5个代码块，建议保留3个核心代码，2个辅助代码移到代码库"
}}
```

只输出 JSON，不要其他内容。
'''

    def __init__(self, config: Dict = None):
        self.config = config or {}
        
    def evaluate(self, extraction_result: ExtractionResult, blog_title: str = "") -> CodeEvaluationResult:
        """
        评估所有代码块
        
        Args:
            extraction_result: Code Extractor 的输出
            blog_title: 博客标题，提供上下文
            
        Returns:
            CodeEvaluationResult: 评估结果
        """
        if not extraction_result.code_blocks:
            return CodeEvaluationResult(
                evaluations=[],
                overall_score=10.0,
                needs_refactoring=False,
                summary="没有代码块需要评估"
            )
        
        # 构建代码块信息
        code_blocks_info = self._format_code_blocks(extraction_result.code_blocks)
        
        # 构建 prompt
        prompt = self.EVALUATION_PROMPT.format(
            blog_title=blog_title or "技术博客",
            code_blocks_info=code_blocks_info
        )
        
        # 调用 Claude CLI
        try:
            result = subprocess.run(
                ['claude', '-p', '--model', 'sonnet', '--output-format', 'text'],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Claude CLI 错误: {result.stderr}")
            
            response = result.stdout.strip()
            
            # 解析 JSON 响应
            return self._parse_response(response, extraction_result.code_blocks)
            
        except subprocess.TimeoutExpired:
            print("  ⚠ 代码评估超时，使用默认评估")
            return self._default_evaluation(extraction_result.code_blocks)
        except Exception as e:
            print(f"  ⚠ 代码评估失败: {e}，使用默认评估")
            return self._default_evaluation(extraction_result.code_blocks)
    
    def _format_code_blocks(self, blocks: List[CodeBlock]) -> str:
        """格式化代码块信息供 LLM 评估"""
        info = []
        for block in blocks:
            info.append(f"### 代码块 {block.index}")
            info.append(f"- 语言: {block.language}")
            info.append(f"- 行数: {block.lines}")
            info.append(f"- 上下文: {block.context[:100] if block.context else '无'}")
            info.append("```" + block.language)
            # 截取前50行防止过长
            lines = block.content.split('\n')
            if len(lines) > 50:
                info.append('\n'.join(lines[:50]))
                info.append(f"# ... (省略 {len(lines) - 50} 行)")
            else:
                info.append(block.content)
            info.append("```")
            info.append("")
        return '\n'.join(info)
    
    def _parse_response(self, response: str, blocks: List[CodeBlock]) -> CodeEvaluationResult:
        """解析 LLM 响应"""
        # 提取 JSON
        try:
            # 尝试找到 JSON 块
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0]
            else:
                json_str = response
            
            data = json.loads(json_str.strip())
            
            evaluations = []
            for eval_data in data.get('evaluations', []):
                evaluations.append(CodeEvaluation(
                    block_index=eval_data.get('block_index', 0),
                    classification=eval_data.get('classification', 'auxiliary'),
                    importance_score=eval_data.get('importance_score', 5.0),
                    can_simplify=eval_data.get('can_simplify', True),
                    simplify_suggestion=eval_data.get('simplify_suggestion', ''),
                    keep_in_blog=eval_data.get('keep_in_blog', True),
                    reason=eval_data.get('reason', '')
                ))
            
            # 更新原始代码块的分类信息
            for eval_item in evaluations:
                if eval_item.block_index < len(blocks):
                    blocks[eval_item.block_index].classification = eval_item.classification
                    blocks[eval_item.block_index].importance_score = eval_item.importance_score
                    blocks[eval_item.block_index].can_simplify = eval_item.can_simplify
                    blocks[eval_item.block_index].simplify_suggestion = eval_item.simplify_suggestion
            
            return CodeEvaluationResult(
                evaluations=evaluations,
                overall_score=data.get('overall_score', 7.0),
                needs_refactoring=data.get('needs_refactoring', False),
                summary=data.get('summary', '')
            )
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  ⚠ JSON 解析失败: {e}")
            return self._default_evaluation(blocks)
    
    def _default_evaluation(self, blocks: List[CodeBlock]) -> CodeEvaluationResult:
        """默认评估（当 LLM 调用失败时）"""
        evaluations = []
        for block in blocks:
            # 简单规则：超过30行的标记为可精简
            can_simplify = block.lines > 30
            keep_in_blog = block.lines <= 50
            
            evaluations.append(CodeEvaluation(
                block_index=block.index,
                classification="auxiliary" if block.lines > 40 else "core",
                importance_score=7.0 if block.lines <= 30 else 5.0,
                can_simplify=can_simplify,
                simplify_suggestion="保留核心逻辑，移除辅助代码" if can_simplify else "",
                keep_in_blog=keep_in_blog,
                reason="基于代码长度的默认评估"
            ))
            
            # 更新代码块
            block.classification = evaluations[-1].classification
            block.importance_score = evaluations[-1].importance_score
            block.can_simplify = evaluations[-1].can_simplify
        
        return CodeEvaluationResult(
            evaluations=evaluations,
            overall_score=7.0,
            needs_refactoring=any(e.can_simplify for e in evaluations),
            summary=f"默认评估：{len(blocks)} 个代码块"
        )
    
    def get_evaluation_summary(self, result: CodeEvaluationResult) -> str:
        """生成评估摘要"""
        lines = []
        lines.append(f"📊 代码评估结果:")
        lines.append(f"  - 总体评分: {result.overall_score}/10")
        lines.append(f"  - 需要重构: {'是' if result.needs_refactoring else '否'}")
        lines.append(f"  - 摘要: {result.summary}")
        lines.append("")
        
        for eval_item in result.evaluations:
            status = "✓ 保留" if eval_item.keep_in_blog else "→ 移到代码库"
            lines.append(f"  [{eval_item.block_index}] {eval_item.classification} "
                        f"(重要性: {eval_item.importance_score}) {status}")
            if eval_item.simplify_suggestion:
                lines.append(f"      建议: {eval_item.simplify_suggestion[:60]}...")
        
        return '\n'.join(lines)


if __name__ == "__main__":
    from code_extractor import CodeExtractor
    
    test_md = '''
# 测试博客

## 核心算法

```python
def core_algorithm(x):
    """核心算法实现"""
    return x * 2
```

## 辅助函数

```python
def helper_function():
    # 这是一个很长的辅助函数
    data = []
    for i in range(100):
        data.append(i)
    # ... 很多行代码
    return data
```
'''
    
    extractor = CodeExtractor()
    extraction = extractor.extract(test_md)
    
    evaluator = CodeEvaluator()
    evaluation = evaluator.evaluate(extraction, "测试博客")
    print(evaluator.get_evaluation_summary(evaluation))
