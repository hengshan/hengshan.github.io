"""
Code Extractor 模块
从博客内容中提取代码块，分析元数据
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class CodeBlock:
    """代码块数据结构"""
    content: str
    language: str
    lines: int
    index: int  # 在博客中的顺序
    start_line: int  # 在原文中的起始行
    context: str = ""  # 代码块前的上下文（标题等）
    
    # 由 Code Evaluator 填充
    classification: Optional[str] = None  # "core" | "auxiliary" | "example"
    importance_score: Optional[float] = None
    can_simplify: Optional[bool] = None
    simplify_suggestion: Optional[str] = None


@dataclass 
class ExtractionResult:
    """提取结果"""
    code_blocks: List[CodeBlock]
    total_code_lines: int
    total_text_lines: int
    code_ratio: float
    languages: List[str]


class CodeExtractor:
    """从博客 Markdown 中提取代码块"""
    
    # 代码块正则：匹配 ```language ... ```
    CODE_BLOCK_PATTERN = re.compile(
        r'^```(\w*)\n(.*?)^```',
        re.MULTILINE | re.DOTALL
    )
    
    def __init__(self):
        pass
    
    def extract(self, markdown_content: str) -> ExtractionResult:
        """
        从 Markdown 内容中提取所有代码块
        
        Args:
            markdown_content: 博客的 Markdown 内容
            
        Returns:
            ExtractionResult: 包含所有代码块和统计信息
        """
        code_blocks = []
        lines = markdown_content.split('\n')
        total_lines = len(lines)
        
        # 查找所有代码块
        for idx, match in enumerate(self.CODE_BLOCK_PATTERN.finditer(markdown_content)):
            language = match.group(1) or 'text'
            content = match.group(2).strip()
            
            # 计算起始行号
            start_pos = match.start()
            start_line = markdown_content[:start_pos].count('\n') + 1
            
            # 获取上下文（代码块前的标题或段落）
            context = self._get_context(markdown_content, start_pos)
            
            block = CodeBlock(
                content=content,
                language=language,
                lines=len(content.split('\n')),
                index=idx,
                start_line=start_line,
                context=context
            )
            code_blocks.append(block)
        
        # 计算统计信息
        total_code_lines = sum(b.lines for b in code_blocks)
        total_text_lines = total_lines - total_code_lines
        code_ratio = total_code_lines / total_lines if total_lines > 0 else 0
        languages = list(set(b.language for b in code_blocks))
        
        return ExtractionResult(
            code_blocks=code_blocks,
            total_code_lines=total_code_lines,
            total_text_lines=total_text_lines,
            code_ratio=code_ratio,
            languages=languages
        )
    
    def _get_context(self, content: str, pos: int, max_chars: int = 200) -> str:
        """获取代码块前的上下文"""
        # 向前查找最近的标题或段落
        before = content[:pos]
        lines = before.split('\n')
        
        context_lines = []
        for line in reversed(lines[-5:]):  # 最多看前5行
            line = line.strip()
            if line:
                context_lines.insert(0, line)
                if line.startswith('#'):  # 找到标题就停
                    break
        
        return '\n'.join(context_lines)[-max_chars:]
    
    def get_code_summary(self, result: ExtractionResult) -> str:
        """生成代码提取摘要"""
        summary = []
        summary.append(f"📊 代码提取摘要:")
        summary.append(f"  - 代码块数量: {len(result.code_blocks)}")
        summary.append(f"  - 代码总行数: {result.total_code_lines}")
        summary.append(f"  - 文字总行数: {result.total_text_lines}")
        summary.append(f"  - 代码占比: {result.code_ratio:.1%}")
        summary.append(f"  - 语言类型: {', '.join(result.languages)}")
        summary.append("")
        
        for block in result.code_blocks:
            summary.append(f"  [{block.index}] {block.language}: {block.lines} 行")
            if block.context:
                ctx = block.context[:50] + "..." if len(block.context) > 50 else block.context
                summary.append(f"      上下文: {ctx}")
        
        return '\n'.join(summary)


if __name__ == "__main__":
    # 测试
    test_md = '''
# 测试博客

这是一段介绍文字。

## 代码示例

下面是核心算法：

```python
def hello():
    print("Hello, World!")
    return True
```

这是辅助函数：

```python
def helper():
    # 辅助代码
    pass
```
'''
    
    extractor = CodeExtractor()
    result = extractor.extract(test_md)
    print(extractor.get_code_summary(result))
