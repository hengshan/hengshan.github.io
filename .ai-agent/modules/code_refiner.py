"""
Code Refiner 模块
根据评估结果：
1. 精简博客中的代码
2. 生成完整代码库
"""

import subprocess
import re
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from .code_extractor import CodeBlock, ExtractionResult
from .code_evaluator import CodeEvaluation, CodeEvaluationResult


@dataclass
class RefinerOutput:
    """Refiner 输出"""
    refined_blog: str  # 精简后的博客内容
    code_repo_path: str  # 代码库路径
    code_files: Dict[str, str]  # 文件名 -> 内容
    readme_content: str  # 代码库 README
    changes_made: List[str]  # 所做的更改列表


class CodeRefiner:
    """代码重构器"""
    
    SIMPLIFY_PROMPT = '''请精简以下代码，保留核心逻辑，移除辅助代码。

## 原始代码 ({language})
```{language}
{original_code}
```

## 上下文
{context}

## 精简要求
- 保留核心算法骨架（约 {target_lines} 行）
- 用注释标记省略的部分，如 `# ... (数据处理代码省略)`
- 保留关键的类/函数签名
- 保留重要的注释
- 确保代码骨架能展示核心思想

## 输出
只输出精简后的代码，不要其他解释。用 ```{language} 包裹。
'''

    def __init__(self, config: Dict = None, code_repo_base: str = None):
        self.config = config or {}
        self.code_repo_base = code_repo_base or os.path.expanduser("~/projects/blog-code")
        
    def refine(
        self, 
        blog_content: str,
        extraction_result: ExtractionResult,
        evaluation_result: CodeEvaluationResult,
        blog_slug: str,
        blog_title: str = "",
        source_url: str = ""
    ) -> RefinerOutput:
        """
        执行代码重构
        
        Args:
            blog_content: 原始博客内容
            extraction_result: 代码提取结果
            evaluation_result: 代码评估结果
            blog_slug: 博客文件名 slug（用于创建代码库文件夹）
            blog_title: 博客标题
            source_url: 论文/来源 URL
            
        Returns:
            RefinerOutput: 包含精简博客和代码库
        """
        changes_made = []
        code_files = {}
        refined_blog = blog_content
        
        # 收集需要处理的代码块
        blocks_to_simplify = []
        blocks_for_repo = []
        
        for eval_item in evaluation_result.evaluations:
            block = extraction_result.code_blocks[eval_item.block_index]
            
            if eval_item.can_simplify and block.lines > 30:
                blocks_to_simplify.append((block, eval_item))
                
            if not eval_item.keep_in_blog or block.lines > 20:
                blocks_for_repo.append((block, eval_item))
        
        # 1. 精简博客中的代码
        for block, eval_item in blocks_to_simplify:
            simplified = self._simplify_code(block, eval_item)
            if simplified:
                refined_blog = self._replace_code_block(
                    refined_blog, block, simplified
                )
                changes_made.append(
                    f"精简代码块 {block.index} ({block.language}): "
                    f"{block.lines} 行 → {len(simplified.split(chr(10)))} 行"
                )
        
        # 2. 生成代码库文件
        if blocks_for_repo:
            code_files = self._generate_code_files(
                blocks_for_repo, 
                extraction_result.languages
            )
            changes_made.append(f"生成 {len(code_files)} 个代码文件到代码库")
        
        # 3. 生成 README
        readme_content = self._generate_readme(
            blog_title=blog_title,
            blog_slug=blog_slug,
            source_url=source_url,
            code_files=code_files,
            blocks_for_repo=blocks_for_repo
        )
        
        # 4. 创建代码库目录并保存
        repo_path = self._save_code_repo(
            blog_slug=blog_slug,
            code_files=code_files,
            readme_content=readme_content
        )
        
        return RefinerOutput(
            refined_blog=refined_blog,
            code_repo_path=repo_path,
            code_files=code_files,
            readme_content=readme_content,
            changes_made=changes_made
        )
    
    def _simplify_code(self, block: CodeBlock, eval_item: CodeEvaluation) -> Optional[str]:
        """使用 LLM 精简单个代码块"""
        if block.lines <= 30:
            return None  # 不需要精简
        
        # 目标行数：原来的 40-60%
        target_lines = max(15, int(block.lines * 0.4))
        
        prompt = self.SIMPLIFY_PROMPT.format(
            language=block.language,
            original_code=block.content,
            context=block.context or "技术博客代码示例",
            target_lines=target_lines
        )
        
        try:
            result = subprocess.run(
                ['claude', '-p', '--model', 'sonnet', '--output-format', 'text'],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return None
            
            response = result.stdout.strip()
            
            # 提取代码块
            pattern = rf'```{block.language}\n(.*?)```'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
            
            # 尝试通用匹配
            if '```' in response:
                code = response.split('```')[1]
                if code.startswith(block.language):
                    code = code[len(block.language):].strip()
                return code.split('```')[0].strip()
            
            return None
            
        except Exception as e:
            print(f"  ⚠ 代码精简失败: {e}")
            return None
    
    def _replace_code_block(self, content: str, block: CodeBlock, new_code: str) -> str:
        """在博客内容中替换代码块"""
        # 构建原始代码块的正则
        original = f"```{block.language}\n{re.escape(block.content)}\n```"
        replacement = f"```{block.language}\n{new_code}\n```"
        
        # 尝试精确替换
        if original in content:
            return content.replace(original, replacement, 1)
        
        # 模糊替换（处理空白差异）
        pattern = rf'```{block.language}\s*\n.*?\n```'
        matches = list(re.finditer(pattern, content, re.DOTALL))
        
        if block.index < len(matches):
            match = matches[block.index]
            return content[:match.start()] + replacement + content[match.end():]
        
        return content  # 无法替换，返回原内容
    
    def _generate_code_files(
        self, 
        blocks: List[Tuple[CodeBlock, CodeEvaluation]],
        languages: List[str]
    ) -> Dict[str, str]:
        """生成代码库文件"""
        files = {}
        
        # 按语言分组
        by_language = {}
        for block, eval_item in blocks:
            lang = block.language
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append((block, eval_item))
        
        # 生成文件
        for lang, lang_blocks in by_language.items():
            ext = self._get_extension(lang)
            
            if len(lang_blocks) == 1:
                # 单个代码块，直接用 main.ext
                block, _ = lang_blocks[0]
                filename = f"main{ext}"
                files[filename] = self._format_code_file(block, lang)
            else:
                # 多个代码块，按分类组织
                core_blocks = [(b, e) for b, e in lang_blocks if e.classification == "core"]
                aux_blocks = [(b, e) for b, e in lang_blocks if e.classification != "core"]
                
                if core_blocks:
                    content = "\n\n".join(
                        self._format_code_file(b, lang) for b, _ in core_blocks
                    )
                    files[f"core{ext}"] = content
                
                if aux_blocks:
                    content = "\n\n".join(
                        self._format_code_file(b, lang) for b, _ in aux_blocks
                    )
                    files[f"utils{ext}"] = content
        
        return files
    
    def _format_code_file(self, block: CodeBlock, language: str) -> str:
        """格式化单个代码文件"""
        header = []
        
        if language == "python":
            header.append('"""')
            if block.context:
                header.append(block.context)
            header.append(f"代码块 {block.index}")
            header.append('"""')
            header.append("")
        elif language in ["cuda", "cpp", "c"]:
            header.append("/*")
            if block.context:
                header.append(f" * {block.context}")
            header.append(f" * 代码块 {block.index}")
            header.append(" */")
            header.append("")
        
        return "\n".join(header) + block.content
    
    def _get_extension(self, language: str) -> str:
        """获取文件扩展名"""
        ext_map = {
            "python": ".py",
            "cuda": ".cu",
            "cpp": ".cpp",
            "c": ".c",
            "javascript": ".js",
            "typescript": ".ts",
            "bash": ".sh",
            "shell": ".sh",
            "rust": ".rs",
            "go": ".go",
        }
        return ext_map.get(language, ".txt")
    
    def _generate_readme(
        self,
        blog_title: str,
        blog_slug: str,
        source_url: str,
        code_files: Dict[str, str],
        blocks_for_repo: List[Tuple[CodeBlock, CodeEvaluation]]
    ) -> str:
        """生成代码库 README"""
        lines = []
        lines.append(f"# {blog_title or blog_slug}")
        lines.append("")
        lines.append(f"博客配套代码")
        lines.append("")
        
        if source_url:
            lines.append(f"**论文/来源**: {source_url}")
            lines.append("")
        
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        lines.append("## 文件说明")
        lines.append("")
        for filename in code_files.keys():
            lines.append(f"- `{filename}`: ")
        lines.append("")
        
        lines.append("## 代码块来源")
        lines.append("")
        for block, eval_item in blocks_for_repo:
            lines.append(f"- 代码块 {block.index} ({block.language}, {block.lines} 行)")
            lines.append(f"  - 分类: {eval_item.classification}")
            lines.append(f"  - 重要性: {eval_item.importance_score}/10")
            if block.context:
                lines.append(f"  - 上下文: {block.context[:80]}")
        lines.append("")
        
        lines.append("## 注意")
        lines.append("")
        lines.append("这些代码是从论文/教程中提炼的示例代码，仅供学习参考。")
        lines.append("如需在生产环境使用，请进行充分测试。")
        
        return "\n".join(lines)
    
    def _save_code_repo(
        self,
        blog_slug: str,
        code_files: Dict[str, str],
        readme_content: str
    ) -> str:
        """保存代码库到磁盘"""
        repo_path = Path(self.code_repo_base) / blog_slug
        repo_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 README
        readme_path = repo_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # 保存代码文件
        for filename, content in code_files.items():
            file_path = repo_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return str(repo_path)
    
    def get_refine_summary(self, output: RefinerOutput) -> str:
        """生成重构摘要"""
        lines = []
        lines.append("📊 代码重构结果:")
        lines.append(f"  - 代码库路径: {output.code_repo_path}")
        lines.append(f"  - 生成文件数: {len(output.code_files)}")
        lines.append("")
        lines.append("  更改列表:")
        for change in output.changes_made:
            lines.append(f"    - {change}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    from code_extractor import CodeExtractor
    from code_evaluator import CodeEvaluator
    
    test_md = '''---
title: Test Blog
---

# 测试博客

## 核心算法

```python
def core_algorithm(x):
    """核心算法实现"""
    # 这里有很多行代码
    result = x * 2
    return result
```
'''
    
    extractor = CodeExtractor()
    extraction = extractor.extract(test_md)
    
    evaluator = CodeEvaluator()
    evaluation = evaluator.evaluate(extraction, "测试博客")
    
    refiner = CodeRefiner()
    output = refiner.refine(
        blog_content=test_md,
        extraction_result=extraction,
        evaluation_result=evaluation,
        blog_slug="2026-01-30-test",
        blog_title="测试博客",
        source_url="https://example.com"
    )
    
    print(refiner.get_refine_summary(output))
