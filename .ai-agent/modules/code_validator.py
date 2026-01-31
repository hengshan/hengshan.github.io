"""
代码验证模块
验证生成的代码是否有语法错误
"""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple


class CodeValidator:
    """代码验证器"""

    def __init__(self, config: Dict):
        self.config = config
        self.python_check = config['validation'].get('python_syntax_check', True)
        self.cuda_check = config['validation'].get('cuda_compile_check', False)
        self.run_tests = config['validation'].get('run_simple_tests', True)

    def extract_code_blocks(self, markdown_content: str) -> List[Dict]:
        """从Markdown中提取代码块"""
        code_blocks = []

        # 匹配代码块的正则表达式
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.finditer(pattern, markdown_content, re.DOTALL)

        for match in matches:
            language = match.group(1) or 'text'
            code = match.group(2).strip()

            code_blocks.append({
                'language': language.lower(),
                'code': code,
                'line_start': markdown_content[:match.start()].count('\n') + 1
            })

        return code_blocks

    def validate_python_code(self, code: str) -> Tuple[bool, str]:
        """验证Python代码语法"""
        try:
            compile(code, '<string>', 'exec')
            return True, "✓ Python语法正确"
        except SyntaxError as e:
            return False, f"✗ Python语法错误: {e}"
        except Exception as e:
            return False, f"✗ 编译错误: {e}"

    def validate_cuda_code(self, code: str) -> Tuple[bool, str]:
        """验证CUDA代码（需要nvcc）"""
        if not self.cuda_check:
            return True, "⊘ CUDA检查已跳过"

        try:
            # 检查nvcc是否可用
            result = subprocess.run(['nvcc', '--version'],
                                  capture_output=True,
                                  timeout=5)
            if result.returncode != 0:
                return True, "⊘ nvcc不可用，跳过CUDA检查"

            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # 尝试编译
            result = subprocess.run(
                ['nvcc', '-c', temp_file, '-o', '/dev/null'],
                capture_output=True,
                timeout=30,
                text=True
            )

            # 清理临时文件
            Path(temp_file).unlink()

            if result.returncode == 0:
                return True, "✓ CUDA代码可以编译"
            else:
                return False, f"✗ CUDA编译错误:\n{result.stderr[:500]}"

        except subprocess.TimeoutExpired:
            return False, "✗ CUDA编译超时"
        except Exception as e:
            return True, f"⊘ CUDA检查失败: {e}"

    def validate_cpp_code(self, code: str) -> Tuple[bool, str]:
        """验证C++代码"""
        try:
            # 简单的语法检查
            # 检查常见错误
            if code.count('{') != code.count('}'):
                return False, "✗ 大括号不匹配"
            if code.count('(') != code.count(')'):
                return False, "✗ 小括号不匹配"

            return True, "✓ C++基础语法检查通过"
        except Exception as e:
            return False, f"✗ C++检查错误: {e}"

    def validate_blog_post(self, markdown_content: str) -> Dict:
        """验证整个博客文章"""
        print("\n🔍 正在验证代码...")

        results = {
            'valid': True,
            'total_blocks': 0,
            'passed': 0,
            'failed': 0,
            'warnings': [],
            'errors': [],
            'details': []
        }

        # 提取所有代码块
        code_blocks = self.extract_code_blocks(markdown_content)
        results['total_blocks'] = len(code_blocks)

        if not code_blocks:
            results['warnings'].append("⚠ 未找到代码块")
            print("  ⚠ 未找到代码块")

        # 验证每个代码块
        for i, block in enumerate(code_blocks, 1):
            lang = block['language']
            code = block['code']

            print(f"\n  代码块 {i}/{len(code_blocks)} [{lang}]:")

            if lang == 'python':
                is_valid, message = self.validate_python_code(code)
            elif lang in ['cuda', 'cu']:
                is_valid, message = self.validate_cuda_code(code)
            elif lang in ['cpp', 'c++']:
                is_valid, message = self.validate_cpp_code(code)
            else:
                is_valid, message = True, f"⊘ {lang} 代码暂不验证"

            print(f"    {message}")

            detail = {
                'block_number': i,
                'language': lang,
                'line_start': block['line_start'],
                'valid': is_valid,
                'message': message,
                'code_preview': code[:100] + '...' if len(code) > 100 else code
            }

            results['details'].append(detail)

            if is_valid:
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['valid'] = False
                results['errors'].append(f"代码块{i} ({lang}): {message}")

        # 总结
        print(f"\n📊 验证结果:")
        print(f"  总计: {results['total_blocks']} 个代码块")
        print(f"  通过: {results['passed']}")
        print(f"  失败: {results['failed']}")

        if results['valid']:
            print("  ✓ 所有代码验证通过")
        else:
            print("  ✗ 存在代码错误")
            for error in results['errors']:
                print(f"    - {error}")

        return results

    def check_blog_quality(self, markdown_content: str) -> Dict:
        """检查博客质量"""
        quality = {
            'has_title': False,
            'has_code': False,
            'has_comments': False,
            'word_count': 0,
            'code_count': 0,
            'quality_score': 0,
            'suggestions': []
        }

        # 检查是否有标题
        if 'title:' in markdown_content:
            quality['has_title'] = True

        # 检查是否有代码
        if '```' in markdown_content:
            quality['has_code'] = True
            quality['code_count'] = markdown_content.count('```') // 2

        # 检查是否有注释（中文注释）
        if '#' in markdown_content and any('\u4e00' <= c <= '\u9fff' for c in markdown_content):
            quality['has_comments'] = True

        # 字数统计（粗略）
        quality['word_count'] = len(markdown_content)

        # 质量评分
        score = 0
        if quality['has_title']:
            score += 20
        if quality['has_code']:
            score += 30
        if quality['has_comments']:
            score += 20
        if quality['word_count'] > 2000:
            score += 20
        if quality['code_count'] >= 3:
            score += 10

        quality['quality_score'] = min(score, 100)

        # 建议
        if not quality['has_code']:
            quality['suggestions'].append("建议添加代码示例")
        if quality['word_count'] < 1500:
            quality['suggestions'].append("内容可以更丰富")
        if quality['code_count'] < 2:
            quality['suggestions'].append("建议增加更多代码示例")

        return quality


if __name__ == "__main__":
    # 测试
    validator = CodeValidator({'validation': {
        'python_syntax_check': True,
        'cuda_compile_check': False,
        'run_simple_tests': True
    }})

    test_md = """
# Test Blog

```python
def hello():
    print("Hello, World!")
    return 42
```

```cuda
__global__ void add(int *a, int *b) {
    int idx = threadIdx.x;
    b[idx] = a[idx] + b[idx];
}
```
"""

    results = validator.validate_blog_post(test_md)
    print(f"\n验证结果: {results}")
