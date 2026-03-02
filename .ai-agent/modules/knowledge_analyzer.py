"""
知识体系分析器
分析用户现有的博客和教程，构建知识图谱
"""

from pathlib import Path
from typing import Dict, List, Set
import re


class KnowledgeAnalyzer:
    """知识体系分析器"""

    def __init__(self, config: Dict):
        self.config = config
        self.cuda_path = Path(config['knowledge_base']['cuda_tutorials']).expanduser()
        self.blog_root = Path('.')

    def analyze_cuda_progress(self) -> Dict:
        """分析CUDA学习进度"""
        progress = {
            'total_files': 0,
            'topics': set(),
            'completed_lessons': [],
            'current_level': 'beginner',
            'keywords': set()
        }

        if not self.cuda_path.exists():
            return progress

        # 统计CUDA文件
        cu_files = list(self.cuda_path.glob('*.cu'))
        progress['total_files'] = len(cu_files)

        # 分析文件名提取主题
        for f in cu_files:
            name = f.stem

            # 提取课程编号
            if match := re.match(r'(\d{4})-', name):
                lesson = match.group(1)
                progress['completed_lessons'].append(lesson)

            # 提取主题关键词
            keywords = name.lower().split('-')[1:]
            progress['keywords'].update(keywords)

            # 识别主题
            if 'matmul' in name or 'gemm' in name:
                progress['topics'].add('矩阵乘法')
            if 'shared' in name:
                progress['topics'].add('共享内存')
            if 'warp' in name:
                progress['topics'].add('Warp编程')
            if 'cg' in name or 'cluster' in name:
                progress['topics'].add('Cooperative Groups')
            if 'stream' in name or 'async' in name:
                progress['topics'].add('异步编程')

        # 判断水平
        completed_count = len(set(progress['completed_lessons']))
        if completed_count < 5:
            progress['current_level'] = 'beginner'
        elif completed_count < 10:
            progress['current_level'] = 'intermediate'
        else:
            progress['current_level'] = 'advanced'

        return progress

    def analyze_blog_topics(self) -> Dict:
        """分析已发布博客主题"""
        topics = {
            'total_posts': 0,
            'categories': {},
            'keywords': set(),
            'recent_topics': []
        }

        # 扫描所有博客分类
        for cat in self.config['content']['categories']:
            cat_dir = Path(cat['output_dir'])
            if not cat_dir.exists():
                continue

            posts = list(cat_dir.glob('*.markdown'))
            cat_name = cat['name']

            if cat_name not in topics['categories']:
                topics['categories'][cat_name] = []

            for post in posts:
                try:
                    with open(post, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # 提取标题
                        title_match = re.search(r'title:\s*["\'](.+?)["\']', content)
                        if title_match:
                            title = title_match.group(1)
                            topics['categories'][cat_name].append(title)
                            topics['recent_topics'].append({
                                'title': title,
                                'category': cat_name,
                                'file': post.name
                            })

                        # 提取关键词
                        keywords = re.findall(r'\b[A-Z]{2,}\b', content)  # 大写缩写
                        topics['keywords'].update(k.lower() for k in keywords)

                except Exception as e:
                    print(f"Warning: 无法分析 {post}: {e}")

            topics['total_posts'] += len(posts)

        # 只保留最近10篇
        topics['recent_topics'] = sorted(
            topics['recent_topics'],
            key=lambda x: x['file'],
            reverse=True
        )[:10]

        return topics

    def get_knowledge_summary(self) -> str:
        """获取知识体系摘要"""
        cuda_progress = self.analyze_cuda_progress()
        blog_topics = self.analyze_blog_topics()

        summary = []

        summary.append("# 用户知识体系摘要\n")

        # CUDA进度
        summary.append("## CUDA学习进度")
        summary.append(f"- 完成文件数: {cuda_progress['total_files']}")
        summary.append(f"- 当前水平: {cuda_progress['current_level']}")
        summary.append(f"- 涉及主题: {', '.join(cuda_progress['topics'])}")
        summary.append("")

        # 博客统计
        summary.append("## 已发布博客")
        summary.append(f"- 总计: {blog_topics['total_posts']} 篇")
        for cat, posts in blog_topics['categories'].items():
            summary.append(f"- {cat}: {len(posts)} 篇")

        # 最近话题
        summary.append("\n## 最近博客话题")
        for topic in blog_topics['recent_topics'][:5]:
            summary.append(f"- [{topic['category']}] {topic['title']}")

        # 知识空白（建议话题）
        summary.append("\n## 可能的知识空白/进阶方向")

        all_keywords = cuda_progress['keywords'] | blog_topics['keywords']

        suggestions = []
        if 'flash' not in all_keywords and 'attention' not in all_keywords:
            suggestions.append("- Flash Attention实现")
        if 'quantization' not in all_keywords:
            suggestions.append("- 模型量化技术")
        if 'distributed' not in all_keywords:
            suggestions.append("- 分布式训练")
        if 'triton' not in all_keywords:
            suggestions.append("- Triton GPU编程")
        if 'rl' not in all_keywords and 'reinforcement' not in all_keywords:
            suggestions.append("- 强化学习算法")

        if suggestions:
            summary.extend(suggestions)
        else:
            summary.append("- 知识体系较为完整，可以深入现有主题")

        return '\n'.join(summary)

    def suggest_next_topic(self, available_topics: List[Dict]) -> Dict:
        """根据知识体系建议下一个话题"""
        cuda_progress = self.analyze_cuda_progress()
        blog_topics = self.analyze_blog_topics()

        # 打分系统
        for topic in available_topics:
            score = topic.get('score', 0)

            topic_text = (topic.get('title', '') + ' ' +
                         topic.get('summary', '')).lower()

            # 如果与CUDA进度相关，加分
            if cuda_progress['current_level'] == 'intermediate':
                if any(kw in topic_text for kw in ['optimization', 'shared memory', 'warp']):
                    score += 5

            # 如果是知识空白，加分
            all_keywords = cuda_progress['keywords'] | blog_topics['keywords']
            if 'flash attention' in topic_text and 'flash' not in all_keywords:
                score += 10
            if 'triton' in topic_text and 'triton' not in all_keywords:
                score += 8

            # 避免重复最近的话题
            recent_titles = [t['title'].lower() for t in blog_topics['recent_topics'][:3]]
            if any(title in topic_text for title in recent_titles):
                score -= 5

            topic['adjusted_score'] = score

        # 返回得分最高的
        return max(available_topics, key=lambda x: x.get('adjusted_score', 0))


if __name__ == "__main__":
    import yaml

    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    analyzer = KnowledgeAnalyzer(config)

    print("=" * 60)
    print(analyzer.get_knowledge_summary())
    print("=" * 60)
