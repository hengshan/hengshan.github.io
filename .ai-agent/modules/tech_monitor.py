"""
技术信息源监控模块
监控arXiv、GitHub、技术博客等获取最新技术动态
"""

import requests
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict
import yaml


class TechMonitor:
    """技术信息源监控器"""

    def __init__(self, sources_config_path: str, config: dict = None):
        with open(sources_config_path, 'r', encoding='utf-8') as f:
            self.sources = yaml.safe_load(f)
        self.config = config or {}

    def fetch_arxiv_papers(self, max_results: int = 8) -> List[Dict]:
        """从arXiv获取论文（分类浏览 + 专题搜索）"""
        papers = []
        seen_urls = set()
        base_url = self.sources['arxiv']['base_url']

        # 1. 按分类获取最新论文
        for category_info in self.sources['arxiv']['categories']:
            cat_id = category_info['id']
            keywords = category_info.get('keywords', [])

            query = f"cat:{cat_id}"
            params = {
                'search_query': query,
                'start': 0,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }

            try:
                response = requests.get(base_url, params=params, timeout=15)
                feed = feedparser.parse(response.content)

                for entry in feed.entries:
                    if entry.link in seen_urls:
                        continue
                    title_summary = (entry.title + ' ' + entry.summary).lower()
                    if any(kw.lower() in title_summary for kw in keywords) or not keywords:
                        seen_urls.add(entry.link)
                        papers.append({
                            'source': 'arxiv',
                            'title': entry.title,
                            'summary': entry.summary,
                            'url': entry.link,
                            'published': entry.published,
                            'category': category_info['name'],
                            'keywords': keywords
                        })
            except Exception as e:
                print(f"  ⚠ arxiv {cat_id} 获取失败: {e}")

        # 2. 专题关键词搜索（按相关性排序，覆盖更广时间范围）
        topic_searches = self.sources['arxiv'].get('topic_searches', [])
        for topic in topic_searches:
            query = topic['query']
            topic_max = topic.get('max_results', 5)
            params = {
                'search_query': f"all:{query}",
                'start': 0,
                'max_results': topic_max,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }

            try:
                response = requests.get(base_url, params=params, timeout=15)
                feed = feedparser.parse(response.content)

                for entry in feed.entries:
                    if entry.link in seen_urls:
                        continue
                    seen_urls.add(entry.link)
                    papers.append({
                        'source': 'arxiv',
                        'title': entry.title,
                        'summary': entry.summary,
                        'url': entry.link,
                        'published': entry.published,
                        'category': topic['name'],
                        'keywords': query.split(' OR ')
                    })
            except Exception as e:
                print(f"  ⚠ arxiv topic '{topic['name']}' 搜索失败: {e}")

        return papers

    def fetch_github_trending(self) -> List[Dict]:
        """获取GitHub trending和重要更新"""
        trending = []

        for repo_info in self.sources['github']['repos']:
            owner = repo_info['owner']
            name = repo_info['name']

            # GitHub API: 获取最近的releases和commits
            # 注意：需要GITHUB_TOKEN环境变量
            try:
                # 获取最新release
                url = f"https://api.github.com/repos/{owner}/{name}/releases/latest"
                response = requests.get(url)
                if response.status_code == 200:
                    release = response.json()
                    trending.append({
                        'source': 'github',
                        'repo': f"{owner}/{name}",
                        'title': release['name'],
                        'description': release['body'][:500],  # 截取前500字符
                        'url': release['html_url'],
                        'published': release['published_at'],
                        'type': 'release',
                        'topics': repo_info.get('topics', [])
                    })
            except Exception as e:
                print(f"Error fetching {owner}/{name}: {e}")

        return trending

    def fetch_blog_posts(self) -> List[Dict]:
        """从技术博客RSS获取最新文章"""
        posts = []

        for blog in self.sources.get('blogs', []):
            try:
                feed = feedparser.parse(blog['url'])
                for entry in feed.entries[:5]:  # 每个博客取前5篇
                    # 检查关键词匹配
                    title_summary = (entry.title + ' ' + entry.summary).lower()
                    keywords = blog.get('keywords', [])

                    if any(kw.lower() in title_summary for kw in keywords) or not keywords:
                        posts.append({
                            'source': 'blog',
                            'blog_name': blog['name'],
                            'title': entry.title,
                            'summary': entry.get('summary', '')[:500],
                            'url': entry.link,
                            'published': entry.get('published', ''),
                            'keywords': keywords
                        })
            except Exception as e:
                print(f"Error fetching {blog['name']}: {e}")

        return posts

    def fetch_openreview_papers(self, max_results: int = 10) -> List[Dict]:
        """从OpenReview获取最新论文"""
        papers = []

        if 'openreview' not in self.sources:
            return papers

        try:
            api_url = self.sources['openreview'].get('api_url', 'https://api.openreview.net/notes')
            limit = self.sources['openreview'].get('limit', 10)

            for venue_info in self.sources['openreview'].get('venues', []):
                venue_id = venue_info['id']
                keywords = venue_info.get('keywords', [])

                # 查询该会议的论文
                params = {
                    'invitation': venue_id + '/-/Blind_Submission',
                    'details': 'replyCount,writable',
                    'limit': limit
                }

                response = requests.get(api_url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    notes = data.get('notes', [])

                    for note in notes[:max_results]:
                        content = note.get('content', {})
                        title = content.get('title', 'No title')
                        abstract = content.get('abstract', '')

                        # 检查关键词匹配（如果有）
                        text = (title + ' ' + abstract).lower()
                        if keywords and not any(kw.lower() in text for kw in keywords):
                            continue

                        papers.append({
                            'source': 'openreview',
                            'venue': venue_info['name'],
                            'title': title,
                            'summary': abstract[:500] if abstract else '',
                            'url': f"https://openreview.net/forum?id={note.get('id', '')}",
                            'published': note.get('tmdate', '')[:10],
                            'keywords': keywords
                        })

        except Exception as e:
            print(f"Error fetching OpenReview: {e}")

        return papers

    def aggregate_and_rank(self, papers: List[Dict], trending: List[Dict],
                          posts: List[Dict], openreview_papers: List[Dict],
                          category_weights: Dict, target_category: str = None) -> List[Dict]:
        """聚合所有信息并根据用户偏好排序

        Args:
            papers: arXiv论文列表
            trending: GitHub trending列表
            posts: 博客文章列表
            openreview_papers: OpenReview论文列表
            category_weights: 类别权重字典
            target_category: 目标类别，如果指定则给匹配的内容大幅加分
        """
        all_items = papers + trending + posts + openreview_papers

        # 去重：同一篇论文（URL 规范化后相同）只保留一次
        import re
        seen_urls = set()
        unique_items = []
        for item in all_items:
            url = item.get('url', '')
            # 规范化 arxiv URL（去版本号）
            norm_url = re.sub(r'(arxiv\.org/abs/\d+\.\d+)v\d+', r'\1', url).rstrip('/')
            if norm_url and norm_url in seen_urls:
                continue
            if norm_url:
                seen_urls.add(norm_url)
            unique_items.append(item)
        all_items = unique_items

        # 排除关键词过滤
        exclude_keywords = self.config.get('exclude_keywords', [])
        if exclude_keywords:
            filtered_items = []
            excluded_count = 0
            for item in all_items:
                item_text = (item.get('title', '') + ' ' +
                            item.get('summary', '') + ' ' +
                            item.get('description', '')).lower()
                matched_exclude = [kw for kw in exclude_keywords if kw.lower() in item_text]
                if matched_exclude:
                    excluded_count += 1
                else:
                    filtered_items.append(item)
            if excluded_count > 0:
                print(f"  🚫 已过滤 {excluded_count} 个不相关话题（医学/生物等）")
            all_items = filtered_items

        # 类别关键词映射
        category_keywords = {
            'CUDA/GPU编程': ['cuda', 'gpu', 'kernel', 'tensor core', 'tensorcore', 'nvidia', 'gpgpu'],
            'ML/DL算法实现': ['transformer', 'attention', 'neural', 'deep learning', 'llm', 'bert', 'gpt'],
            '强化学习': ['reinforcement', 'rl', 'ppo', 'dqn', 'policy', 'reward', 'agent', 'mdp'],
            '推理优化': ['inference', 'flash attention', 'quantization', 'pruning', 'distillation', 'tensorrt'],
            '优化与科学计算': ['optimization', 'gradient', 'convex', 'numerical', 'solver', 'linear algebra'],
            'Statistical Modeling': ['bayesian', 'causal inference', 'regression', 'time series', 'forecasting',
                                     'statistical', 'hypothesis testing', 'markov chain', 'monte carlo', 'mcmc',
                                     'gaussian process', 'survival analysis', 'mixed model', 'hierarchical model',
                                     'variational inference', 'probabilistic', 'stochastic'],
            'Spatial Intelligence': [
                                     # Geospatial AI & Remote Sensing (高优先)
                                     'remote sensing', 'satellite imagery', 'satellite image', 'aerial image',
                                     'geospatial', 'earth observation', 'sentinel', 'landsat',
                                     'hyperspectral', 'multispectral', 'synthetic aperture radar',
                                     'land cover', 'land use', 'crop classification',
                                     'building extraction', 'road extraction', 'building detection',
                                     'change detection remote', 'urban mapping',
                                     # LiDAR & Point Cloud
                                     'lidar', 'point cloud classification', 'point cloud segmentation',
                                     'airborne laser', 'als point cloud',
                                     # Photogrammetry & Mapping
                                     'photogrammetry', 'orthophoto', 'structure from motion',
                                     'digital elevation model', 'dem generation', 'dsm',
                                     'drone mapping', 'uav mapping', 'opendronemap',
                                     # GIS & Spatial Analysis
                                     'gis deep learning', 'torchgeo', 'geo-foundation model', 'geosam',
                                     # 3D Vision (保留但不过于宽泛)
                                     'nerf', 'gaussian splatting', '3d gaussian',
                                     'slam', 'visual odometry',
                                     'radiance field', 'view synthesis', 'novel view',
                                     'scene reconstruction', 'depth estimation outdoor',
                                     'stereo matching', 'multiview stereo']
        }

        # 加载排除关键词
        exclude_keywords = self.config.get('exclude_keywords', [])

        # 排除包含过滤词的内容
        if exclude_keywords:
            filtered_items = []
            for item in all_items:
                item_text_check = (item.get('title', '') + ' ' +
                                   item.get('summary', '') + ' ' +
                                   item.get('description', '')).lower()
                excluded = False
                for ekw in exclude_keywords:
                    if ekw.lower() in item_text_check:
                        print(f"  ⊘ 排除 (匹配 '{ekw}'): {item.get('title', '')[:60]}")
                        excluded = True
                        break
                if not excluded:
                    filtered_items.append(item)
            all_items = filtered_items

        # 简单评分系统
        for item in all_items:
            score = 0

            # 根据分类权重评分
            item_text = (item.get('title', '') + ' ' +
                        item.get('summary', '') + ' ' +
                        item.get('description', '') + ' ' +
                        str(item.get('keywords', [])) + ' ' +
                        str(item.get('topics', []))).lower()

            # 记录匹配的类别
            matched_categories = []

            for category, keywords in category_keywords.items():
                if any(kw in item_text for kw in keywords):
                    weight = category_weights.get(category, 0.1)
                    score += weight * 10
                    matched_categories.append(category)

            # 如果指定了目标类别，给匹配的内容大幅加分
            if target_category:
                target_keywords = category_keywords.get(target_category, [])
                matched_target_kws = [kw for kw in target_keywords if kw in item_text]
                if matched_target_kws:
                    score += 50  # 目标类别匹配加50分
                    item['matched_target'] = True
                    # Geospatial 高优先关键词额外加分
                    geospatial_priority_kws = [
                        'remote sensing', 'satellite image', 'aerial image', 'geospatial',
                        'earth observation', 'hyperspectral', 'multispectral',
                        'synthetic aperture radar', 'land cover', 'land use',
                        'building extraction', 'road extraction', 'crop classification',
                        'lidar', 'point cloud classification', 'point cloud segmentation',
                        'photogrammetry', 'digital elevation model', 'drone mapping',
                        'torchgeo', 'sentinel', 'landsat'
                    ]
                    geo_matches = sum(1 for kw in geospatial_priority_kws if kw in item_text)
                    score += geo_matches * 15  # 每匹配一个 geospatial 关键词额外加15分
                else:
                    item['matched_target'] = False

            item['matched_categories'] = matched_categories

            # 时效性加分（最近3天的内容）
            try:
                pub_date = datetime.strptime(item.get('published', '')[:10], '%Y-%m-%d')
                days_old = (datetime.now() - pub_date).days
                if days_old <= 3:
                    score += 5
                elif days_old <= 7:
                    score += 2
            except:
                pass

            item['score'] = score

        # 按分数排序
        all_items.sort(key=lambda x: x['score'], reverse=True)

        return all_items

    def get_daily_recommendations(self, category_weights: Dict, top_n: int = 10,
                                    target_category: str = None) -> List[Dict]:
        """获取每日推荐技术话题

        Args:
            category_weights: 类别权重字典
            top_n: 返回的推荐数量
            target_category: 目标类别，如果指定则优先推荐该类别的内容
        """
        print("📡 正在监控技术信息源...")

        papers = self.fetch_arxiv_papers()
        print(f"  ✓ 从arXiv获取了 {len(papers)} 篇论文")

        trending = self.fetch_github_trending()
        print(f"  ✓ 从GitHub获取了 {len(trending)} 个更新")

        posts = self.fetch_blog_posts()
        print(f"  ✓ 从技术博客获取了 {len(posts)} 篇文章")

        openreview_papers = self.fetch_openreview_papers()
        if openreview_papers:
            print(f"  ✓ 从OpenReview获取了 {len(openreview_papers)} 篇论文")

        ranked = self.aggregate_and_rank(papers, trending, posts, openreview_papers,
                                         category_weights, target_category)

        if target_category:
            print(f"\n🎯 针对'{target_category}'类别，推荐以下 {min(top_n, len(ranked))} 个话题：")
        else:
            print(f"\n🎯 根据你的偏好，推荐以下 {min(top_n, len(ranked))} 个话题：")
        for i, item in enumerate(ranked[:top_n], 1):
            print(f"  {i}. [{item['source']}] {item['title'][:60]}... (评分: {item['score']:.1f})")

        return ranked[:top_n]


if __name__ == "__main__":
    # 测试
    monitor = TechMonitor('../sources/tech_sources.yaml')
    weights = {
        'CUDA/GPU编程': 0.3,
        'ML/DL算法实现': 0.25,
        '强化学习': 0.2,
        '推理优化': 0.15,
        '优化与科学计算': 0.1,
        'Spatial Intelligence': 0.2
    }

    print("=" * 60)
    print("测试1: 无指定类别")
    print("=" * 60)
    recommendations = monitor.get_daily_recommendations(weights)

    print("\n" + "=" * 60)
    print("测试2: 指定 Spatial Intelligence 类别")
    print("=" * 60)
    recommendations_spatial = monitor.get_daily_recommendations(
        weights, target_category='Spatial Intelligence'
    )
