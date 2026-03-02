#!/usr/bin/env python3
"""
AI博客自动生成系统 - 主程序
每天自动生成高质量AI技术博客

用法:
  python main.py                       # 生成今日博客并发送审阅邮件
  python main.py --dry-run             # 测试运行（不发送邮件）
  python main.py --send-review         # 发送最新草稿的审阅邮件
  python main.py --send-review --draft FILE  # 发送指定草稿的审阅邮件
  python main.py --publish             # 发布已审阅的草稿
  python main.py --category CUDA       # 指定生成类别
"""

import argparse
import yaml
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import shutil
import os

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print("✓ 已加载 .env 配置文件")
except ImportError:
    # python-dotenv 未安装，跳过
    pass

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from modules.tech_monitor import TechMonitor
from modules.content_generator import ContentGenerator
from modules.email_sender import EmailSender
from modules.code_validator import CodeValidator
from modules.blog_evaluator import BlogEvaluator
from modules.code_extractor import CodeExtractor
from modules.code_evaluator import CodeEvaluator
from modules.code_refiner import CodeRefiner
from modules.blog_refiner import BlogRefiner


class BlogGenerationSystem:
    """AI博客生成系统"""

    def __init__(self, config_path: str = '.ai-agent/config.yaml', dry_run: bool = False):
        print("🚀 AI博客生成系统启动中...")

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化模块
        self.tech_monitor = TechMonitor('.ai-agent/sources/tech_sources.yaml', config=self.config)
        self.content_generator = ContentGenerator(self.config)
        # dry-run模式下跳过邮件验证
        self.email_sender = EmailSender(self.config, skip_validation=dry_run)
        self.code_validator = CodeValidator(self.config)
        self.blog_evaluator = BlogEvaluator(self.config)
        
        # 代码处理模块（Phase 2）
        self.code_extractor = CodeExtractor()
        self.code_evaluator = CodeEvaluator(self.config)
        self.code_refiner = CodeRefiner(
            config=self.config,
            code_repo_base=os.path.expanduser("~/projects/blog-code")
        )
        
        # 博客改进模块（Phase 3）
        self.blog_refiner = BlogRefiner(self.config)

        # 类别权重
        self.category_weights = {
            cat['name']: cat['weight']
            for cat in self.config['content']['categories']
        }

        # 生成历史记录（集中式去重）
        self.history_path = Path('.ai-agent/generation_history.json')
        self.generation_history = self._load_history()

        print("✓ 系统初始化完成\n")

    def _load_history(self) -> dict:
        """加载生成历史"""
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"generated": []}

    def _save_history(self):
        """保存生成历史"""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(self.generation_history, f, ensure_ascii=False, indent=2)

    def _record_generation(self, topic_url: str, topic_title: str, filename: str, score: float):
        """记录一次成功的博客生成"""
        normalized_url = self._normalize_url(topic_url)
        title_slug = self._title_to_slug(topic_title)
        self.generation_history["generated"].append({
            "url": normalized_url,
            "original_url": topic_url,
            "title": topic_title,
            "title_slug": title_slug,
            "filename": filename,
            "score": score,
            "generated_at": datetime.now().isoformat()
        })
        self._save_history()

    @staticmethod
    def _normalize_url(url: str) -> str:
        """规范化 URL，去掉 arxiv 版本号等差异"""
        import re
        # arxiv: 去掉版本号 (v1, v2, ...)
        url = re.sub(r'(arxiv\.org/abs/\d+\.\d+)v\d+', r'\1', url)
        return url.rstrip('/')

    @staticmethod
    def _title_to_slug(title: str) -> str:
        """将标题转为 slug 用于匹配"""
        import re
        slug = title.lower().strip()
        slug = re.sub(r'[^a-z0-9\s]', '', slug)
        slug = re.sub(r'\s+', '-', slug)
        return slug[:50]  # 取前50字符足够匹配

    def _is_topic_already_generated(self, topic_url: str, topic_title: str = '') -> bool:
        """检查某个话题是否已经生成过博客

        使用三层去重机制：
        1. 集中式历史记录（generation_history.json）— 最可靠
        2. 草稿/已发布文件的 source_url 匹配
        3. 文件名 slug 匹配
        """
        normalized_url = self._normalize_url(topic_url)
        title_slug = self._title_to_slug(topic_title) if topic_title else ''

        # === 第一层：查历史记录（最快最可靠）===
        for entry in self.generation_history.get("generated", []):
            # URL 匹配
            if normalized_url and (entry.get("url") == normalized_url or
                                   entry.get("original_url") == topic_url):
                return True
            # 标题 slug 匹配（防止同一话题 URL 略有不同）
            if title_slug and entry.get("title_slug") and \
               title_slug[:30] == entry.get("title_slug", "")[:30]:
                return True

        # === 第二层：扫描文件（兜底，覆盖历史记录引入前的旧文件）===
        def _check_file(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read(1000)
                    # source_url 字段匹配
                    for line in content.split('\n'):
                        if line.startswith('source_url:'):
                            file_url = line.split(':', 1)[1].strip()
                            file_url_normalized = self._normalize_url(file_url)
                            if normalized_url and (
                                file_url_normalized == normalized_url or
                                file_url == topic_url
                            ):
                                return True
                            break  # source_url 只有一行
                    # 文件名 slug 匹配
                    if title_slug and title_slug[:30] in filepath.stem:
                        return True
            except:
                pass
            return False

        # 检查草稿目录
        drafts_dir = Path(self.config['drafts']['save_location'])
        if drafts_dir.exists():
            for draft_file in drafts_dir.glob('*.markdown'):
                if _check_file(draft_file):
                    return True

        # 检查所有已发布的博客目录
        for cat in self.config['content']['categories']:
            posts_dir = Path(cat['output_dir'])
            if posts_dir.exists():
                for post_file in posts_dir.glob('*.markdown'):
                    if _check_file(post_file):
                        return True

        return False

    def generate_daily_blog(self, specified_category: str = None, dry_run: bool = False):
        """生成每日博客"""
        print("=" * 60)
        print(f"  AI博客生成 - {datetime.now().strftime('%Y年%m月%d日')}")
        print("=" * 60)

        # 1. 监控技术源，获取推荐话题
        # 如果指定了类别，传递给推荐系统以优先获取该类别的内容
        recommendations = self.tech_monitor.get_daily_recommendations(
            self.category_weights,
            top_n=10,
            target_category=specified_category
        )

        if not recommendations:
            print("❌ 没有找到合适的技术话题")
            return False

        # 2. 选择话题和类别
        if specified_category:
            # 如果指定了类别，先过滤出符合该类别的话题
            filtered_topics = [
                topic for topic in recommendations
                if self._determine_category(topic) == specified_category
            ]

            if not filtered_topics:
                print(f"⚠️  未找到符合'{specified_category}'类别的话题")
                print(f"💡 将从所有话题中选择，并使用指定类别生成")
                topics_to_consider = recommendations
            else:
                topics_to_consider = filtered_topics
                print(f"✓ 找到 {len(filtered_topics)} 个符合'{specified_category}'的话题")

            category = specified_category
        else:
            # 未指定类别，考虑所有话题
            topics_to_consider = recommendations
            category = None

        # 3. 从候选话题中选择未生成过的话题
        selected_topic = None
        skipped_count = 0

        for topic in topics_to_consider:
            if self._is_topic_already_generated(topic['url'], topic.get('title', '')):
                skipped_count += 1
                print(f"⏭️  跳过已生成: {topic['title'][:50]}...")
                continue

            selected_topic = topic
            break

        if not selected_topic:
            print(f"⚠️  所有 {len(topics_to_consider)} 个候选话题都已生成过")
            print(f"💡 本次跳过生成，避免重复内容")
            return False

        if skipped_count > 0:
            print(f"✓ 跳过了 {skipped_count} 个已生成的话题")

        # 4. 确定最终分类
        if not category:
            category = self._determine_category(selected_topic)

        print(f"\n🎯 选定话题: {selected_topic['title'][:60]}...")
        print(f"📂 分类: {category}")
        print(f"🔗 来源: {selected_topic['url']}")

        # 3. 生成博客内容
        try:
            blog_data = self.content_generator.generate_blog_post(
                selected_topic,
                category
            )
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return False

        # 4. 验证代码
        validation_results = self.code_validator.validate_blog_post(blog_data['content'])

        if not validation_results['valid']:
            print("\n⚠ 代码验证未通过，但仍将保存草稿供人工审阅")

        # 5. 质量检查
        quality = self.code_validator.check_blog_quality(blog_data['content'])
        print(f"\n📊 质量评分: {quality['quality_score']}/100")
        if quality['suggestions']:
            print("💡 改进建议:")
            for suggestion in quality['suggestions']:
                print(f"  - {suggestion}")

        # 5.5 AI 深度评估（使用 Claude CLI）
        evaluation = None
        try:
            evaluation = self.blog_evaluator.evaluate_blog(
                blog_data['content'],
                selected_topic
            )
            print(f"\n🎯 AI评估总分: {evaluation.get('overall_score', 'N/A')}/10")
            
            # 显示简要评估结果
            struct = evaluation.get('structure', {})
            print(f"   文字/代码比例: {struct.get('text_ratio', 0):.0%} / {struct.get('code_ratio', 0):.0%}")
            print(f"   比例合理: {'✓' if struct.get('balanced', True) else '✗ 需调整'}")
            
            if evaluation.get('suggestions'):
                print("   改进建议:")
                for s in evaluation['suggestions'][:2]:
                    print(f"     - {s}")
        except Exception as e:
            print(f"\n⚠️ AI评估失败: {e}，将跳过评估")

        # 5.6 代码处理（Phase 2: 提取、评估、重构）
        code_repo_path = None
        try:
            print("\n🔧 代码处理中...")
            
            # 5.6.1 提取代码块
            extraction_result = self.code_extractor.extract(blog_data['content'])
            print(f"   提取到 {len(extraction_result.code_blocks)} 个代码块")
            print(f"   代码占比: {extraction_result.code_ratio:.1%}")
            
            # 5.6.2 评估代码块（判断核心/辅助）
            if extraction_result.code_blocks and extraction_result.code_ratio > 0.5:
                print("   正在评估代码块...")
                code_eval_result = self.code_evaluator.evaluate(
                    extraction_result,
                    blog_data.get('tech_topic', {}).get('title', '')
                )
                print(f"   代码评估完成: {code_eval_result.summary}")
                
                # 5.6.3 重构代码（精简博客 + 生成代码库）
                if code_eval_result.needs_refactoring:
                    print("   正在重构代码...")
                    
                    # 从文件名提取 slug
                    blog_slug = blog_data['filename'].replace('.markdown', '')
                    
                    refiner_output = self.code_refiner.refine(
                        blog_content=blog_data['content'],
                        extraction_result=extraction_result,
                        evaluation_result=code_eval_result,
                        blog_slug=blog_slug,
                        blog_title=blog_data.get('tech_topic', {}).get('title', ''),
                        source_url=blog_data.get('tech_topic', {}).get('url', '')
                    )
                    
                    # 更新博客内容为精简版
                    blog_data['content'] = refiner_output.refined_blog
                    code_repo_path = refiner_output.code_repo_path
                    
                    print(f"   ✓ 代码重构完成")
                    for change in refiner_output.changes_made:
                        print(f"     - {change}")
                    print(f"   📁 代码库: {code_repo_path}")
                else:
                    print("   ✓ 代码结构合理，无需重构")
            else:
                print("   ✓ 代码占比正常，跳过重构")
                
        except Exception as e:
            print(f"\n⚠️ 代码处理失败: {e}，将使用原始内容")

        # 5.6.5 先保存草稿（防止后续迭代超时丢失）
        draft_path = self.content_generator.save_draft(blog_data)
        self._save_metadata(blog_data, draft_path, validation_results, quality)
        print(f"  ✓ 草稿已保存: {draft_path}")

        # 5.7 评估结果诊断（不再 refine 循环，改为写 PROMPT_IMPROVEMENTS.md）
        # 理由：per-post refine 循环浪费 token，经常 timeout，且无法修复 prompt 本身的问题。
        # 正确做法：prompt 写好，评估做诊断，不达标时记录原因给人工改 prompt。
        TARGET_SCORE = 8.0
        if evaluation:
            current_score = evaluation.get('overall_score', 0)
            structure = evaluation.get('structure', {})
            text_ratio = structure.get('text_ratio', 0)

            if current_score >= TARGET_SCORE and 0.40 <= text_ratio <= 0.60:
                print(f"\n✓ 博客质量达标 (评分: {current_score}/10, 文字占比: {text_ratio:.0%})")
            else:
                print(f"\n⚠ 博客评分 {current_score}/10（目标 {TARGET_SCORE}），记录改进建议...")
                suggestions = evaluation.get('suggestions', [])
                self._log_prompt_improvements(
                    topic=selected_topic,
                    score=current_score,
                    category=selected_category,
                    suggestions=suggestions,
                    evaluation=evaluation,
                )

        # 6. 重新保存草稿（迭代改进后的版本）
        draft_path = self.content_generator.save_draft(blog_data)
        self._save_metadata(blog_data, draft_path, validation_results, quality)

        # 7. 记录到生成历史（防止后续 run 重复选同一话题）
        current_score = evaluation.get('overall_score', 0) if evaluation else 0
        self._record_generation(
            topic_url=selected_topic['url'],
            topic_title=selected_topic.get('title', ''),
            filename=blog_data['filename'],
            score=current_score
        )

        # 8. 自动发布或发送审阅邮件
        auto_publish_config = self.config.get('auto_publish', {})
        auto_publish_enabled = auto_publish_config.get('enabled', False)
        publish_threshold = auto_publish_config.get('threshold', 7.0)
        send_email = auto_publish_config.get('send_email', True)
        
        if not dry_run:
            # 照常发送邮件通知
            if send_email:
                self.email_sender.send_draft_review(blog_data, draft_path, evaluation)
                print(f"\n📧 审阅邮件已发送")
            
            # 评分达标则自动发布
            if auto_publish_enabled and current_score >= publish_threshold:
                print(f"\n🤖 评分 {current_score}/10 >= 阈值 {publish_threshold}，自动发布...")
                self.publish_draft(Path(draft_path).name)
                print("\n" + "=" * 60)
                print("✅ 博客已自动发布!")
                print(f"📄 草稿位置: {draft_path}")
                print(f"📊 评分: {current_score}/10")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print(f"⚠️ 评分 {current_score}/10 < 阈值 {publish_threshold}，未自动发布")
                print(f"📄 草稿位置: {draft_path}")
                print(f"💡 手动发布: python .ai-agent/main.py --publish")
                print("=" * 60)
        else:
            print(f"\n🔍 试运行模式：跳过发布")
            print(f"   草稿已保存到: {draft_path}")

        return True

    def send_review_email(self, draft_filename: str = None):
        """单独发送草稿审阅邮件"""
        print("\n📧 准备发送审阅邮件...\n")

        # 查找草稿
        drafts_dir = Path(self.config['drafts']['save_location'])

        if draft_filename:
            draft_path = drafts_dir / draft_filename
        else:
            # 找最新的草稿
            drafts = sorted(drafts_dir.glob('*.markdown'),
                          key=lambda p: p.stat().st_mtime,
                          reverse=True)
            if not drafts:
                print("❌ 未找到草稿文件")
                return False
            draft_path = drafts[0]

        if not draft_path.exists():
            print(f"❌ 草稿文件不存在: {draft_path}")
            return False

        print(f"📄 草稿: {draft_path.name}")

        # 读取草稿内容
        with open(draft_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 构建博客数据
        import re
        title_match = re.search(r'title:\s*["\'](.+?)["\']', content)
        title = title_match.group(1) if title_match else "未知标题"

        category_match = re.search(r'category:\s*(\w+)', content)
        category = category_match.group(1) if category_match else "AI"

        source_match = re.search(r'source_url:\s*(.+)', content)
        source_url = source_match.group(1) if source_match else "unknown"

        blog_data = {
            'filename': draft_path.name,
            'category': category,
            'tech_topic': {
                'title': title,
                'url': source_url.strip()
            },
            'word_count': len(content),
            'has_code': '```' in content
        }

        # 发送邮件
        success = self.email_sender.send_draft_review(blog_data, str(draft_path))

        if success:
            print("\n✅ 审阅邮件已发送！")
            print("\n接下来:")
            print("  1. 查收邮件，审阅内容")
            print(f"  2. 如需修改: vim {draft_path}")
            print("  3. 发布博客: python .ai-agent/main.py --publish")

        return success

    def _is_draft_already_published(self, draft_filename: str) -> bool:
        """检查草稿是否已经发布到 _posts 目录"""
        for cat in self.config['content']['categories']:
            posts_dir = Path(cat['output_dir'])
            if (posts_dir / draft_filename).exists():
                return True
        # 也检查 generation_history 中的 status 字段
        for entry in self.generation_history.get("generated", []):
            if entry.get("filename") == draft_filename and entry.get("status") == "published":
                return True
        return False

    def _find_latest_unpublished_draft(self) -> Path | None:
        """从 generation_history 中找最新的未发布草稿，而非盲目按 mtime"""
        drafts_dir = Path(self.config['drafts']['save_location'])

        # 优先从历史记录中找（按时间倒序）
        for entry in reversed(self.generation_history.get("generated", [])):
            filename = entry.get("filename", "")
            if not filename:
                continue
            if entry.get("status") == "published":
                continue
            draft_path = drafts_dir / filename
            if draft_path.exists() and not self._is_draft_already_published(filename):
                return draft_path

        # 兜底：按 mtime 找，但排除已发布的
        drafts = sorted(drafts_dir.glob('*.markdown'), key=lambda p: p.stat().st_mtime, reverse=True)
        for draft in drafts:
            if not self._is_draft_already_published(draft.name):
                return draft

        return None

    def publish_draft(self, draft_filename: str = None):
        """发布已审阅的草稿"""
        print("\n📤 准备发布博客...\n")

        # 查找草稿
        drafts_dir = Path(self.config['drafts']['save_location'])

        if draft_filename:
            draft_path = drafts_dir / draft_filename
        else:
            # 找最新的未发布草稿
            draft_path = self._find_latest_unpublished_draft()
            if not draft_path:
                print("❌ 未找到未发布的草稿文件")
                return False

        if not draft_path.exists():
            print(f"❌ 草稿文件不存在: {draft_path}")
            return False

        # 检查是否已发布（防止重复发布）
        if self._is_draft_already_published(draft_path.name):
            print(f"⏭️  草稿已发布过，跳过: {draft_path.name}")
            return False

        print(f"📄 草稿: {draft_path.name}")

        # 读取草稿内容
        with open(draft_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 确定目标目录
        category_line = [line for line in content.split('\n') if line.startswith('category:')]
        if not category_line:
            print("❌ 无法确定博客分类")
            return False

        category_tag = category_line[0].split(':', 1)[1].strip()

        # 找到对应的输出目录
        output_dir = None
        for cat in self.config['content']['categories']:
            if cat['category_tag'] == category_tag:
                output_dir = Path(cat['output_dir'])
                break

        if not output_dir:
            print(f"❌ 未找到分类 {category_tag} 的输出目录")
            return False

        # 复制到博客目录
        target_path = output_dir / draft_path.name
        shutil.copy(draft_path, target_path)

        print(f"✓ 博客已复制到: {target_path}")

        # 标记为已发布
        for entry in self.generation_history.get("generated", []):
            if entry.get("filename") == draft_path.name:
                entry["status"] = "published"
                self._save_history()
                break

        # Git操作
        if self.config['git'].get('auto_commit', False):
            self._git_commit_and_push(target_path, content)

        print("\n✅ 博客发布成功!")
        print(f"📁 位置: {target_path}")
        print("\n接下来:")
        print("  1. 运行 git status 查看更改")
        print("  2. 运行 git add . && git commit -m '添加博客' && git push")
        print("  3. GitHub Pages 将自动部署")

        return True

    def _determine_category(self, topic: Dict) -> str:
        """根据话题内容确定分类"""
        text = (topic.get('title', '') + ' ' +
               topic.get('summary', '') + ' ' +
               str(topic.get('keywords', []))).lower()

        # 评分
        scores = {}

        if any(kw in text for kw in ['cuda', 'gpu', 'kernel', 'tensorcore']):
            scores['CUDA/GPU编程'] = 10
        if any(kw in text for kw in ['nerf', 'gaussian splatting', '3d reconstruction',
                                      'slam', 'point cloud', 'spatial', '3d vision',
                                      'depth estimation', 'pose estimation', 'voxel',
                                      'mesh', 'geometry', 'camera', 'lidar', 'rgbd']):
            scores['Spatial Intelligence'] = 9
        if any(kw in text for kw in ['transformer', 'attention', 'llm', 'neural']):
            scores['ML/DL算法实现'] = 8
        if any(kw in text for kw in ['reinforcement', 'rl', 'policy', 'dqn']):
            scores['强化学习'] = 9
        if any(kw in text for kw in ['inference', 'optimization', 'quantization']):
            scores['推理优化'] = 7
        if any(kw in text for kw in ['optimization', 'gradient', 'numerical']):
            scores['优化与科学计算'] = 6

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        return 'ML/DL算法实现'  # 默认分类

    def _log_prompt_improvements(self, topic: Dict, score: float,
                                  category: str, suggestions: list,
                                  evaluation: Dict):
        """当评估分数不达标时，将改进建议写入 PROMPT_IMPROVEMENTS.md。
        不再触发 refine 循环——改进 prompt 本身才是正确方式。"""
        improvements_path = Path('.ai-agent/PROMPT_IMPROVEMENTS.md')
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        lines = []
        if not improvements_path.exists():
            lines.append("# Prompt Improvement Log\n")
            lines.append("每次生成评分低于阈值时自动记录，供人工改进提示词使用。\n\n")
        
        lines.append(f"## [{now}] Score {score}/10 — {category}\n")
        lines.append(f"**Topic**: {topic.get('title', 'unknown')}\n\n")
        
        depth = evaluation.get('content_depth', {})
        code_q = evaluation.get('code_quality', {})
        struct = evaluation.get('structure', {})
        
        lines.append(f"**Scores**: depth={depth.get('score','?')} | code={code_q.get('score','?')} | structure={struct.get('score','?')}\n\n")
        
        if suggestions:
            lines.append("**Suggestions (fix in prompt):**\n")
            for s in suggestions:
                lines.append(f"- {s}\n")
        
        if depth.get('comments'):
            lines.append(f"\n**Depth comment**: {depth['comments']}\n")
        
        lines.append("\n---\n\n")
        
        with open(improvements_path, 'a', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"  📝 改进建议已记录到 .ai-agent/PROMPT_IMPROVEMENTS.md")
        print(f"  （不触发 refine 循环，请人工根据建议改进 prompts/ 目录下的模板）")

    def _save_metadata(self, blog_data: Dict, draft_path: str,
                      validation: Dict, quality: Dict):
        """保存生成的元数据"""
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'draft_path': draft_path,
            'blog_data': {
                'filename': blog_data['filename'],
                'category': blog_data['category'],
                'word_count': blog_data['word_count'],
                'has_code': blog_data['has_code']
            },
            'tech_topic': blog_data['tech_topic'],
            'validation': validation,
            'quality': quality
        }

        metadata_path = Path(draft_path).with_suffix('.meta.yaml')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, allow_unicode=True)

    def _git_commit_and_push(self, file_path: Path, content: str):
        """Git提交和推送（仅在最终发布时调用一次）"""
        import subprocess

        # 提取标题
        title_line = [line for line in content.split('\n') if line.startswith('title:')]
        title = title_line[0].split(':', 1)[1].strip().strip('"') if title_line else "新博客"

        commit_msg = self.config['git']['commit_message_template'].format(
            title=title,
            date=datetime.now().strftime('%Y-%m-%d')
        )

        try:
            # 确保在 main 分支上（防止 detached HEAD）
            try:
                current = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                        capture_output=True, text=True, check=True)
                if current.stdout.strip() == 'HEAD':
                    print("⚠ Detached HEAD 检测到，切回 main...")
                    subprocess.run(['git', 'stash', '--include-untracked'], capture_output=True)
                    subprocess.run(['git', 'checkout', 'main'], check=True, capture_output=True)
                    subprocess.run(['git', 'stash', 'pop'], capture_output=True)
            except subprocess.CalledProcessError:
                pass

            # 先拉取远端更新，避免 non-fast-forward 错误
            try:
                subprocess.run(['git', 'pull', '--rebase', '--autostash'], check=True,
                             capture_output=True, text=True)
            except subprocess.CalledProcessError:
                print("⚠ git pull --rebase 失败，尝试 merge...")
                try:
                    subprocess.run(['git', 'rebase', '--abort'], capture_output=True)
                    subprocess.run(['git', 'pull', '--no-rebase'], check=True,
                                 capture_output=True, text=True)
                except subprocess.CalledProcessError:
                    print("⚠ git pull 也失败，尝试继续...")

            # Git add（目标文件 + generation history）
            subprocess.run(['git', 'add', str(file_path)], check=True)
            subprocess.run(['git', 'add', '.ai-agent/generation_history.json'], capture_output=True)

            # 检查是否有实际变更需要提交
            result = subprocess.run(['git', 'diff', '--cached', '--quiet'],
                                   capture_output=True)
            if result.returncode == 0:
                print("✓ 文件无变更，跳过 commit")
                return

            # Git commit
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            print("✓ Git commit 完成")

            # 推送
            try:
                subprocess.run(['git', 'push'], check=True)
                print("✓ 已推送到GitHub")
            except subprocess.CalledProcessError:
                # push 失败时再次尝试 rebase
                print("⚠ push 失败，尝试 rebase 后重试...")
                subprocess.run(['git', 'pull', '--rebase', '--autostash'], check=True)
                subprocess.run(['git', 'push'], check=True)
                print("✓ rebase 后推送成功")

        except subprocess.CalledProcessError as e:
            print(f"⚠ Git操作失败: {e}")


def _acquire_lock():
    """获取文件锁，防止并发执行"""
    import fcntl
    lock_path = Path(__file__).parent / '.generation.lock'
    lock_file = open(lock_path, 'w')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        return lock_file
    except (IOError, OSError):
        # 检查持锁进程是否还活着
        try:
            with open(lock_path, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)  # 检查进程是否存在
            print(f"⚠ 另一个实例正在运行 (PID {pid})，退出")
            lock_file.close()
            return None
        except (ValueError, ProcessLookupError, PermissionError):
            # 持锁进程已死，强制获取锁
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(str(os.getpid()))
            lock_file.flush()
            return lock_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI博客自动生成系统')
    parser.add_argument('--publish', action='store_true', help='发布已审阅的草稿')
    parser.add_argument('--send-review', action='store_true', help='发送草稿审阅邮件')
    parser.add_argument('--dry-run', action='store_true', help='试运行（不发送邮件）')
    parser.add_argument('--category', type=str, help='指定生成类别')
    parser.add_argument('--draft', type=str, help='指定草稿文件名（默认使用最新）')

    args = parser.parse_args()

    # 切换到项目根目录
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)

    # 获取文件锁，防止并发执行
    lock_file = _acquire_lock()
    if lock_file is None:
        sys.exit(1)

    # 初始化系统
    system = BlogGenerationSystem(dry_run=(args.dry_run or args.send_review))

    if args.send_review:
        # 发送审阅邮件模式
        success = system.send_review_email(args.draft)
    elif args.publish:
        # 发布模式
        success = system.publish_draft(args.draft)
    else:
        # 生成模式
        success = system.generate_daily_blog(
            specified_category=args.category,
            dry_run=args.dry_run
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
