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
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict
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
        self.tech_monitor = TechMonitor('.ai-agent/sources/tech_sources.yaml')
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

        print("✓ 系统初始化完成\n")

    def _is_topic_already_generated(self, topic_url: str) -> bool:
        """检查某个话题是否已经生成过博客"""
        # 检查草稿目录
        drafts_dir = Path(self.config['drafts']['save_location'])
        if drafts_dir.exists():
            for draft_file in drafts_dir.glob('*.markdown'):
                try:
                    with open(draft_file, 'r', encoding='utf-8') as f:
                        content = f.read(500)  # 只读前500字符
                        if topic_url in content or f"source_url: {topic_url}" in content:
                            return True
                except:
                    pass

        # 检查所有已发布的博客目录
        for cat in self.config['content']['categories']:
            posts_dir = Path(cat['output_dir'])
            if posts_dir.exists():
                for post_file in posts_dir.glob('*.markdown'):
                    try:
                        with open(post_file, 'r', encoding='utf-8') as f:
                            content = f.read(500)
                            if topic_url in content or f"source_url: {topic_url}" in content:
                                return True
                    except:
                        pass

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
            if self._is_topic_already_generated(topic['url']):
                skipped_count += 1
                print(f"⏭️  跳过已生成: {topic['title'][:50]}...")
                continue

            selected_topic = topic
            break

        if not selected_topic:
            print(f"⚠️  所有 {len(topics_to_consider)} 个候选话题都已生成过")
            print(f"💡 将重新生成评分最高的话题（可能产生不同内容）")
            selected_topic = topics_to_consider[0]

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

        # 5.7 Phase 3: 迭代改进博客（最多2次）
        MAX_ITERATIONS = 2
        TARGET_SCORE = 8.0
        
        for iteration in range(1, MAX_ITERATIONS + 1):
            # 检查是否需要改进
            if evaluation:
                current_score = evaluation.get('overall_score', 0)
                structure = evaluation.get('structure', {})
                text_ratio = structure.get('text_ratio', 0)
                
                # 检查是否已达标
                if current_score >= TARGET_SCORE and 0.40 <= text_ratio <= 0.60:
                    print(f"\n✓ 博客质量达标 (评分: {current_score}/10, 文字占比: {text_ratio:.0%})")
                    break
                
                # 需要改进
                print(f"\n🔄 迭代改进 ({iteration}/{MAX_ITERATIONS})...")
                print(f"   当前评分: {current_score}/10, 目标: {TARGET_SCORE}/10")
                
                try:
                    refine_result = self.blog_refiner.refine(
                        blog_content=blog_data['content'],
                        evaluation=evaluation,
                        iteration=iteration
                    )
                    
                    if refine_result.converged:
                        print(f"   ✓ 已达标，停止迭代")
                        break
                    
                    # 更新博客内容
                    blog_data['content'] = refine_result.refined_content
                    
                    print(f"   改动:")
                    for change in refine_result.changes_made:
                        print(f"     - {change}")
                    
                    # 重新评估
                    print(f"   重新评估中...")
                    evaluation = self.blog_evaluator.evaluate_blog(
                        blog_data['content'],
                        selected_topic
                    )
                    new_score = evaluation.get('overall_score', 0)
                    print(f"   新评分: {new_score}/10")
                    
                    if new_score >= TARGET_SCORE:
                        print(f"   ✓ 评分达标，停止迭代")
                        break
                        
                except Exception as e:
                    print(f"   ⚠ 迭代改进失败: {e}")
                    break
            else:
                break

        # 6. 保存草稿
        draft_path = self.content_generator.save_draft(blog_data)

        # 7. 保存元数据
        self._save_metadata(blog_data, draft_path, validation_results, quality)

        # 8. 自动发布或发送审阅邮件
        auto_publish_config = self.config.get('auto_publish', {})
        auto_publish_enabled = auto_publish_config.get('enabled', False)
        publish_threshold = auto_publish_config.get('threshold', 7.0)
        send_email = auto_publish_config.get('send_email', True)
        
        current_score = evaluation.get('overall_score', 0) if evaluation else 0
        
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

    def publish_draft(self, draft_filename: str = None):
        """发布已审阅的草稿"""
        print("\n📤 准备发布博客...\n")

        # 查找最新草稿
        drafts_dir = Path(self.config['drafts']['save_location'])

        if draft_filename:
            draft_path = drafts_dir / draft_filename
        else:
            # 找最新的草稿
            drafts = sorted(drafts_dir.glob('*.markdown'), key=lambda p: p.stat().st_mtime, reverse=True)
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
        """Git提交和推送"""
        import subprocess

        # 提取标题
        title_line = [line for line in content.split('\n') if line.startswith('title:')]
        title = title_line[0].split(':', 1)[1].strip().strip('"') if title_line else "新博客"

        commit_msg = self.config['git']['commit_message_template'].format(
            title=title,
            date=datetime.now().strftime('%Y-%m-%d')
        )

        try:
            # Git add
            subprocess.run(['git', 'add', str(file_path)], check=True)

            # Git commit
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)

            print("✓ Git commit 完成")

            # 询问是否push
            response = input("\n是否推送到GitHub? (y/n): ")
            if response.lower() == 'y':
                subprocess.run(['git', 'push'], check=True)
                print("✓ 已推送到GitHub")

        except subprocess.CalledProcessError as e:
            print(f"⚠ Git操作失败: {e}")


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
