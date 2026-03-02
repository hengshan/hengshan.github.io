"""
邮件发送模块
发送博客草稿审阅邮件
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from typing import Dict


class EmailSender:
    """邮件发送器"""

    def __init__(self, config: Dict, skip_validation: bool = False):
        self.smtp_server = config['email']['smtp_server']
        self.smtp_port = config['email']['smtp_port']
        self.from_email = os.environ.get(config['email']['from_email_env'])
        self.password = os.environ.get(config['email']['password_env'])
        self.to_email = os.environ.get(config['email']['to_email_env'])
        self.subject_template = config['email']['subject_template']

        # dry-run模式下跳过验证
        if not skip_validation and not all([self.from_email, self.password, self.to_email]):
            raise ValueError(
                "请设置邮件相关的环境变量:\n"
                f"  {config['email']['from_email_env']}\n"
                f"  {config['email']['password_env_env']}\n"
                f"  {config['email']['to_email_env']}"
            )

    def send_draft_review(self, blog_data: Dict, draft_path: str, evaluation: Dict = None) -> bool:
        """发送博客草稿审阅邮件
        
        Args:
            blog_data: 博客数据
            draft_path: 草稿路径
            evaluation: AI评估结果（可选）
        """
        print(f"\n📧 正在发送审阅邮件到 {self.to_email}...")

        # 读取草稿内容
        with open(draft_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取标题
        title_line = [line for line in content.split('\n') if line.startswith('title:')]
        title = title_line[0].split(':', 1)[1].strip().strip('"') if title_line else "未知标题"

        # 构建邮件主题
        subject = self.subject_template.format(
            date=datetime.now().strftime('%Y-%m-%d'),
            topic=title[:30]
        )

        # 构建HTML邮件内容
        html_content = self._create_html_email(blog_data, draft_path, content, evaluation)

        # 创建邮件
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.from_email
        msg['To'] = self.to_email

        # 添加HTML内容
        msg.attach(MIMEText(html_content, 'html', 'utf-8'))

        # 发送邮件
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.from_email, self.password)
                server.send_message(msg)

            print("  ✓ 邮件发送成功")
            return True

        except Exception as e:
            print(f"  ✗ 邮件发送失败: {e}")
            return False

    def _create_evaluation_html(self, evaluation: Dict) -> str:
        """创建评估报告的HTML部分"""
        if not evaluation:
            return ''
        
        overall = evaluation.get('overall_score', 7)
        depth = evaluation.get('content_depth', {})
        code = evaluation.get('code_quality', {})
        struct = evaluation.get('structure', {})
        
        # 根据分数决定颜色
        def score_color(score):
            if score >= 8:
                return '#48bb78'  # 绿色
            elif score >= 6:
                return '#ed8936'  # 橙色
            else:
                return '#e53e3e'  # 红色
        
        stars = '⭐' * min(overall, 10)
        
        html = f"""
    <div style="background: linear-gradient(135deg, #f6f9fc 0%, #edf2f7 100%); 
                border: 2px solid #667eea; border-radius: 10px; 
                padding: 25px; margin: 25px 0;">
        <h2 style="color: #667eea; margin-top: 0; border-bottom: 2px solid #667eea; padding-bottom: 10px;">
            🔍 AI 质量评估报告
        </h2>
        
        <!-- 总分 -->
        <div style="text-align: center; padding: 20px; background: white; border-radius: 8px; margin: 15px 0;">
            <div style="font-size: 48px; font-weight: bold; color: {score_color(overall)};">
                {overall}/10
            </div>
            <div style="font-size: 24px;">{stars}</div>
            <div style="color: #718096; margin-top: 10px;">总体评分</div>
        </div>
        
        <!-- 详细评分 -->
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0;">
            <!-- 内容深度 -->
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: {score_color(depth.get('score', 7))};">
                    {depth.get('score', 7)}/10
                </div>
                <div style="color: #718096; font-size: 14px;">📚 内容深度</div>
            </div>
            <!-- 代码质量 -->
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: {score_color(code.get('score', 7))};">
                    {code.get('score', 7)}/10
                </div>
                <div style="color: #718096; font-size: 14px;">💻 代码质量</div>
            </div>
            <!-- 结构平衡 -->
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: {score_color(struct.get('score', 7))};">
                    {struct.get('score', 7)}/10
                </div>
                <div style="color: #718096; font-size: 14px;">📐 结构平衡</div>
            </div>
        </div>
        
        <!-- 详细分析 -->
        <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h3 style="color: #667eea; margin-top: 0;">📊 详细分析</h3>
            
            <p><strong>📚 内容深度:</strong> {depth.get('comments', 'N/A')}</p>
            
            <p><strong>💻 代码质量:</strong> 
                可运行: {'✅ 是' if code.get('runnable', True) else '❌ 否'}
                {' | 问题: ' + ', '.join(code.get('issues', [])[:2]) if code.get('issues') else ''}
            </p>
            
            <p><strong>📐 文字/代码比例:</strong> 
                文字 {struct.get('text_ratio', 0.5):.0%} / 代码 {struct.get('code_ratio', 0.5):.0%}
                {'✅ 比例合理' if struct.get('balanced', True) else '⚠️ 需要调整'}
            </p>
            <p style="color: #718096; font-size: 14px;">{struct.get('comments', '')}</p>
        </div>
        
        <!-- 总结 -->
        <div style="background: #667eea; color: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <strong>📝 AI 总结:</strong> {evaluation.get('summary', 'N/A')}
        </div>
        
        <!-- 改进建议 -->
        {"<div style='background: #fffaf0; border-left: 4px solid #ed8936; padding: 15px; border-radius: 5px;'><strong>💡 改进建议:</strong><ul style='margin: 10px 0; padding-left: 20px;'>" + ''.join([f"<li>{s}</li>" for s in evaluation.get('suggestions', [])[:5]]) + "</ul></div>" if evaluation.get('suggestions') else ''}
    </div>
"""
        return html

    def _create_html_email(self, blog_data: Dict, draft_path: str, content: str, evaluation: Dict = None) -> str:
        """创建HTML格式的审阅邮件"""
        # 提取元信息
        tech_topic = blog_data['tech_topic']
        category = blog_data['category']
        word_count = blog_data['word_count']
        has_code = blog_data['has_code']

        # 代码块数量
        code_blocks = content.count('```')

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .meta {{
            background: #f7fafc;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .meta-item {{
            margin: 10px 0;
        }}
        .meta-label {{
            font-weight: bold;
            color: #667eea;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            border: 2px solid #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #718096;
            font-size: 14px;
        }}
        .action-buttons {{
            margin: 30px 0;
            text-align: center;
        }}
        .button {{
            display: inline-block;
            padding: 12px 24px;
            margin: 0 10px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s;
        }}
        .button-approve {{
            background: #48bb78;
            color: white;
        }}
        .button-edit {{
            background: #ed8936;
            color: white;
        }}
        .preview {{
            background: #f7fafc;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin: 20px 0;
            max-height: 400px;
            overflow-y: auto;
        }}
        .preview pre {{
            background: #2d3748;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e2e8f0;
            color: #718096;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📝 AI博客草稿待审阅</h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">
            {datetime.now().strftime('%Y年%m月%d日 %H:%M')}
        </p>
    </div>

    <div class="meta">
        <div class="meta-item">
            <span class="meta-label">📂 分类:</span> {category}
        </div>
        <div class="meta-item">
            <span class="meta-label">📌 话题:</span> {tech_topic['title']}
        </div>
        <div class="meta-item">
            <span class="meta-label">🔗 来源:</span>
            <a href="{tech_topic['url']}">{tech_topic['url']}</a>
        </div>
        <div class="meta-item">
            <span class="meta-label">📄 文件:</span> {draft_path}
        </div>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{word_count:,}</div>
            <div class="stat-label">字数</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{code_blocks // 2}</div>
            <div class="stat-label">代码块</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{'✓' if has_code else '✗'}</div>
            <div class="stat-label">包含代码</div>
        </div>
    </div>

    {self._create_evaluation_html(evaluation) if evaluation else ''}

    <div class="action-buttons">
        <p><strong>审阅步骤：</strong></p>
        <ol style="text-align: left; max-width: 500px; margin: 20px auto;">
            <li>在本地打开草稿文件进行审阅</li>
            <li>如需修改，直接编辑草稿文件</li>
            <li>运行 <code>python .ai-agent/main.py --publish</code> 发布</li>
        </ol>
    </div>

    <div class="preview">
        <h3>📄 内容预览（前500字）</h3>
        <pre>{content[:500]}...</pre>
    </div>

    <div class="footer">
        <p>
            🤖 本邮件由AI博客生成系统自动发送<br>
            草稿位置: {draft_path}<br>
            生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
        <p>
            <strong>命令快速参考：</strong><br>
            查看草稿: <code>cat {draft_path}</code><br>
            编辑草稿: <code>vim {draft_path}</code><br>
            发布博客: <code>cd ~/projects/hengshan.github.io && python .ai-agent/main.py --publish</code>
        </p>
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    # 测试
    import yaml

    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    sender = EmailSender(config)

    # 测试数据
    test_blog = {
        'filename': 'test.md',
        'category': 'CUDA/GPU编程',
        'tech_topic': {
            'title': '测试博客',
            'url': 'https://example.com'
        },
        'word_count': 1500,
        'has_code': True
    }

    # sender.send_draft_review(test_blog, '../templates/blog_template.md')
