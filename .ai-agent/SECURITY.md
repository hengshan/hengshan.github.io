# 安全配置说明

## ⚠️ 重要：保护你的API密钥和密码

### 正确的配置方式

1. **创建实际配置文件**（包含真实密钥）
   ```bash
   cp .ai-agent/.env.example .ai-agent/.env
   vim .ai-agent/.env  # 填入真实的API密钥和密码
   ```

2. **设置文件权限**
   ```bash
   chmod 600 .ai-agent/.env  # 只有你可以读写
   ```

3. **验证配置**
   ```bash
   # .env.example 应该只包含示例值
   cat .ai-agent/.env.example  # ❌ 不应该有真实密码

   # .env 包含真实配置（且被gitignore）
   cat .ai-agent/.env  # ✅ 真实的API密钥和密码
   ```

## 🔒 安全检查清单

- [ ] `.env.example` 只包含示例值（如 `your-api-key-here`）
- [ ] 真实配置在 `.env` 文件中
- [ ] `.ai-agent/` 在项目根目录的 `.gitignore` 中
- [ ] `.env` 在 `.ai-agent/.gitignore` 中
- [ ] `.env` 文件权限设置为 600

## 验证 Git 忽略配置

```bash
# 检查这些路径是否被忽略
git check-ignore .ai-agent/
git check-ignore .ai-agent/.env
git check-ignore drafts/

# 都应该返回路径，表示被忽略
```

## 文件分层保护

```
项目根目录
├── .gitignore          ← 忽略整个 .ai-agent/ 和 drafts/
└── .ai-agent/
    ├── .gitignore      ← 额外忽略 .env（双重保护）
    ├── .env.example    ← 模板（不含真实密钥）
    └── .env            ← 真实配置（被忽略，不会提交）
```

## 如果不小心提交了敏感信息

**立即采取措施：**

1. **更换所有API密钥和密码**
   - Claude API: https://console.anthropic.com/
   - Gmail应用密码: 重新生成新的

2. **从Git历史中移除**
   ```bash
   # 警告：这会重写Git历史
   git filter-branch --force --index-filter \
     "git rm -rf --cached --ignore-unmatch .ai-agent/.env" \
     --prune-empty --tag-name-filter cat -- --all

   git push origin --force --all
   ```

3. **检查GitHub**
   - 确认文件已从所有分支删除
   - 如果是public repo，假设密钥已泄露

## 最佳实践

✅ **DO（应该做）**
- 使用 `.env` 文件存储真实配置
- 定期备份 `.env` 到安全位置（加密）
- 使用不同的API密钥在不同环境（开发/生产）
- 定期轮换API密钥

❌ **DON'T（不要做）**
- 不要在 `.env.example` 中填入真实密码
- 不要将 `.env` 文件提交到Git
- 不要在代码中硬编码API密钥
- 不要共享 `.env` 文件（即使是私有仓库）

## 环境变量管理工具

考虑使用专业工具：
- **1Password** / **Bitwarden**: 密码管理器
- **Pass**: 命令行密码管理
- **SOPS**: 加密配置文件

## 应急联系

如果怀疑密钥泄露：
1. 立即登录 https://console.anthropic.com/ 删除旧密钥
2. 生成新的API密钥
3. 更新本地 `.env` 文件
4. 检查账单是否有异常使用

---

**记住**: 安全配置是使用AI Agent的第一步，也是最重要的一步！
