# 用 Git 给 RoomMind 做贡献

这份说明给第一次往这个仓库交代码的同事。所有命令都在仓库根目录执行。尖括号里的内容换成你自己的，例如 `<分支名>`。

远程仓库是 `git@github.com:2p1c/RoomMind.git`，主分支是 `main`。功能做在自己的分支上，用 Pull Request 合进 `main`。不要在 `main` 上直接改，也不要强制推送 `main`。

## 0. 准备

电脑上要有 Git，以及 [GitHub CLI](https://cli.github.com/)。第一次用 `gh` 时登录一次：

```bash
gh auth login
```

选 GitHub.com、SSH。后面创建 PR 会用到它。

## 1. 克隆

```bash
git clone git@github.com:2p1c/RoomMind.git
cd RoomMind
git status
```

`git status` 应显示 `On branch main`，且没有要提交的改动。

没有 SSH 密钥、克隆失败时，改用：

```bash
git clone https://github.com/2p1c/RoomMind.git
```

## 2. 从最新的 main 开分支

分支名用小写和连字符，并带类型前缀：

| 前缀 | 什么时候用 |
| --- | --- |
| `feat/` | 新功能 |
| `fix/` | 修 bug |
| `docs/` | 只改文档 |
| `refactor/` | 行为不变，只调整结构 |
| `chore/` | 杂务，例如忽略规则、示例图 |

一条分支只做一件事。例如给感知页加一个按钮，用 `feat/dino-score-filter`，不要用 `update` 或 `test`。

先更新 `main`，再开分支：

```bash
git switch main
git pull --ff-only origin main
git switch -c feat/你的功能名
```

也可以直接跑脚本。它会拒绝用 `main` 当分支名：

```bash
chmod +x contributing/new-branch.sh
./contributing/new-branch.sh feat/你的功能名
```

之后用 `git status` 确认当前分支是你刚建的名字，不是 `main`。

## 3. 哪些文件不要提交

`.gitignore` 已经挡住了其中一部分。提交前仍要看 `git status`，下面这些即使出现也不要 `git add`：

| 路径 | 原因 |
| --- | --- |
| `agent/.env` | API 密钥 |
| `.env`、任何带密钥的文件 | 同上 |
| `.venv/` | 本机 Python 环境 |
| `agent/node_modules/`、`web/node_modules/`、`agent/dist/` | 安装出来的依赖和编译结果 |
| `__pycache__/`、`*.pyc` | Python 字节码缓存 |
| `models/`、`*.pt` | 模型权重，体积大且每台机器自己下载 |
| `roomind.egg-info/` | 安装 `roomind` 命令时生成的元数据 |
| `.DS_Store` | macOS 目录缓存 |

`test/tmp/` 里是推理生成的图。当前忽略规则被注释掉了，所以 `git status` 会看见它们。日常开发不要加入这些图。只有这次任务就是更新示例结果图时才提交。

不确定某个文件该不该进仓库时，不要先 `git add .`。把下面这段发给 Agent：

```text
我准备提交 RoomMind。请只读 git status 和 git diff，列出不应该提交的文件，尤其是密钥、.env、虚拟环境、node_modules、__pycache__、models、权重和 test/tmp 里的临时结果图。不要执行 git add 或 git commit。
```

## 4. 提交

改完并自己测过之后再提交。先看改了什么：

```bash
git status
git diff
```

只加入这次任务相关的文件。不要用 `git add .`，那样容易把密钥和临时图一起加进去。

```bash
git add app/dino/detector.py
git add web/public/app.js
```

路径换成你真正改过的文件。确认暂存区：

```bash
git diff --cached --stat
```

提交说明用仓库里已有的格式：`类型: 说明`。类型与分支前缀相同，说明写这次改动的目的，用中文，一句话说完。

```bash
git commit -m "$(cat <<'EOF'
feat: 让 DINO 结果按置信度过滤

EOF
)"
```

可以参考的写法：

```text
feat: 增加感知页和 Agent 对话页
fix: 推理改为 float32
docs: 说明 Agent 的运行方式和添加工具的步骤
chore: 停止跟踪 VLM 的 Python 字节码缓存
```

不要写 `update`、`fix bug`、`暂存`。不要提交空说明。

提交之后确认还在自己的分支上，并且没有误加文件：

```bash
git status
git log -1
```

想让 Agent 代写说明并提交时，把范围写死：

```text
请提交我在分支 feat/你的功能名 上的改动。先看 git status 和 git diff。不要加入 .env、.venv、node_modules、__pycache__、models、*.pt 和 test/tmp。用「类型: 中文说明」写提交说明，说明要写目的。不要 push，也不要改 git config。
```

## 5. 推送前测试

推送前至少跑和你改动对应的检查。下面的命令都从仓库根目录开始。某一步失败就先修，不要推送。

只改了 Agent：

```bash
cd agent
npm test
cd ..
```

只改了感知页：

```bash
cd web
npx tsc --noEmit
cd ..
```

改了 Python 感知代码：

```bash
source .venv/bin/activate
python3 -m py_compile app/main.py
```

你改过的每个 Python 文件都加到这条命令里。然后按根目录 README 的「在电脑上测试」启动相关服务，把这次功能点一次。例如改了 DINO 页面，就打开 http://127.0.0.1:8080 ，上传一张图跑一次 DINO。

也可以先跑仓库里的检查脚本。它会拒绝暂存区里的密钥和依赖目录，并在依赖已经安装时跑 Agent 测试和页面类型检查：

```bash
chmod +x contributing/pre-push-check.sh
./contributing/pre-push-check.sh
```

脚本代替不了你在页面上点一遍。模型推理仍然要按 README 手工看一次结果。

## 6. 推送分支

```bash
git push -u origin HEAD
```

第一次推这个分支需要 `-u`，之后在同一分支上只要：

```bash
git push
```

推送前再看一眼：

```bash
git status
git log --oneline origin/main..HEAD
```

第二行应只列出你这次的提交。如果里面有别人的提交，或者分支名是 `main`，停下来问同事，不要继续。

## 7. 提交 Pull Request

```bash
gh pr create --base main --title "feat: 让 DINO 结果按置信度过滤" --body "$(cat <<'EOF'
## 改了什么
- 一句话写行为变化

## 怎么测的
- [ ] cd agent && npm test
- [ ] 在页面上跑了一次相关功能

EOF
)"
```

标题和提交说明用同一句话。命令成功后终端会打印 PR 链接，把链接发给评审的人。

想让 Agent 创建 PR 时：

```text
请为当前分支创建一个指向 main 的 Pull Request。先看 git status、git log origin/main..HEAD。不要把 .env、模型权重和 test/tmp 临时图推上去。标题用「类型: 中文说明」。正文写改了什么，以及我已经做过的测试。使用 gh pr create，完成后把 PR URL 发给我。
```

PR 打开之后，评审意见在 GitHub 上。继续改时留在同一分支：改代码、再提交、再 `git push`。不要新开一个 PR，除非评审明确要求拆开。

## 做完一轮之后

下次再做新功能，从最新的 `main` 重新开分支，不要在旧功能分支上接着做无关的事：

```bash
git switch main
git pull --ff-only origin main
git switch -c feat/下一个功能
```
