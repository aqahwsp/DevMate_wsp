# DevMate

DevMate 是一个基于 **Deep Agents / LangChain / FastAPI** 的智能编程助手项目。它把 **MCP 网络搜索**、**本地知识库 RAG**、**分阶段 Agent 工作流**、**Skills 复用**、**受控本地 Python 执行**、**Web UI** 和 **CLI** 集成到同一个工程里，适合做代码生成、项目搭建、已有仓库修改、需求分解、验证与交付打包。

本 README 以**当前项目代码实现**为准，重点说明：

- 项目提供了哪些功能
- 每个 CLI 命令能做什么
- 如何启动 Web 端、MCP 搜索服务、RAG 索引
- 运行过程中会生成什么状态和产物
- 常见使用方式和命令示例

---

## 1. 项目定位

DevMate 的目标不是简单地“问答”，而是把一次开发任务拆成一条可观测、可回放、可验证的执行链：

1. **Planner**：理解需求并给出计划
2. **Researcher**：查询网络资料与本地知识库
3. **Builder**：生成或修改代码 / 文档 / 目录结构
4. **Verifier**：做验证、回看与必要修复
5. **Packager**：整理结果、输出总结、打包交付物

整个流程由单个共享 Agent 驱动，并带有状态持久化、文件变更跟踪、可选深度优化回环和“立即输出”控制。

---

## 2. 功能总览

### 2.1 面向用户的功能

1. **自然语言生成项目或修改项目**
   - 支持一句话描述需求
   - 支持多文件工程生成
   - 支持在已有工作区上继续修改

2. **CLI 交互与单轮执行**
   - 单次执行：传入 `--prompt`
   - 交互执行：不传 `--prompt` 进入交互模式
   - 运行中支持输入 `/finish` 触发立即收尾打包

3. **Web UI 图形界面**
   - 浏览器输入需求发起任务
   - 查看当前状态 / 当前阶段 / Agent 回复
   - 查看变更文件、结果文件、交付 zip
   - 支持上传初始文件并在本次任务中注入
   - 支持“立即输出”按钮

4. **基于 MCP 的 Web 搜索**
   - 提供独立的 `mcp-search` 服务
   - 通过 Streamable HTTP 暴露 `search_web`
   - 由主 Agent 通过 MCP Client 调用
   - 搜索服务底层使用 Tavily

5. **本地知识库 RAG**
   - 把 `docs/` 或工作区中的文档摄入到 Chroma
   - 通过向量检索获取相关片段
   - 支持把 Researcher 阶段结论缓存为知识文档

6. **Skill 管理与沉淀**
   - 读取技能目录中的 `SKILL.md`
   - 任务成功后可自动保存技能模式
   - 支持列出已有技能
   - 支持把工作区生成的技能迁移回托管目录

7. **受控本地 Python 工具（可选）**
   - Builder / Verifier 可按需运行本地 Python
   - 允许读项目上下文、写工作区内部文件
   - 禁止向工作区外写文件
   - 禁止生成子进程
   - 适合做轻量检查、脚本执行、pytest 验证

8. **交付物打包**
   - 任务结束后可生成 zip 交付包
   - Web 端可直接下载
   - CLI 会输出 zip 路径

9. **运行状态持久化**
   - Web/CLI 任务状态写入 JSON 快照
   - 进程重启后可恢复最近状态
   - CLI/Web 均有独立状态目录

10. **工作区文件预览与下载**
    - Web 端可列出工作区结果文件
    - 可在线预览文本文件
    - 可下载结果文件和打包产物

### 2.2 工程内部能力

1. **分阶段工作流调度**
2. **工作区变更检测与文件操作记录**
3. **重复输出 / 循环输出检测**
4. **深度优化回环（Verifier → Researcher → Builder）**
5. **LangSmith 可观测性接入**
6. **Docker / docker-compose 启动**
7. **配置统一放在 `config.toml`**
8. **依赖由 `pyproject.toml` + `uv` 管理**
9. **受控读写根目录约束，避免路径逃逸**
10. **更清晰的 Web JSON 请求错误处理**
11. **特色啦啦队：自动检测当输入信息过多时（视作困难任务），程序会自动添加鼓励的话语附加到输入信息末尾，以此提升单次模型回话效果**

---

## 3. 核心工作流

默认执行路径如下：

```text
Planner -> Researcher -> Builder -> Verifier -> Packager
```

当开启深度优化时，Verifier 如果发现问题，运行时可以回退到更前阶段继续优化：

```text
Planner
  -> Researcher
  -> Builder
  -> Verifier
      -> (可选) Researcher / Builder 再优化
  -> Packager
```

补充说明：

- **立即输出**：CLI 输入 `/finish` 或 Web 点击“立即输出”后，系统会尽快结束当前阶段并进入 Packager。
- **Planner 回退模式**：如果当前 Agent 没有 Planner 工具，会使用 Planner 提示词回退方案，直接让模型以 Planner 身份输出计划。
- **验证方式**：
  - 开启本地 Python 工具时，可做更真实的本地检查
  - 未开启时，Verifier 主要基于代码与上下文做静态审查

---

## 4. 项目结构

以当前仓库为准，核心目录如下：

```text
.
├── AGENTS.md
├── Dockerfile
├── docker-compose.yml
├── config.toml
├── README.md
├── requirements.md
├── pyproject.toml
├── uv.lock
├── docs/
├── src/devmate/
│   ├── cli.py
│   ├── config.py
│   ├── local_python_tool.py
│   ├── mcp_server.py
│   ├── rag.py
│   ├── runtime.py
│   ├── schemas.py
│   ├── search_service.py
│   ├── skills.py
│   ├── state_store.py
│   ├── webapp.py
│   └── workflow.py
├── tests/
└── workspace/
```

### 关键模块说明

- `src/devmate/cli.py`
  - Typer CLI 入口
  - 暴露 `ingest`、`chat`、`web`、`mcp-search`、`list-skills`

- `src/devmate/runtime.py`
  - 项目核心运行时
  - 负责 Agent 组装、阶段执行、工作区跟踪、打包、技能沉淀

- `src/devmate/webapp.py`
  - FastAPI Web UI 与 API
  - 管理任务状态、上传文件、工作区预览和下载

- `src/devmate/mcp_server.py`
  - MCP 搜索服务
  - 暴露 `search_web`

- `src/devmate/rag.py`
  - 文档摄入、向量存储、相似检索

- `src/devmate/skills.py`
  - Skill 读取、保存、迁移

- `src/devmate/local_python_tool.py`
  - 受控本地 Python 执行器

- `src/devmate/state_store.py`
  - Web / CLI 运行状态持久化

---

## 5. 环境要求

- Python **3.13**
- `uv`
- 可用的大模型 API Key
- Tavily API Key（如果要启用真实 Web 搜索）
- LangSmith Key（如果要启用完整链路追踪）

---

## 6. 安装与初始化

### 6.1 安装依赖

```bash
uv sync --extra dev
```

### 6.2 配置文件

默认会读取当前目录下的 `config.toml`。

也可以通过以下方式指定配置：

```bash
uv run devmate chat --config-path ./config.toml --prompt "构建一个 FastAPI 服务"
```

或使用环境变量：

```bash
export DEVMATE_CONFIG_PATH=/absolute/path/to/config.toml
```

### 6.3 关键配置项

`config.toml` 里当前支持的主要分节如下：

- `[model]`
  - `provider`
  - `ai_base_url`
  - `api_key`
  - `model_name`
  - `embedding_model_name`
  - `temperature`
  - `max_tokens`
  - `timeout_seconds`
  - `embedding_provider`

- `[search]`
  - `tavily_api_key`
  - `mcp_url`
  - `max_results`
  - `allow_mock_search`

- `[langsmith]`
  - `enabled`
  - `langchain_tracing_v2`
  - `langchain_api_key`
  - `langsmith_api_key`
  - `project`
  - `endpoint`

- `[skills]`
  - `skills_dir`
  - `auto_save_on_success`

- `[rag]`
  - `docs_dir`
  - `persist_dir`
  - `research_cache_dir`
  - `collection_name`
  - `top_k`
  - `chunk_size`
  - `chunk_overlap`

- `[app]`
  - `workspace_dir`
  - `state_dir`
  - `host`
  - `port`
  - `log_level`

- `[mcp]`
  - `host`
  - `port`
  - `streamable_http_path`

- `[workflow]`
  - `artifacts_dir`
  - `deep_optimization_default`
  - `local_python_tool_default`
  - `max_deep_optimization_rounds`
  - `standard_builder_repair_rounds`

---

## 7. 快速开始

### Docker 启动

```bash
docker compose build --no-cache
docker compose up
```
项目提供了：

- `Dockerfile`
- `docker-compose.yml`

#### 一键启动

```bash
docker compose up --build
```

默认会启动两个服务：

- `search-mcp`
  - MCP 搜索服务
  - 端口 `8001`

- `app`
  - Web UI
  - 端口 `8080`

启动后访问：

```text
http://localhost:8080
```
---

### 7.1 构建本地知识库

先把 `docs/` 中的 markdown / txt 文档摄入向量库：

```bash
uv run devmate ingest
```

### 7.2 启动 MCP 搜索服务

```bash
uv run devmate mcp-search --host 0.0.0.0 --port 8001
```

默认健康检查：

```text
GET /healthz
```

### 7.3 启动 Web UI

```bash
uv run devmate web --host 0.0.0.0 --port 8080
```

打开浏览器：

```text
http://localhost:8080
```

### 7.4 从 CLI 直接执行一次任务

```bash
uv run devmate chat --prompt "我想构建一个展示附近徒步路线的网站项目。"
```

---

## 8. CLI 可操作指南

CLI 入口定义在 `pyproject.toml`：

```toml
[project.scripts]
devmate = "devmate.cli:app"
```

因此你可以用下面两种方式调用：

```bash
uv run devmate <command>
```

或：

```bash
uv run python -m devmate <command>
```

### 8.1 命令总览

| 命令 | 作用 |
|---|---|
| `devmate ingest` | 摄入本地文档到向量库 |
| `devmate chat` | 执行一次任务，或进入交互模式 |
| `devmate web` | 启动 FastAPI Web UI |
| `devmate mcp-search` | 启动 MCP 搜索服务 |
| `devmate list-skills` | 列出当前可用技能 |

---

### 8.2 `devmate ingest`

作用：

- 读取知识库目录中的 `.md` / `.txt` 文件
- 切分文本块
- 写入 Chroma 向量库

命令：

```bash
uv run devmate ingest
```

可选参数：

```bash
uv run devmate ingest --config-path ./config.toml
```

适用场景：

- 修改了 `docs/` 中的知识文档后重新建索引
- 首次部署项目时初始化 RAG 数据

---

### 8.3 `devmate chat`

作用：

- 单轮执行任务
- 或进入 CLI 交互模式连续执行多轮任务

#### 单轮执行

```bash
uv run devmate chat --prompt "构建一个 FastAPI 服务"
```

#### 交互模式

```bash
uv run devmate chat
```

进入后输入需求，输入 `exit` 或 `quit` 退出。

#### 可选参数总览

```bash
uv run devmate chat \
  --prompt "构建一个博客后台" \
  --config-path ./config.toml \
  --deep-optimization \
  --local-python-tool \
  --max-deep-optimization-rounds 2 \
  --upload-initial-file ./seed_spec.md \
  --upload-initial-file ./api_contract.json \
  --include-initial-files
```

#### `chat` 支持的参数

- `--prompt <text>`
  - 单次任务提示词
  - 不传时进入交互模式

- `--config-path <path>`
  - 指定配置文件路径

- `--deep-optimization / --no-deep-optimization`
  - 开启或关闭 Verifier 驱动的深度优化回环

- `--local-python-tool / --no-local-python-tool`
  - 开启或关闭本地 Python 工具
  - 默认关闭

- `--max-deep-optimization-rounds <int>`
  - 深度优化的最大轮数
  - 最小值为 `0`

- `--upload-initial-file <path>`
  - 把本地文件复制到 `workspace/data/uploads/cli`
  - 可重复传入多个文件

- `--include-initial-files / --no-include-initial-files`
  - 是否把已暂存的初始文件注入到当前运行上下文

#### 运行过程中的特殊控制

任务正在跑的时候，CLI 支持输入：

```text
/finish
```

也支持：

```text
finish
/f
```

作用：

- 请求系统尽快结束当前阶段
- 直接进入最终收尾与打包
- 适合已经生成出可用结果、但你不想继续优化时使用

#### 单轮任务的典型示例

1. **普通生成**

```bash
uv run devmate chat --prompt "构建一个包含 health check 的 FastAPI 服务"
```

2. **开启深度优化**

```bash
uv run devmate chat \
  --prompt "生成一个博客后台" \
  --deep-optimization \
  --max-deep-optimization-rounds 2
```

3. **开启本地 Python 检查**

```bash
uv run devmate chat \
  --prompt "修复当前项目中的单元测试" \
  --local-python-tool
```

4. **注入初始文件作为上下文**

```bash
uv run devmate chat \
  --upload-initial-file ./specs/product_requirement.md \
  --upload-initial-file ./contracts/openapi.json \
  --include-initial-files \
  --prompt "基于这些初始文件继续完善项目"
```

#### `chat` 的输出内容

单轮执行完成后，CLI 会输出：

- 最终回复文本
- 变更文件列表
- 交付 zip 路径（如果生成）
- 验证是否通过
- 本地 Python 工具是否开启
- 深度优化轮数使用情况
- 是否触发立即输出
- 是否保存了新技能

---

### 8.4 `devmate web`

作用：

- 启动 FastAPI Web UI
- 提供上传、发起任务、轮询状态、预览文件、下载产物等能力

命令：

```bash
uv run devmate web --host 0.0.0.0 --port 8080
```

可选参数：

- `--host <host>`：绑定地址，默认 `0.0.0.0`
- `--port <port>`：绑定端口，默认 `8080`
- `--config-path <path>`：配置文件路径

---

### 8.5 `devmate mcp-search`

作用：

- 启动独立 MCP 搜索服务
- 暴露 `search_web` 工具给主 Agent 调用

命令：

```bash
uv run devmate mcp-search --host 0.0.0.0 --port 8001
```

可选参数：

- `--host <host>`：绑定地址，默认 `0.0.0.0`
- `--port <port>`：绑定端口，默认 `8001`
- `--config-path <path>`：配置文件路径

---

### 8.6 `devmate list-skills`

作用：

- 列出当前技能目录下所有可识别技能

命令：

```bash
uv run devmate list-skills
```

可选参数：

```bash
uv run devmate list-skills --config-path ./config.toml
```

---

## 9. Web 端使用指南

### 9.1 页面能力

当前 Web 页面包含以下功能：

- 需求输入框
- 深度优化开关
- 本地 Python 工具开关
- 最大深度优化次数
- 初始文件上传
- 是否带入初始文件的复选框
- 开始生成按钮
- 立即输出按钮
- 当前状态 / 当前阶段展示
- Agent 回复展示
- 交付信息展示
- 变更文件列表
- 结果文件列表
- 文件预览
- 结果 zip 下载

### 9.2 Web API 一览

| 方法 | 路径 | 作用 |
|---|---|---|
| `GET` | `/` | Web UI 页面 |
| `GET` | `/healthz` | 健康检查 |
| `GET` | `/api/uploads` | 列出已上传初始文件 |
| `POST` | `/api/uploads` | 上传初始文件 |
| `POST` | `/api/generate` | 启动后台任务 |
| `POST` | `/api/generate-sync` | 同步执行任务 |
| `GET` | `/api/jobs/{job_id}` | 查询后台任务状态 |
| `POST` | `/api/jobs/{job_id}/immediate-output` | 请求立即输出 |
| `GET` | `/api/workspace` | 列出工作区文件 |
| `GET` | `/api/file/{relative_path}` | 读取文本文件内容 |
| `GET` | `/api/download/{relative_path}` | 下载工作区文件 |
| `GET` | `/api/artifact/{relative_path}` | 下载工作流产物 |

### 9.3 JSON 请求说明

发起任务的请求体模型为：

```json
{
  "prompt": "构建一个项目",
  "deep_optimization": false,
  "local_python_tool_enabled": false,
  "max_deep_optimization_rounds": 0,
  "include_initial_files": false,
  "initial_file_paths": []
}
```

如果请求体不是合法 JSON，后端会返回更明确的错误：

- `400`：请求体不是合法 JSON
- `422`：字段校验失败

---

## 10. RAG / MCP / Skills 说明

### 10.1 MCP 搜索

- 服务由 `src/devmate/mcp_server.py` 提供
- 工具名：`search_web`
- 传输方式：**Streamable HTTP**
- 底层搜索服务：Tavily

工具参数：

- `query`
- `max_results`
- `topic`（`general` / `news` / `finance`）

### 10.2 RAG 知识库

知识库支持：

- 读取 `docs/` 下的 `.md` / `.txt`
- 文档切块
- Chroma 本地持久化
- 相似度检索
- Researcher 阶段结果缓存回知识库

相关目录默认位于：

```text
workspace/docs
workspace/data/chroma
workspace/docs/research_cache
```

### 10.3 Skills

技能目录由 `config.toml` 中的 `[skills].skills_dir` 控制。

当前项目支持：

- 列出技能
- 保存技能模式
- 自动从成功运行中沉淀技能
- 把工作区发现的技能移动到托管技能目录
- 把新的 Markdown 技能文件同步回源目录

---

## 11. 受控本地 Python 工具说明

当开启 `--local-python-tool` 或 Web 中对应开关时，Builder / Verifier 可调用本地 Python。

当前安全约束包括：

- 允许读取项目上下文文件
- 允许写入**工作区内部**文件
- **禁止**写入工作区外路径
- **禁止**通过 `subprocess` 等方式生成子进程
- 超时和输出长度都有上限控制

适用场景：

- 执行小段 Python 代码
- 跑模块 / 脚本
- 做本地验证
- 在 Builder / Verifier 中辅助检查结果

---

## 12. 工作区、状态与产物目录

### 12.1 工作区

默认工作区：

```text
workspace/
```

### 12.2 运行状态

默认状态目录：

```text
workspace/data/runtime_state/
```

其中会区分：

```text
workspace/data/runtime_state/web_jobs/
workspace/data/runtime_state/cli_runs/
```

每次任务会写入：

- `*.json`：任务快照
- `*.control.json`：控制信号（例如立即输出）
- `*.log`：某些场景下的日志文件

### 12.3 工作流产物

默认工作流产物目录：

```text
workspace/data/workflow_runs/
```

常见内容包括：

- Planner 产物
- Researcher 产物
- Builder 产物
- Verifier 产物
- 最终打包 zip

### 12.4 初始文件上传目录

CLI：

```text
workspace/data/uploads/cli/
```

Web：

```text
workspace/data/uploads/web/
```

---

## 13. 常用命令速查

### 安装与检查

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
```

### RAG / MCP / Web

```bash
uv run devmate ingest
uv run devmate mcp-search --host 0.0.0.0 --port 8001
uv run devmate web --host 0.0.0.0 --port 8080
```

### CLI 执行

```bash
uv run devmate chat --prompt "构建一个 FastAPI 服务"
uv run devmate chat --prompt "修复当前项目测试" --local-python-tool
uv run devmate chat --prompt "优化项目结构" --deep-optimization --max-deep-optimization-rounds 2
```

### 技能

```bash
uv run devmate list-skills
```

---

## 14. 开发与测试

项目当前已包含测试，覆盖了若干关键行为，例如：

- CLI 状态快照持久化
- Web 任务状态恢复
- Web 回复持久化节流
- 非法 JSON 请求的清晰错误提示
- 流式输出期间工作区扫描节流
- 本地 Python 工具的安全限制
- 运行时只读 / 只写边界约束
- 技能迁移逻辑

建议本地执行：

```bash
uv run ruff check .
uv run pytest
```

---

## 15. 常见使用建议

1. **先 ingest，再跑任务**
   - 如果你依赖本地规范或模板，先执行 `devmate ingest`

2. **大任务建议开深度优化**
   - 对“生成整站”“重构项目”这类复杂任务更适合

3. **需要真实验证时开本地 Python 工具**
   - 例如修测试、做轻量脚本检查、验证代码可运行性

4. **需要给模型喂上下文时先上传初始文件**
   - 比如产品需求文档、OpenAPI 契约、设计说明

5. **结果差不多时可以用立即输出**
   - 减少多余优化轮次，直接收尾打包

---

## 16. 适合的典型任务

- 生成一个新的 Python / FastAPI / 前端项目
- 在现有工作区上做增量修改
- 根据需求文档生成多文件骨架
- 结合本地规范和网络资料输出方案
- 修复测试或补齐 README / 文档
- 让 Agent 把成功任务沉淀成可复用 Skill

---

## 17. 一句话总结

如果你希望把“搜索资料、读取本地文档、生成代码、验证、打包、沉淀技能”串成一条完整的开发执行链，DevMate 就是这个仓库当前实现的核心能力集合。
