# Splitter

一个面向本地知识库构建的 RAG 项目，当前已经包含：

- 多格式文档导入：`pdf`、`docx`、`md`、`txt`
- 父子块切分
  - 父块：按格式走不同策略
  - 子块：递归切分与语义切分的启发式路由
- 存储分层
  - 父块：MongoDB
  - 子块：Qdrant `child_chunks_hybrid`
- 在线问答链路
  - query rewrite
  - hybrid retrieval
  - rerank
  - parent recall
- Ragas 合成评测链路
- FastAPI 服务化入口

## 1. 环境要求

- Python `3.11+`
- 本地可访问的 MongoDB
- 本地可访问的 Qdrant

当前默认约定：

- MongoDB：`localhost:27017`
- Qdrant：`localhost:6333`

## 2. 安装依赖

项目使用 `uv` 管理依赖，推荐这样初始化：

```powershell
uv sync
```

如果你已经有虚拟环境，也可以直接用项目里的：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果项目里没有 `requirements.txt`，仍然推荐直接使用：

```powershell
uv sync
```

## 3. 配置 `.env`

在项目根目录创建 `.env`，至少补齐 DeepSeek 配置：

```env
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

可选服务级配置：

```env
APP_NAME=Splitter API
APP_VERSION=1.0.0
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
TASK_MAX_WORKERS=4

DENSE_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
SPARSE_MODEL_NAME=Qdrant/bm25
RERANKER_MODEL_NAME=BAAI/bge-reranker-base
```

说明：

- `DEEPSEEK_*` 是在线问答和部分评测必须配置
- `DENSE_MODEL_NAME` / `SPARSE_MODEL_NAME` / `RERANKER_MODEL_NAME` 都有默认值，不配也能跑
- `TASK_MAX_WORKERS` 控制导入任务线程池大小

## 4. 本地依赖准备

### MongoDB

项目默认会连本地 MongoDB，并把父块及导入状态写进去。

你之前使用的本地配置是：

- 用户名：`admin`
- 密码：`123456`

只要你的本地 Mongo 容器已经按这个方式启动，项目就能直接连。

### Qdrant

项目默认使用：

- collection：`child_chunks_hybrid`

如果你之前是旧版本，只建过 `child_chunks`，需要重新做离线导入，才能让在线检索正常工作。

## 5. 离线导入

### 单文件导入

```powershell
.\.venv\Scripts\python.exe .\main.py --file .\data\demo.pdf
```

### 批量导入

```powershell
.\.venv\Scripts\python.exe .\main.py --data-dir .\data
```

批量模式会扫描目录下支持的文件类型：

- `.pdf`
- `.docx`
- `.md`
- `.txt`

### 导入后的效果

- 父块写入 MongoDB
- 子块写入 Qdrant `child_chunks_hybrid`
- 内容哈希用于去重

## 6. 交互式问答

如果你想直接在终端里持续提问：

```powershell
.\.venv\Scripts\python.exe .\rag_chat.py
```

进入循环后：

- 输入问题直接问
- 输入 `exit` 或 `quit` 退出

回答时会打印：

- `Answer`
- 命中的 `Sources`

## 7. FastAPI 服务启动

推荐启动方式：

```powershell
.\.venv\Scripts\python.exe .\serve.py
```

这会使用统一 settings 读取：

- `API_HOST`
- `API_PORT`
- `LOG_LEVEL`

默认等价于：

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

启动后默认地址：

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

## 8. API 一览

### 健康检查

```http
GET /v1/health
GET /v1/ready
```

- `/v1/health`：服务进程是否存活
- `/v1/ready`：Mongo 和 Qdrant 是否可用

### 在线问答

```http
POST /v1/chat/query
Content-Type: application/json
```

请求体：

```json
{
  "question": "光线对人像摄影有什么影响？"
}
```

响应体：

```json
{
  "answer": "……",
  "source_items": [
    {
      "parent_id": "parent-1",
      "source": "demo.pdf",
      "file_path": "C:/docs/demo.pdf"
    }
  ]
}
```

### 同步导入接口

单文件：

```http
POST /v1/ingest/file
Content-Type: multipart/form-data
```

批量：

```http
POST /v1/ingest/batch
Content-Type: application/json
```

批量请求体：

```json
{
  "data_dir": "C:/WORK/Source/Python/Splitter/data"
}
```

### 长任务导入接口

提交单文件任务：

```http
POST /v1/tasks/ingest/file
Content-Type: multipart/form-data
```

提交批量任务：

```http
POST /v1/tasks/ingest/batch
Content-Type: application/json
```

查询任务：

```http
GET /v1/tasks/{task_id}
```

任务状态目前支持：

- `pending`
- `running`
- `succeeded`
- `failed`

## 9. curl 示例

### 问答

```powershell
curl -X POST http://127.0.0.1:8000/v1/chat/query `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"光线对人像摄影有什么影响？\"}"
```

### 批量导入任务

```powershell
curl -X POST http://127.0.0.1:8000/v1/tasks/ingest/batch `
  -H "Content-Type: application/json" `
  -d "{\"data_dir\":\"C:/WORK/Source/Python/Splitter/data\"}"
```

### 查询任务

```powershell
curl http://127.0.0.1:8000/v1/tasks/<task_id>
```

## 10. 统一错误响应

当前 API 错误响应统一格式为：

```json
{
  "code": "DEPENDENCY_UNAVAILABLE",
  "message": "MongoDB is unavailable",
  "details": {}
}
```

并且响应头会带：

- `X-Request-ID`

常见错误码包括：

- `BAD_REQUEST`
- `VALIDATION_ERROR`
- `FILE_NOT_FOUND`
- `DIRECTORY_NOT_FOUND`
- `UNSUPPORTED_FILE_TYPE`
- `TASK_NOT_FOUND`
- `NO_CONTEXT_RETRIEVED`
- `COLLECTION_NOT_READY`
- `DEPENDENCY_UNAVAILABLE`
- `INGEST_CONFLICT`
- `MODEL_INITIALIZATION_ERROR`
- `INTERNAL_ERROR`

## 11. Ragas 评测

项目已经有合成测试集和评测 CLI。

例如只跑 `baseline`：

```powershell
.\.venv\Scripts\python.exe -m evals.cli --dataset .\artifacts\evals\synthetic.jsonl --output-dir .\artifacts\evals --test-size 10 --experiment baseline
```

查看实验结果：

- `.\artifacts\evals\baseline\metrics.json`
- `.\artifacts\evals\baseline\summary.md`
- `.\artifacts\evals\baseline\trace.jsonl`

## 12. 常见问题

### 1. `child_chunks_hybrid` 不存在

说明你当前本地只导入过旧版本数据，或者还没跑过新的离线导入。

先执行：

```powershell
.\.venv\Scripts\python.exe .\main.py --data-dir .\data
```

如果 `ingested_files` 已经把老数据标成完成，需要先清理旧导入记录再重跑。

### 2. `Question>` 后卡住

先检查：

- Qdrant 是否可访问
- Mongo 是否可访问
- `.env` 是否配置正确
- reranker 模型是否首次下载过慢

也可以先用：

```powershell
curl http://127.0.0.1:8000/v1/ready
```

确认依赖是否健康。

### 3. DeepSeek 配置报错

如果启动时报缺少以下任意配置：

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`

请先检查项目根目录 `.env`。

### 4. Mongo/Qdrant 连接失败

先确认本地容器是否正常运行。

Mongo 和 Qdrant 只要任意一个不可用，`/v1/ready` 就不会返回 `ready`。

### 5. 评测只生成了 `synthetic.jsonl`

这通常说明：

- 评测样本生成成功了
- 但后续某个指标计算阶段失败了

优先检查：

- DeepSeek 配置
- API 配额
- 依赖服务是否可用

## 13. 推荐使用顺序

如果你是第一次跑这个项目，建议顺序是：

1. 启动 MongoDB 和 Qdrant
2. 配好 `.env`
3. 先跑离线导入
4. 再跑 `rag_chat.py` 或 FastAPI
5. 最后再跑评测

## 14. 当前阶段说明

目前这套服务已经完成：

- FastAPI 基础服务化
- 长任务导入
- 结构化日志与 request/task id
- CLI 与 API 统一到共享 runtime/service
- 领域异常体系
- 服务级 settings 与标准启动入口

后续还可以继续补：

- 更强的 readiness 检查
- 鉴权
- 评测接口服务化
- Docker / 部署文档
