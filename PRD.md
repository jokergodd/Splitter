# Splitter — 智能文档问答系统产品需求文档（PRD）

> **版本**：v1.0  
> **日期**：2026-04-22  
> **状态**：评审稿  
> **撰写人**：产品团队  
> **适用范围**：后端 / 前端 / 算法 / 测试 / 运维

---

## 1. 文档信息

| 项 | 内容 |
|---|---|
| 产品名称 | Splitter（中文代号：「智析」） |
| 产品定位 | 面向企业与个人用户的本地知识库智能问答系统 |
| 目标市场 | 需要私有化文档管理的中小型企业、研发团队、法务/财务部门 |
| 核心语言 | 中文为主，兼容英文混合场景 |
| 交付形态 | 本地部署（Python 服务 + React 前端），后续规划 SaaS 化 |

---

## 2. 背景与目标

### 2.1 背景

组织内部沉淀了大量非结构化文档（制度文件、技术规范、合同、研究报告），但存在以下痛点：

- **找不到**：传统关键词搜索无法理解语义，用户必须用精确词汇才能命中。
- **读不完**：文档动辄数十页，人工阅读筛选信息成本极高。
- **记不住**：跨文档关联信息分散在不同文件中，人脑难以整合。
- **不敢信**：通用大模型对组织内部知识「一无所知」，回答 hallucination 严重。

### 2.2 产品目标

| 目标层级 | 描述 | 衡量标准 |
|---|---|---|
| **核心目标** | 让用户用自然语言提问，即可获得基于私有文档库的确切回答 | 端到端回答可用率 ≥ 85% |
| **体验目标** | 上传文档 → 可提问，全流程 ≤ 5 分钟；单次问答延迟 ≤ 8 秒 | 首响时间（TTFB）≤ 3s |
| **信任目标** | 每个回答必须标注来源文档及具体段落，支持溯源验证 | 100% 回答附带引用 |
| **扩展目标** | 支持多格式、大批量、自动化接入，降低知识库维护成本 | 批量上传吞吐量 ≥ 10 页/秒 |

### 2.3 成功指标（OKR）

- **O1**：上线首月内，单知识库平均文档数达到 50+ 份，用户周活跃问答次数 ≥ 20 次。
- **O2**：RAGAS 评测中，Faithfulness ≥ 0.80，Answer Relevancy ≥ 0.85，Context Recall ≥ 0.75。
- **O3**：生产环境零 P0 故障，API 可用性 ≥ 99.5%。

---

## 3. 用户画像与场景

### 3.1 用户画像

| 角色 | 特征 | 核心诉求 |
|---|---|---|
| **知识库管理员** | IT/行政/文档管理专员，负责维护文档库 | 批量上传、去重、状态监控、失败重试 |
| **业务查询者** | 普通员工，对制度、流程、规范有疑问 | 自然语言提问、快速得到准确答案、来源可追溯 |
| **分析师/研究员** | 需要跨文档整合信息，撰写报告 | 多文档关联问答、高质量上下文整合、导出功能 |
| **系统集成者** | 需要将问答能力嵌入其他业务系统 | 稳定 REST API、清晰错误码、异步任务回调 |

### 3.2 关键使用场景

#### 场景 A：制度速查（高频）
> 小张入职第二天，想确认「出差住宿标准是多少」。他打开系统，直接提问，系统在 3 秒内给出答案，并引用《差旅管理办法.pdf》第 3 页相关条款。

#### 场景 B：批量归档（中频）
> 法务部李经理有 200 份历史合同需要入库。她通过批量上传功能一次性提交，系统在后台异步处理，15 分钟后全部可检索，系统自动去重并跳过已处理的文件。

#### 场景 C：跨文档综合分析（低频，高价值）
> 研究员需要对比三个项目的技术方案优劣，通过连续追问获得跨文档的信息整合，最终导出问答记录作为报告素材。

---

## 4. 功能需求

### 4.1 功能架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                      用户交互层                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Web 前端 │  │ CLI 工具 │  │ REST API │  │ 评估平台 │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                      服务编排层                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐    │
│  │ ChatService  │ │IngestService │ │ TaskService      │    │
│  │  问答编排    │ │  摄入编排    │ │ 异步任务管理     │    │
│  └──────────────┘ └──────────────┘ └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                      领域能力层                               │
│  查询改写 → 混合检索 → 重排序 → 父块召回 → 答案生成             │
│  文档加载 → 清洗 → 父子分块 → 嵌入 → 存储/去重                 │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                      基础设施层                               │
│        MongoDB (元数据/父块)    Qdrant (向量/子块)           │
│        DeepSeek LLM            HuggingFace 嵌入模型          │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 文档摄入模块（Ingestion）

#### 4.2.1 单文件上传
- **需求编号**：ING-001
- **描述**：用户通过 API 或前端上传单个文件，系统同步完成处理并返回状态。
- **输入**：文件（PDF / DOCX / MD / TXT），支持最大 50MB
- **处理流程**：加载 → 清洗 → 父块切分 → 子块切分（启发式路由）→ 嵌入 → MongoDB + Qdrant 存储
- **输出**：文件处理结果（成功/失败）、文档 ID、子块数量
- **去重逻辑**：基于 SHA256 内容哈希，已存在且状态为 completed 的文件直接跳过，返回已存在提示

#### 4.2.2 批量上传
- **需求编号**：ING-002
- **描述**：用户上传文件夹或选择多个文件，系统提交后台异步任务处理。
- **输入**：文件列表（支持拖拽目录自动遍历）
- **处理流程**：提交任务 → 任务注册 → 后台线程池并行处理 → 任务状态可查询
- **输出**：任务 ID，用户通过轮询或后续 WebSocket 获取进度
- **异常处理**：单文件失败不影响整体任务，失败原因记录到任务详情

#### 4.2.3 格式支持与解析策略

| 格式 | 解析器 | 元数据保留 | 特殊处理 |
|---|---|---|---|
| PDF | PyMuPDF | 页码、文件名 | 超长无结构 PDF 启用语义切分 |
| DOCX | Docx2txt | 文件名 | 同 PDF 启发式路由 |
| Markdown | UnstructuredMarkdown + Recursive | 标题层级、文件名 | 优先按标题切分父块 |
| TXT | TextLoader | 文件名 | 递归字符切分 |

#### 4.2.4 分块策略（Chunking）
- **需求编号**：ING-003
- **父块切分**：Markdown 优先按标题切分（MarkdownHeaderTextSplitter），超长段落回退到递归切分；其他格式直接递归切分。
- **子块切分**：启发式自动路由——对于「长 + 低行密度 + 无结构标记」的父块，使用语义切分（SemanticChunker）；其余使用递归切分。支持配置强制模式：`auto`（默认）、`recursive`、`semantic`。
- **并行优化**：子块切分使用 ThreadPoolExecutor 并行加速。

#### 4.2.5 嵌入与存储
- **需求编号**：ING-004
- **稠密向量**：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，384 维，支持中文语义编码。
- **稀疏向量**：`fastembed.SparseTextEmbedding`（`Qdrant/bm25`），用于关键词匹配增强。
- **缓存机制**：`CachedEmbeddings` 内存缓存，避免同 session 内重复编码。
- **Qdrant 存储**：单 collection 存储子块，每个点包含 dense + sparse 双向量， cosine 距离，ID 为 content_hash:child_id 的 UUID5 确定性生成。
- **MongoDB 存储**：文件摄入状态（processing/completed/failed）+ 父块全文 + 元数据。

### 4.3 智能问答模块（Retrieval & Answering）

#### 4.3.1 查询改写（Query Rewrite）
- **需求编号**：QA-001
- **描述**：用户原始 query 可能口语化、不完整或歧义，LLM 将其改写为最多 4 个等价搜索 query，提升召回率。
- **约束**：改写必须保持原意，不得引入新信息；单 query 长度不超过 100 字。

#### 4.3.2 混合检索（Hybrid Retrieval）
- **需求编号**：QA-002
- **描述**：对每个改写 query，分别执行稠密 + 稀疏向量检索，Qdrant 内部使用 RRF 融合返回候选子块。
- **多 query 合并**：多个改写 query 的候选结果按 child ID 去重，保留最高 RRF 分数。
- **Top-K 配置**：每路检索默认 prefetch 20，合并后取 Top 20 进入重排序。

#### 4.3.3 重排序（Reranking）
- **需求编号**：QA-003
- **模型**：`BAAI/bge-reranker-base`
- **逻辑**：将合并后的候选子块与用户原始 query 组成 pair，计算相关性分数，按 rerank_score 重新排序。
- **Top-K**：重排序后取 Top 10 进入父块召回。

#### 4.3.4 父块召回（Parent Recall）
- **需求编号**：QA-004
- **描述**：将子块命中结果按父块聚合，每个父块只保留得分最高的子块作为代表，按该得分排序。
- **全文获取**：从 MongoDB 取出父块的完整原文，作为 LLM 的最终上下文。
- **父块上限**：默认最多取 5 个父块，防止上下文溢出。

#### 4.3.5 答案生成
- **需求编号**：QA-005
- **模型**：DeepSeek（默认 DeepSeek-V3，可配置）
- **Prompt 策略**：System prompt 要求模型严格基于提供的上下文回答，不得编造；每个信息点后标注引用编号 [1][2] 等。
- **输出格式**：结构化 JSON，包含 `answer`（正文，Markdown 格式）和 `sources`（来源列表，含文档名、页码、原文片段）。
- **兜底策略**：当检索无结果或得分过低时，返回「未找到相关信息」而非编造答案。

### 4.4 API 服务模块

#### 4.4.1 接口清单

| 方法 | 路径 | 功能 | 同步/异步 |
|---|---|---|---|
| POST | `/v1/chat/query` | 单轮/多轮问答 | 同步 |
| POST | `/v1/ingest/file` | 单文件上传摄入 | 同步 |
| POST | `/v1/ingest/batch` | 批量摄入 | 同步（提交任务） |
| POST | `/v1/tasks/ingest/file` | 异步单文件任务 | 异步 |
| POST | `/v1/tasks/ingest/batch` | 异步批量任务 | 异步 |
| GET | `/v1/tasks/{task_id}` | 查询任务状态 | 同步 |
| GET | `/v1/health` | 健康检查 | 同步 |
| GET | `/v1/ready` | 就绪探针 | 同步 |

#### 4.4.2 请求/响应规范
- 统一使用 Pydantic 模型校验，Content-Type: `application/json`（文件上传除外）。
- 全局错误响应格式：`{ "code": "ERROR_CODE", "message": "人类可读描述", "details": {} }`
- 请求 ID 追踪：通过 `X-Request-ID` Header 透传，日志全程关联。

#### 4.4.3 错误码体系

| 错误码 | HTTP 状态 | 场景 | 用户提示 |
|---|---|---|---|
| `COLLECTION_NOT_READY` | 503 | Qdrant collection 未创建或不可访问 | 系统初始化中，请稍后重试 |
| `NO_CONTEXT_RETRIEVED` | 404 | 检索未命中任何相关文档 | 未找到与问题相关的文档，请尝试更换关键词或上传相关文档 |
| `DEPENDENCY_UNAVAILABLE` | 503 | DeepSeek / MongoDB / Qdrant 连接失败 | 依赖服务暂不可用，请联系管理员 |
| `TASK_NOT_FOUND` | 404 | 查询了不存在的任务 ID | 任务不存在或已过期 |
| `VALIDATION_ERROR` | 422 | 请求参数校验失败 | 参数格式错误，请参考 API 文档 |
| `INTERNAL_ERROR` | 500 | 未预期的内部异常 | 系统内部错误，请联系管理员 |

### 4.5 Web 前端模块

#### 4.5.1 页面结构
- **聊天主界面**：类 ChatGPT 的对话式交互，消息气泡区分用户/AI，AI 消息支持 Markdown 渲染。
- **来源引用面板**：AI 回答旁展开显示引用的原文片段及文档来源。
- **侧边栏**：
  - 知识库文件列表（已上传文档）
  - 拖拽上传区域（支持单文件 + 批量）
  - 移动端可收起

#### 4.5.2 交互细节
- **上传进度**：前端显示上传进度条，后台处理后自动刷新文件列表。
- **历史记录**：前端本地保存最近 20 轮对话记录（localStorage），支持新建会话。
- **重新生成**：对 AI 回答不满意，可点击重新生成，保留同一问题上下文。
- **复制/点赞/点踩**：每个 AI 消息提供快捷操作，点赞/点踩数据用于后续模型优化。

### 4.6 评估与监控模块

#### 4.6.1 RAGAS 自动评估
- **能力**：基于 MongoDB 中的父块记录，自动生成合成评测数据集。
- **消融实验**：支持 baseline、no-rewrite、no-rerank、no-multi-query 等实验对比。
- **指标**：
  - 检索：Hit@K、Context Precision、Context Recall
  - 生成：Answer Relevancy、Faithfulness
- **输出**：每个实验生成 `summary.md`（人类可读报告）、`trace.jsonl`（每样本全流程 trace）、`metrics.json`（结构化指标）。

#### 4.6.2 运行时监控
- **结构化日志**：所有关键操作记录 event_name、duration_ms、request_id、user_id。
- **健康探针**：`/v1/health`（存活）、`/v1/ready`（依赖就绪）。
- **关键指标暴露**：计划接入 Prometheus metrics（QPS、P99 延迟、检索召回率、生成 token 数等）。

---

## 5. 非功能需求

### 5.1 性能要求

| 指标 | 目标值 | 说明 |
|---|---|---|
| 单文件处理吞吐 | ≥ 10 页/秒 | 不含嵌入模型冷启动 |
| 问答首响时间（TTFB） | ≤ 3 秒 | 从请求到达至首字符返回 |
| 问答完整延迟 | ≤ 8 秒 | 普通复杂度问题（≤ 5 个父块） |
| API P99 延迟 | ≤ 10 秒 | 含异常路径 |
| 并发支持 | ≥ 20 QPS | 单实例，后续可水平扩展 |

### 5.2 可用性与可靠性

- **服务可用性**：≥ 99.5%（排除计划维护窗口）。
- **故障隔离**：单文件处理失败不影响同批次其他文件；单组件故障（如重排序器）有优雅降级路径。
- **数据持久化**：MongoDB 和 Qdrant 数据需定期备份策略（由运维侧保障，PRD 侧提出需求）。
- **幂等性**：同一文件重复上传，基于 SHA256 去重，返回一致结果。

### 5.3 安全需求

- **传输安全**：API 通信强制 HTTPS（生产环境）。
- **输入安全**：文件上传限制类型与大小，防止恶意文件；文件内容解析需防崩溃（如畸形 PDF）。
- **Prompt 安全**：LLM prompt 中不包含系统级敏感信息；防范 prompt injection（基础过滤 + 指令隔离）。
- **鉴权**（Phase 1 后规划）：当前版本为内网部署，暂不做用户鉴权，但接口需预留 `X-User-ID` / `Authorization` 扩展点。

### 5.4 可扩展性

- **模型可替换**：LLM、嵌入模型、重排序模型均通过配置化注入，不硬编码。
- **新格式接入**：loader 层遵循统一接口，新增格式仅需实现 `load()` 和 `normalize_metadata()`。
- **检索策略可插拔**：检索、重排序、父块召回各阶段通过 dataclass config 开关组合。

### 5.5 兼容性

- **浏览器**：Chrome ≥ 110, Edge ≥ 110, Firefox ≥ 110, Safari ≥ 16。
- **Python 版本**：3.11+
- **操作系统**：开发/部署目标为 Linux（Ubuntu 22.04+），Windows 11 为开发兼容。

---

## 6. 数据模型

### 6.1 核心实体

#### 文件摄入记录（MongoDB）
```json
{
  "_id": "ObjectId",
  "file_name": "差旅管理办法.pdf",
  "file_path": "/data/差旅管理办法.pdf",
  "file_type": "pdf",
  "content_hash": "sha256:abc123...",
  "status": "completed",
  "chunk_count": 12,
  "parent_chunks": [
    {
      "parent_id": "uuid",
      "content": "全文内容...",
      "metadata": { "page": 3, "section": "住宿标准" }
    }
  ],
  "created_at": "2026-04-22T10:00:00Z",
  "updated_at": "2026-04-22T10:00:30Z"
}
```

#### 子块向量点（Qdrant）
```json
{
  "id": "uuid5(content_hash:child_id)",
  "vector": {
    "dense": [0.1, 0.2, ...],
    "sparse": { "indices": [...], "values": [...] }
  },
  "payload": {
    "content": "子块文本片段",
    "parent_id": "uuid",
    "content_hash": "sha256:abc123...",
    "source": "差旅管理办法.pdf",
    "file_type": "pdf",
    "metadata": { "page": 3 }
  }
}
```

#### 任务记录（内存 + 持久化扩展点）
```json
{
  "task_id": "uuid",
  "task_type": "file_ingest | batch_ingest",
  "status": "pending | running | completed | failed",
  "files_total": 10,
  "files_processed": 7,
  "files_failed": 1,
  "result": { "file_results": [...] },
  "error": null,
  "created_at": "...",
  "updated_at": "..."
}
```

---

## 7. 接口详细设计（关键接口）

### 7.1 问答接口

**请求**：
```http
POST /v1/chat/query
Content-Type: application/json
X-Request-ID: req-123

{
  "query": "出差住宿标准是多少？",
  "session_id": "sess-abc",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "options": {
    "top_k": 5,
    "enable_rewrite": true,
    "enable_rerank": true
  }
}
```

**成功响应（200）**：
```json
{
  "answer": "根据《差旅管理办法》规定，一线城市住宿标准为每晚不超过 **500 元**，二线城市为每晚不超过 **350 元**。[1]",
  "sources": [
    {
      "index": 1,
      "document": "差旅管理办法.pdf",
      "page": 3,
      "snippet": "住宿标准：一线城市 500 元/晚，二线城市 350 元/晚..."
    }
  ],
  "metadata": {
    "rewrite_queries": ["出差住宿费用限额", "差旅酒店标准"],
    "retrieval_time_ms": 1200,
    "generation_time_ms": 2500
  }
}
```

### 7.2 异步批量摄入接口

**请求**：
```http
POST /v1/tasks/ingest/batch
Content-Type: multipart/form-data
X-Request-ID: req-456

files: [file1.pdf, file2.docx, ...]
```

**响应（202 Accepted）**：
```json
{
  "task_id": "task-uuid-789",
  "status": "pending",
  "message": "Batch ingestion task submitted successfully"
}
```

**状态查询**：
```http
GET /v1/tasks/task-uuid-789
```

```json
{
  "task_id": "task-uuid-789",
  "status": "completed",
  "progress": {
    "total": 10,
    "completed": 9,
    "failed": 1
  },
  "results": [...],
  "error": null,
  "duration_ms": 45000
}
```

---

## 8. 前端-后端集成规划

> **现状**：前端 `rag-frontend/src/api/ragApi.js` 当前全部为 mock 数据，未对接真实 API。

### 8.1 集成任务清单

| 优先级 | 任务 | 说明 |
|---|---|---|
| P0 | 替换 `sendQuery` | 对接 `POST /v1/chat/query`，流式响应支持见 Phase 2 |
| P0 | 替换 `uploadFile` | 对接 `POST /v1/ingest/file` 或 `/v1/tasks/ingest/file` |
| P0 | 替换 `uploadBatch` | 对接 `POST /v1/tasks/ingest/batch`，轮询任务状态 |
| P0 | 替换 `getFileList` | 对接文件列表查询（需新增 API 或从 MongoDB 暴露） |
| P1 | 错误处理 | 对接后端统一错误码，前端给出对应提示 |
| P1 | 进度展示 | 异步任务进度百分比展示 |
| P2 | 流式输出 | 后续迭代支持 SSE / WebSocket 流式返回 |

---

## 9. 里程碑与排期

### Phase 1：MVP 可用（当前 → 2 周后）
- [ ] 前端-后端 API 完全打通（P0 集成任务）
- [ ] 修复测试套件导入错误（`services.errors` → `services.exceptions`）
- [ ] 补充 API 启动入口（`api/app.py` 增加 `uvicorn.run` 或 `pyproject.toml` scripts）
- [ ] 完善 README（部署步骤、环境变量、快速开始）
- [ ] 核心链路端到端通过（上传 → 提问 → 回答 → 引用）

### Phase 2：体验优化（第 3-4 周）
- [ ] 流式输出（SSE）降低首响感知延迟
- [ ] 前端会话管理（多会话、历史记录持久化到后端）
- [ ] 文件列表与管理 API（删除、重新处理）
- [ ] 单测修复 + 覆盖率提升至 ≥ 80%

### Phase 3：生产加固（第 5-6 周）
- [ ] Docker 化部署（Dockerfile + docker-compose.yml）
- [ ] 用户鉴权（JWT / API Key）
- [ ] 可观测性（Prometheus + Grafana 指标大盘）
- [ ] 限流与熔断（API 层 rate limiting）

### Phase 4：智能化升级（第 7-8 周及以后）
- [ ] LangGraph 多 Agent 编排（路由 Agent、检索 Agent、总结 Agent）
- [ ] 多轮对话上下文压缩与优化
- [ ] 反馈闭环（点赞/点踩 → 微调数据沉淀）
- [ ] 知识库自动更新监控（目录监听、增量同步）

---

## 10. 风险与应对

| 风险 | 影响 | 概率 | 应对策略 |
|---|---|---|---|
| 前端-后端联调延迟 | Phase 1 延期 | 中 | 后端先提供 Swagger/OpenAPI 文档，前后端可并行；mock 层保留作为 fallback |
| 嵌入模型/重排序模型下载慢或失败 | 首次启动失败 | 高 | 文档明确预下载步骤；支持 HF 镜像（HF_ENDPOINT）；提供离线模型包 |
| DeepSeek API 限流/故障 | 问答不可用 | 中 | 实现 fallback LLM 配置（如本地 Ollama）；请求重试 + 指数退避 |
| Qdrant/MongoDB 单机性能瓶颈 | 高并发延迟飙升 | 低 | Phase 3 规划集群部署；当前单实例通过缓存和批量查询优化 |
| RAGAS 评估数据集质量差 | 指标失真 | 中 | 结合人工抽检 50 条以上；优化合成数据生成 prompt；引入真实用户问答日志作为补充 |
| 长文档召回精度不足 | 用户体验差 | 中 | 持续优化切分策略；引入 sliding window / Hierarchical 检索；A/B 测试不同策略 |

---

## 11. 附录

### 11.1 术语表

| 术语 | 解释 |
|---|---|
| RAG | Retrieval-Augmented Generation，检索增强生成 |
| Parent-Child Chunking | 父块切分（较大粒度，存储全文）+ 子块切分（较小粒度，用于向量检索） |
| RRF | Reciprocal Rank Fusion，倒数排名融合，用于合并多路检索结果 |
| RAGAS | RAG Assessment 框架，用于自动化评估 RAG 系统质量 |
| TTFB | Time To First Byte，首字节响应时间 |

### 11.2 相关文档索引

- `docs/superpowers/specs/` — 各模块详细设计规格（分块、检索、存储、错误处理等）
- `docs/superpowers/plans/` — 迭代计划与路线图
- `docs/evals/ragas-evaluation.md` — 评估报告规范

### 11.3 环境变量清单

| 变量名 | 必填 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 是 | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | 是 | DeepSeek API 基地址 |
| `DEEPSEEK_MODEL` | 是 | 模型名称，如 `deepseek-chat` |
| `HF_ENDPOINT` | 否 | HuggingFace 镜像地址 |
| `HF_TOKEN` | 否 | HuggingFace Token |
| `MONGODB_URI` | 是 | MongoDB 连接串 |
| `QDRANT_URL` | 是 | Qdrant 服务地址 |

---

> **评审记录**
> - v1.0（2026-04-22）：初稿完成，待技术评审与排期确认。
