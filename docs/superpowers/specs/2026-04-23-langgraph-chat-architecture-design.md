# LangGraph Chat Architecture Design

日期：2026-04-23

## 背景

当前项目已经具备两套可工作的能力边界：

- `FastAPI` 已经作为服务入口存在，能够承接问答、导入、任务查询与健康检查。
- 在线问答链路已经异步化到 service 层，核心能力包括 query rewrite、hybrid retrieval、rerank、parent recall 和 answer generation。

但目前问答编排仍主要集中在 `rag_demo.answering` 一类函数式流程中，存在几个问题：

- 编排逻辑与阶段能力耦合较紧，后续扩展条件分支、降级策略、阶段观测会越来越重。
- `FastAPI` 是异步入口，但问答内部还没有形成清晰的“哪些节点是原生 async，哪些节点是同步/CPU 边界”的统一模型。
- 后续如果要加入流式输出、图级 tracing、阶段级重试、条件路由、缓存或 agent 化能力，现有线性函数链路会变得不够清晰。

因此，本阶段设计目标不是重做整个 RAG 项目，而是在保留已有服务化基础的前提下，把在线问答链路演进为：

- `FastAPI` 作为统一异步入口
- `LangGraph` 作为在线问答编排层
- 选择性异步化 IO 节点
- 同步或 CPU 密集节点保持同步执行

## 目标

本次设计目标如下：

1. 用 `LangGraph` 接管在线问答链路的编排，但不改造文档导入链路。
2. 继续以 `FastAPI` 作为统一入口，对外暴露稳定的 HTTP API。
3. 把问答流程拆成可观察、可测试、可降级的节点。
4. 明确异步边界：IO 节点优先原生 async；同步/CPU 密集节点保持同步或显式隔离执行。
5. 保持现有 `api -> services -> runtime -> rag_demo` 的整体分层思路，采用增量式重构。

## 非目标

本阶段明确不做以下事情：

- 不把文档导入链路图化。
- 不追求全链路“所有函数都改成 `async def`”。
- 不立即替换所有底层三方库为原生 async 版本。
- 不引入多 agent、多图协作、长期记忆等超前能力。
- 不调整现有外部 API 协议，除非为兼容 `LangGraph` 输出增加少量内部字段。

## 范围

本次设计只覆盖在线问答主链路：

- `POST /v1/chat/query`
- 问答流程内部的编排、节点拆分、状态流转、降级策略与异步边界

本次设计不覆盖：

- `/v1/ingest/*`
- `/v1/tasks/*`
- 离线导入 pipeline 图化
- 评测链路图化

## 推荐方案

建议采用“分层重构型”方案，而不是轻量包裹型或全 agent 化方案。

### 方案 A：轻量包裹型

做法：

- 保留 `ChatService -> rag_demo.answering` 主体不变
- 在外层加一个非常薄的 `LangGraph`
- 每个节点只是现有函数的简单包装

优点：

- 改动小
- 上线风险低

缺点：

- 图节点职责不清
- 后续扩展价值有限
- 本质上只是“换了编排壳”

### 方案 B：分层重构型

做法：

- 把在线问答链路拆分为明确的图节点
- `LangGraph` 只负责状态流转和条件路由
- 各节点内部调用现有 `rag_demo/*` 能力函数或新抽取的 node service
- `ChatService` 退化为 graph 调用入口

优点：

- 节点边界清晰
- 便于可观测、测试、降级和后续扩展
- 风险可控，适合当前项目阶段

缺点：

- 需要一次中等规模重构

### 方案 C：全 agent 化扩展型

做法：

- 直接围绕 agent workflow 重构问答系统
- 预留工具调用、多图协作和长期记忆

优点：

- 扩展空间大

缺点：

- 对当前项目明显过重
- 会提前引入复杂度

### 结论

推荐采用方案 B。

它最适合当前项目：既不会只是表面接入 `LangGraph`，也不会把整个 RAG 项目一次性推入过重的 agent 架构。

## 总体架构

建议把问答相关部分划分为四层：

1. API 层
2. Application 层
3. Graph 层
4. Infrastructure/Runtime 层

调用关系如下：

`FastAPI Router -> ChatGraphService -> LangGraph App -> Node Executors -> Runtime/Storage`

### API 层

职责：

- 处理 HTTP 请求与响应
- 参数校验
- 认证、限流、日志、异常映射等横切逻辑
- 调用应用层服务

这一层不直接理解 RAG 编排细节，也不直接拼装底层模型或存储对象。

### Application 层

建议新增一个面向用例的入口服务，例如：

- `services/chat_graph_service.py`

职责：

- 接收 API 入参
- 构造 graph 输入
- 调用 `LangGraph` app
- 把 graph 最终状态映射成 API 响应模型

这一层是 API 与 Graph 的适配层，不承担复杂编排。

### Graph 层

建议新增：

- `graphs/chat/state.py`
- `graphs/chat/nodes.py`
- `graphs/chat/builder.py`
- `graphs/chat/models.py`

职责：

- 定义在线问答的状态对象
- 定义节点执行函数
- 定义主链路和条件路由
- 输出编译后的 graph app

这一层是本次设计的核心。

### Infrastructure/Runtime 层

保留：

- `runtime/container.py`
- `runtime/settings.py`
- `rag_demo/*`

职责：

- 提供 LLM、embeddings、reranker、Mongo、Qdrant 等依赖
- 复用现有检索、重排、回查、生成能力
- 管理资源初始化与关闭

## 目录演进建议

建议在当前项目基础上增量演进为：

```text
api/
  routers/
  schemas.py
  dependencies.py

services/
  chat_service.py
  chat_graph_service.py
  ingest_service.py
  task_service.py

graphs/
  chat/
    state.py
    models.py
    nodes.py
    builder.py

runtime/
  container.py
  settings.py

rag_demo/
  answering.py
  retrieval.py
  rerank.py
  parent_recall.py
  query_rewrite.py
```

说明：

- `chat_service.py` 可先保留为兼容层，再逐步退化为旧接口适配器。
- `rag_demo.answering.py` 在迁移期可以继续存在，最终变薄或退化为兼容入口。
- 不建议把 `LangGraph` 逻辑放进 `api/` 或 `runtime/`，避免层次污染。

## 问答图设计

建议把在线问答图拆成以下节点。

### 1. `prepare_query`

职责：

- 标准化输入
- 写入 request 上下文
- 初始化状态对象与阶段计时容器

输入：

- 原始问题
- 请求级参数，如 `top_k`、`candidate_limit`、`max_queries`、`parent_limit`
- request id、trace id 等上下文

输出：

- `question`
- `normalized_question`
- `request_context`
- `timings`

### 2. `rewrite_query`

职责：

- 调用 LLM 进行 query rewrite
- 生成多条候选检索 query

降级规则：

- 如果 rewrite 被关闭，则直接回退到原始 query
- 如果 rewrite 执行失败，则记录错误并回退到原始 query

输出：

- `rewritten_queries`
- `rewrite_status`

### 3. `retrieve_candidates`

职责：

- 对多条 query 执行 hybrid retrieval
- 聚合 child chunk 候选结果

建议：

- 如果 Qdrant async client 已可用，则原生 async 执行
- 若底层仍有同步调用，则只在该节点边界显式做隔离

输出：

- `retrieved_children`
- `per_query_hits`

### 4. `merge_candidates`

职责：

- 合并多 query 的检索结果
- 按 child chunk 去重
- 保留最佳分数

特点：

- 纯内存计算
- 保持同步实现即可

输出：

- `merged_children`

### 5. `rerank_candidates`

职责：

- 使用 cross-encoder reranker 对候选进行重排

建议：

- 如果 reranker 为本地同步模型，则保持同步节点
- 如有必要，可在单独 worker 边界执行，但不要伪装成原生 async

降级规则：

- reranker 不可用时，可直接跳过并使用 merge 后顺序

输出：

- `reranked_children`
- `rerank_status`

### 6. `recall_parents`

职责：

- 根据 child 命中回查 parent chunk
- 拼装最终上下文

建议：

- 该节点优先原生 async
- 如果 Mongo 仍为同步库，则作为明确的同步边界处理

输出：

- `parent_chunks`

### 7. `generate_answer`

职责：

- 基于 parent chunks 构造 prompt
- 调用 LLM 生成最终答案

建议：

- 使用 async LLM 调用路径

输出：

- `answer`
- `generation_status`

### 8. `build_response`

职责：

- 归一化 sources
- 组装返回体
- 收敛阶段时间、错误和 trace 摘要

特点：

- 纯内存处理
- 保持同步实现即可

输出：

- `source_items`
- `response_payload`

## 状态模型设计

建议定义统一的 `ChatGraphState`，用于表示一次在线问答的全生命周期状态。

建议字段包括：

- `question`
- `normalized_question`
- `rewritten_queries`
- `retrieved_children`
- `per_query_hits`
- `merged_children`
- `reranked_children`
- `parent_chunks`
- `answer`
- `source_items`
- `response_payload`
- `errors`
- `timings`
- `request_context`
- `rewrite_status`
- `rerank_status`
- `generation_status`

设计原则：

- 图负责管理状态，不让路由层和 service 层持有大量中间态。
- 节点只读写与自己职责直接相关的字段。
- 节点输出结构稳定，便于做节点单测和图级追踪。

## 异步化边界

### 设计原则

不追求“所有节点异步”，而是追求“该异步的地方异步，不该异步的地方保持同步”。

### 应优先原生 async 的节点

- `rewrite_query`
- `retrieve_candidates`
- `recall_parents`
- `generate_answer`

原因：

- 它们大概率涉及 LLM、Qdrant、Mongo 等外部 IO
- 原生 async 能避免阻塞事件循环

### 应保持同步的节点

- `merge_candidates`
- `build_response`

原因：

- 它们主要是内存内数据处理
- 强行 async 没有收益，只会增加复杂度

### 需谨慎处理的节点

- `rerank_candidates`
- 本地 embedding 相关步骤
- 其它本地模型推理

处理原则：

- 如果底层是同步/CPU 密集，就保持同步节点
- 如需与 async 图协同，可在明确边界处使用线程池或 `asyncio.to_thread(...)`
- 不把同步模型推理误标成“原生 async”

## 降级与容错

问答图需要明确支持以下容错策略：

1. query rewrite 失败时，回退到原始 query。
2. reranker 不可用时，跳过 rerank 阶段。
3. parent recall 后没有上下文时，抛出明确业务错误，而不是返回空答案。
4. 节点需要记录阶段失败原因，便于日志、调试与后续可观测性。

不建议在本阶段加入复杂重试策略；优先先把阶段边界与失败语义建立清楚。

## 迁移策略

建议按以下顺序增量迁移。

### 第一阶段：抽取状态对象

先定义 `ChatGraphState` 与图级输入输出模型，不改外部 API。

目标：

- 固化问答链路的中间状态
- 为图化重构建立稳定载体

### 第二阶段：节点化现有问答阶段

把当前 `answer_query_async` 内部阶段拆成独立节点函数。

目标：

- 让 rewrite、retrieve、merge、rerank、recall、generate 具备独立边界
- 尽可能复用现有 `rag_demo/*` 能力

### 第三阶段：引入 LangGraph 主链路

构建线性主图：

`prepare -> rewrite -> retrieve -> merge -> rerank -> recall -> generate -> response`

目标：

- 用图接管当前问答编排
- 保持 API 路由与外部行为稳定

### 第四阶段：让 ChatService 调用 graph

让 `ChatService` 或新建的 `ChatGraphService` 作为 graph 调用入口。

目标：

- 对 API 层隐藏图细节
- 保证 `/v1/chat/query` 对外协议尽量不变

### 第五阶段：补充可观测与降级细节

补充：

- 阶段耗时
- 阶段状态
- 节点失败信息
- 条件跳转原因

目标：

- 为后续 tracing、流式输出和复杂路由预留空间

## 对现有模块的影响

### `api/routers/chat.py`

影响：

- 基本不改协议
- 依赖的 service 实现从当前 chat service 迁移到 graph service

### `services/chat_service.py`

影响：

- 从“直接执行问答流程”逐步转为“graph 调用入口”或兼容适配层

### `rag_demo/answering.py`

影响：

- 逐步从总控函数演化为节点能力承载层
- 最终可保留为兼容包装或被内部节点完全替代

### `runtime/container.py`

影响：

- 继续负责统一装配 LLM、Mongo、Qdrant、embeddings、reranker
- 不承担图编排职责

## 测试策略

本阶段至少补三层测试。

### 1. 节点级测试

验证：

- 各节点输入输出是否稳定
- 降级逻辑是否符合预期
- 同步节点与异步节点边界是否明确

### 2. 图级测试

验证：

- 主链路状态流转是否正确
- 在 rewrite 失败、rerank 跳过、无上下文等场景下，图行为是否符合设计

### 3. API 回归测试

验证：

- `/v1/chat/query` 的外部响应结构不回归
- 错误码和异常映射不回归

## 风险与约束

### 风险

1. 如果节点拆分过细，图会变得啰嗦，反而增加维护成本。
2. 如果节点拆分过粗，`LangGraph` 只是形式接入，收益有限。
3. 如果同步/CPU 密集节点误进入事件循环，会影响 `FastAPI` 并发性能。
4. 迁移期可能同时存在旧问答入口与新图入口，需要明确兼容边界。

### 约束

1. 现有导入链路不纳入本次图化范围。
2. 外部 API 行为应尽量稳定。
3. 迁移期间优先复用现有运行时装配和领域能力，避免大规模重写。

## 结论

对当前项目来说，最优路径不是“把整个 RAG 全部推倒重做成全异步图系统”，而是：

- 保留 `FastAPI` 作为统一异步入口
- 只让 `LangGraph` 接管在线问答链路
- 采用节点级选择性异步化
- 对同步/CPU 密集步骤保持同步执行
- 用增量式迁移替代大爆炸式重构

这条路径能在控制复杂度和风险的前提下，把当前项目从“已有异步服务骨架”推进到“具有清晰编排边界、清晰异步边界、可持续扩展的问答架构”。
