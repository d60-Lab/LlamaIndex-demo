# LlamaIndex：企业级 LLM 应用开发框架

## 目录
- [概述](#概述)
- [核心概念](#核心概念)
- [架构设计](#架构设计)
- [核心功能](#核心功能)
- [LlamaAgents 智能代理](#llamaagents-智能代理)
- [高级特性](#高级特性)
- [最佳实践](#最佳实践)
- [生态系统](#生态系统)
- [实战场景](#实战场景)
- [快速开始](#快速开始)

---

## 概述

LlamaIndex（原名 GPT Index）是一个专为生产环境设计的开源 LLM 应用开发框架。它解决了构建 RAG（检索增强生成）应用和 AI 代理系统中的核心挑战：**如何让 LLM 高效地访问和理解私有数据**。

### 为什么选择 LlamaIndex？

- 🚀 **开箱即用**：5 行代码即可构建基础 RAG 应用
- 🔌 **丰富集成**：支持 160+ 数据源和 50+ LLM 提供商
- 🏗️ **生产就绪**：完整的可观测性、评估和部署工具链
- 🤖 **智能代理**：内置 ReAct、Function Calling 等代理模式
- 📈 **可扩展性**：从原型到百万级文档索引的平滑扩展

---

## 核心概念

### RAG（检索增强生成）工作流

```
用户查询 → 数据检索 → 上下文增强 → LLM 生成 → 结构化输出
```

LlamaIndex 在每个环节都提供了优化工具：

1. **数据摄取**：统一接口处理非结构化数据
2. **索引构建**：多种索引策略适配不同场景
3. **智能检索**：混合检索、重排序、查询变换
4. **生成优化**：提示工程、响应合成、流式输出

---

## 架构设计

### 三层架构

```
┌─────────────────────────────────────────┐
│         应用层 (Application Layer)       │
│  Query Engines | Agents | Chat Engines  │
├─────────────────────────────────────────┤
│        索引层 (Indexing Layer)          │
│  Vector | Graph | Tree | Summary Index  │
├─────────────────────────────────────────┤
│       数据层 (Data Layer)                │
│  Readers | Parsers | Transformers       │
└─────────────────────────────────────────┘
```

### 1. 数据连接层

**支持的数据源**（160+ 种）：
- 📄 **文档**：PDF, Word, Markdown, HTML
- 🗄️ **数据库**：PostgreSQL, MongoDB, MySQL, Redis
- ☁️ **云服务**：Google Drive, Notion, Slack, Confluence
- 🌐 **API**：Web scraping, RSS feeds, REST APIs
- 📊 **结构化数据**：CSV, Excel, JSON, Parquet

**关键特性**：
```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.notion import NotionPageReader

# 统一接口加载不同数据源
documents = SimpleDirectoryReader("./docs").load_data()
notion_docs = NotionPageReader(token="xxx").load_data()

# 自动文档解析和元数据提取
for doc in documents:
    print(f"File: {doc.metadata['file_name']}")
    print(f"Size: {len(doc.text)} chars")
```

### 2. 索引系统

#### 向量索引（Vector Store Index）
最常用的索引类型，适合语义搜索：

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents,
    embed_model="local:BAAI/bge-small-en-v1.5"  # 可配置嵌入模型
)
```

#### 树形索引（Tree Index）
层级化组织，适合大规模文档摘要：

```python
from llama_index.core import TreeIndex

tree_index = TreeIndex.from_documents(documents)
# 自动构建摘要树，支持自顶向下查询
```

#### 知识图谱索引（Knowledge Graph Index）
提取实体关系，支持结构化推理：

```python
from llama_index.core import KnowledgeGraphIndex

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=5,
    include_embeddings=True
)
```

#### 列表索引（List Index）
简单顺序遍历，适合小数据集：

```python
from llama_index.core import SummaryIndex

list_index = SummaryIndex.from_documents(documents)
```

### 3. 查询引擎

**检索策略**：

```python
# 1. 基础向量检索
query_engine = index.as_query_engine(similarity_top_k=5)

# 2. 混合检索（向量 + 关键词）
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)
query_engine = RetrieverQueryEngine(retriever=retriever)

# 3. 带重排序的检索
from llama_index.postprocessor.cohere_rerank import CohereRerank

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[CohereRerank(top_n=3)]
)
```

**响应合成模式**：

```python
# compact: 合并上下文后一次性生成
# refine: 迭代式生成，逐步细化答案
# tree_summarize: 树形聚合多个响应
# simple_summarize: 截断上下文后生成

query_engine = index.as_query_engine(
    response_mode="compact",
    streaming=True  # 流式输出
)
```

---

## 核心功能

### 数据分块（Chunking）

合理的分块策略直接影响检索质量：

```python
from llama_index.core.node_parser import SentenceSplitter

# 1. 基于句子的分块
splitter = SentenceSplitter(
    chunk_size=1024,        # token 数量
    chunk_overlap=200,      # 重叠部分，保持上下文连贯
    paragraph_separator="\n\n"
)

# 2. 基于语义的分块
from llama_index.core.node_parser import SemanticSplitterNodeParser

semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95
)

# 3. 保持代码结构的分块
from llama_index.core.node_parser import CodeSplitter

code_splitter = CodeSplitter(
    language="python",
    chunk_lines=40,
    chunk_overlap_lines=5
)
```

### 元数据过滤

精确控制检索范围：

```python
from llama_index.core.vector_stores import MetadataFilters, FilterCondition

filters = MetadataFilters(
    filters=[
        {"key": "category", "value": "technical"},
        {"key": "date", "value": "2024", "operator": ">="}
    ],
    condition=FilterCondition.AND
)

query_engine = index.as_query_engine(filters=filters)
```

### 查询变换

优化用户查询以提升检索效果：

```python
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

# HyDE: 生成假设性文档再检索
hyde = HyDEQueryTransform(include_original=True)
query_engine = index.as_query_engine(query_transform=hyde)

# 多查询生成
from llama_index.core.indices.query.query_transform import MultiQueryTransform
multi_query = MultiQueryTransform(num_queries=3)
```

---

## LlamaAgents 智能代理

### 代理模式对比

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| **ReAct** | 推理-行动循环 | 需要多步推理的任务 |
| **Function Calling** | 结构化工具调用 | API 集成、数据库操作 |
| **OpenAI Agents** | 原生函数调用 | OpenAI 模型专用 |
| **LLMCompiler** | 并行任务执行 | 需要高效率的复杂工作流 |

### ReAct 代理实现

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.openai import OpenAI

# 1. 定义工具函数
def multiply(a: float, b: float) -> float:
    """两个数相乘"""
    return a * b

def search_docs(query: str) -> str:
    """搜索文档库"""
    return query_engine.query(query).response

# 2. 创建工具
multiply_tool = FunctionTool.from_defaults(fn=multiply)
search_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="document_search",
    description="搜索公司内部文档，回答业务相关问题"
)

# 3. 初始化代理
agent = ReActAgent.from_tools(
    tools=[multiply_tool, search_tool],
    llm=OpenAI(model="gpt-4-turbo"),
    verbose=True,
    max_iterations=10
)

# 4. 执行任务
response = agent.chat(
    "查找我们 Q3 的销售额，然后将其乘以 1.15 预测 Q4 增长"
)
```

### 多代理协作

```python
from llama_index.core.agent import AgentRunner
from llama_index.core.workflow import Workflow

# 定义专门的代理
research_agent = ReActAgent.from_tools([search_tool], llm=llm)
analysis_agent = ReActAgent.from_tools([calculator_tool], llm=llm)
writer_agent = ReActAgent.from_tools([document_tool], llm=llm)

# 工作流编排
class ReportWorkflow(Workflow):
    async def run(self, topic: str):
        # 研究阶段
        research = await research_agent.achat(f"研究主题: {topic}")
        
        # 分析阶段
        analysis = await analysis_agent.achat(
            f"分析以下数据: {research.response}"
        )
        
        # 撰写阶段
        report = await writer_agent.achat(
            f"基于以下分析撰写报告: {analysis.response}"
        )
        
        return report.response

workflow = ReportWorkflow()
result = await workflow.run("2024年AI行业趋势")
```

### 记忆系统

```python
from llama_index.core.memory import ChatMemoryBuffer

# 短期记忆（会话级别）
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

agent = ReActAgent.from_tools(
    tools=tools,
    memory=memory,
    llm=llm
)

# 长期记忆（持久化）
from llama_index.core.storage.chat_store import SimpleChatStore

chat_store = SimpleChatStore()
memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user_123"
)
```

---

## 高级特性

### 1. 多模态处理

```python
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader

# 加载图像文档
image_documents = SimpleDirectoryReader(
    input_files=["chart.png", "diagram.jpg"]
).load_data()

# 多模态查询
multimodal_llm = OpenAIMultiModal(model="gpt-4-vision-preview")
response = multimodal_llm.complete(
    prompt="分析这张销售图表的趋势",
    image_documents=image_documents
)
```

### 2. 结构化输出（Structured Output）

```python
from pydantic import BaseModel, Field
from llama_index.program.openai import OpenAIPydanticProgram

class CompanyInfo(BaseModel):
    """公司信息"""
    name: str = Field(description="公司名称")
    founded_year: int = Field(description="成立年份")
    industry: str = Field(description="所属行业")
    revenue: float = Field(description="年收入（百万美元）")
    employees: int = Field(description="员工数量")

# 创建提取程序
program = OpenAIPydanticProgram.from_defaults(
    output_cls=CompanyInfo,
    prompt_template_str=(
        "从以下文本中提取公司信息：\n"
        "{text}\n"
        "返回结构化的JSON数据。"
    ),
    verbose=True
)

# 执行提取
company_info = program(text=document.text)
print(f"公司: {company_info.name}")
print(f"收入: ${company_info.revenue}M")
```

### 3. 子问题查询（Sub Question Query）

将复杂问题分解为多个子问题：

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

# 为不同数据源创建查询引擎
sales_engine = sales_index.as_query_engine()
marketing_engine = marketing_index.as_query_engine()
product_engine = product_index.as_query_engine()

# 包装为工具
tools = [
    QueryEngineTool.from_defaults(
        query_engine=sales_engine,
        name="sales_data",
        description="包含销售数据和业绩指标"
    ),
    QueryEngineTool.from_defaults(
        query_engine=marketing_engine,
        name="marketing_data",
        description="包含营销活动和ROI数据"
    ),
    QueryEngineTool.from_defaults(
        query_engine=product_engine,
        name="product_data",
        description="包含产品特性和用户反馈"
    )
]

# 创建子问题查询引擎
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools,
    verbose=True
)

# 自动分解复杂查询
response = query_engine.query(
    "对比Q1和Q2的销售表现，分析营销活动的效果，"
    "并评估用户对新产品的反馈"
)
```

### 4. 路由查询（Router Query）

根据问题类型动态选择最佳查询引擎：

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# 创建路由器
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        QueryEngineTool.from_defaults(
            query_engine=vector_engine,
            description="用于语义搜索和概念理解"
        ),
        QueryEngineTool.from_defaults(
            query_engine=sql_engine,
            description="用于结构化数据查询和统计分析"
        ),
        QueryEngineTool.from_defaults(
            query_engine=graph_engine,
            description="用于关系推理和实体关联"
        )
    ]
)
```

### 5. 评估框架

```python
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator
)

# 定义评估器
faithfulness = FaithfulnessEvaluator(llm=llm)
relevancy = RelevancyEvaluator(llm=llm)
correctness = CorrectnessEvaluator(llm=llm)

# 评估查询结果
response = query_engine.query("LlamaIndex的主要特性是什么？")

faith_result = faithfulness.evaluate_response(response=response)
rel_result = relevancy.evaluate_response(
    query="LlamaIndex的主要特性是什么？",
    response=response
)

print(f"忠实度分数: {faith_result.score}")
print(f"相关性分数: {rel_result.score}")
```

### 6. 批量评估

```python
from llama_index.core.evaluation import BatchEvalRunner

# 准备测试集
questions = [
    "什么是RAG？",
    "LlamaIndex支持哪些数据源？",
    "如何优化检索性能？"
]

# 批量评估
runner = BatchEvalRunner(
    {"faithfulness": faithfulness, "relevancy": relevancy},
    workers=8
)

eval_results = await runner.aevaluate_queries(
    query_engine=query_engine,
    queries=questions
)

# 生成报告
print(eval_results)
```

---

## 最佳实践

### 1. 数据准备最佳实践

```python
from llama_index.core import Document
from llama_index.core.schema import TextNode

# ✅ 推荐：添加丰富的元数据
documents = [
    Document(
        text=content,
        metadata={
            "source": "internal_wiki",
            "department": "engineering",
            "last_updated": "2024-01-15",
            "author": "john@example.com",
            "version": "2.0",
            "tags": ["api", "authentication"]
        }
    )
]

# ✅ 推荐：自定义文档ID便于更新
documents = [
    Document(
        text=content,
        id_="doc_123",  # 自定义ID
        metadata=metadata
    )
]

# 增量更新
index.refresh_ref_docs(documents)  # 只更新变化的文档
```

### 2. 分块策略选择

```python
# 场景1：技术文档 - 保持代码完整性
from llama_index.core.node_parser import CodeSplitter
parser = CodeSplitter(language="python", chunk_lines=50)

# 场景2：对话记录 - 保持对话完整
from llama_index.core.node_parser import SentenceSplitter
parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separator="\n\n"  # 按对话分隔
)

# 场景3：学术论文 - 语义分块
from llama_index.core.node_parser import SemanticSplitterNodeParser
parser = SemanticSplitterNodeParser(
    embed_model=embed_model,
    breakpoint_percentile_threshold=95
)
```

### 3. 索引优化

```python
# ✅ 使用持久化存储避免重复构建
from llama_index.core import StorageContext, load_index_from_storage

# 首次构建
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./storage")

# 后续加载
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# ✅ 异步构建提升性能
index = await VectorStoreIndex.afrom_documents(documents)

# ✅ 批量插入优化
index.insert_nodes(nodes, show_progress=True)
```

### 4. 检索优化

```python
# ✅ 混合检索策略
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

# 融合检索结果
retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=5,
    mode="relative_score"  # 相对分数融合
)

# ✅ 重排序提升精度
from llama_index.postprocessor.cohere_rerank import CohereRerank

reranker = CohereRerank(api_key="xxx", top_n=3)
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker]
)
```

### 5. 成本优化

```python
# ✅ 使用缓存减少API调用
from llama_index.core import set_global_handler

set_global_handler("simple")  # 启用缓存

# ✅ 选择性使用强模型
from llama_index.llms.openai import OpenAI

# 检索和初步处理用小模型
cheap_llm = OpenAI(model="gpt-3.5-turbo")
index = VectorStoreIndex.from_documents(documents, llm=cheap_llm)

# 最终生成用强模型
expensive_llm = OpenAI(model="gpt-4-turbo")
query_engine = index.as_query_engine(llm=expensive_llm)

# ✅ 本地嵌入模型
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
```

### 6. 可观测性配置

```python
# ✅ 集成 LangSmith 追踪
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "xxx"

from llama_index.core import set_global_handler
set_global_handler("langsmith")

# ✅ 自定义回调
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

# 应用到索引
index = VectorStoreIndex.from_documents(
    documents,
    callback_manager=callback_manager
)

# ✅ 性能监控
from llama_index.core.callbacks import TokenCountingHandler

token_counter = TokenCountingHandler()
callback_manager = CallbackManager([token_counter])

# 查询后检查token使用
print(f"Embedding Tokens: {token_counter.total_embedding_token_count}")
print(f"LLM Prompt Tokens: {token_counter.prompt_llm_token_count}")
print(f"LLM Completion Tokens: {token_counter.completion_llm_token_count}")
```

---

## 生态系统

### LLM 提供商支持

| 提供商 | 模型示例 | 安装包 |
|--------|----------|--------|
| OpenAI | GPT-4, GPT-3.5 | `llama-index-llms-openai` |
| Anthropic | Claude 3 | `llama-index-llms-anthropic` |
| Google | Gemini | `llama-index-llms-gemini` |
| Cohere | Command | `llama-index-llms-cohere` |
| Azure OpenAI | GPT-4 | `llama-index-llms-azure-openai` |
| 本地模型 | Llama 2, Mistral | `llama-index-llms-ollama` |

### 向量数据库集成

| 数据库 | 特点 | 适用规模 |
|--------|------|----------|
| Pinecone | 托管服务，易用 | 中大型 |
| Weaviate | 开源，功能丰富 | 大型 |
| Chroma | 轻量级，本地优先 | 小中型 |
| Milvus | 高性能，可扩展 | 大型 |
| Qdrant | Rust实现，快速 | 中大型 |
| FAISS | Meta开源，内存型 | 小中型 |

```python
# Pinecone 示例
from llama_index.vector_stores.pinecone import PineconeVectorStore
import pinecone

pinecone.init(api_key="xxx", environment="us-west1-gcp")
vector_store = PineconeVectorStore(pinecone_index=index)

# Chroma 示例
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

chroma_client = chromadb.Client()
vector_store = ChromaVectorStore(chroma_collection=collection)
```

### 框架集成

```python
# 与 LangChain 互操作
from langchain.agents import Tool
from llama_index.langchain_helpers.agents import LlamaToolkit

toolkit = LlamaToolkit(index=index)
tools = toolkit.get_tools()

# 与 FastAPI 集成
from fastapi import FastAPI
app = FastAPI()

@app.post("/query")
async def query(question: str):
    response = await query_engine.aquery(question)
    return {"answer": response.response}

# 与 Streamlit 集成
import streamlit as st

st.title("企业知识库问答")
question = st.text_input("请输入问题：")
if question:
    with st.spinner("思考中..."):
        response = query_engine.query(question)
        st.write(response.response)
```

---

## 实战场景

### 场景 1：企业知识库问答系统

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. 数据加载
documents = SimpleDirectoryReader(
    input_dir="./company_docs",
    recursive=True,
    required_exts=[".pdf", ".docx", ".md"]
).load_data()

# 2. 文档处理
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)

# 3. 构建索引
index = VectorStoreIndex(
    nodes=nodes,
    embed_model=OpenAIEmbedding(model="text-embedding-3-small")
)

# 4. 创建查询引擎（带引用）
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact",
    llm=OpenAI(model="gpt-4"),
)

# 5. 查询并显示来源
response = query_engine.query("公司的休假政策是什么？")
print(f"答案: {response.response}")
print("\n来源文档:")
for node in response.source_nodes:
    print(f"- {node.metadata['file_name']} (相关度: {node.score:.2f})")
```

### 场景 2：SQL 数据库智能查询

```python
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine

# 1. 连接数据库
engine = create_engine("postgresql://user:pass@localhost/sales_db")
sql_database = SQLDatabase(engine, include_tables=["orders", "customers"])

# 2. 创建NL2SQL查询引擎
query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["orders", "customers"],
    llm=OpenAI(model="gpt-4")
)

# 3. 自然语言查询
response = query_engine.query(
    "列出2024年Q1销售额超过10万的客户，按销售额降序排列"
)
print(response.response)
print(f"\n生成的SQL: {response.metadata['sql_query']}")
```

### 场景 3：多文档对比分析

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import SubQuestionQueryEngine

# 1. 为每个文档创建索引
doc_2023 = SimpleDirectoryReader(input_files=["report_2023.pdf"]).load_data()
doc_2024 = SimpleDirectoryReader(input_files=["report_2024.pdf"]).load_data()

index_2023 = VectorStoreIndex.from_documents(doc_2023)
index_2024 = VectorStoreIndex.from_documents(doc_2024)

# 2. 创建查询工具
tools = [
    QueryEngineTool.from_defaults(
        query_engine=index_2023.as_query_engine(),
        name="report_2023",
        description="2023年度业务报告"
    ),
    QueryEngineTool.from_defaults(
        query_engine=index_2024.as_query_engine(),
        name="report_2024",
        description="2024年度业务报告"
    )
]

# 3. 子问题引擎自动分解对比问题
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools,
    verbose=True
)

response = query_engine.query(
    "对比2023和2024年的营收增长率和利润率，分析变化原因"
)
```

### 场景 4：客服机器人（带工单系统）

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

# 1. 定义工具函数
def search_knowledge_base(query: str) -> str:
    """搜索知识库回答常见问题"""
    response = kb_query_engine.query(query)
    return response.response

def create_ticket(issue: str, priority: str) -> str:
    """创建工单"""
    ticket_id = tickets_api.create(issue=issue, priority=priority)
    return f"已创建工单 #{ticket_id}，我们会尽快处理"

def check_order_status(order_id: str) -> str:
    """查询订单状态"""
    status = orders_db.get_status(order_id)
    return f"订单 {order_id} 状态: {status}"

# 2. 创建工具
tools = [
    FunctionTool.from_defaults(fn=search_knowledge_base),
    FunctionTool.from_defaults(fn=create_ticket),
    FunctionTool.from_defaults(fn=check_order_status)
]

# 3. 初始化客服代理
agent = ReActAgent.from_tools(
    tools=tools,
    llm=OpenAI(model="gpt-4"),
    system_prompt=(
        "你是一个专业的客服代理。"
        "优先从知识库查找答案，如果无法解决则创建工单。"
        "始终保持礼貌和专业。"
    ),
    max_iterations=5
)

# 4. 处理用户请求
response = agent.chat("我的订单 #12345 什么时候发货？")
```

### 场景 5：代码库问答助手

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import CodeSplitter
from llama_index.readers.file import FlatReader

# 1. 加载代码文件
reader = FlatReader()
documents = reader.load_data(
    input_dir="./src",
    include_exts=[".py", ".js", ".ts"]
)

# 2. 使用代码分块器
splitter = CodeSplitter(
    language="python",
    chunk_lines=100,
    chunk_overlap_lines=10,
    max_chars=2000
)
nodes = splitter.get_nodes_from_documents(documents)

# 3. 添加函数级元数据
for node in nodes:
    # 提取函数名、类名等
    code = node.text
    if "def " in code:
        func_name = code.split("def ")[1].split("(")[0]
        node.metadata["function_name"] = func_name

# 4. 构建索引
index = VectorStoreIndex(nodes=nodes)

# 5. 查询
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact"
)

response = query_engine.query(
    "如何实现用户认证？请给出相关代码示例"
)
```

---

## 快速开始

### 安装

```bash
# 基础安装
pip install llama-index

# 完整安装（包含所有集成）
pip install llama-index[all]

# 按需安装特定集成
pip install llama-index-llms-anthropic
pip install llama-index-vector-stores-pinecone
pip install llama-index-readers-notion
```

### 5 分钟快速入门

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1. 加载数据（将文档放在 ./data 目录）
documents = SimpleDirectoryReader("data").load_data()

# 2. 构建索引
index = VectorStoreIndex.from_documents(documents)

# 3. 查询
query_engine = index.as_query_engine()
response = query_engine.query("文档的主要内容是什么？")

print(response)
```

### 配置环境变量

```bash
# .env 文件
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
COHERE_API_KEY=xxx
PINECONE_API_KEY=xxx
PINECONE_ENVIRONMENT=us-west1-gcp
```

```python
# 在代码中加载
from dotenv import load_dotenv
load_dotenv()
```

### 持久化示例

```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
import os

PERSIST_DIR = "./storage"

# 首次运行：构建并保存索引
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # 后续运行：加载已有索引
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
```

---

## 学习资源

### 官方文档
- 📚 [完整文档](https://docs.llamaindex.ai/)
- 🎓 [教程中心](https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html)
- 💡 [示例代码库](https://github.com/run-llama/llama_index/tree/main/docs/examples)

### 社区资源
- 💬 [Discord 社区](https://discord.gg/dGcwcsnxhU)
- 🐙 [GitHub 仓库](https://github.com/run-llama/llama_index)
- 🎥 [YouTube 教程](https://www.youtube.com/@LlamaIndex)

### 推荐学习路径
1. **初级**：官方快速入门 → 基础RAG应用
2. **中级**：查询引擎优化 → 多种索引策略
3. **高级**：智能代理开发 → 多代理协作系统
4. **专家**：生产部署 → 性能优化与监控

---

## 总结

LlamaIndex 通过其**模块化架构**和**丰富的工具链**，已成为构建生产级 LLM 应用的首选框架。它特别适合：

✅ 需要处理**大量私有数据**的企业应用  
✅ 需要**复杂推理和工具调用**的 AI 代理  
✅ 需要**高度定制化**的 RAG 系统  
✅ 需要**快速原型到生产部署**的项目

通过合理运用其索引策略、检索优化、代理系统和评估框架，开发者可以构建出高质量、可扩展的智能应用。
