# LlamaIndex 高级功能详解

## 索引策略

LlamaIndex 提供多种索引类型以适应不同的使用场景：

### 1. 向量索引（Vector Store Index）

最常用的索引类型，适合语义搜索：

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents,
    embed_model="local:BAAI/bge-small-en-v1.5"
)
```

### 2. 树形索引（Tree Index）

层级化组织，适合大规模文档摘要：

```python
from llama_index.core import TreeIndex

tree_index = TreeIndex.from_documents(documents)
```

### 3. 知识图谱索引（Knowledge Graph Index）

提取实体关系，支持结构化推理：

```python
from llama_index.core import KnowledgeGraphIndex

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=5,
    include_embeddings=True
)
```

## 查询优化

### 混合检索

结合向量检索和关键词检索：

```python
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=5,
    mode="relative_score"
)
```

### 重排序

使用重排序模型提升检索精度：

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[CohereRerank(top_n=3)]
)
```

## 智能代理

LlamaIndex 支持多种代理模式：

### ReAct 代理

推理-行动循环模式：

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

def multiply(a: float, b: float) -> float:
    """两个数相乘"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

agent = ReActAgent.from_tools(
    tools=[multiply_tool],
    llm=OpenAI(model="gpt-4-turbo"),
    verbose=True
)
```

### Function Calling

结构化工具调用：

```python
from llama_index.core.tools import QueryEngineTool

search_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="document_search",
    description="搜索公司内部文档"
)
```

## 评估框架

LlamaIndex 提供完整的评估工具链：

```python
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator
)

faithfulness = FaithfulnessEvaluator(llm=llm)
relevancy = RelevancyEvaluator(llm=llm)
correctness = CorrectnessEvaluator(llm=llm)
```

这些评估指标帮助开发者量化 RAG 系统的性能，持续优化系统表现。