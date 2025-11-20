# LlamaIndex 生产部署指南

## 环境配置

### 1. 环境变量设置

创建 `.env` 文件：

```bash
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
COHERE_API_KEY=xxx
PINECONE_API_KEY=xxx
```

### 2. 依赖管理

使用 rye 管理项目依赖：

```bash
rye init llamaindex-demo
rye add llama-index openai python-dotenv
rye sync
```

## 生产优化

### 1. 索引持久化

```python
from llama_index.core import StorageContext, load_index_from_storage

# 保存索引
index.storage_context.persist(persist_dir="./storage")

# 加载索引
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

### 2. 缓存策略

```python
from llama_index.core import set_global_handler

set_global_handler("simple")  # 启用缓存
```

### 3. 成本优化

使用不同模型组合：

```python
from llama_index.llms.openai import OpenAI

# 检索用小模型
cheap_llm = OpenAI(model="gpt-3.5-turbo")
index = VectorStoreIndex.from_documents(documents, llm=cheap_llm)

# 生成用强模型
expensive_llm = OpenAI(model="gpt-4-turbo")
query_engine = index.as_query_engine(llm=expensive_llm)
```

## 部署架构

### 1. FastAPI 后端

```python
from fastapi import FastAPI
from llama_index.core import VectorStoreIndex

app = FastAPI()
query_engine = index.as_query_engine()

@app.post("/query")
async def query(question: str):
    response = await query_engine.aquery(question)
    return {"answer": response.response}
```

### 2. Streamlit 前端

```python
import streamlit as st
from llama_index.core import VectorStoreIndex

st.title("企业知识库问答")
question = st.text_input("请输入问题：")

if question:
    with st.spinner("思考中..."):
        response = query_engine.query(question)
        st.write(response.response)
```

## 监控与观测

### 1. 性能监控

```python
from llama_index.core.callbacks import TokenCountingHandler

token_counter = TokenCountingHandler()
callback_manager = CallbackManager([token_counter])

# 查询后检查token使用
print(f"Embedding Tokens: {token_counter.total_embedding_token_count}")
print(f"LLM Prompt Tokens: {token_counter.prompt_llm_token_count}")
```

### 2. 错误处理

```python
import logging
from llama_index.core import VectorStoreIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    response = query_engine.query("问题")
except Exception as e:
    logger.error(f"查询失败: {e}")
    # 实现降级策略
```

## 扩展性考虑

### 1. 分布式处理

```python
# 使用异步处理提升性能
index = await VectorStoreIndex.afrom_documents(documents)

# 批量插入优化
index.insert_nodes(nodes, show_progress=True)
```

### 2. 向量数据库

使用专业向量数据库：

```python
from llama_index.vector_stores.pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(pinecone_index=index)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(vector_store=vector_store)
)
```

这些最佳实践确保 LlamaIndex 应用在生产环境中稳定、高效地运行。