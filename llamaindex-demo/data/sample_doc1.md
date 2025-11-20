# LlamaIndex 入门指南

## 什么是 LlamaIndex？

LlamaIndex 是一个专为生产环境设计的开源 LLM 应用开发框架。它解决了构建 RAG（检索增强生成）应用和 AI 代理系统中的核心挑战：如何让 LLM 高效地访问和理解私有数据。

## 核心特性

- **开箱即用**：5 行代码即可构建基础 RAG 应用
- **丰富集成**：支持 160+ 数据源和 50+ LLM 提供商
- **生产就绪**：完整的可观测性、评估和部署工具链
- **智能代理**：内置 ReAct、Function Calling 等代理模式
- **可扩展性**：从原型到百万级文档索引的平滑扩展

## 使用场景

1. **企业知识库问答**：将公司内部文档转换为智能问答系统
2. **代码库助手**：帮助开发者理解和查询大型代码库
3. **研究助理**：处理学术论文和研究资料
4. **客服机器人**：结合知识库提供智能客服支持

## 快速开始

### 安装

```bash
pip install llama-index
```

### 基础示例

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 加载数据
documents = SimpleDirectoryReader("data").load_data()

# 构建索引
index = VectorStoreIndex.from_documents(documents)

# 查询
query_engine = index.as_query_engine()
response = query_engine.query("LlamaIndex 的主要特性是什么？")
```

这个简单的示例展示了 LlamaIndex 的核心工作流程：数据加载 → 索引构建 → 查询检索。