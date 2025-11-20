# LlamaIndex 演示项目

这是一个基于 LlamaIndex 的企业级 LLM 应用开发框架演示项目，展示了如何构建生产就绪的 RAG（检索增强生成）系统和智能代理。

## 🚀 快速开始

### 环境准备

1. **安装 Rye**（推荐）：
```bash
curl -sSf https://rye.astral.sh/get | bash
```

2. **克隆项目并初始化**：
```bash
git clone <repository-url>
cd llamaindex-demo
rye sync
```

3. **配置环境变量**：
```bash
cp .env.example .env
# 编辑 .env 文件，添加你的 API 密钥
```

### 运行示例

#### 1. 基础 RAG 应用
```bash
cd llamaindex-demo
rye run python examples/basic_rag.py
```

#### 2. 智能代理
```bash
rye run python examples/intelligent_agent.py
```

#### 3. 多模态处理
```bash
rye run python examples/multimodal_demo.py
```

#### 4. 评估框架
```bash
rye run python examples/evaluation_demo.py
```

#### 5. 企业知识库系统
```bash
# 命令行版本
rye run python src/enterprise_kb.py

# Web 界面
rye run streamlit run web_app.py
```

## 📁 项目结构

```
llamaindex-demo/
├── data/                   # 示例文档数据
│   ├── sample_doc1.md
│   ├── sample_doc2.md
│   └── sample_doc3.md
├── examples/               # 示例代码
│   ├── basic_rag.py       # 基础 RAG 应用
│   ├── intelligent_agent.py # 智能代理
│   ├── multimodal_demo.py  # 多模态处理
│   └── evaluation_demo.py  # 评估框架
├── src/                    # 核心代码
│   └── enterprise_kb.py   # 企业知识库
├── web_app.py             # Streamlit Web 应用
├── README.md              # 项目文档
├── .env.example           # 环境变量示例
└── pyproject.toml         # 项目配置
```

## 🎯 功能特性

### 1. 基础 RAG 应用
- ✅ 文档加载和处理
- ✅ 向量索引构建
- ✅ 智能检索和生成
- ✅ 来源引用显示

### 2. 智能代理系统
- ✅ ReAct 代理模式
- ✅ 多工具集成
- ✅ 记忆系统
- ✅ 复杂任务处理

### 3. 多模态处理
- ✅ 图像理解
- ✅ 文本-图像混合检索
- ✅ 多模态问答

### 4. 评估框架
- ✅ 忠实度评估
- ✅ 相关性评估
- ✅ 正确性评估
- ✅ 批量评估

### 5. 企业知识库
- ✅ 文档管理
- ✅ 元数据增强
- ✅ 过滤检索
- ✅ Web 界面
- ✅ 使用分析

## 🔧 配置说明

### 环境变量

在 `.env` 文件中配置以下变量：

```bash
# 必需
OPENAI_API_KEY=sk-your-openai-api-key

# 可选
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key
COHERE_API_KEY=your-cohere-api-key
PINECONE_API_KEY=your-pinecone-api-key
```

### 自定义配置

可以在代码中自定义以下配置：

- **嵌入模型**: `text-embedding-3-small`, `text-embedding-3-large`
- **LLM 模型**: `gpt-3.5-turbo`, `gpt-4-turbo`
- **分块大小**: 默认 512 tokens
- **检索数量**: 默认 3 个结果

## 📊 使用示例

### 基础查询

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 构建索引
index = VectorStoreIndex.from_documents(documents)

# 查询
query_engine = index.as_query_engine()
response = query_engine.query("LlamaIndex 的主要特性是什么？")
print(response.response)
```

### 智能代理

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

def multiply(a: float, b: float) -> float:
    return a * b

tool = FunctionTool.from_defaults(fn=multiply)
agent = ReActAgent.from_tools([tool])

response = agent.chat("计算 120 乘以 1.15")
```

## 🧪 测试和评估

项目包含完整的评估框架，可以衡量 RAG 系统的性能：

- **忠实度**: 回答是否基于检索到的上下文
- **相关性**: 回答是否与查询相关
- **正确性**: 回答是否准确（需要参考答案）
- **语义相似度**: 与参考答案的语义相似程度

运行评估：
```bash
rye run python examples/evaluation_demo.py
```

## 🌐 Web 应用

启动 Web 界面：
```bash
rye run streamlit run web_app.py
```

Web 应用功能：
- 💬 智能问答界面
- 📄 文档管理
- 📊 使用分析
- 🔧 高级配置

## 📚 学习资源

- [LlamaIndex 官方文档](https://docs.llamaindex.ai/)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
- [RAG 系统最佳实践](https://docs.llamaindex.ai/en/stable/optimizing/rag/rag_optimizing.html)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 🆘 常见问题

### Q: 如何添加新的文档类型？
A: 在 `SimpleDirectoryReader` 中添加 `required_exts` 参数支持新的文件扩展名。

### Q: 如何优化检索性能？
A: 可以通过以下方式优化：
- 使用混合检索
- 调整分块策略
- 使用重排序模型
- 选择合适的向量数据库

### Q: 如何处理中文文档？
A: 确保使用支持中文的嵌入模型，如 `text-embedding-3-small` 对中文支持良好。

---

🚀 开始探索 LlamaIndex 的强大功能吧！