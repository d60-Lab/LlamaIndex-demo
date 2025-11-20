"""
简单的 Ollama 测试
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

def test_simple():
    print("🧪 简单 Ollama 测试")
    
    try:
        # 测试 LLM
        llm = Ollama(model="deepseek-r1", request_timeout=120.0)
        response = llm.complete("你好！")
        print(f"✅ LLM 测试成功: {response.text[:50]}...")
        
        # 测试嵌入
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
        )
        embedding = embed_model.get_text_embedding("测试文本")
        print(f"✅ 嵌入测试成功，维度: {len(embedding)}")
        
        # 测试完整的 RAG 流程
        print("\n📚 测试完整 RAG 流程...")
        documents = SimpleDirectoryReader("./data").load_data()
        print(f"加载了 {len(documents)} 个文档")
        
        # 使用较小的 chunk 避免嵌入问题
        from llama_index.core.node_parser import SentenceSplitter
        parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
        nodes = parser.get_nodes_from_documents(documents[:1])  # 只用第一个文档测试
        
        print(f"分割为 {len(nodes)} 个节点")
        
        # 构建索引
        index = VectorStoreIndex.from_documents(
            documents=documents[:1],  # 只用第一个文档
            embed_model=embed_model
        )
        
        # 查询
        query_engine = index.as_query_engine(llm=llm)
        response = query_engine.query("LlamaIndex 是什么？")
        
        print(f"✅ RAG 测试成功!")
        print(f"回答: {response.response[:100]}...")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple()
