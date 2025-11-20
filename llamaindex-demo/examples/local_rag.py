"""
纯本地 RAG 应用示例
使用 Ollama 本地模型，无需外部 API
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# 加载环境变量
load_dotenv()

def main():
    """纯本地 RAG 应用主函数"""
    
    print("🚀 纯本地 RAG 应用演示")
    print("=" * 50)
    
    # 1. 初始化本地模型
    print("🤖 初始化本地模型...")
    
    try:
        # LLM 模型
        llm = Ollama(
            model="deepseek-r1",
            base_url="http://localhost:11434",
            temperature=0.1,
            request_timeout=120.0
        )
        print(f"✅ LLM 模型: {llm.model}")
        
        # 嵌入模型
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        print(f"✅ 嵌入模型: {embed_model.model_name}")
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        print("💡 请确保 Ollama 正在运行: ollama serve")
        return
    
    # 2. 加载文档
    print("\n📁 加载文档数据...")
    try:
        documents = SimpleDirectoryReader("./data").load_data()
        print(f"✅ 成功加载 {len(documents)} 个文档")
    except Exception as e:
        print(f"❌ 文档加载失败: {e}")
        return
    
    # 3. 文档分块（使用较小的块大小）
    print("\n🔧 处理文档分块...")
    parser = SentenceSplitter(
        chunk_size=256,        # 较小的块大小
        chunk_overlap=25,      # 重叠部分
        paragraph_separator="\n\n"
    )
    
    try:
        nodes = parser.get_nodes_from_documents(documents)
        print(f"✅ 文档被分割为 {len(nodes)} 个节点")
    except Exception as e:
        print(f"❌ 文档分块失败: {e}")
        return
    
    # 4. 构建索引
    print("\n🏗️ 构建向量索引...")
    try:
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embed_model
        )
        print("✅ 索引构建完成")
    except Exception as e:
        print(f"❌ 索引构建失败: {e}")
        print("💡 尝试使用更小的文档集或检查嵌入模型")
        return
    
    # 5. 创建查询引擎
    print("\n🔍 初始化查询引擎...")
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
        llm=llm,
    )
    
    # 6. 交互查询
    print("\n🚀 本地 RAG 系统已就绪！开始提问吧（输入 'quit' 退出）:")
    print("=" * 60)
    
    # 预设一些测试问题
    test_questions = [
        "LlamaIndex 的主要特性是什么？",
        "如何优化 RAG 系统的检索性能？",
        "什么是 ReAct 代理？"
    ]
    
    print("💡 你可以尝试以下问题:")
    for i, q in enumerate(test_questions, 1):
        print(f"  {i}. {q}")
    print()
    
    while True:
        try:
            question = input("❓ 请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
                
            if not question:
                continue
            
            # 如果用户输入数字，使用预设问题
            if question.isdigit() and 1 <= int(question) <= len(test_questions):
                question = test_questions[int(question) - 1]
                print(f"🎯 使用预设问题: {question}")
            
            print("🤔 思考中...")
            start_time = os.times()[4]
            
            response = query_engine.query(question)
            
            end_time = os.times()[4]
            query_time = end_time - start_time
            
            print(f"\n💡 回答:")
            print(response.response)
            
            print(f"\n⏱️ 用时: {query_time:.2f}秒")
            
            # 显示来源文档
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"\n📚 参考来源:")
                for i, node in enumerate(response.source_nodes, 1):
                    file_name = node.metadata.get('file_name', '未知文件')
                    score = getattr(node, 'score', 0)
                    snippet = node.text[:100] + "..." if len(node.text) > 100 else node.text
                    print(f"  {i}. {file_name} (相关度: {score:.2f})")
                    print(f"     片段: {snippet}")
                    
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 出现错误: {e}")
            print("💡 请检查 Ollama 服务是否正常运行")

if __name__ == "__main__":
    main()
