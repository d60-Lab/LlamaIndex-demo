"""
基础 RAG 应用示例
演示如何使用 LlamaIndex 构建简单的文档问答系统
"""

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# 加载环境变量
load_dotenv()

def main():
    """基础 RAG 应用主函数"""
    
    # 1. 数据加载
    print("📁 加载文档数据...")
    documents = SimpleDirectoryReader("./data").load_data()
    print(f"✅ 成功加载 {len(documents)} 个文档")
    
    # 2. 文档分块
    print("🔧 处理文档分块...")
    parser = SentenceSplitter(
        chunk_size=512,        # token 数量
        chunk_overlap=50,      # 重叠部分，保持上下文连贯
        paragraph_separator="\n\n"
    )
    nodes = parser.get_nodes_from_documents(documents)
    print(f"✅ 文档被分割为 {len(nodes)} 个节点")
    
    # 3. 构建索引
    print("🏗️ 构建向量索引...")
    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=OpenAIEmbedding(model="text-embedding-3-small")
    )
    print("✅ 索引构建完成")
    
    # 4. 创建查询引擎
    print("🔍 初始化查询引擎...")
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
        llm=OpenAI(model="gpt-3.5-turbo"),
    )
    
    # 5. 交互查询
    print("\n🚀 RAG 系统已就绪！开始提问吧（输入 'quit' 退出）:")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n❓ 请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
                
            if not question:
                continue
                
            print("🤔 思考中...")
            response = query_engine.query(question)
            
            print(f"\n💡 回答:")
            print(response.response)
            
            # 显示来源文档
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"\n📚 参考来源:")
                for i, node in enumerate(response.source_nodes, 1):
                    file_name = node.metadata.get('file_name', '未知文件')
                    score = getattr(node, 'score', 0)
                    print(f"  {i}. {file_name} (相关度: {score:.2f})")
                    
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 出现错误: {e}")

if __name__ == "__main__":
    main()
