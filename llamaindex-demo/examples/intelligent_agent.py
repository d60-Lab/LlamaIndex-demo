import os
import random
import time

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# 加载环境变量
load_dotenv()


def simulate_database_query(query: str) -> str:
    """模拟数据库查询"""
    # 模拟一些销售数据
    sales_data = {
        "Q1 2024": "120万美元",
        "Q2 2024": "135万美元",
        "Q3 2024": "148万美元",
        "Q4 2024": "162万美元",
        "年度总计": "565万美元",
    }

    if "Q1" in query:
        return f"2024年Q1销售额为{sales_data['Q1 2024']}"
    elif "Q2" in query:
        return f"2024年Q2销售额为{sales_data['Q2 2024']}"
    elif "Q3" in query:
        return f"2024年Q3销售额为{sales_data['Q3 2024']}"
    elif "Q4" in query:
        return f"2024年Q4销售额为{sales_data['Q4 2024']}"
    else:
        return f"2024年年度总销售额为{sales_data['年度总计']}"


def multiply(a: float, b: float) -> float:
    """两个数相乘"""
    return a * b


def add(a: float, b: float) -> float:
    """两个数相加"""
    return a + b


def get_current_time() -> str:
    """获取当前时间"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def generate_sales_forecast(current_sales: str, growth_rate: float) -> str:
    """生成销售预测"""
    # 提取数字
    import re

    numbers = re.findall(r"[\d.]+", current_sales)
    if numbers:
        base_sales = float(numbers[0])
        forecast = base_sales * growth_rate
        return f"基于当前销售额{base_sales}万美元和增长率{growth_rate}，预测销售额为{forecast:.2f}万美元"
    return "无法解析当前销售额数据"


def create_knowledge_base():
    """创建知识库"""
    print("📚 初始化知识库...")
    documents = SimpleDirectoryReader("./data").load_data()
    
    # 使用 Ollama 嵌入模型
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    # 使用较小的文档块
    from llama_index.core.node_parser import SentenceSplitter
    parser = SentenceSplitter(chunk_size=256, chunk_overlap=25)
    nodes = parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
    
    # 使用 Ollama LLM
    llm = Ollama(
        model="deepseek-r1",
        base_url="http://localhost:11434",
        temperature=0.1,
        request_timeout=120.0
    )
    
    query_engine = index.as_query_engine(
        similarity_top_k=3, 
        llm=llm
    )
    return query_engine


def main():
    """智能代理主函数"""

    # 1. 创建知识库
    kb_query_engine = create_knowledge_base()

    # 2. 定义工具函数
    print("🔧 配置代理工具...")

    # 知识库搜索工具
    search_tool = QueryEngineTool.from_defaults(
        query_engine=kb_query_engine,
        name="knowledge_search",
        description="搜索知识库，回答关于LlamaIndex、RAG系统、AI应用开发等问题",
    )

    # 数据库查询工具
    database_tool = FunctionTool.from_defaults(
        fn=simulate_database_query,
        name="database_query",
        description="查询销售数据库，获取各季度销售数据",
    )

    # 计算工具
    multiply_tool = FunctionTool.from_defaults(
        fn=multiply, name="multiply", description="计算两个数的乘积"
    )

    add_tool = FunctionTool.from_defaults(
        fn=add, name="add", description="计算两个数的和"
    )

    # 时间工具
    time_tool = FunctionTool.from_defaults(
        fn=get_current_time, name="get_current_time", description="获取当前日期和时间"
    )

    # 预测工具
    forecast_tool = FunctionTool.from_defaults(
        fn=generate_sales_forecast,
        name="sales_forecast",
        description="基于当前销售额和增长率预测未来销售额",
    )

    # 3. 设置记忆系统
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    # 4. 初始化 ReAct 代理
    print("🤖 初始化智能代理...")
    
    # 使用 Ollama LLM
    llm = Ollama(
        model="deepseek-r1",
        base_url="http://localhost:11434",
        temperature=0.1,
        request_timeout=120.0
    )
    
    agent = ReActAgent(
        tools=[
            search_tool,
            database_tool,
            multiply_tool,
            add_tool,
            time_tool,
            forecast_tool,
        ],
        llm=llm,
        memory=memory,
        verbose=True,
        max_iterations=10,
        system_prompt=(
            "你是一个专业的AI助手，专门帮助用户处理业务分析和技术问题。"
            "你可以搜索知识库、查询数据库、进行数学计算和生成预测。"
            "请始终提供准确、有用的回答，并在需要时使用适当的工具。"
        ),
    )

    print("✅ 智能代理已就绪！")
    print("\n🚀 开始对话吧（输入 'quit' 退出）:")
    print("=" * 60)
    print("💡 你可以尝试以下类型的提问:")
    print("  - 技术问题: LlamaIndex的主要特性是什么？")
    print("  - 数据查询: 查询Q3的销售额")
    print("  - 计算任务: 计算120乘以1.15")
    print("  - 复合任务: 查询Q3销售额，然后预测增长15%后的结果")
    print("  - 时间查询: 现在是什么时间？")
    print("=" * 60)

    # 5. 交互对话
    while True:
        try:
            user_input = input("\n👤 用户: ").strip()

            if user_input.lower() in ["quit", "exit", "退出"]:
                print("🤖 代理: 再见！很高兴为您服务。")
                break

            if not user_input:
                continue

            print("🤖 代理: ", end="", flush=True)
            response = agent.chat(user_input)

        except KeyboardInterrupt:
            print("\n🤖 代理: 再见！很高兴为您服务。")
            break
        except Exception as e:
            print(f"\n❌ 出现错误: {e}")
            print("🤖 代理: 抱歉，我遇到了一些问题。请重新尝试。")


if __name__ == "__main__":
    main()
