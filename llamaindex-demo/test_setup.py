"""
测试脚本 - 验证环境和依赖
"""

import os
import sys
from pathlib import Path

def test_environment():
    """测试环境配置"""
    print("🧪 测试环境配置")
    print("=" * 40)
    
    # 检查 Python 版本
    python_version = sys.version
    print(f"🐍 Python 版本: {python_version}")
    
    # 检查当前工作目录
    cwd = Path.cwd()
    print(f"📁 当前目录: {cwd}")
    
    # 检查必要文件
    required_files = [
        "data/sample_doc1.md",
        "data/sample_doc2.md", 
        "data/sample_doc3.md",
        "examples/basic_rag.py",
        "src/enterprise_kb.py",
        "web_app.py"
    ]
    
    print("\n📋 检查必要文件:")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
    
    # 检查环境变量
    print("\n🔐 检查环境变量:")
    env_vars = ["OPENAI_API_KEY"]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {'*' * 10}...{value[-4:]}")
        else:
            print(f"  ⚠️  {var}: 未设置")

def test_imports():
    """测试依赖导入"""
    print("\n📦 测试依赖导入")
    print("=" * 40)
    
    dependencies = [
        ("llama_index", "LlamaIndex 核心库"),
        ("llama_index.core", "LlamaIndex 核心模块"),
        ("llama_index.llms.openai", "OpenAI LLM"),
        ("llama_index.embeddings.openai", "OpenAI 嵌入"),
        ("streamlit", "Streamlit Web 框架"),
        ("fastapi", "FastAPI Web 框架"),
        ("pandas", "Pandas 数据处理"),
        ("PIL", "Pillow 图像处理"),
        ("dotenv", "python-dotenv 环境变量")
    ]
    
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"  ✅ {module}: {description}")
        except ImportError as e:
            print(f"  ❌ {module}: {description} - {e}")

def test_basic_functionality():
    """测试基础功能"""
    print("\n🔧 测试基础功能")
    print("=" * 40)
    
    try:
        # 测试文档加载
        from llama_index.core import SimpleDirectoryReader
        
        print("📚 测试文档加载...")
        documents = SimpleDirectoryReader("./data").load_data()
        print(f"  ✅ 成功加载 {len(documents)} 个文档")
        
        # 测试索引构建
        from llama_index.core import VectorStoreIndex
        
        print("🏗️ 测试索引构建...")
        index = VectorStoreIndex.from_documents(documents)
        print("  ✅ 索引构建成功")
        
        # 测试查询引擎
        print("🔍 测试查询引擎...")
        query_engine = index.as_query_engine()
        response = query_engine.query("LlamaIndex 是什么？")
        print(f"  ✅ 查询成功: {response.response[:100]}...")
        
        print("\n🎉 基础功能测试通过！")
        return True
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🌟 LlamaIndex 演示项目环境测试")
    print("=" * 50)
    
    # 测试环境
    test_environment()
    
    # 测试导入
    test_imports()
    
    # 测试基础功能（仅在设置了 API 密钥时）
    if os.getenv("OPENAI_API_KEY"):
        test_basic_functionality()
    else:
        print("\n⚠️ 跳过功能测试（需要 OPENAI_API_KEY）")
    
    print("\n📝 测试完成！")
    print("💡 如果所有测试通过，可以运行 ./run_examples.sh 开始演示")

if __name__ == "__main__":
    main()