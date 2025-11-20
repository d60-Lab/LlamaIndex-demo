#!/bin/bash

# LlamaIndex 演示项目启动脚本

echo "🚀 LlamaIndex 演示项目"
echo "===================="

# 检查环境变量
if [ ! -f ".env" ]; then
    echo "⚠️  未找到 .env 文件，请先复制 .env.example 并配置 API 密钥"
    echo "   cp .env.example .env"
    exit 1
fi

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "📦 初始化虚拟环境..."
    rye sync
fi

echo "📋 可用的演示："
echo "1. 基础 RAG 应用"
echo "2. 智能代理"
echo "3. 多模态处理"
echo "4. 评估框架"
echo "5. 企业知识库 (命令行)"
echo "6. 企业知识库 (Web 界面)"
echo "7. 运行所有演示"

read -p "请选择要运行的演示 (1-7): " choice

case $choice in
    1)
        echo "🔍 启动基础 RAG 应用..."
        rye run python examples/basic_rag.py
        ;;
    2)
        echo "🤖 启动智能代理..."
        rye run python examples/intelligent_agent.py
        ;;
    3)
        echo "🌟 启动多模态处理演示..."
        rye run python examples/multimodal_demo.py
        ;;
    4)
        echo "📊 启动评估框架..."
        rye run python examples/evaluation_demo.py
        ;;
    5)
        echo "🏢 启动企业知识库 (命令行)..."
        rye run python src/enterprise_kb.py
        ;;
    6)
        echo "🌐 启动企业知识库 (Web 界面)..."
        rye run streamlit run web_app.py
        ;;
    7)
        echo "🎯 运行所有演示..."
        echo "正在运行: 基础 RAG 应用"
        rye run python examples/basic_rag.py
        echo ""
        echo "正在运行: 智能代理"
        rye run python examples/intelligent_agent.py
        echo ""
        echo "正在运行: 多模态处理"
        rye run python examples/multimodal_demo.py
        echo ""
        echo "正在运行: 评估框架"
        rye run python examples/evaluation_demo.py
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac