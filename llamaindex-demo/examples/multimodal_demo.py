"""
多模态处理示例
演示如何使用 LlamaIndex 处理图像和文本数据
"""

import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageDocument
from llama_index.llms.openai import OpenAI
import base64
from PIL import Image
import io

# 加载环境变量
load_dotenv()

def create_sample_image():
    """创建一个示例图表图像（模拟销售数据图表）"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # 创建一个简单的柱状图
        width, height = 400, 300
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制坐标轴
        draw.line([(50, 250), (350, 250)], fill='black', width=2)  # X轴
        draw.line([(50, 50), (50, 250)], fill='black', width=2)   # Y轴
        
        # 绘制柱状图
        data = [120, 135, 148, 162]  # Q1-Q4销售额
        colors = ['blue', 'green', 'red', 'orange']
        labels = ['Q1', 'Q2', 'Q3', 'Q4']
        
        for i, (value, color, label) in enumerate(zip(data, colors, labels)):
            x = 80 + i * 70
            bar_height = int(value * 1.2)  # 缩放因子
            draw.rectangle([x-20, 250-bar_height, x+20, 250], fill=color)
            draw.text((x-15, 260), label, fill='black')
            draw.text((x-15, 235-bar_height), str(value), fill='black')
        
        # 添加标题
        draw.text((120, 20), "2024年季度销售额（万美元）", fill='black')
        
        # 保存图像
        img_path = "./data/sales_chart.png"
        img.save(img_path)
        print(f"✅ 创建示例图表: {img_path}")
        return img_path
        
    except ImportError:
        print("❌ PIL 库未安装，无法创建示例图像")
        return None

def analyze_image_with_text(image_path: str, question: str):
    """使用多模态LLM分析图像"""
    print(f"🔍 分析图像: {image_path}")
    print(f"❓ 问题: {question}")
    
    try:
        # 加载图像文档
        image_documents = SimpleDirectoryReader(
            input_files=[image_path]
        ).load_data()
        
        # 初始化多模态LLM
        multimodal_llm = OpenAIMultiModal(model="gpt-4-vision-preview")
        
        # 分析图像
        response = multimodal_llm.complete(
            prompt=question,
            image_documents=image_documents
        )
        
        print(f"💡 分析结果:")
        print(response.text)
        return response.text
        
    except Exception as e:
        print(f"❌ 图像分析失败: {e}")
        return None

def create_mixed_index():
    """创建包含文本和图像的混合索引"""
    print("📚 创建混合模态索引...")
    
    # 加载文本文档
    text_documents = SimpleDirectoryReader(
        input_dir="./data",
        required_exts=[".md"]
    ).load_data()
    
    # 加载图像文档
    image_documents = []
    image_files = ["./data/sales_chart.png"]
    
    for img_file in image_files:
        if os.path.exists(img_file):
            img_doc = ImageDocument(image_path=img_file)
            # 为图像添加描述性元数据
            img_doc.metadata = {
                "file_type": "image",
                "description": "2024年季度销售额柱状图",
                "content": "销售数据可视化"
            }
            image_documents.append(img_doc)
    
    # 合并所有文档
    all_documents = text_documents + image_documents
    
    # 创建索引
    index = VectorStoreIndex.from_documents(all_documents)
    query_engine = index.as_query_engine(
        llm=OpenAI(model="gpt-4-turbo"),
        similarity_top_k=3
    )
    
    print(f"✅ 混合索引创建完成，包含 {len(text_documents)} 个文本文档和 {len(image_documents)} 个图像文档")
    return query_engine

def text_to_image_analysis():
    """文本到图像的分析示例"""
    print("\n🎨 文本到图像分析示例")
    print("=" * 50)
    
    # 创建示例图像
    image_path = create_sample_image()
    if not image_path:
        print("❌ 无法创建示例图像，跳过多模态演示")
        return
    
    # 分析图像的不同方面
    questions = [
        "这张图表展示了什么数据？",
        "哪个季度的销售额最高？",
        "分析这个销售趋势，并给出预测",
        "用中文描述图表的主要信息"
    ]
    
    for question in questions:
        print(f"\n" + "="*50)
        analyze_image_with_text(image_path, question)

def multimodal_qa_demo():
    """多模态问答演示"""
    print("\n🤖 多模态问答演示")
    print("=" * 50)
    
    query_engine = create_mixed_index()
    
    test_questions = [
        "LlamaIndex的主要特性是什么？",
        "如何优化RAG系统的检索性能？",
        "销售数据表现如何？",
        "对比不同季度的销售表现"
    ]
    
    for question in test_questions:
        print(f"\n❓ 问题: {question}")
        print("🤔 思考中...")
        
        try:
            response = query_engine.query(question)
            print(f"💡 回答:")
            print(response.response)
            
            # 显示来源信息
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"\n📚 参考来源:")
                for i, node in enumerate(response.source_nodes, 1):
                    file_type = node.metadata.get('file_type', 'text')
                    file_name = node.metadata.get('file_name', '未知文件')
                    score = getattr(node, 'score', 0)
                    print(f"  {i}. {file_name} ({file_type}, 相关度: {score:.2f})")
                    
        except Exception as e:
            print(f"❌ 查询失败: {e}")

def main():
    """多模态处理主函数"""
    print("🌟 LlamaIndex 多模态处理演示")
    print("=" * 60)
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 请设置 OPENAI_API_KEY 环境变量")
        return
    
    try:
        # 文本到图像分析
        text_to_image_analysis()
        
        # 多模态问答
        multimodal_qa_demo()
        
        print("\n🎉 多模态演示完成！")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")

if __name__ == "__main__":
    main()