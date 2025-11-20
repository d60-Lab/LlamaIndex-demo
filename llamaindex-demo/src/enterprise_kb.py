"""
企业知识库问答系统
基于 LlamaIndex 构建的企业级智能问答系统
"""

import os
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.vector_stores import MetadataFilters, FilterCondition

# 加载环境变量
load_dotenv()

@dataclass
class QueryResult:
    """查询结果数据类"""
    answer: str
    sources: List[Dict]
    confidence: float
    query_time: float

class EnterpriseKnowledgeBase:
    """企业知识库类"""
    
    def __init__(self, 
                 persist_dir: str = "./storage",
                 embed_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-3.5-turbo"):
        
        self.persist_dir = persist_dir
        self.embed_model = OpenAIEmbedding(model=embed_model)
        self.llm = OpenAI(model=llm_model)
        self.index = None
        self.query_engine = None
        
    def initialize(self, data_dir: str = "./data", force_rebuild: bool = False):
        """初始化知识库"""
        print("🚀 初始化企业知识库...")
        
        # 检查是否需要重建索引
        if force_rebuild or not os.path.exists(self.persist_dir):
            print("📚 构建新索引...")
            self._build_index(data_dir)
        else:
            print("📂 加载已有索引...")
            self._load_index()
        
        # 创建查询引擎
        self._create_query_engine()
        print("✅ 知识库初始化完成")
    
    def _build_index(self, data_dir: str):
        """构建新的索引"""
        # 加载文档
        documents = SimpleDirectoryReader(
            data_dir,
            recursive=True,
            required_exts=[".md", ".pdf", ".txt", ".docx"]
        ).load_data()
        
        if not documents:
            raise ValueError(f"在目录 {data_dir} 中未找到文档")
        
        # 增强文档元数据
        documents = self._enhance_documents(documents)
        
        # 文档分块
        parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,
            paragraph_separator="\n\n"
        )
        nodes = parser.get_nodes_from_documents(documents)
        
        # 构建索引
        self.index = VectorStoreIndex(
            nodes=nodes,
            embed_model=self.embed_model
        )
        
        # 持久化索引
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print(f"💾 索引已保存到 {self.persist_dir}")
    
    def _load_index(self):
        """加载已有索引"""
        storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
        self.index = load_index_from_storage(storage_context)
    
    def _enhance_documents(self, documents: List[Document]) -> List[Document]:
        """增强文档元数据"""
        enhanced_docs = []
        
        for doc in documents:
            # 提取文件信息
            file_path = doc.metadata.get('file_path', '')
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1]
            
            # 增强元数据
            enhanced_metadata = {
                **doc.metadata,
                'file_name': file_name,
                'file_type': file_ext,
                'doc_size': len(doc.text),
                'department': self._infer_department(file_name, doc.text),
                'category': self._infer_category(file_name, doc.text),
                'indexed_at': str(pd.Timestamp.now())
            }
            
            enhanced_doc = Document(
                text=doc.text,
                metadata=enhanced_metadata,
                id_=f"doc_{hash(doc.text)}"
            )
            enhanced_docs.append(enhanced_doc)
        
        return enhanced_docs
    
    def _infer_department(self, file_name: str, content: str) -> str:
        """推断文档所属部门"""
        content_lower = content.lower()
        file_name_lower = file_name.lower()
        
        departments = {
            'engineering': ['技术', '开发', '工程', '代码', 'api', '系统'],
            'sales': ['销售', '客户', '合同', '订单', '业绩'],
            'marketing': ['市场', '营销', '推广', '品牌', '活动'],
            'hr': ['人事', '招聘', '员工', '培训', '薪酬'],
            'finance': ['财务', '预算', '成本', '收入', '报表'],
            'product': ['产品', '需求', '功能', '设计', '用户']
        }
        
        for dept, keywords in departments.items():
            if any(keyword in content_lower or keyword in file_name_lower 
                   for keyword in keywords):
                return dept
        
        return 'general'
    
    def _infer_category(self, file_name: str, content: str) -> str:
        """推断文档类别"""
        content_lower = content.lower()
        
        categories = {
            'tutorial': ['教程', '指南', '入门', '如何', '步骤'],
            'documentation': ['文档', '说明', '参考', '手册'],
            'policy': ['政策', '规定', '制度', '流程'],
            'report': ['报告', '总结', '分析', '统计'],
            'meeting': ['会议', '纪要', '讨论', '决策']
        }
        
        for category, keywords in categories.items():
            if any(keyword in content_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _create_query_engine(self):
        """创建查询引擎"""
        # 创建检索器
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5
        )
        
        # 创建查询引擎
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ]
        )
    
    def query(self, 
              question: str, 
              filters: Optional[Dict] = None,
              top_k: int = 3) -> QueryResult:
        """执行查询"""
        import time
        start_time = time.time()
        
        try:
            # 应用过滤器
            if filters:
                metadata_filters = MetadataFilters(
                    filters=[
                        {"key": k, "value": v} 
                        for k, v in filters.items()
                    ],
                    condition=FilterCondition.AND
                )
                self.query_engine = self.index.as_query_engine(
                    filters=metadata_filters,
                    similarity_top_k=top_k
                )
            
            # 执行查询
            response = self.query_engine.query(question)
            
            # 提取来源信息
            sources = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes[:top_k]:
                    sources.append({
                        'file_name': node.metadata.get('file_name', '未知'),
                        'department': node.metadata.get('department', '未知'),
                        'category': node.metadata.get('category', '未知'),
                        'relevance': getattr(node, 'score', 0),
                        'snippet': node.text[:200] + "..." if len(node.text) > 200 else node.text
                    })
            
            # 计算置信度
            confidence = self._calculate_confidence(response, sources)
            
            query_time = time.time() - start_time
            
            return QueryResult(
                answer=response.response,
                sources=sources,
                confidence=confidence,
                query_time=query_time
            )
            
        except Exception as e:
            return QueryResult(
                answer=f"查询过程中出现错误: {str(e)}",
                sources=[],
                confidence=0.0,
                query_time=time.time() - start_time
            )
    
    def _calculate_confidence(self, response, sources: List[Dict]) -> float:
        """计算回答置信度"""
        if not sources:
            return 0.0
        
        # 基于来源相关度计算置信度
        relevance_scores = [src['relevance'] for src in sources if src['relevance'] > 0]
        if not relevance_scores:
            return 0.0
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        return min(avg_relevance, 1.0)
    
    def add_documents(self, file_paths: List[str]):
        """添加新文档"""
        print(f"📄 添加 {len(file_paths)} 个新文档...")
        
        # 加载新文档
        new_documents = SimpleDirectoryReader(
            input_files=file_paths
        ).load_data()
        
        # 增强元数据
        new_documents = self._enhance_documents(new_documents)
        
        # 添加到索引
        for doc in new_documents:
            self.index.insert(doc)
        
        # 保存更新后的索引
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print("✅ 文档添加完成")
    
    def get_statistics(self) -> Dict:
        """获取知识库统计信息"""
        try:
            # 获取索引中的节点数量
            docstore = self.index.docstore
            node_count = len(docstore.docs)
            
            # 获取存储大小
            storage_size = 0
            if os.path.exists(self.persist_dir):
                for root, dirs, files in os.walk(self.persist_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            storage_size += os.path.getsize(file_path)
            
            return {
                'total_documents': node_count,
                'storage_size_mb': round(storage_size / (1024 * 1024), 2),
                'storage_path': self.persist_dir,
                'embedding_model': self.embed_model.model_name,
                'llm_model': self.llm.model
            }
            
        except Exception as e:
            return {'error': str(e)}

def interactive_demo():
    """交互式演示"""
    print("🏢 企业知识库问答系统")
    print("=" * 50)
    
    # 初始化知识库
    kb = EnterpriseKnowledgeBase()
    
    try:
        kb.initialize()
        
        # 显示统计信息
        stats = kb.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"  文档数量: {stats.get('total_documents', '未知')}")
        print(f"  存储大小: {stats.get('storage_size_mb', '未知')} MB")
        print(f"  嵌入模型: {stats.get('embedding_model', '未知')}")
        
        print("\n🚀 系统已就绪！开始提问吧（输入 'quit' 退出）:")
        print("💡 提示: 你可以使用过滤器，如 'department:engineering 技术问题'")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n❓ 请输入问题: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if not user_input:
                    continue
                
                # 解析过滤器
                filters = {}
                question = user_input
                
                if ':' in user_input:
                    parts = user_input.split(':', 1)
                    if len(parts) == 2 and len(parts[0].split()) == 1:
                        filter_key = parts[0].strip()
                        filter_value = parts[1].split()[0].strip()
                        question = ' '.join(parts[1].split()[1:])
                        filters[filter_key] = filter_value
                
                print("🤔 思考中...")
                result = kb.query(question, filters=filters)
                
                print(f"\n💡 回答:")
                print(result.answer)
                
                print(f"\n📊 置信度: {result.confidence:.2f} | ⏱️ 用时: {result.query_time:.2f}s")
                
                if result.sources:
                    print(f"\n📚 参考来源:")
                    for i, source in enumerate(result.sources, 1):
                        print(f"  {i}. {source['file_name']} "
                              f"({source['department']} | {source['category']}) "
                              f"[相关度: {source['relevance']:.2f}]")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 出现错误: {e}")
    
    except Exception as e:
        print(f"❌ 初始化失败: {e}")

if __name__ == "__main__":
    # 需要安装 pandas 用于时间戳
    try:
        import pandas as pd
    except ImportError:
        print("⚠️ 需要安装 pandas: pip install pandas")
        exit(1)
    
    interactive_demo()