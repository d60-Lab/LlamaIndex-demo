"""
企业知识库 Web 应用
基于 Streamlit 的用户界面
"""

import streamlit as st
import time
import os
from typing import Dict, List
import pandas as pd

# 设置页面配置
st.set_page_config(
    page_title="企业知识库问答系统",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入企业知识库
from src.enterprise_kb import EnterpriseKnowledgeBase

def initialize_session_state():
    """初始化会话状态"""
    if 'kb' not in st.session_state:
        st.session_state.kb = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.title("🏢 知识库设置")
    
    # 初始化按钮
    if st.sidebar.button("🚀 初始化知识库", type="primary"):
        with st.sidebar.spinner("正在初始化知识库..."):
            try:
                kb = EnterpriseKnowledgeBase()
                kb.initialize()
                st.session_state.kb = kb
                st.session_state.initialized = True
                st.sidebar.success("✅ 知识库初始化成功！")
                
                # 显示统计信息
                stats = kb.get_statistics()
                st.sidebar.info(f"📊 统计信息:\n"
                              f"文档数量: {stats.get('total_documents', '未知')}\n"
                              f"存储大小: {stats.get('storage_size_mb', '未知')} MB")
            except Exception as e:
                st.sidebar.error(f"❌ 初始化失败: {e}")
    
    # 知识库状态
    if st.session_state.initialized:
        st.sidebar.success("🟢 知识库已就绪")
        
        # 显示高级选项
        with st.sidebar.expander("🔧 高级选项"):
            # 部门过滤器
            departments = ['all', 'engineering', 'sales', 'marketing', 'hr', 'finance', 'product']
            selected_dept = st.selectbox("部门过滤", departments)
            
            # 文档类别过滤器
            categories = ['all', 'tutorial', 'documentation', 'policy', 'report', 'meeting']
            selected_category = st.selectbox("类别过滤", categories)
            
            # 相似度阈值
            similarity_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.7, 0.1)
            
            # 返回结果数量
            top_k = st.slider("返回结果数", 1, 10, 3)
            
            return {
                'department': selected_dept if selected_dept != 'all' else None,
                'category': selected_category if selected_category != 'all' else None,
                'similarity_threshold': similarity_threshold,
                'top_k': top_k
            }
    else:
        st.sidebar.warning("🔴 知识库未初始化")
        return None

def render_chat_interface():
    """渲染聊天界面"""
    st.title("💬 智能问答")
    
    if not st.session_state.initialized:
        st.warning("⚠️ 请先在侧边栏初始化知识库")
        return
    
    # 获取高级选项
    filters = {}
    advanced_options = render_sidebar()
    if advanced_options:
        if advanced_options['department']:
            filters['department'] = advanced_options['department']
        if advanced_options['category']:
            filters['category'] = advanced_options['category']
    
    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.chat_message("user").write(message['content'])
            else:
                st.chat_message("assistant").write(message['content'])
                
                # 显示来源信息
                if message.get('sources'):
                    with st.expander("📚 参考来源"):
                        for i, source in enumerate(message['sources'], 1):
                            st.write(f"**{i}. {source['file_name']}**")
                            st.write(f"   部门: {source['department']} | 类别: {source['category']}")
                            st.write(f"   相关度: {source['relevance']:.2f}")
                            st.write(f"   内容片段: {source['snippet'][:200]}...")
                            st.divider()
                
                # 显示元数据
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("置信度", f"{message.get('confidence', 0):.2f}")
                with col2:
                    st.metric("响应时间", f"{message.get('query_time', 0):.2f}s")
                with col3:
                    st.metric("来源数量", len(message.get('sources', [])))
    
    # 用户输入
    user_input = st.chat_input("请输入您的问题...")
    
    if user_input:
        # 添加用户消息到历史
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # 显示用户消息
        st.chat_message("user").write(user_input)
        
        # 处理查询
        with st.chat_message("assistant"):
            with st.spinner("🤔 思考中..."):
                try:
                    result = st.session_state.kb.query(
                        user_input,
                        filters=filters,
                        top_k=advanced_options['top_k'] if advanced_options else 3
                    )
                    
                    # 显示回答
                    st.write(result.answer)
                    
                    # 显示来源
                    if result.sources:
                        with st.expander("📚 参考来源"):
                            for i, source in enumerate(result.sources, 1):
                                st.write(f"**{i}. {source['file_name']}**")
                                st.write(f"   部门: {source['department']} | 类别: {source['category']}")
                                st.write(f"   相关度: {source['relevance']:.2f}")
                                st.write(f"   内容片段: {source['snippet'][:200]}...")
                                st.divider()
                    
                    # 显示指标
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("置信度", f"{result.confidence:.2f}")
                    with col2:
                        st.metric("响应时间", f"{result.query_time:.2f}s")
                    with col3:
                        st.metric("来源数量", len(result.sources))
                    
                    # 添加助手消息到历史
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': result.answer,
                        'sources': result.sources,
                        'confidence': result.confidence,
                        'query_time': result.query_time
                    })
                    
                except Exception as e:
                    st.error(f"❌ 查询失败: {e}")

def render_document_management():
    """渲染文档管理界面"""
    st.title("📄 文档管理")
    
    if not st.session_state.initialized:
        st.warning("⚠️ 请先在侧边栏初始化知识库")
        return
    
    # 知识库统计
    stats = st.session_state.kb.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("文档数量", stats.get('total_documents', 0))
    with col2:
        st.metric("存储大小", f"{stats.get('storage_size_mb', 0)} MB")
    with col3:
        st.metric("嵌入模型", stats.get('embedding_model', 'Unknown'))
    with col4:
        st.metric("LLM模型", stats.get('llm_model', 'Unknown'))
    
    # 文档上传
    st.subheader("📤 添加新文档")
    uploaded_files = st.file_uploader(
        "选择文档文件",
        type=['txt', 'md', 'pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("上传文档"):
        with st.spinner("正在上传文档..."):
            try:
                # 保存上传的文件
                file_paths = []
                for file in uploaded_files:
                    file_path = f"./data/{file.name}"
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    file_paths.append(file_path)
                
                # 添加到知识库
                st.session_state.kb.add_documents(file_paths)
                st.success(f"✅ 成功添加 {len(file_paths)} 个文档")
                
            except Exception as e:
                st.error(f"❌ 上传失败: {e}")

def render_analytics():
    """渲染分析界面"""
    st.title("📊 使用分析")
    
    if not st.session_state.initialized:
        st.warning("⚠️ 请先在侧边栏初始化知识库")
        return
    
    # 查询统计
    if st.session_state.chat_history:
        # 提取查询数据
        queries = [msg for msg in st.session_state.chat_history if msg['role'] == 'user']
        responses = [msg for msg in st.session_state.chat_history if msg['role'] == 'assistant']
        
        st.subheader("📈 查询统计")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("总查询数", len(queries))
            avg_confidence = sum(r.get('confidence', 0) for r in responses) / len(responses) if responses else 0
            st.metric("平均置信度", f"{avg_confidence:.2f}")
        
        with col2:
            avg_time = sum(r.get('query_time', 0) for r in responses) / len(responses) if responses else 0
            st.metric("平均响应时间", f"{avg_time:.2f}s")
            total_sources = sum(len(r.get('sources', [])) for r in responses)
            st.metric("总引用数", total_sources)
        
        # 查询历史
        st.subheader("📜 查询历史")
        if queries:
            history_data = []
            for i, query in enumerate(queries):
                response = responses[i] if i < len(responses) else {}
                history_data.append({
                    '时间': f"查询 {i+1}",
                    '问题': query['content'][:50] + "..." if len(query['content']) > 50 else query['content'],
                    '置信度': f"{response.get('confidence', 0):.2f}",
                    '响应时间': f"{response.get('query_time', 0):.2f}s",
                    '来源数': len(response.get('sources', []))
                })
            
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)
    else:
        st.info("📝 暂无查询记录")

def main():
    """主函数"""
    initialize_session_state()
    
    # 页面导航
    page = st.sidebar.selectbox(
        "选择页面",
        ["💬 智能问答", "📄 文档管理", "📊 使用分析"]
    )
    
    if page == "💬 智能问答":
        render_chat_interface()
    elif page == "📄 文档管理":
        render_document_management()
    elif page == "📊 使用分析":
        render_analytics()

if __name__ == "__main__":
    main()
