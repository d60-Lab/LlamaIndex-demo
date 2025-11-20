"""
LLM 工厂类
支持多种 LLM 提供商：Ollama、DeepSeek、OpenAI、Poe 等
"""

import os
from typing import Optional
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

# 加载环境变量
load_dotenv()

class LLMFactory:
    """LLM 工厂类，根据配置创建不同的 LLM 实例"""
    
    @staticmethod
    def get_llm(provider: Optional[str] = None, model: Optional[str] = None):
        """获取 LLM 实例"""
        provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        
        if provider.lower() == "ollama":
            return LLMFactory._create_ollama_llm(model)
        elif provider.lower() == "deepseek":
            return LLMFactory._create_deepseek_llm(model)
        elif provider.lower() == "openai":
            return LLMFactory._create_openai_llm(model)
        elif provider.lower() == "poe":
            return LLMFactory._create_poe_llm(model)
        else:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")
    
    @staticmethod
    def get_embedding_model(provider: Optional[str] = None, model: Optional[str] = None):
        """获取嵌入模型实例"""
        provider = provider or os.getenv("EMBEDDING_PROVIDER", "ollama")
        
        if provider.lower() == "ollama":
            return LLMFactory._create_ollama_embedding(model)
        elif provider.lower() == "openai":
            return LLMFactory._create_openai_embedding(model)
        else:
            raise ValueError(f"不支持的嵌入模型提供商: {provider}")
    
    @staticmethod
    def _create_ollama_llm(model: Optional[str] = None):
        """创建 Ollama LLM"""
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = model or os.getenv("OLLAMA_MODEL", "deepseek-r1")
        
        try:
            return Ollama(
                model=model_name,
                base_url=base_url,
                temperature=0.1,
                request_timeout=120.0
            )
        except Exception as e:
            raise ValueError(f"无法连接到 Ollama: {e}\n请确保 Ollama 已安装并运行在 {base_url}")
    
    @staticmethod
    def _create_ollama_embedding(model: Optional[str] = None):
        """创建 Ollama 嵌入模型"""
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        
        try:
            return OllamaEmbedding(
                model_name=model_name,
                base_url=base_url
            )
        except Exception as e:
            raise ValueError(f"无法连接到 Ollama 嵌入模型: {e}\n请确保 Ollama 已安装并运行在 {base_url}")
    
    @staticmethod
    def _create_poe_llm(model: Optional[str] = None):
        """创建 Poe LLM"""
        api_key = os.getenv("POE_API_KEY")
        base_url = os.getenv("POE_BASE_URL", "https://api.poe.com")
        
        if not api_key:
            raise ValueError("请设置 POE_API_KEY 环境变量")
        
        model_name = model or os.getenv("POE_MODEL", "claude-3-haiku-20240307")
        
        # Poe 使用 OpenAI 兼容接口
        return OpenAI(
            model=model_name,
            api_key=api_key,
            api_base=base_url,
            temperature=0.1,
            max_tokens=4096
        )
    
    @staticmethod
    def _create_deepseek_llm(model: Optional[str] = None):
        """创建 DeepSeek LLM"""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        
        if not api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
        
        # DeepSeek 模型映射
        model_mapping = {
            "fast": "deepseek-chat",
            "reasoning": "deepseek-reasoner",
            "code": "deepseek-coder"
        }
        
        if model and model in model_mapping:
            model_name = model_mapping[model]
        else:
            model_name = model or "deepseek-chat"
        
        return OpenAI(
            model=model_name,
            api_key=api_key,
            api_base=base_url,
            temperature=0.1,
            max_tokens=4096
        )
    
    @staticmethod
    def _create_openai_llm(model: Optional[str] = None):
        """创建 OpenAI LLM"""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
        
        model_name = model or "gpt-3.5-turbo"
        
        return OpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.1,
            max_tokens=4096
        )
    
    @staticmethod
    def _create_openai_embedding(model: Optional[str] = None):
        """创建 OpenAI 嵌入模型"""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("嵌入模型需要设置 OPENAI_API_KEY 环境变量")
        
        model_name = model or "text-embedding-3-small"
        
        return OpenAIEmbedding(
            model=model_name,
            api_key=api_key
        )
    
    @staticmethod
    def get_available_providers():
        """获取可用的提供商列表"""
        providers = []
        
        # 检查 Ollama（本地服务）
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                providers.append("ollama")
        except:
            pass
        
        if os.getenv("DEEPSEEK_API_KEY"):
            providers.append("deepseek")
        
        if os.getenv("OPENAI_API_KEY"):
            providers.append("openai")
        
        if os.getenv("POE_API_KEY"):
            providers.append("poe")
        
        return providers
    
    @staticmethod
    def get_provider_info():
        """获取提供商信息"""
        info = {
            "available_providers": LLMFactory.get_available_providers(),
            "current_llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
            "current_embedding_provider": os.getenv("EMBEDDING_PROVIDER", "ollama"),
            "models": {
                "ollama": {
                    "chat": "deepseek-r1",
                    "embedding": "nomic-embed-text"
                },
                "deepseek": {
                    "chat": "deepseek-chat",
                    "reasoner": "deepseek-reasoner", 
                    "coder": "deepseek-coder"
                },
                "openai": {
                    "chat": "gpt-3.5-turbo",
                    "advanced": "gpt-4-turbo",
                    "embedding": "text-embedding-3-small"
                },
                "poe": {
                    "chat": "claude-3-haiku-20240307",
                    "advanced": "claude-3-sonnet-20240229"
                }
            }
        }
        return info

def test_llm_connection():
    """测试 LLM 连接"""
    print("🧪 测试 LLM 连接")
    print("=" * 40)
    
    try:
        # 获取提供商信息
        info = LLMFactory.get_provider_info()
        print(f"📋 可用提供商: {info['available_providers']}")
        print(f"🎯 当前 LLM 提供商: {info['current_llm_provider']}")
        print(f"🔤 当前嵌入提供商: {info['current_embedding_provider']}")
        
        # 测试 LLM
        llm = LLMFactory.get_llm()
        print(f"✅ LLM 初始化成功: {llm.model}")
        
        # 测试简单对话
        response = llm.complete("你好，请简单介绍一下你自己。")
        print(f"💬 测试对话: {response.text[:100]}...")
        
        # 测试嵌入模型
        embedding = LLMFactory.get_embedding_model()
        print(f"✅ 嵌入模型初始化成功: {embedding.model_name}")
        
        # 测试嵌入
        text_embedding = embedding.get_text_embedding("这是一个测试文本")
        print(f"🔢 嵌入维度: {len(text_embedding)}")
        
        print("\n🎉 所有连接测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return False

if __name__ == "__main__":
    test_llm_connection()