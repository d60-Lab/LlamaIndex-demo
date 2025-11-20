"""
评估框架示例
演示如何使用 LlamaIndex 的评估工具来衡量 RAG 系统性能
"""

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    BatchEvalRunner
)
from llama_index.llms.openai import OpenAI
import pandas as pd
import asyncio
from typing import List, Dict
import json

# 加载环境变量
load_dotenv()

class RAGEvaluator:
    """RAG 系统评估器"""
    
    def __init__(self, query_engine):
        self.query_engine = query_engine
        self.llm = OpenAI(model="gpt-4-turbo")
        
        # 初始化评估器
        self.faithfulness_evaluator = FaithfulnessEvaluator(llm=self.llm)
        self.relevancy_evaluator = RelevancyEvaluator(llm=self.llm)
        self.correctness_evaluator = CorrectnessEvaluator(llm=self.llm)
        self.semantic_similarity_evaluator = SemanticSimilarityEvaluator(llm=self.llm)
    
    def evaluate_single_query(self, query: str, reference_answer: str = None):
        """评估单个查询"""
        print(f"🔍 评估查询: {query}")
        
        # 获取系统回答
        response = self.query_engine.query(query)
        
        # 评估结果
        results = {
            "query": query,
            "system_response": response.response,
            "reference_answer": reference_answer
        }
        
        try:
            # 忠实度评估（回答是否基于检索到的上下文）
            faith_result = self.faithfulness_evaluator.evaluate_response(response=response)
            results["faithfulness_score"] = faith_result.score
            results["faithfulness_feedback"] = faith_result.feedback
            
        except Exception as e:
            print(f"❌ 忠实度评估失败: {e}")
            results["faithfulness_score"] = 0
            results["faithfulness_feedback"] = "评估失败"
        
        try:
            # 相关性评估（回答是否与查询相关）
            rel_result = self.relevancy_evaluator.evaluate_response(
                query=query, 
                response=response
            )
            results["relevancy_score"] = rel_result.score
            results["relevancy_feedback"] = rel_result.feedback
            
        except Exception as e:
            print(f"❌ 相关性评估失败: {e}")
            results["relevancy_score"] = 0
            results["relevancy_feedback"] = "评估失败"
        
        try:
            # 正确性评估（如果有参考答案）
            if reference_answer:
                corr_result = self.correctness_evaluator.evaluate_response(
                    query=query,
                    response=response,
                    reference=reference_answer
                )
                results["correctness_score"] = corr_result.score
                results["correctness_feedback"] = corr_result.feedback
            else:
                results["correctness_score"] = None
                results["correctness_feedback"] = "无参考答案"
                
        except Exception as e:
            print(f"❌ 正确性评估失败: {e}")
            results["correctness_score"] = 0
            results["correctness_feedback"] = "评估失败"
        
        try:
            # 语义相似度评估（如果有参考答案）
            if reference_answer:
                sem_result = self.semantic_similarity_evaluator.evaluate_response(
                    query=query,
                    response=response,
                    reference=reference_answer
                )
                results["semantic_similarity_score"] = sem_result.score
            else:
                results["semantic_similarity_score"] = None
                
        except Exception as e:
            print(f"❌ 语义相似度评估失败: {e}")
            results["semantic_similarity_score"] = 0
        
        return results
    
    async def batch_evaluate(self, test_dataset: List[Dict]):
        """批量评估查询"""
        print(f"📊 开始批量评估 {len(test_dataset)} 个查询...")
        
        # 准备查询和参考答案列表
        queries = [item["query"] for item in test_dataset]
        reference_answers = [item.get("reference_answer") for item in test_dataset]
        
        # 创建批量评估器
        runner = BatchEvalRunner(
            {
                "faithfulness": self.faithfulness_evaluator,
                "relevancy": self.relevancy_evaluator,
                "correctness": self.correctness_evaluator,
                "semantic_similarity": self.semantic_similarity_evaluator
            },
            workers=4
        )
        
        # 执行批量评估
        eval_results = await runner.aevaluate_queries(
            query_engine=self.query_engine,
            queries=queries,
            reference=reference_answers
        )
        
        return eval_results
    
    def generate_report(self, eval_results):
        """生成评估报告"""
        print("\n📋 评估报告")
        print("=" * 60)
        
        if isinstance(eval_results, dict) and "faithfulness" in eval_results:
            # 批量评估结果
            self._generate_batch_report(eval_results)
        else:
            # 单个评估结果
            self._generate_single_report(eval_results)
    
    def _generate_single_report(self, result):
        """生成单个查询的评估报告"""
        print(f"查询: {result['query']}")
        print(f"系统回答: {result['system_response']}")
        
        if result.get('reference_answer'):
            print(f"参考答案: {result['reference_answer']}")
        
        print(f"\n📊 评估分数:")
        print(f"  忠实度: {result.get('faithfulness_score', 'N/A')}/1.0")
        print(f"  相关性: {result.get('relevancy_score', 'N/A')}/1.0")
        print(f"  正确性: {result.get('correctness_score', 'N/A')}/1.0")
        print(f"  语义相似度: {result.get('semantic_similarity_score', 'N/A')}/1.0")
        
        print(f"\n💬 反馈:")
        if result.get('faithfulness_feedback'):
            print(f"  忠实度: {result['faithfulness_feedback']}")
        if result.get('relevancy_feedback'):
            print(f"  相关性: {result['relevancy_feedback']}")
        if result.get('correctness_feedback'):
            print(f"  正确性: {result['correctness_feedback']}")
    
    def _generate_batch_report(self, eval_results):
        """生成批量评估报告"""
        # 计算平均分数
        metrics = {}
        for metric_name, results in eval_results.items():
            scores = [r.score for r in results if hasattr(r, 'score')]
            if scores:
                metrics[metric_name] = {
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        print("📊 批量评估统计:")
        for metric_name, stats in metrics.items():
            print(f"  {metric_name}:")
            print(f"    平均分数: {stats['average']:.3f}")
            print(f"    最高分数: {stats['max']:.3f}")
            print(f"    最低分数: {stats['min']:.3f}")
            print(f"    评估数量: {stats['count']}")
        
        # 详细结果
        print(f"\n📋 详细结果:")
        for metric_name, results in eval_results.items():
            print(f"\n{metric_name} 详细结果:")
            for i, result in enumerate(results):
                if hasattr(result, 'query') and hasattr(result, 'score'):
                    print(f"  {i+1}. 查询: {result.query[:50]}...")
                    print(f"     分数: {result.score:.3f}")
                    if hasattr(result, 'feedback') and result.feedback:
                        print(f"     反馈: {result.feedback[:100]}...")

def create_test_dataset():
    """创建测试数据集"""
    return [
        {
            "query": "LlamaIndex的主要特性是什么？",
            "reference_answer": "LlamaIndex的主要特性包括开箱即用、丰富集成、生产就绪、智能代理和可扩展性。"
        },
        {
            "query": "如何优化RAG系统的检索性能？",
            "reference_answer": "可以通过混合检索、重排序、查询变换、优化分块策略和使用专业向量数据库来优化RAG系统的检索性能。"
        },
        {
            "query": "LlamaIndex支持哪些类型的索引？",
            "reference_answer": "LlamaIndex支持向量索引、树形索引、知识图谱索引、列表索引等多种索引类型。"
        },
        {
            "query": "什么是ReAct代理？",
            "reference_answer": "ReAct代理是一种推理-行动循环的代理模式，适合需要多步推理的任务。"
        },
        {
            "query": "如何评估RAG系统的性能？",
            "reference_answer": "可以使用忠实度、相关性、正确性等评估指标来衡量RAG系统的性能。"
        }
    ]

def create_query_engine():
    """创建查询引擎"""
    print("🏗️ 初始化查询引擎...")
    
    # 加载文档
    documents = SimpleDirectoryReader("./data").load_data()
    
    # 创建索引
    index = VectorStoreIndex.from_documents(documents)
    
    # 创建查询引擎
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        llm=OpenAI(model="gpt-3.5-turbo")
    )
    
    print("✅ 查询引擎初始化完成")
    return query_engine

async def main():
    """评估演示主函数"""
    print("🎯 LlamaIndex 评估框架演示")
    print("=" * 60)
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 请设置 OPENAI_API_KEY 环境变量")
        return
    
    try:
        # 创建查询引擎
        query_engine = create_query_engine()
        
        # 初始化评估器
        evaluator = RAGEvaluator(query_engine)
        
        # 创建测试数据集
        test_dataset = create_test_dataset()
        
        print(f"\n📝 测试数据集包含 {len(test_dataset)} 个查询")
        
        # 单个查询评估演示
        print("\n🔍 单个查询评估演示")
        print("=" * 40)
        sample_query = test_dataset[0]
        single_result = evaluator.evaluate_single_query(
            query=sample_query["query"],
            reference_answer=sample_query["reference_answer"]
        )
        evaluator.generate_report(single_result)
        
        # 批量评估演示
        print("\n📊 批量评估演示")
        print("=" * 40)
        batch_results = await evaluator.batch_evaluate(test_dataset)
        evaluator.generate_report(batch_results)
        
        # 保存评估结果
        report = {
            "evaluation_type": "batch",
            "total_queries": len(test_dataset),
            "timestamp": pd.Timestamp.now().isoformat(),
            "results": batch_results
        }
        
        with open("./evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 评估报告已保存到: ./evaluation_report.json")
        print("\n🎉 评估演示完成！")
        
    except Exception as e:
        print(f"❌ 评估过程中出现错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())
