"""
海洋渔业 谈判辅助RAG系统 - 检索模块
功能：基于混合检索（向量检索 + BM25关键词检索）实现相关度排序
技术选型：OpenAI Embedding + Faiss + rank-bm25
"""

import os
import sys
import json
import logging
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import openai  

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import faiss
    from rank_bm25 import BM25Okapi
except ImportError:
    logger.error("缺少必要的库。请运行相应的pip install命令")
    sys.exit(1)


class EmbeddingGenerator:
    """文本向量生成器 - 使用 OpenAI API"""
    
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small", 
                 timeout: int = 30, base_url: str = None):
        """
        初始化向量生成器
        
        Args:
            api_key: OpenAI API 密钥
            model_name: 使用的 OpenAI Embedding 模型名称
            timeout: API请求的超时时间（秒）
            base_url: API基础地址（第三方URL）
        """
        self.model_name = model_name
        
        # 初始化 OpenAI 客户端，设置超时和基础URL
        self.client = openai.OpenAI(
            api_key=api_key,
            timeout=timeout,
            base_url=base_url  # 使用传入的base_url
        ) 
        logger.info(f"Embedding Generator 初始化成功，使用模型: {self.model_name}")
        if base_url:
            logger.info(f"使用自定义API地址: {base_url}")

    def get_embedding(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        通过 OpenAI API 获取文本向量，支持分批处理以避免超时。
        """
        logger.info("生成分块向量...")
        all_embeddings = []
        
        # 分批处理逻辑
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"  > 正在处理批次: {i // batch_size + 1} / {num_batches} (文本数: {len(batch)})")
            
            try:
                # 调用 OpenAI Embedding API
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                
                # 提取向量并转换为 NumPy 数组
                embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(embeddings)
                
            except Exception as e:
                logger.error(f"OpenAI Embedding API 调用失败 (批次 {i // batch_size + 1}): {e}")
                raise 

        logger.info("向量生成完成")
        return np.array(all_embeddings, dtype=np.float32)


class HybridRetriever:
    """混合检索器：向量检索 (Faiss) + 关键词检索 (BM25)"""

    def __init__(self, 
                 chunks: List[Dict], 
                 api_key: str, 
                 model_name: str = "text-embedding-3-small",
                 index_path: str = "preprocessed_data/faiss_index.bin",
                 base_url: str = None):
        """
        初始化混合检索器
        
        Args:
            chunks: 包含所有文本分块的列表
            api_key: OpenAI API 密钥
            model_name: Embedding 模型名称
            index_path: Faiss索引的保存路径
            base_url: API基础地址（第三方URL）
        """
        logger.info(f"初始化混合检索器，共 {len(chunks)} 个分块")
        self.chunks = chunks
        self.texts = [chunk["text"] for chunk in chunks]
        self.index_path = index_path
        
        # 初始化向量生成器，传入base_url
        logger.info("初始化向量生成器...")
        self.embedding_generator = EmbeddingGenerator(
            api_key=api_key, 
            model_name=model_name,
            base_url=base_url
        )
        
        # 初始化检索索引
        self._initialize_retrieval_index()

    def _initialize_retrieval_index(self):
        """初始化        初始化向量索引和关键词索引
        """
        
        # 2.1 Faiss 向量索引
        if Path(self.index_path).exists():
            logger.info(f"加载现有 Faiss 索引: {self.index_path}")
            self.faiss_index = faiss.read_index(self.index_path)
            self.embedding_dim = self.faiss_index.d
        else:
            logger.info("未找到 Faiss 索引，开始生成向量...")
            
            # 生成所有文本的向量
            text_embeddings = self.embedding_generator.get_embedding(self.texts)
            self.embedding_dim = text_embeddings.shape[1]
            
            # 创建 Faiss 索引
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            self.faiss_index.add(text_embeddings)
            
            # 保存索引
            logger.info(f"保存 Faiss 索引到: {self.index_path}")
            faiss.write_index(self.faiss_index, self.index_path)
            
        logger.info(f"Faiss 索引初始化成功，维度: {self.embedding_dim}, 数量: {self.faiss_index.ntotal}")

        # 2.2 BM25 关键词索引
        logger.info("初始化 BM25 关键词索引...")
        tokenized_corpus = [text.split(" ") for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 索引初始化成功")


    def _vector_search(self, query: str, top_k: int) -> List[int]:
        """执行 Faiss 向量检索，返回索引列表"""
        query_embedding = self.embedding_generator.get_embedding([query])[0]
        D, I = self.faiss_index.search(np.expand_dims(query_embedding, axis=0), top_k)
        return I[0].tolist()

    def _bm25_search(self, query: str, top_k: int) -> List[int]:
        """执行 BM25 关键词检索，返回索引列表"""
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # 获取 top_k 索引
        top_k_indices = np.argsort(doc_scores)[::-1][:top_k].tolist()
        
        return top_k_indices

    def _combine_results(self, vector_indices: List[int], bm25_indices: List[int], top_k: int) -> List[Dict]:
        """合并检索results并重排序"""
        
        # 1. 收集所有不重复的索引
        all_indices = list(set(vector_indices + bm25_indices))
        
        # 2. 基于简单的融合权重计算新的分数
        vector_weight = 0.6
        bm25_weight = 0.4
        
        # 为每个文档创建分数映射
        combined_scores = {}
        
        # 计算向量检索分数 (基于 rank)
        for rank, index in enumerate(vector_indices):
            score = vector_weight * (1.0 / (rank + 1.0))
            combined_scores[index] = combined_scores.get(index, 0.0) + score
            
        # 计算 BM25 关键词检索分数 (基于 rank)
        for rank, index in enumerate(bm25_indices):
            score = bm25_weight * (1.0 / (rank + 1.0))
            combined_scores[index] = combined_scores.get(index, 0.0) + score
            
        # 3. 按合并分数排序
        sorted_indices = sorted(combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True)
        
        # 4. 构建最终结果
        final_results = []
        for index in sorted_indices[:top_k]:
            chunk = self.chunks[index].copy()
            chunk["score"] = combined_scores[index]
            final_results.append(chunk)
            
        logger.info(f"融合检索结果：返回 {len(final_results)} 个分块")
        return final_results


    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        执行混合检索
        """
        
        # 1. 向量检索
        vector_indices = self._vector_search(query, top_k=top_k)
        
        # 2. 关键词检索
        bm25_indices = self._bm25_search(query, top_k=top_k)
        
        # 3. 合并和重排
        final_results = self._combine_results(vector_indices, bm25_indices, top_k=top_k)
        
        return final_results


# ============ 使用示例 ============
if __name__ == "__main__":
    # 假设已有preprocessed_data/chunks.json
    chunks_file_path = "preprocessed_data/chunks.json"
    if not os.path.exists(chunks_file_path):
        logger.error(f"测试需要 {chunks_file_path} 文件，请先运行 data_preprocessing.py")
        sys.exit(1)
        
    with open(chunks_file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    logger.info(f"加载了 {len(chunks)} 个分块")
    
    # 从环境变量中获取配置，增加默认值处理
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # 从环境变量获取API URL，如果未设置则使用你指定的默认地址
    openai_api_url = os.getenv("OPENAI_API_URL", "https://api.chatanywhere.tech")
    openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    if not openai_api_key:
        logger.error("请设置 OPENAI_API_KEY 环境变量以使用 API 向量模型")
        sys.exit(1)
    
    # 打印当前使用的API地址，方便调试
    logger.info(f"当前使用的API地址: {openai_api_url}")
        
    # 初始化混合检索器，传入API URL
    retriever = HybridRetriever(
        chunks=chunks,
        api_key=openai_api_key,
        model_name=openai_embedding_model,
        base_url=openai_api_url  # 确保传递到检索器
    )
    
    # 测试查询
    test_queries = [
        "若渔船在东太平洋因不可抗力原因原因未能遵守禁渔期，是否可以申请豁免？",
        "IATTC管辖区域内捕捞活动活动受影响的规定",
        "禁渔期的定义和适用范围"
    ]
    
    for query in test_queries:
        logger.info(f"\n[测试查询]: {query}")
        results = retriever.retrieve(query, top_k=3)
        for i, res in enumerate(results):
            logger.info(f"  结果 {i+1} (分数: {res['score']:.4f}): {res['metadata']['source_document']} - {res['metadata']['clause_number']}")