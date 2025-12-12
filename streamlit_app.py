"""Streamlit 界面"""

import streamlit as st
import os
import sys
import json
import logging
from typing import List, Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from retrieval_module import HybridRetriever
    from generation_module import RAGGenerator, PromptBuilder
except ImportError:
    logger.error("缺少必要的模块文件")
    sys.exit(1)


class RAGSystem:
    """RAG系统类"""

    def __init__(self, chunks_file: str = "preprocessed_data/chunks.json"):
        """初始化RAG系统"""
        self.chunks_file = chunks_file
        self.chunks = None
        self.retriever = None
        self.generator = None
        self.conversation_history = []

        # ========== 核心修改1：定义默认URL ==========
        # 设定默认API地址（你的目标URL），优先读环境变量，没有则用默认值
        self.openai_api_url = os.getenv("OPENAI_API_URL", "https://api.chatanywhere.tech")
        logger.info(f"当前使用的API地址: {self.openai_api_url}")

        logger.info("初始化RAG系统...")
        self._initialize_system()

    def _initialize_system(self):
        """初始化系统：加载分块、初始化检索器和生成器"""

        # 步骤1：加载分块
        if not os.path.exists(self.chunks_file):
            logger.error(f"未找到分块文件: {self.chunks_file}")
            logger.info("请先运行 data_preprocessing.py 进行数据预处理")
            sys.exit(1)

        logger.info(f"加载分块文件: {self.chunks_file}")
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        logger.info(f"成功加载 {len(self.chunks)} 个分块")

        # 步骤2：初始化混合检索器
        logger.info("初始化检索器...")
        try:
            # 从环境变量获取 API 密钥（您已设置）
            openai_api_key = os.getenv("OPENAI_API_KEY") 
            # 允许用户通过环境变量指定 embedding 模型，默认为 small
            openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            
            if not openai_api_key:
                 logger.error("未设置 OPENAI_API_KEY 环境变量！")
                 sys.exit(1)
                 
            # ========== 核心修改2：检索器传递URL（带默认值） ==========
            self.retriever = HybridRetriever(
                chunks=self.chunks,
                api_key=openai_api_key, 
                model_name=openai_embedding_model,
                base_url=self.openai_api_url  # 使用带默认值的URL
            )
            logger.info("检索器初始化成功")
        except Exception as e:
             logger.error(f"初始化检索器失败: {e}")
             sys.exit(1)

        # 步骤3：初始化生成器
        logger.info("初始化生成器...")
        try:
            # ========== 核心修改3：生成器传递URL（带默认值） ==========
            self.generator = RAGGenerator(
                use_api=True,
                api_url=self.openai_api_url,  # 使用带默认值的URL
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-5-mini"  # 保持你的模型名称
            )
            logger.info("生成器初始化成功")
        except Exception as e:
            logger.error(f"初始化生成器失败: {e}")
            sys.exit(1)

        logger.info("\n✓ 系统初始化完成！\n")

    def process_question(self, question: str):
        """处理用户问题的完整流程：检索 -> 生成"""
        # 检索
        retrieved_chunks = self.retriever.retrieve(question, top_k=5)

        if not retrieved_chunks:
            st.write("✗ 未找到相关内容，无法生成答案")
            return

        # 生成
        try:
            result = self.generator.generate_answer(question, retrieved_chunks)
            st.write("最终答案:")
            st.write(result["answer"])
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            st.write(f"✗ 生成答案失败: {e}")


@st.cache_resource
def load_rag_system():
    """
    加载并缓存RAG系统。
    使用st.cache_resource装饰器，确保系统只在第一次加载时初始化。
    """
    logger.info("--- 正在加载或从缓存中读取RAG系统 ---")
    system = RAGSystem(chunks_file="preprocessed_data/chunks.json")
    return system


# Streamlit 应用界面
st.title("海洋渔业谈判辅助RAG系统")

# 加载 RAG 系统 (此操作会被缓存，只有第一次访问网页时会真正执行初始化)
rag_system = load_rag_system()

# 用户输入框
user_question = st.text_input("请输入问题：")

# 提问按钮
if st.button("提问"):
    if user_question:
        rag_system.process_question(user_question)
    else:
        st.write("请输入问题！")

# 侧边栏可以放一些配置
with st.sidebar:
    st.header("配置")
    # 新增：显示当前使用的API地址（方便调试）
    st.write(f"当前API地址: {rag_system.openai_api_url}")
    st.write("这里可以放一些配置选项")