"""
海洋渔业谈判辅助RAG系统 - 生成模块
功能：基于检索结果生成结构化答案（包含思考过程）
技术选型：transformers + 大模型推理 (DeepSeek / GPT等)
"""

import os
import sys
import json 
import logging
from typing import List, Dict
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    logger.error("缺少必要的库")
    sys.exit(1)


class PromptBuilder:
    """提示词构建器（最终版：IATTC专家+固定格式+结论维度）"""
    
    @staticmethod
    def build_prompt(
        question: str,
        retrieved_chunks: List[Dict],
        system_role: str = None
    ) -> str:
        """
        构建结构化提示词（严格匹配IATTC专家+固定输出格式要求）
        
        Args:
            question: 用户问题
            retrieved_chunks: 检索到的相关分块列表
            system_role: 系统角色定义
            
        Returns:
            完整的提示词
        """
        if system_role is None:
            # 核心：新增IATTC法律分析专家身份 + 结论维度要求
            system_role = """你是一名专业的IATTC法律分析专家，需严格遵循以下规则回答问题：
1. 仅基于提供的背景知识（<context>内内容）回答，禁止使用任何外部知识；
2. 思考过程需体现“实体提取→定位依据→逐步推理→得出结论”的完整逻辑；
3. 回答需包含“判断依据、具体内容解析、总结、结论”四个核心维度，格式严格固定。
"""
        
        # 提取检索文本（拼接所有条款全文，用于<context>块）
        retrieved_text = "\n\n".join([
            f"【{chunk['metadata'].get('source_document', '未知决议')} {chunk['metadata'].get('clause_number', '未标注条款')}】\n{chunk['text']}"
            for chunk in retrieved_chunks
        ])
        
        # 构建完整提示词（严格匹配要求的格式：<context> + ### 思考过程/回答 + ² 符号）
        prompt = f"""{system_role}

<context>
{retrieved_text}
</context>

### 用户问题：
{question}

### 输出要求（严格遵守格式，不可增删/修改符号）：
### 思考过程：
1. 分析用户问题，识别核心实体：{{组织}}, {{措施}}, {{条款}}。
2. 从背景知识中定位相关依据。
3. 逐步推理：逐条验证用户问题中的条件是否符合条款要求。
4. 得出结论：明确用户问题的最终答案方向。

### 回答：
² 判断依据：仅标注决议编号+条款号（如C-24-01决议第17条）
² 具体内容解析：分点拆解条款核心条件（用1.2.3.），仅用中文，禁止堆砌英文
² 总结：分点说明核心条件（每点≤20字）
² 结论：针对用户问题给出明确、唯一的最终结论（如“符合条件可申请豁免”/“不符合条件不可申请豁免”）
"""
        return prompt


class LocalModelGenerator:
    """本地模型生成器 (适用于DeepSeek等本地部署的模型)"""
    
    def __init__(self, model_name: str, use_gpu: bool = True):
        """
        初始化本地模型生成器
        
        Args:
            model_name: 模型名称或路径
            use_gpu: 是否使用GPU
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")
        
        try:
            logger.info(f"加载模型: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        基于提示词生成文本
        
        Args:
            prompt: 提示词
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: nucleus采样参数
            
        Returns:
            生成的文本
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 移除输入部分，只保留生成部分
            result = generated_text[len(prompt):].strip()
            
            return result
        except Exception as e:
            logger.error(f"生成文本失败: {e}")
            return ""


class APIModelGenerator:
    """API模型生成器 (用于OpenAI兼容的API)"""
    
    def __init__(
        self,
        api_url: str = None,
        api_key: str = None,
        model: str = None
    ):
        """
        初始化API模型生成器
        
        Args:
            api_url: API地址 (默认从环境变量读取)
            api_key: API密钥 (默认从环境变量读取)
            model: 模型名称 (默认从环境变量读取)
        """
        self.api_url = api_url or os.getenv("OPENAI_API_URL", "https://api.chatanywhere.tech")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        
        if not self.api_key:
            logger.error("未设置OPENAI_API_KEY环境变量")
            raise ValueError("OPENAI_API_KEY not set")
        
        logger.info(f"API配置 - URL: {self.api_url}, 模型: {self.model}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """
        通过API生成文本
        
        Args:
            prompt: 提示词
            max_tokens: 最大token数
            temperature: 温度参数
            
        Returns:
            生成的文本
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            logger.info(f"调用API: {self.api_url}/v1/chat/completions")
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"API返回错误: {response.status_code} - {response.text}")
                return ""
            
            result = response.json()["choices"][0]["message"]["content"]
            logger.info("API生成成功")
            
            return result
            
        except Exception as e:
            logger.error(f"API请求失败: {e}")
            return ""


class RAGGenerator:
    """RAG生成器 - 整合检索和生成"""
    
    def __init__(
        self,
        use_api: bool = True,
        api_url: str = None,
        api_key: str = None,
        model: str = None,
        use_gpu: bool = True
    ):
        """
        初始化RAG生成器
        
        Args:
            use_api: 是否使用API模式
            api_url: API地址
            api_key: API密钥
            model: 模型名称
            use_gpu: 本地模式是否使用GPU
        """
        self.use_api = use_api
        
        if use_api:
            logger.info("使用API模式")
            self.generator = APIModelGenerator(api_url, api_key, model)
        else:
            logger.info("使用本地模型模式")
            model_name = model or "deepseek-ai/deepseek-7b-chat"
            self.generator = LocalModelGenerator(model_name, use_gpu)
    
    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict]
    ) -> Dict:
        """
        生成结构化答案
        
        Args:
            question: 用户问题
            retrieved_chunks: 检索到的分块
            
        Returns:
            {
                "question": "用户问题",
                "answer": "生成的答案",
                "retrieved_chunks": [... ],
                "metadata": {... }
            }
        """
        logger.info(f"开始生成答案，问题: {question}")
        
        # 构建提示词
        prompt = PromptBuilder.build_prompt(question, retrieved_chunks)
        logger.info(f"提示词长度: {len(prompt)} 字符")
        
        # 生成答案
        answer = self.generator.generate(prompt)
        
        if not answer:
            answer = "生成失败，请检查API配置或模型状态"
        
        result = {
            "question": question,
            "answer": answer,
            "retrieved_chunks": [
                {
                    "text": chunk["text"][:200] + "...",
                    "source": chunk["metadata"]["source_document"],
                    "clause": chunk["metadata"].get("clause_number", "N/A"),
                    "score": chunk.get("score", 0)
                }
                for chunk in retrieved_chunks
            ],
            "metadata": {
                "generator_type": "API" if self.use_api else "LocalModel",
                "num_retrieved_chunks": len(retrieved_chunks)
            }
        }
        
        return result
    
    def save_result(self, result: Dict, output_file: str = "generation_result.json"):
        """
        保存生成结果
        
        Args:
            result: 生成结果字典
            output_file: 输出文件名
        """
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_path}")


# ============ 使用示例 ============
if __name__ == "__main__":
    # 示例：使用API模式
    logger.info("=== RAG生成模块测试 ===\n")
    
    try:
        # 初始化生成器
        generator = RAGGenerator(
            use_api=True,
            api_url=os.getenv("OPENAI_API_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL")
        )
        
        # 模拟检索结果
        mock_chunks = [
            {
                "text": "第17条：a. 若不可抗力（force majeure）事件导致渔船在两个禁渔期之外的一段至少连续75天期间内无法出海，缔约方（CPC）可请求按第3段和第17条第1款规定的方式申请减少的禁渔期豁免。b. 仅在捕鱼作业过程中因机械和/或结构故障、火灾或爆炸而致使船舶失能的情况，才被视为不可抗力。c. 本豁免适用于遵守第3段所规定任一禁渔期的船队的船舶。",
                "metadata": {
                    "source_document": "C-24-01决议",
                    "clause_number": "第17条"
                },
                "score": 0.95
            }
        ]
        
        # 生成答案
        question = "若渔船在东太平洋因不可抗力原因未能遵守禁渔期，且在其管辖区域内的捕捞活动受到了影响，是否可以申请豁免？"
        result = generator.generate_answer(question, mock_chunks)
        
        print("\n=== 生成结果 ===\n")
        print(f"问题: {result['question']}\n")
        print(f"答案:\n{result['answer']}\n")
        print(f"检索的分块数: {result['metadata']['num_retrieved_chunks']}")
        
        # 保存结果
        generator.save_result(result)
        
    except Exception as e:
        logger.error(f"测试失败: {e}")