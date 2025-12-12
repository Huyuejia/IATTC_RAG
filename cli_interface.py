"""
海洋渔业谈判辅助RAG系统 - 命令行交互界面
功能：提供用户友好的命令行交互体验
"""

import os
import sys
import json
import logging
from typing import List, Dict
from pathlib import Path

logging.basicConfig(level=logging. INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from retrieval_module import HybridRetriever
    from generation_module import RAGGenerator, PromptBuilder
except ImportError:
    logger.error("缺少必要的模块文件")
    sys.exit(1)


class RAGCLI:
    """RAG系统命令行交互界面"""
    
    def __init__(self, chunks_file: str = "preprocessed_data/chunks.json"):
        """
        初始化CLI
        
        Args:
            chunks_file: 预处理后的分块文件路径
        """
        self. chunks_file = chunks_file
        self.chunks = None
        self.retriever = None
        self.generator = None
        self.conversation_history = []
        
        logger.info("初始化RAG系统...")
        self._initialize_system()
    
    def _initialize_system(self):
        """初始化系统：加载分块、初始化检索器和生成器"""
        
        # 步骤1：加载分块
        if not os.path.exists(self. chunks_file):
            logger. error(f"未找到分块文件: {self.chunks_file}")
            logger.info("请先运行 data_preprocessing.py 进行数据预处理")
            sys.exit(1)
        
        logger.info(f"加载分块文件: {self.chunks_file}")
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        logger.info(f"成功加载 {len(self.chunks)} 个分块")
        
        # 步骤2：初始化混合检索器
        logger.info("初始化混合检索器...")
        try:
            # 使用Hugging Face镜像中的模型
            model_name = "BAAI/bge-m3"
            self.retriever = HybridRetriever(
                self.chunks,
                model_name=model_name,
                use_gpu=True
            )
            logger.info("混合检索器初始化成功")
        except Exception as e:
            logger.error(f"初始化检索器失败: {e}")
            logger.info("尝试在CPU模式下继续...")
            try:
                self.retriever = HybridRetriever(
                    self.chunks,
                    model_name=model_name,
                    use_gpu=False
                )
            except Exception as e2:
                logger.error(f"CPU模式也失败了: {e2}")
                sys.exit(1)
        
        # 步骤3：初始化生成器
        logger.info("初始化生成器...")
        try:
            self.generator = RAGGenerator(
                use_api=True,
                api_url=os.getenv("OPENAI_API_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-5-mini" #修改模型名称为gpt-5-mini
            )
            logger.info("生成器初始化成功")
        except Exception as e:
            logger.error(f"初始化生成器失败: {e}")
            sys.exit(1)
        
        logger.info("\n✓ 系统初始化完成！\n")
    
    def _print_banner(self):
        """打印欢迎信息"""
        banner = """
╔═══════════════════════════════════════════════════════════╗
║   海洋渔业谈判辅助RAG系统                                  ║
║   Fishery Negotiation Assistance RAG System                ║
╚═══════════════════════════════════════════════════════════╝

使用说明：
  • 输入问题，系统将自动进行检索和生成答案
  • 输入 'help'   查看帮助信息
  • 输入 'history' 查看对话历史
  • 输入 'exit'   退出系统
  • 输入 'save'   保存对话历史

"""
        print(banner)
    
    def _print_help(self):
        """打印帮助信息"""
        help_text = """
═══════════════════════════════════════════════════════════

【命令列表】

1. 输入问题 (直接输入中文或英文问题)
   示例: 若渔船在东太平洋因不可抗力原因未能遵守禁渔期，是否可以申请豁免？

2. help
   显示此帮助信息

3.  history
   显示最近的对话历史

4. retrieve <问题文本>
   仅执行检索，不生成答案
   示例: retrieve 禁渔期的定义

5. analyze <问题文本>
   显示检索过程的详细分析

6. save
   将对话历史保存为JSON文件

7. clear
   清空对话历史

8. exit (或 quit, q)
   退出系统

═══════════════════════════════════════════════════════════
"""
        print(help_text)
    
    def _print_section(self, title: str, content: str = ""):
        """打印格式化的章节"""
        print(f"\n【{title}】")
        if content:
            print(content)
    
    def retrieve_and_display(self, question: str, show_details: bool = False):
        """
        执行检索并显示结果
        
        Args:
            question: 用户问题
            show_details: 是否显示详细分析
        """
        self._print_section("检索过程")
        print(f"问题: {question}\n")
        
        try:
            results = self.retriever.retrieve(question, top_k=5)
            
            print(f"找到 {len(results)} 个相关分块:\n")
            
            for i, chunk in enumerate(results, 1):
                source = chunk["metadata"]. get("source_document", "Unknown")
                clause = chunk["metadata"].get("clause_number", "N/A")
                score = chunk. get("score", 0)
                text_preview = chunk["text"][:150] + "..." if len(chunk["text"]) > 150 else chunk["text"]
                
                print(f"  [{i}] 匹配度: {score:.1%}")
                print(f"      来源: {source}")
                print(f"      条款: {clause}")
                print(f"      文本: {text_preview}\n")
            
            if show_details:
                self._print_section("详细分析", "这是完整的检索结果分析")
            
            return results
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            print(f"✗ 检索失败: {e}")
            return []
    
    def generate_answer(self, question: str, retrieved_chunks: List[Dict]):
        """
        生成答案并显示
        
        Args:
            question: 用户问题
            retrieved_chunks: 检索到的分块
        """
        self._print_section("答案生成")
        print("正在调用AI模型生成答案.. .\n")
        
        try:
            result = self.generator.generate_answer(question, retrieved_chunks)
            
            # 显示答案
            self._print_section("最终答案")
            print(result["answer"])
            
            # 保存到对话历史
            self.conversation_history.append({
                "question": question,
                "answer": result["answer"],
                "retrieved_chunks_count": len(retrieved_chunks),
                "timestamp": self._get_timestamp()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            print(f"✗ 生成答案失败: {e}")
            return None
    
    def process_question(self, question: str):
        """
        处理用户问题的完整流程：检索 -> 生成
        
        Args:
            question: 用户问题
        """
        # 检索
        retrieved_chunks = self.retrieve_and_display(question)
        
        if not retrieved_chunks:
            print("✗ 未找到相关内容，无法生成答案")
            return
        
        # 生成
        self.generate_answer(question, retrieved_chunks)
    
    def show_history(self, limit: int = 5):
        """显示对话历史"""
        if not self.conversation_history:
            print("还没有对话历史")
            return
        
        self._print_section("对话历史")
        recent = self.conversation_history[-limit:]
        
        for i, item in enumerate(recent, 1):
            print(f"\n[{i}] {item['timestamp']}")
            print(f"    问题: {item['question'][:80]}...")
            print(f"    检索分块数: {item['retrieved_chunks_count']}")
    
    def save_history(self, output_file: str = "conversation_history.json"):
        """保存对话历史"""
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 对话历史已保存到: {output_path}")
    
    @staticmethod
    def _get_timestamp() -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime. now().strftime("%Y-%m-%d %H:%M:%S")
    
    def run(self):
        """启动交互式CLI"""
        self._print_banner()
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n📝 输入问题 (输入 'help' 查看帮助): ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input. lower() in ['exit', 'quit', 'q']:
                    print("\n再见！")
                    break
                
                elif user_input.lower() == 'help':
                    self._print_help()
                
                elif user_input.lower() == 'history':
                    self.show_history()
                
                elif user_input.lower() == 'save':
                    self.save_history()
                
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("✓ 对话历史已清空")
                
                elif user_input.lower(). startswith('retrieve '):
                    question = user_input[9:].strip()
                    if question:
                        self.retrieve_and_display(question)
                    else:
                        print("✗ 请提供问题文本")
                
                elif user_input.lower().startswith('analyze '):
                    question = user_input[8:].strip()
                    if question:
                        self.retrieve_and_display(question, show_details=True)
                    else:
                        print("✗ 请提供问题文本")
                
                else:
                    # 普通问题
                    self.process_question(user_input)
            
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                logger.error(f"处理请求时出错: {e}")
                print(f"✗ 出错: {e}")


# ============ 主程序入口 ============
if __name__ == "__main__":
    try:
        # 设置Hugging Face镜像地址
        # 这会让sentence-transformers库通过镜像下载模型
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        cli = RAGCLI()
        cli.run()
    except Exception as e:
        logger.error(f"启动失败: {e}")
        sys.exit(1)