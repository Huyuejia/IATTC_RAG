"""
海洋渔业谈判辅助RAG系统 - 数据预处理模块
功能：从PDF文件中提取文本和表格数据，进行结构化处理
技术选型：PyMuPDF (fitz) + pdfplumber
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    import pdfplumber
    import pandas as pd
except ImportError:
    logger.error("缺少必要的库。请运行: pip install PyMuPDF pdfplumber pandas")
    sys.exit(1)


class PDFPreprocessor:
    """PDF数据预处理器"""
    
    def __init__(self, pdf_folder: str = "pdfs", chunk_size: int = 500):
        """
        初始化PDF预处理器
        
        Args:
            pdf_folder: PDF文件所在的文件夹路径
            chunk_size: 分块大小（字符数）
        """
        self.pdf_folder = pdf_folder
        self.chunk_size = chunk_size
        self.chunks = []  # 存储处理后的分块
        
        # 创建输出目录
        self.output_dir = "preprocessed_data"
        os. makedirs(self.output_dir, exist_ok=True)
    
    def get_pdf_files(self) -> List[str]:
        """
        获取pdf文件夹下的所有PDF文件
        
        Returns:
            PDF文件路径列表
        """
        pdf_dir = Path(self.pdf_folder)
        if not pdf_dir.exists():
            logger.error(f"文件夹 {self.pdf_folder} 不存在")
            return []
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        return [str(f) for f in pdf_files]
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        使用PyMuPDF从PDF提取文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            (文本内容, 元数据字典)
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            metadata = {
                "source_document": os.path.basename(pdf_path),
                "total_pages": len(doc),
                "document_title": doc.metadata.get("title", "Unknown")
            }
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                text += f"\n--- 第 {page_num + 1} 页 ---\n{page_text}"
            
            doc.close()
            logger.info(f"成功提取 {pdf_path} 的文本，共 {len(text)} 字符")
            return text, metadata
            
        except Exception as e:
            logger.error(f"提取 {pdf_path} 的文本失败: {e}")
            return "", {}
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[str]:
        """
        使用pdfplumber从PDF提取表格，转换为Markdown格式
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            Markdown格式的表格列表
        """
        tables_md = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables):
                            # 转换为DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])
                            # 转换为Markdown
                            md_table = df.to_markdown(index=False)
                            tables_md.append(f"表格 (第{page_num+1}页-{table_idx+1}):\n{md_table}")
            
            logger.info(f"从 {pdf_path} 提取了 {len(tables_md)} 个表格")
            
        except Exception as e:
            logger.warning(f"提取 {pdf_path} 的表格失败: {e}")
        
        return tables_md
    
    def remove_noise(self, text: str) -> str:
        """
        清理文本中的噪音（页眉、页脚、多余空白等）
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        # 移除多余的空行
        text = re.sub(r'\n\s*\n+', '\n', text)
        
        # 移除常见的页眉页脚模式
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 跳过纯数字行（页码）
            if re. match(r'^\d+$', line. strip()):
                continue
            # 跳过过短的行（可能是页眉）
            if len(line. strip()) < 3:
                continue
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # 移除多余的空格
        text = re.sub(r' +', ' ', text)
        
        return text. strip()
    
    def identify_clauses(self, text: str) -> List[Tuple[int, int, str]]:
        """
        识别法规中的条款结构（如"3.1", "第3条", "Article 3"等）
        
        Args:
            text: 文本内容
            
        Returns:
            [(起始位置, 结束位置, 条款号), ... ] 列表
        """
        clauses = []
        
        # 匹配模式：数字. 数字（如3.1, 3.2）
        pattern1 = r'(\d+\.\d+)'
        # 匹配模式：第X条（中文）
        pattern2 = r'第(\d+)条'
        pattern2 = r'第\s*([一二三四五六七八九十百零\d]+)\s*条'
        # 匹配模式：Article X（英文）
        pattern3 = r'Article\s+(\d+)'
        # 匹配模式：Section X (英文)
        pattern4 = r'Section\s+(\d+)'
        # 匹配模式：带括号的数字或字母，如 (1), (a)
        pattern5 = r'^\s*\(([a-zA-Z0-9]+)\)' # 使用^要求在行首
        
        for match in re.finditer(pattern1, text):
            clauses.append((match.start(), match.end(), match.group(1)))
        
        for match in re.finditer(pattern2, text):
            clauses.append((match.start(), match.end(), f"第{match.group(1)}条"))
            clauses.append((match.start(), match.end(), f"第 {match.group(1)} 条"))
        
        for match in re.finditer(pattern3, text):
            clauses.append((match.start(), match.end(), f"Article {match.group(1)}"))

        for match in re.finditer(pattern4, text):
            clauses.append((match.start(), match.end(), f"Section {match.group(1)}"))

        for match in re.finditer(pattern5, text, re.MULTILINE):
            clauses.append((match.start(), match.end(), f"({match.group(1)})"))
        
        # 按起始位置排序
        clauses.sort(key=lambda x: x[0])
        logger.info(f"识别出 {len(clauses)} 个条款")
        
        return clauses
    
    def smart_chunking(self, text: str, metadata: Dict, clauses: List[Tuple[int, int, str]]) -> List[Dict]:
        """
        智能分块：优先按照条款结构分割，若无法识别则按大小分割
        
        Args:
            text: 文本内容
            metadata: 元数据
            clauses: 条款位置列表
            
        Returns:
            分块列表，每个分块包含文本、元数据等
        """
        chunks = []
        
        if clauses:
            # 基于条款结构进行分块
            for i, (start, end, clause_num) in enumerate(clauses):
                # 获取从当前条款到下一个条款的文本
                chunk_start = start
                if i + 1 < len(clauses):
                    chunk_end = clauses[i + 1][0]
                else:
                    chunk_end = len(text)
                
                chunk_text = text[chunk_start:chunk_end]. strip()
                
                if len(chunk_text) > 0:
                    chunk_metadata = metadata.copy()
                    chunk_metadata["clause_number"] = clause_num
                    chunk_metadata["chunk_index"] = len(chunks)
                    
                    chunks.append({
                        "text": chunk_text,
                        "metadata": chunk_metadata
                    })
        else:
            # 回退方案：按大小分块
            logger.warning("未识别到条款结构，使用大小分块")
            for i in range(0, len(text), self.chunk_size):
                chunk_text = text[i:i + self.chunk_size].strip()
                
                if len(chunk_text) > 0:
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = len(chunks)
                    chunk_metadata["char_range"] = f"{i}-{i + len(chunk_text)}"
                    
                    chunks.append({
                        "text": chunk_text,
                        "metadata": chunk_metadata
                    })
        
        logger.info(f"生成了 {len(chunks)} 个分块")
        return chunks
    
    def process_single_pdf(self, pdf_path: str) -> List[Dict]:
        """
        处理单个PDF文件的完整流程
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            处理后的分块列表
        """
        logger.info(f"\n开始处理: {pdf_path}")
        
        # 步骤1：提取文本和元数据
        text, metadata = self.extract_text_from_pdf(pdf_path)
        if not text:
            return []
        
        # 步骤2：提取表格
        tables = self.extract_tables_from_pdf(pdf_path)
        if tables:
            text += "\n\n--- 提取的表格 ---\n" + "\n\n".join(tables)
        
        # 步骤3：清理噪音
        text = self. remove_noise(text)
        
        # 步骤4：识别条款
        clauses = self.identify_clauses(text)
        
        # 步骤5：智能分块
        chunks = self. smart_chunking(text, metadata, clauses)
        
        return chunks
    
    def process_all_pdfs(self) -> List[Dict]:
        """
        批量处理pdfs文件夹下的所有PDF文件
        
        Returns:
            所有分块的列表
        """
        pdf_files = self.get_pdf_files()
        all_chunks = []
        
        for pdf_path in pdf_files:
            chunks = self.process_single_pdf(pdf_path)
            all_chunks.extend(chunks)
        
        logger.info(f"\n=== 处理完成 ===")
        logger.info(f"总共处理了 {len(pdf_files)} 个PDF文件，生成了 {len(all_chunks)} 个分块")
        
        self.chunks = all_chunks
        return all_chunks
    
    def save_chunks(self, output_file: str = "chunks.json") -> str:
        """
        将分块保存为JSON文件
        
        Args:
            output_file: 输出文件名
            
        Returns:
            保存的文件路径
        """
        import json
        
        output_path = os.path.join(self.output_dir, output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分块已保存到: {output_path}")
        return output_path
    
    def get_chunks(self) -> List[Dict]:
        """获取已处理的分块"""
        return self.chunks


# ============ 使用示例 ============
if __name__ == "__main__":
    # 创建预处理器实例
    preprocessor = PDFPreprocessor(pdf_folder="pdfs", chunk_size=500)
    
    # 处理所有PDF文件
    chunks = preprocessor.process_all_pdfs()
    
    # 显示前3个分块的信息
    print("\n=== 前3个分块示例 ===\n")
    for i, chunk in enumerate(chunks[:3]):
        print(f"分块 {i+1}:")
        print(f"  来源: {chunk['metadata']['source_document']}")
        print(f"  条款号: {chunk['metadata']. get('clause_number', 'N/A')}")
        print(f"  文本长度: {len(chunk['text'])} 字符")
        print(f"  文本预览: {chunk['text'][:100]}.. .\n")
    
    # 保存分块
    preprocessor.save_chunks()
    
    print(f"✓ 数据预处理完成！生成了 {len(chunks)} 个分块。")