import langextract as lx
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# ====================================================================
# 1. Configuration and Setup
# ====================================================================
LLM_API_KEY = None
LLM_BASE_URL = None
LLM_PROVIDER = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

COLUMN_MAPPING = {
    "report_text": ["病理诊断", "病理诊断信息", "诊断内容", "报告内容"],
    "id": ["序号", "编号", "ID"],
    "pathology_id": ["病理号", "病理编号", "原病理号"],
    "gender": ["性别"],
    "age": ["年龄"],
    "specimen_type": ["标本名称", "标本类型"],
    "location": ["取材部位", "部位"],
    "clinical_diagnosis": ["临床诊断"],
    "date": ["申请时间", "送检时间", "签发时间"],
}


def configure_environment():

    global LLM_API_KEY, LLM_BASE_URL, LLM_PROVIDER

    """配置环境变量"""
    load_dotenv()

    # 设置API密钥:cite[1]
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

    if not LLM_API_KEY:
        logger.warning("未找到LLM_API_KEY环境变量，请检查.env文件配置")
    else:
        logger.info(f"API密钥已加载，使用Provider: {LLM_PROVIDER}")


# ====================================================================
# 2. 重新定义数据模型以适应LangExtract格式:cite[1]
# ====================================================================


def create_pathology_examples():
        """创建病理报告提取示例，基于真实样本数据"""
        
        examples = []
        
        # 示例1：基于第一个样本
        example_text1 = """标本类型：左乳6点肿物扩大切除标本
    肿物大小2.0×2.0×1.3cm；
    组织学类型：浸润性癌，非特殊类型；
    组织学分级：Ⅲ级
    （腺管形成3分 + 核多形性3分 + 核分裂象3分 = 9分）
    肿瘤累及范围：脉管侵犯：（-） 
    保乳手术切缘情况：上切缘：请参见病理报告F2025-017418； 下切缘：请参见病理报告F2025-017418； 内切缘：请参见病理报告F2025-017418； 外切缘：请参见病理报告F2025-017418； 
    伴发病变：
    肿瘤间质浸润淋巴细胞（sTILs）：50%。
    免疫组化及特殊染色检查：（特检编号: A2025-26052）
    浸润性癌细胞：ER（-）, PR（-）, HER2（0，无染色）, Ki67（90%+）, AR（-）, GATA-3（部分弱+）, E-Cadherin（+）, P120（膜+）, CK5/6（-）, EGFR（+）, P63（散在弱+）, SOX-10（部分弱+）, AE1/AE3（+）。
    pTNM分期：pT1cN0Mx
    补充说明：淋巴结信息参考病理报告F2025-017433综合评估。"""

        examples.append(
            lx.data.ExampleData(
                text=example_text1,
                extractions=[
                    lx.data.Extraction(
                        extraction_class="specimen_type",
                        extraction_text="左乳6点肿物扩大切除标本",
                        attributes={"location": "左乳", "procedure": "扩大切除"},
                    ),
                    lx.data.Extraction(
                        extraction_class="tumor_size",
                        extraction_text="2.0×2.0×1.3cm",
                        attributes={"dimensions": "2.0×2.0×1.3cm", "unit": "cm"},
                    ),
                    lx.data.Extraction(
                        extraction_class="histologic_type",
                        extraction_text="浸润性癌，非特殊类型",
                        attributes={"type": "浸润性癌", "subtype": "非特殊类型"},
                    ),
                    lx.data.Extraction(
                        extraction_class="histologic_grade",
                        extraction_text="Ⅲ级",
                        attributes={"grade": "Ⅲ级", "score": "9分"},
                    ),
                    lx.data.Extraction(
                        extraction_class="er_status",
                        extraction_text="ER（-）",
                        attributes={"result": "阴性"},
                    ),
                    lx.data.Extraction(
                        extraction_class="her2_status",
                        extraction_text="HER2（0，无染色）",
                        attributes={"score": "0", "result": "阴性", "description": "无染色"},
                    ),
                    lx.data.Extraction(
                        extraction_class="ki67_rate",
                        extraction_text="Ki67（90%+）",
                        attributes={"percentage": "90%", "intensity": "+"},
                    ),
                    lx.data.Extraction(
                        extraction_class="margin_status",
                        extraction_text="上切缘：请参见病理报告F2025-017418",
                        attributes={"margin": "上切缘", "status": "需参考其他报告"},
                    ),
                    lx.data.Extraction(
                        extraction_class="stils_rate",
                        extraction_text="肿瘤间质浸润淋巴细胞（sTILs）：50%",
                        attributes={"percentage": "50%"},
                    ),
                    lx.data.Extraction(
                        extraction_class="ptnm_stage",
                        extraction_text="pT1cN0Mx",
                        attributes={"T": "T1c", "N": "N0", "M": "Mx"},
                    ),
                ],
            )
        )
        
        # 示例2：基于第二个样本
        example_text2 = """标本类型：右乳癌保乳根治标本
    肿瘤大小：单灶，大小1.7×1.5×0.8cm
    组织学类型：（冰冻剩余组织）浸润性癌，非特殊类型,伴部分中级别导管原位癌
    组织学分级：Ⅲ级
    （腺管形成3分 + 核多形性3分 + 核分裂象2分 = 8分）
    肿瘤累及范围：脉管侵犯：（-） 皮肤：（-） 
    保乳手术切缘情况：上切缘：详见报告F2025-017452 下切缘：详见报告F2025-017452 内切缘：详见报告F2025-017452 外切缘：详见报告F2025-017452 
    伴发病变：其余乳腺组织硬化性腺病，间质纤维化，散在微钙化
    免疫组化及特殊染色检查：（特检编号: A2025-25887）浸润性癌细胞ER（95%+，染色中等-强）, PR（约50%+，染色中等-强）, HER2（2+）, Ki67（约20%+）， E-Cadherin（+）, CK5/6（-）, P120（胞膜+）, GATA-3（+）,  CgA（-）, SYN（-）； P63（-）, Calponin（-）示癌巢周围肌上皮缺失。原位癌细胞ER（95%+，染色中等-强）, PR（约50%+，染色"""
        
        examples.append(
            lx.data.ExampleData(
                text=example_text2,
                extractions=[
                    lx.data.Extraction(
                        extraction_class="tumor_size",
                        extraction_text="1.7×1.5×0.8cm",
                        attributes={"dimensions": "1.7×1.5×0.8cm", "unit": "cm", "foci": "单灶"},
                    ),
                    lx.data.Extraction(
                        extraction_class="histologic_type",
                        extraction_text="浸润性癌，非特殊类型,伴部分中级别导管原位癌",
                        attributes={"invasive_type": "浸润性癌，非特殊类型", "dcis_type": "中级别导管原位癌"},
                    ),
                    lx.data.Extraction(
                        extraction_class="histologic_grade",
                        extraction_text="Ⅲ级",
                        attributes={"grade": "Ⅲ级", "score": "8分"},
                    ),
                    lx.data.Extraction(
                        extraction_class="er_status",
                        extraction_text="ER（95%+，染色中等-强）",
                        attributes={"percentage": "95%+", "intensity": "中等-强", "result": "阳性"},
                    ),
                    lx.data.Extraction(
                        extraction_class="pr_status",
                        extraction_text="PR（约50%+，染色中等-强）",
                        attributes={"percentage": "约50%+", "intensity": "中等-强", "result": "阳性"},
                    ),
                    lx.data.Extraction(
                        extraction_class="her2_status",
                        extraction_text="HER2（2+）",
                        attributes={"score": "2+", "result": "需进一步检测"},
                    ),
                    lx.data.Extraction(
                        extraction_class="ki67_rate",
                        extraction_text="Ki67（约20%+）",
                        attributes={"percentage": "约20%+", "intensity": "+"},
                    ),
                    lx.data.Extraction(
                        extraction_class="myoepithelial_status",
                        extraction_text="P63（-）, Calponin（-）示癌巢周围肌上皮缺失",
                        attributes={"p63": "阴性", "calponin": "阴性", "result": "肌上皮缺失"},
                    ),
                ],
            )
        )
        
        return examples


def create_pathology_graph_examples():
        """创建病理报告关系提取示例，基于真实样本数据"""
        
        examples = []
        
        # 示例1：基于第一个样本
        example_text1 = """标本类型：左乳6点肿物扩大切除标本
    肿物大小2.0×2.0×1.3cm；
    组织学类型：浸润性癌，非特殊类型；
    组织学分级：Ⅲ级
    （腺管形成3分 + 核多形性3分 + 核分裂象3分 = 9分）
    肿瘤累及范围：脉管侵犯：（-） 
    免疫组化及特殊染色检查：（特检编号: A2025-26052）
    浸润性癌细胞：ER（-）, PR（-）, HER2（0，无染色）, Ki67（90%+）"""
        
        examples.append(
            lx.data.ExampleData(
                text=example_text1,
                extractions=[
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="组织学分级：Ⅲ级",
                        attributes={
                            "head": "浸润性癌",
                            "relationship": "具有分级",
                            "tail": "Ⅲ级",
                            "type": "分级关系",
                            "score": "9分"
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="ER（-）",
                        attributes={
                            "head": "ER",
                            "relationship": "表达状态",
                            "tail": "阴性",
                            "type": "免疫组化标记",
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="HER2（0，无染色）",
                        attributes={
                            "head": "HER2",
                            "relationship": "表达状态",
                            "tail": "阴性",
                            "type": "免疫组化标记",
                            "score": "0"
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="Ki67（90%+）",
                        attributes={
                            "head": "Ki67",
                            "relationship": "增殖指数",
                            "tail": "90%",
                            "type": "增殖标记",
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="脉管侵犯：（-）",
                        attributes={
                            "head": "肿瘤",
                            "relationship": "脉管侵犯状态",
                            "tail": "阴性",
                            "type": "侵犯关系",
                        },
                    ),
                ],
            )
        )
        
        # 示例2：基于第二个样本
        example_text2 = """标本类型：右乳癌保乳根治标本
    肿瘤大小：单灶，大小1.7×1.5×0.8cm
    组织学类型：（冰冻剩余组织）浸润性癌，非特殊类型,伴部分中级别导管原位癌
    组织学分级：Ⅲ级
    （腺管形成3分 + 核多形性3分 + 核分裂象2分 = 8分）
    免疫组化及特殊染色检查：浸润性癌细胞ER（95%+，染色中等-强）, PR（约50%+，染色中等-强）, HER2（2+）, Ki67（约20%+）"""
        
        examples.append(
            lx.data.ExampleData(
                text=example_text2,
                extractions=[
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="组织学分级：Ⅲ级",
                        attributes={
                            "head": "浸润性癌",
                            "relationship": "具有分级",
                            "tail": "Ⅲ级",
                            "type": "分级关系",
                            "score": "8分"
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="ER（95%+，染色中等-强）",
                        attributes={
                            "head": "ER",
                            "relationship": "表达状态",
                            "tail": "阳性",
                            "type": "免疫组化标记",
                            "percentage": "95%+",
                            "intensity": "中等-强"
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="PR（约50%+，染色中等-强）",
                        attributes={
                            "head": "PR",
                            "relationship": "表达状态",
                            "tail": "阳性",
                            "type": "免疫组化标记",
                            "percentage": "约50%+",
                            "intensity": "中等-强"
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="HER2（2+）",
                        attributes={
                            "head": "HER2",
                            "relationship": "表达状态",
                            "tail": "可疑阳性",
                            "type": "免疫组化标记",
                            "score": "2+"
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="伴部分中级别导管原位癌",
                        attributes={
                            "head": "浸润性癌",
                            "relationship": "伴随病变",
                            "tail": "中级别导管原位癌",
                            "type": "伴随关系",
                        },
                    ),
                ],
            )
        )
        
        # 示例3：基于第三个样本（小叶癌）
        example_text3 = """标本类型：左乳癌保乳切除标本
    肿物大小1.6×1.5×1.0cm；
    组织学类型：多形性浸润性小叶癌；
    组织学分级：Ⅲ级
    （腺管形成3分 + 核多形性3分 + 核分裂象3分 = 9分）
    免疫组化及特殊染色检查：
    浸润性癌细胞：ER（80%+，染色中等-强）, PR（95%+，染色强）, HER2（3+）, Ki67（约20%+）"""
        
        examples.append(
            lx.data.ExampleData(
                text=example_text3,
                extractions=[
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="组织学类型：多形性浸润性小叶癌",
                        attributes={
                            "head": "肿瘤",
                            "relationship": "组织学类型",
                            "tail": "多形性浸润性小叶癌",
                            "type": "类型关系",
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="ER（80%+，染色中等-强）",
                        attributes={
                            "head": "ER",
                            "relationship": "表达状态",
                            "tail": "阳性",
                            "type": "免疫组化标记",
                            "percentage": "80%+",
                            "intensity": "中等-强"
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="HER2（3+）",
                        attributes={
                            "head": "HER2",
                            "relationship": "表达状态",
                            "tail": "阳性",
                            "type": "免疫组化标记",
                            "score": "3+"
                        },
                    ),
                ],
            )
        )
        
        return examples


def create_pathology_prompt():
    """创建病理报告提取提示"""
    prompt = """
    从乳腺癌病理诊断报告中提取以下关键信息：
    1. 肿瘤大小（浸润性肿瘤的最大三维尺寸，格式为XxYxZcm）
    2. 组织学类型（如：浸润性癌，非特殊类型）
    3. 组织学分级（如：I级、II级、III级）
    4. ER状态（雌激素受体表达百分比和强度）
    5. HER2状态（HER2表达状态和评分）
    6. Ki67指数（增殖百分比）
    7. 切缘状态（保乳手术切缘的总体状态）
    
    要求：
    - 使用报告中的精确文本进行提取
    - 不要释义或改写原文
    - 为每个提取项提供有意义的属性
    - 确保所有数值和结果准确对应原文
    """
    return prompt


def create_pathology_graph_prompt():
    prompt = """
    从病理报告中提取关键实体之间的关系，形成(主语, 关系, 宾语)三元组。
    重点提取：
    - 实体与分期/分级的关系
    - 取材与切缘的对应关系  
    - 分子标记的表达关系
    - 组织病理学的确认关系
    
    要求：
    - 使用精确的文本提取
    - 保持关系的准确性
    - 为每个三元组提供清晰的属性
    """
    return prompt

# ====================================================================
# 3. 核心处理管道类（修改版）
# ====================================================================


class ReportProcessingPipeline:
    """
    病理报告处理核心类，使用新版LangExtract API
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None
        self.column_map: Dict[str, str] = {}

    # --- 数据加载方法保持不变 ---
    def _read_file(self):
        """内部方法：根据文件后缀读取CSV或Excel文件"""
        file_path = Path(self.file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件未找到: {self.file_path}")

        if file_path.suffix in [".xlsx", ".xls"]:
            self.df = pd.read_excel(file_path)
            logger.info(f"读取Excel文件: {file_path.name}")
            return

        for encoding in ["utf-8", "gbk", "gb2312"]:
            try:
                self.df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"使用编码'{encoding}'成功加载CSV文件")
                return
            except UnicodeDecodeError:
                continue

        raise UnicodeDecodeError(f"无法使用utf-8/gbk/gb2312解码文件: {self.file_path}")

    def detect_columns(self) -> Dict[str, str]:
        """自动检测文件中的列名并进行标准化映射"""
        if self.df is None:
            self._read_file()

        self.df.columns = self.df.columns.astype(str).str.strip()
        logger.info(f"成功加载文件，共{len(self.df)}行，{len(self.df.columns)}列")

        detected_columns = {}
        for standard_name, possible_names in COLUMN_MAPPING.items():
            for possible_name in possible_names:
                if possible_name in self.df.columns:
                    if standard_name in detected_columns:
                        continue
                    detected_columns[standard_name] = possible_name
                    break

        self.column_map = detected_columns
        return detected_columns

    def load_reports(
        self, text_column: str = None, sample_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """加载、采样并标准化报告数据结构，增强数据清洗"""
        if not self.column_map:
            self.detect_columns()

        report_text_col_actual = text_column or self.column_map.get("report_text")

        if not report_text_col_actual or report_text_col_actual not in self.df.columns:
            raise ValueError(
                f"报告文本列'{report_text_col_actual}'不存在。请检查文件或指定正确的列名。"
            )

        processing_df = self.df.head(sample_size) if sample_size else self.df

        reports = []
        empty_count = 0

        metadata_cols_actual = {
            std_name: actual_name
            for std_name, actual_name in self.column_map.items()
            if std_name != "report_text" and actual_name in processing_df.columns
        }

        for idx, row in processing_df.iterrows():
            report_text = row[report_text_col_actual]

            if pd.isna(report_text) or str(report_text).strip() == "":
                empty_count += 1
                continue

            # 增强数据清洗
            cleaned_text = self._clean_report_text(str(report_text))
            
            if not cleaned_text.strip():
                empty_count += 1
                continue

            metadata = {
                std_name: row[actual_name]
                for std_name, actual_name in metadata_cols_actual.items()
            }

            reports.append(
                {
                    "row_index": idx,
                    "report_text": cleaned_text,
                    "metadata": metadata,
                }
            )

        if empty_count > 0:
            logger.warning(f"跳过{empty_count}条空记录")

        logger.info(f"成功加载{len(reports)}份有效报告")
        return reports

    def _clean_report_text(self, text: str) -> str:
        """清洗报告文本"""
        if not text:
            return ""
        
        # 替换不可见字符和多余空白
        text = text.replace('\xa0', ' ')  # 替换 &nbsp;
        text = text.replace('\u3000', ' ')  # 替换全角空格
        
        # 标准化换行符和空白字符
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                # 合并连续的多个空格
                line = ' '.join(line.split())
                lines.append(line)
        
        cleaned_text = '\n'.join(lines)
        
        # 处理标本类型中的引号
        cleaned_text = cleaned_text.replace('标本类型："', '标本类型：')
        cleaned_text = cleaned_text.replace('"标本类型：', '标本类型：')
        
        return cleaned_text

    # --- 修改的提取方法 ---
    @staticmethod
    def run_ner_extraction(report_text: str) -> Dict[str, Any]:
        """使用新版LangExtract API执行NER提取:cite[1]"""
        try:
            prompt = create_pathology_prompt()
            examples = create_pathology_examples()

            result = lx.extract(
                text_or_documents=report_text,
                prompt_description=prompt,
                examples=examples,
                extraction_passes=2,  # 多次提取提高召回率:cite[9]
                max_workers=4,  # 并行处理
                model_id=LLM_PROVIDER,
                api_key=LLM_API_KEY,  # Only use this for testing/development
                model_url=LLM_BASE_URL,
            )

            # 转换结果为字典格式
            return {
                "extractions": [
                    {
                        "extraction_class": ext.extraction_class,
                        "extraction_text": ext.extraction_text,
                        "attributes": ext.attributes,
                    }
                    for ext in result.extractions
                ],
                "document_id": result.document_id,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"NER提取失败: {str(e)}")
            return {"error": str(e), "status": "error"}

    @staticmethod
    def run_graph_extraction(report_text: str) -> List[Dict[str, Any]]:
        """使用新版LangExtract API执行图提取"""
        try:
            graph_prompt = create_pathology_graph_prompt()

            graph_examples = create_pathology_graph_examples()

            result = lx.extract(
                text_or_documents=report_text,
                prompt_description=graph_prompt,
                examples=graph_examples,
                extraction_passes=2,
                model_id=LLM_PROVIDER,
                api_key=LLM_API_KEY,  # Only use this for testing/development
                model_url=LLM_BASE_URL,
            )

            triples = []
            for ext in result.extractions:
                if hasattr(ext, "attributes") and ext.attributes:
                    triples.append(
                        {
                            "head": ext.attributes.get("head", ""),
                            "relationship": ext.attributes.get("relationship", ""),
                            "tail": ext.attributes.get("tail", ""),
                            "type": ext.attributes.get("type", ""),
                        }
                    )

            return triples

        except Exception as e:
            logger.error(f"图提取失败: {str(e)}")
            return [{"error": str(e)}]

    # --- 批量处理方法保持不变 ---
    def process_reports_batch(
        self,
        text_column: str = None,
        output_file: str = "extraction_results.json",
        sample_size: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """核心：批量处理病理报告，调用抽取函数"""
        reports = self.load_reports(text_column=text_column, sample_size=sample_size)

        if not reports:
            logger.error("没有可处理的报告数据")
            return []

        results = []
        total_reports = len(reports)

        pathology_id_col = self.column_map.get("pathology_id", "N/A")

        for i, report_data in enumerate(reports):
            report_id = report_data["metadata"].get(pathology_id_col, "N/A")
            logger.info(f"处理进度: {i+1}/{total_reports} - 病理号: {report_id}")

            try:
                report_text = report_data["report_text"]

                ner_result = self.run_ner_extraction(report_text)
                graph_result = self.run_graph_extraction(report_text)

                result = {
                    "row_index": report_data["row_index"],
                    "metadata": report_data["metadata"],
                    "report_text_preview": (
                        report_text[:200] + "..."
                        if len(report_text) > 200
                        else report_text
                    ),
                    "ner_extraction": ner_result,
                    "graph_extraction": graph_result,
                    "graph_triples_count": (
                        len(graph_result) if isinstance(graph_result, list) else 0
                    ),
                    "status": "success",
                    "processing_time": datetime.now().isoformat(),
                }
                results.append(result)

                if batch_size and (i + 1) % batch_size == 0:
                    interim_file = f"interim_{output_file}"
                    self._save_results_to_file(results, interim_file)

            except Exception as e:
                logger.error(f"处理失败(行{report_data['row_index']}): {str(e)}")
                results.append(
                    {
                        "row_index": report_data["row_index"],
                        "metadata": report_data["metadata"],
                        "error": str(e),
                        "status": "error",
                        "processing_time": datetime.now().isoformat(),
                    }
                )

        self._save_results_to_file(results, output_file)
        return results

    @staticmethod
    def _save_results_to_file(results: List[Dict[str, Any]], output_file: str):
        """保存结果到JSON文件"""
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"结果已保存到: {output_file}")
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")

    # --- 其他工具方法保持不变 ---
    @staticmethod
    def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成处理摘要报告"""
        total_count = len(results)
        success_count = len([r for r in results if r.get("status") == "success"])

        total_triples = sum(
            [
                r.get("graph_triples_count", 0)
                for r in results
                if r.get("status") == "success"
            ]
        )

        return {
            "total_reports": total_count,
            "successful_processing": success_count,
            "failed_processing": total_count - success_count,
            "success_rate": (
                round(success_count / total_count * 100, 2) if total_count > 0 else 0
            ),
            "total_triples_extracted": total_triples,
            "average_triples_per_report": (
                round(total_triples / success_count, 2) if success_count > 0 else 0
            ),
            "processing_date": datetime.now().isoformat(),
        }

    @staticmethod
    def export_to_csv(
        results: List[Dict[str, Any]], csv_output_file: str = "extraction_results.csv"
    ):
        """将结果导出为CSV格式"""
        csv_data = []

        for result in results:
            metadata = result["metadata"]
            pathology_id = metadata.get(next(iter(metadata.keys()), "病理号"), "")

            row_data = {
                "row_index": result["row_index"],
                "病理号": pathology_id,
                "处理状态": "成功" if result.get("status") == "success" else "失败",
                "提取的三元组数量": result.get("graph_triples_count", 0),
                "错误信息": (
                    result.get("error", "") if result.get("status") == "error" else ""
                ),
            }

            if result.get("status") == "success" and result.get("ner_extraction"):
                ner_data = result["ner_extraction"]
                if "extractions" in ner_data:
                    for extraction in ner_data["extractions"]:
                        class_name = extraction.get("extraction_class", "")
                        row_data[f"NER_{class_name}"] = extraction.get(
                            "extraction_text", ""
                        )

            csv_data.append(row_data)

        df = pd.DataFrame(csv_data)
        Path(csv_output_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_output_file, index=False, encoding="utf-8-sig")
        logger.info(f"CSV格式结果已导出到: {csv_output_file}")


# ====================================================================
# 4. 主执行函数
# ====================================================================


def main():
    """主执行函数"""

    # 1. 配置环境
    configure_environment()

    # 2. 参数设置
    CSV_FILE_PATH = "data/ruijin/Breast-20251021092802.xlsx"
    TEXT_COLUMN = "病理诊断"
    OUTPUT_JSON = "data/ruijin/pathology_extraction_results.json"
    OUTPUT_CSV = "data/ruijin/pathology_extraction_summary.csv"
    SAMPLE_SIZE = 5  # 开始时使用小样本测试
    BATCH_SIZE = 10

    try:
        # 3. 实例化并加载数据
        pipeline = ReportProcessingPipeline(CSV_FILE_PATH)
        detected_columns = pipeline.detect_columns()

        print("\n=== 列映射 ===")
        for std_name, actual_name in detected_columns.items():
            print(f"   {std_name}: {actual_name}")

        # 4. 批量处理
        results = pipeline.process_reports_batch(
            text_column=TEXT_COLUMN,
            output_file=OUTPUT_JSON,
            sample_size=SAMPLE_SIZE,
            batch_size=BATCH_SIZE,
        )

        # 5. 汇总和导出
        summary = pipeline.generate_summary_report(results)
        print("\n=== 处理摘要 ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

        pipeline.export_to_csv(results, OUTPUT_CSV)

    except (FileNotFoundError, ValueError, Exception) as e:
        logger.error(f"主程序执行失败: {str(e)}")


if __name__ == "__main__":
    main()
