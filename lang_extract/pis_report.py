import re
import langextract as lx
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import os
import sys
from datetime import datetime
import asyncio
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langextract_glmprovider.provider import GLMProviderLanguageModel, GLMProviderSchema

# ====================================================================
# 1. Configuration and Setup
# ====================================================================
current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")

LLM_API_KEY = None
LLM_BASE_URL = None
LLM_MODEL_ID = None
CSV_FILE_PATH = "data/ruijin/Breast-20251021092802.xlsx"
TEXT_COLUMN = "病理诊断"
OUTPUT_JSON = f"data/ruijin/{current_datetime}/pathology_extraction_results.json"
OUTPUT_CSV = f"data/ruijin/{current_datetime}/pathology_extraction_summary.csv"
SAMPLE_SIZE = 50  # 开始时使用小样本测试
BATCH_SIZE = 10  # 控制并发请求数

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
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

    global LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_ID

    """配置环境变量"""
    load_dotenv()

    # 设置环境变量
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "glm-4-flash")

    if not LLM_API_KEY:
        logger.warning("未找到LLM_API_KEY环境变量，请检查.env文件配置")
    else:
        logger.info(f"API密钥已加载，使用Provider: {LLM_MODEL_ID}")


def create_pathology_prompt():
    """创建病理报告提取提示"""
    prompt = """
    从乳腺癌病理诊断报告中提取以下关键信息：
    1. 肿瘤标签（存在肿瘤、无肿瘤）
    2. 肿瘤大小（浸润性肿瘤的最大三维尺寸，格式为XxYxZcm）
    3. 组织学主类型（如：未分类、纤维上皮肿瘤、上皮源性肿瘤、间叶源性肿瘤、浸润性癌、原位癌、转移性癌、转移性乳腺癌）
    4. 组织学类型（如：浸润性癌，非特殊类型l;浸润性小叶癌,黏液性癌,浸润性乳头状癌,微乳头状癌,大汗腺癌,化生性癌,淋巴上皮瘤样癌,炎症性癌,透明细胞癌,梭形细胞癌,腺样囊性癌）
    5. 组织学分级（如：I级、II级、III级），并提取相关指标（如：腺管形成3分 + 核多形性3分 + 核分裂象3分 = 9分）
    6. 肿瘤累及范围（如：脉管侵犯：（-） 皮肤：（-） 乳头：（+） 基底：（-） ）
    7. 其余象限乳腺组织：（如：内上象限：（+） 内下象限：（-） 外上象限：（-） 外下象限：（-） ）
    8. 保乳手术切缘情况
    9. 免疫组化及特殊染色检查
    10. pTNM分期(如：pT2N0Mx）
    
    要求：
    - 使用报告中的精确文本进行提取
    - 不要释义或改写原文
    - 为每个提取项提供有意义的属性
    - 确保所有数值和结果准确对应原文
    - 确保所有输出的数值和结果都能在对应原文找到
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


class PathologyExamples:
    """
    A single comprehensive example with a generalized schema for breast cancer pathology reports.
    Based on the provided comprehensive_text input, focusing on 7 key extraction classes.
    """

    @staticmethod
    def _create_extraction(
        class_name: str, text: str, attributes: dict
    ) -> lx.data.Extraction:
        """Helper function to create a standardized Extraction object."""
        return lx.data.Extraction(
            extraction_class=class_name,
            extraction_text=text,
            attributes=attributes,
        )

    @staticmethod
    def get_examples() -> List[lx.data.ExampleData]:
        """
        从乳腺癌病理诊断报告中提取以下关键信息（共10项）：
        1. 肿瘤标签（存在肿瘤、无肿瘤）
        2. 肿瘤大小（浸润性肿瘤的最大三维尺寸，格式为XxYxZcm）
        3. 组织学主类型
        4. 组织学类型
        5. 组织学分级
        6. 肿瘤累及范围
        7. 其余象限乳腺组织
        8. 保乳手术切缘情况
        9. 免疫组化及特殊染色检查
        10. pTNM分期
        """

        comprehensive_text = (
            "标本类型:左乳6点肿物扩大切除标本;\n"
            "肿物大小2.0×2.0×1.3cm;\n"
            "组织学类型:浸润性癌,非特殊类型;\n"
            "组织学分级:Ⅲ级;(腺管形成3分+核多形性3分+核分裂象3分=9分);\n"
            "肿瘤累及范围:脉管侵犯:(阴性);\n"
            "保乳手术切缘情况:上切缘:参考其他报告;下切缘:参考其他报告;内切缘:参考其他报告;外切缘:参考其他报告;\n"
            "伴发病变:无;\n"
            "肿瘤间质浸润淋巴细胞(sTILs):50%;\n"
            "免疫组化及特殊染色检查:(特检编号:A2025-26052);\n"
            "浸润性癌细胞:ER(阴性),PR(阴性),HER2(0,无染色),Ki67(90%+),AR(阴性),GATA-3(部分弱+),"
            "E-Cadherin(阳性),P120(膜+),CK5/6(阴性),EGFR(阳性),P63(散在弱+),SOX-10(部分弱+),AE1/AE3(阳性);\n"
            "pTNM分期:pT1cN0Mx;\n"
            "补充说明:淋巴结信息参考病理报告F2025-017433综合评估;"
        )

        examples = [
            lx.data.ExampleData(
                text=comprehensive_text,
                extractions=[
                    # 1. 肿瘤标签
                    PathologyExamples._create_extraction(
                        "tumor_presence",
                        "存在肿瘤",
                        {"result": "存在肿瘤"},
                    ),
                    # 2. 肿瘤大小
                    PathologyExamples._create_extraction(
                        "tumor_size",
                        "2.0×2.0×1.3cm",
                        {
                            "dimensions": "2.0x2.0x1.3",
                            "unit": "cm",
                        },
                    ),
                    # 3. 组织学主类型
                    PathologyExamples._create_extraction(
                        "histological_main_type",
                        "上皮源性肿瘤",
                        {"category": "上皮源性肿瘤"},
                    ),
                    # 4. 组织学类型
                    PathologyExamples._create_extraction(
                        "histological_type",
                        "浸润性癌,非特殊类型",
                        {
                            "classification": "浸润性癌",
                            "subtype": "非特殊类型",
                        },
                    ),
                    # 5. 组织学分级
                    PathologyExamples._create_extraction(
                        "histological_grade",
                        "Ⅲ级;(腺管形成3分+核多形性3分+核分裂象3分=9分)",
                        {
                            "grade": "III",
                            "tubule_formation": "3",
                            "nuclear_pleomorphism": "3",
                            "mitotic_count": "3",
                            "total_score": "9",
                        },
                    ),
                    # 6. 肿瘤累及范围
                    PathologyExamples._create_extraction(
                        "tumor_extent",
                        "脉管侵犯:(阴性)",
                        {
                            "vascular_invasion": "阴性",
                            "skin": "N/A",
                            "nipple": "N/A",
                            "base": "N/A",
                        },
                    ),
                    # 7. 其余象限乳腺组织
                    PathologyExamples._create_extraction(
                        "other_quadrants",
                        "N/A",
                        {
                            "upper_inner": "N/A",
                            "lower_inner": "N/A",
                            "upper_outer": "N/A",
                            "lower_outer": "N/A",
                        },
                    ),
                    # 8. 保乳手术切缘情况
                    PathologyExamples._create_extraction(
                        "margin_status",
                        "上切缘:参考其他报告;下切缘:参考其他报告;内切缘:参考其他报告;外切缘:参考其他报告;",
                        {
                            "overall_status": "未评估/需参考其他报告",
                            "superior_margin": "参考其他报告",
                            "inferior_margin": "参考其他报告",
                            "medial_margin": "参考其他报告",
                            "lateral_margin": "参考其他报告",
                        },
                    ),
                    # 9. 免疫组化及特殊染色检查
                    PathologyExamples._create_extraction(
                        "immunohistochemistry",
                        "ER(阴性),PR(阴性),HER2(0,无染色),Ki67(90%+)",
                        {
                            "ER": "阴性",
                            "PR": "阴性",
                            "HER2": "0 (无染色)",
                            "Ki67": "90%",
                            "other_markers": "N/A",
                        },
                    ),
                    # 10. pTNM分期
                    PathologyExamples._create_extraction(
                        "pTNM_stage",
                        "pT1cN0Mx",
                        {"T": "1c", "N": "0", "M": "x"},
                    ),
                ],
            )
        ]

        return examples


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
            cleaned_text = clean_report_text(str(report_text))

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

    # --- 修改的提取方法 ---
    @staticmethod
    async def run_ner_extraction(report_text: str) -> Dict[str, Any]:
        """使用新版LangExtract API执行NER提取:cite[1]"""
        try:
            prompt = create_pathology_prompt()
            examples = PathologyExamples().get_examples()

            result = await run_extraction_execution_native_async(
                report_text, prompt, examples
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
    async def run_graph_extraction(report_text: str) -> List[Dict[str, Any]]:
        """使用新版LangExtract API执行图提取"""
        try:
            graph_prompt = create_pathology_graph_prompt()

            graph_examples = PathologyExamples.get_examples()

            result = await run_extraction_execution_native_async(
                report_text, graph_prompt, graph_examples
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

    # --- 异步批量处理方法 ---
    async def process_reports_batch(
        self,
        text_column: str = None,
        output_file: str = "extraction_results.json",
        sample_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_concurrent: int = 3,  # 控制并发数，避免压垮外部服务
        file_mode: str = "w",  # "w" for overwrite, "a" for append
    ) -> List[Dict[str, Any]]:
        """核心：批量处理病理报告，调用抽取函数（异步并行版本）"""
        reports = self.load_reports(text_column=text_column, sample_size=sample_size)

        if not reports:
            logger.error("没有可处理的报告数据")
            return []

        # 创建信号量来限制并发外部服务调用
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single_report(report_data: Dict[str, Any]) -> Dict[str, Any]:
            """处理单个报告的异步函数"""
            report_id = report_data["metadata"].get("pathology_id", "N/A")

            try:
                report_text = report_data["report_text"]

                # 使用信号量限制并发外部服务调用
                async with semaphore:
                    ner_result = await self.run_ner_extraction(report_text)
                    # graph_result = await self.run_graph_extraction(report_text)  # If also async

                result = {
                    "row_index": report_data["row_index"],
                    "metadata": report_data["metadata"],
                    "report_text_preview": (
                        report_text[:200] + "..."
                        if len(report_text) > 200
                        else report_text
                    ),
                    "ner_extraction": ner_result,
                    # "graph_extraction": graph_result,
                    # "graph_triples_count": (
                    #     len(graph_result) if isinstance(graph_result, list) else 0
                    # ),
                    "status": "success",
                    "processing_time": datetime.now().isoformat(),
                }
                return result

            except Exception as e:
                logger.error(f"处理失败(行{report_data['row_index']}): {str(e)}")
                return {
                    "row_index": report_data["row_index"],
                    "metadata": report_data["metadata"],
                    "error": str(e),
                    "status": "error",
                    "processing_time": datetime.now().isoformat(),
                }

        # 并发处理所有报告
        tasks = []
        total_reports = len(reports)

        for i, report_data in enumerate(reports):
            report_id = report_data["metadata"].get("pathology_id", "N/A")
            logger.info(f"提交任务: {i+1}/{total_reports} - 病理号: {report_id}")

            task = asyncio.create_task(process_single_report(report_data))
            tasks.append(task)

        # 等待所有任务完成并收集结果
        results = []
        try:
            # 按完成顺序处理结果
            for i, completed_task in enumerate(asyncio.as_completed(tasks)):
                result = await completed_task
                results.append(result)

                # 记录进度
                completed_count = i + 1

                logger.info("#" * 64)
                logger.info(f"完成进度: {completed_count}/{total_reports}")
                logger.info("#" * 64)

                # 如果指定了batch_size，保存临时结果
                if batch_size and completed_count % batch_size == 0:
                    interim_file = f"interim_{output_file}"
                    # 对于临时文件，使用追加模式积累结果
                    self._save_results_to_file(results, interim_file, mode="a")
                    logger.info(f"已保存临时结果到: {interim_file}")

        except Exception as e:
            logger.error(f"批量处理过程中发生错误: {str(e)}")
            # 取消剩余任务
            for task in tasks:
                if not task.done():
                    task.cancel()

        # 使用指定模式保存最终结果
        self._save_results_to_file(results, output_file, mode=file_mode)
        logger.info(
            f"处理完成，共处理 {len(results)} 个报告，结果保存到: {output_file}"
        )
        return results

    @staticmethod
    def _save_results_to_file(
        results: List[Dict[str, Any]], output_file: str, mode: str = "w"
    ):
        """保存结果到文件 - 支持追加模式和JSON Lines格式"""
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            if mode == "a" and Path(output_file).exists():
                # 读取现有数据
                with open(output_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)

                # 创建现有row_index集合用于去重
                existing_indexes = {
                    item["row_index"] for item in existing_data if "row_index" in item
                }

                # 从新结果中过滤掉重复项
                new_results = [
                    result
                    for result in results
                    if result.get("row_index") not in existing_indexes
                ]

                combined_results = existing_data + new_results
            else:
                combined_results = results

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    combined_results, f, indent=2, ensure_ascii=False, default=str
                )

            logger.info(
                f"已保存 {len(combined_results)} 条结果到: {output_file} (模式: {mode})"
            )

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
            pathology_id = metadata.get("pathology_id")

            row_data = {
                "row_index": result["row_index"],
                "病理号": pathology_id,
                "处理状态": "成功" if result.get("status") == "success" else "失败",
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


async def main():
    """主执行函数"""
    logger.info("*" * 64)
    logger.info("Job Starting ...")
    logger.info("*" * 64)

    # 1. 配置环境
    configure_environment()

    try:
        # 3. 实例化并加载数据
        pipeline = ReportProcessingPipeline(CSV_FILE_PATH)
        detected_columns = pipeline.detect_columns()

        logger.info(f"\n=== 列映射 ===")
        for std_name, actual_name in detected_columns.items():
            logger.info(f"   {std_name}: {actual_name}")

        # 4. 批量处理（异步）
        results = await pipeline.process_reports_batch(
            text_column=TEXT_COLUMN,
            output_file=OUTPUT_JSON,
            sample_size=SAMPLE_SIZE,
            batch_size=BATCH_SIZE,
            max_concurrent=BATCH_SIZE,
            file_mode="w",  # 最终文件使用覆盖模式
        )

        # 5. 汇总和导出
        summary = pipeline.generate_summary_report(results)
        logger.info(f"\n=== 处理摘要 ===")
        for key, value in summary.items():
            logger.info(f"{key}: {value}")

        pipeline.export_to_csv(results, OUTPUT_CSV)
        logger.info("*" * 64)
        logger.info("Job Done!")
        logger.info("*" * 64)
    except (FileNotFoundError, ValueError, Exception) as e:
        logger.error(f"主程序执行失败: {str(e)}")


def clean_report_text(text: str) -> str:
    """
    清洗和标准化病理报告文本。
    主要包括：移除不可见字符、标准化空白和换行、移除外部引用、
    标准化中英文标点符号和常用标记。
    """
    if not text:
        return ""

    # --- 1. 基础字符清理和标准化 ---
    text = text.replace("\xa0", " ")  # 替换 &nbsp;
    text = text.replace("\u3000", " ")  # 替换全角空格

    # 移除标本类型中的引号（保留原有逻辑）
    text = text.replace('标本类型："', "标本类型：")
    text = text.replace('"标本类型：', "标本类型：")

    # --- 2. 移除外部引用和非必要噪声 ---
    # 移除或替换"请参见/详见报告FXXXX"等外部引用文本
    # 这类文本对当前报告的实体提取无用
    text = re.sub(
        r"(请参见|详见)(病理)?报告[A-Z0-9-]+", "参考其他报告", text, flags=re.IGNORECASE
    )

    # --- 3. 标点符号和格式标准化 ---

    # 统一全角冒号/分号/逗号为半角，并确保后面有空格
    text = text.replace("：", ": ")
    text = text.replace("；", "; ")
    text = text.replace("，", ", ")
    text = text.replace("。", ". ")
    text = text.replace("（", "(")
    text = text.replace("）", ")")
    text = text.replace("“", "")
    text = text.replace("”", "")

    # 统一标准化常见的负号/阴性符号，便于模型识别
    text = text.replace("(阴性)", "(-)")
    text = text.replace("(阳性)", "(+)")

    # --- 4. 空白和换行符处理 (精简文本) ---
    text = text.replace("\n", "; ").replace("\r", "; ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s", "", text).strip()
    text = re.sub(r";+", ";", text).strip()

    text = text.replace(":;", ":无;")
    text = text.replace(".;", ";")
    return text


def run_extraction_execution_native(
    report_text: str, prompt: str, examples: list
) -> Dict[str, Any]:
    context_text = clean_report_text(report_text)
    logger.debug(context_text)
    result = lx.extract(
        text_or_documents=context_text,
        model_url=LLM_BASE_URL,
        model_id=LLM_MODEL_ID,
        api_key=LLM_API_KEY,
        prompt_description=prompt,
        examples=examples,
        extraction_passes=3,  # Improves recall through multiple passes
        max_workers=5,  # Parallel processing for speed
        max_char_buffer=1000,  # Smaller contexts for better accuracy
        use_schema_constraints=True,
    )

    return result


async def run_extraction_execution_native_async(
    report_text: str, prompt: str, examples: list
) -> Dict[str, Any]:
    """异步版本的提取执行函数"""
    # 由于lx.extract可能是同步的，我们在线程池中运行它以避免阻塞事件循环
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: run_extraction_execution_native(report_text, prompt, examples)
    )
    return result


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
