from openai import base_url
import langextract as lx
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List

load_dotenv()


def read_markdown_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
    return content


class ChemicalCompound(BaseModel):
    """Represents a chemical compound (单体化合物) with only fields from the provided table."""

    compound_name: str = Field(..., description="Molecular Name")
    molecular_formula: Optional[str] = Field(None, description="MF")
    molecular_weight: Optional[str] = Field(None, description="MW")
    smiles: Optional[str] = Field(None, description="SMILES")
    cas_number: Optional[str] = Field(None, description="CAS Number")
    source_herb: Optional[str] = Field(None, description="Sourse/草药名字")
    ms_type: Optional[str] = Field(None, description="MS")
    extraction_form: Optional[str] = Field(None, description="提取方式")
    reference: Optional[str] = Field(None, description="Reference")
    reference_title: Optional[str] = Field(None, description="Reference Title")
    compound_concentration: Optional[str] = Field(None, description="Concentration")


class TCMFormula(BaseModel):
    """Represents a Chinese medicine compound/formula (中药复方)."""

    name: str = Field(..., description="Formula name, e.g., 银翘散 (Yinqiao San)")
    source_herbs: Optional[str] = Field(None, description="草药来源（配伍药味清单）")
    ms_type: Optional[str] = Field(None, description="MS Type")
    extraction_form: Optional[str] = Field(None, description="复方化合物样品的提取形式")
    reference: Optional[str] = Field(None, description="文献来源（Reference）")
    reference_title: Optional[str] = Field(None, description="文献名字")


# -----------------------------
# Few-shot examples
# -----------------------------
# A) TCM formula (银翘散) — multiple extraction scenarios
tcm_examples: List[lx.data.ExampleData] = [
    lx.data.ExampleData(
        text=(
            "Name: 银翘散 (Yinqiao San) — 水煎剂\n"
            "草药来源（Source Herbs）: 金银花, 连翘, 薄荷, 牛蒡子, 荆芥, 淡豆豉, 桔梗, 甘草, 竹叶, 芦根\n"
            "MS Type: LC–ESI–QTOF（正负离子切换）\n"
            "复方化合物样品的提取形式: （1）水煎剂 Herbal Decoction\n"
            "Reference: DOI: 10.1016/j.heliyon.2024.e36178\n"
            "Reference Title: The therapeutic effect of Yinqiaosan decoction against influenza A virus infection by regulating T cell receptor signaling pathway\n"
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="name", extraction_text="银翘散 (Yinqiao San)"
            ),
            lx.data.Extraction(
                extraction_class="source_herbs",
                extraction_text="金银花, 连翘, 薄荷, 牛蒡子, 荆芥, 淡豆豉, 桔梗, 甘草, 竹叶, 芦根",
            ),
            lx.data.Extraction(
                extraction_class="ms_type",
                extraction_text="LC–ESI–QTOF（正负离子切换）",
            ),
            lx.data.Extraction(
                extraction_class="extraction_form",
                extraction_text="（1）水煎剂 Herbal Decoction",
            ),
            lx.data.Extraction(
                extraction_class="reference",
                extraction_text="DOI: 10.1016/j.heliyon.2024.e36178",
            ),
            lx.data.Extraction(
                extraction_class="reference_title",
                extraction_text="The therapeutic effect of Yinqiaosan decoction against influenza A virus infection by regulating T cell receptor signaling pathway",
            ),
        ],
    )
]


# B) Single chemical compound examples
# compound_examples = [
#     lx.data.ExampleData(
#         text=(
#             "Name: Prolinum\n"
#             "MF: C5H9NO2  MW: 115.13 g/mol\n"
#             "SMILES: C1C[C@H](NC1)C(=O)O\n"
#             "CAS Number: 147-85-3\n"
#             "草药来源（Source）: Arctium lappa L.（牛蒡）\n"
#             "MS Type: UHPLC-Q-TOF-MS/MS\n"
#             "提取方式: Herbal Decoction\n"
#             "Reference: DOI: 10.1016/j.heliyon.2024.e36178\n"
#             "Reference Title: The therapeutic effect of Yinqiaosan decoction against influenza A virus infection by regulating T cell receptor signaling pathway\n"
#             "Concentration: N/A"
#         ),
#         extractions=[
#             lx.data.Extraction(extraction_class="compound_name", extraction_text="Prolinum"),
#             lx.data.Extraction(extraction_class="molecular_formula", extraction_text="C5H9NO2"),
#             lx.data.Extraction(extraction_class="molecular_weight", extraction_text="115.13 g/mol"),
#             lx.data.Extraction(extraction_class="smiles", extraction_text="C1C[C@H](NC1)C(=O)O"),
#             lx.data.Extraction(extraction_class="cas_number", extraction_text="147-85-3"),
#             lx.data.Extraction(extraction_class="source_herb", extraction_text="Arctium lappa L.（牛蒡）"),
#             lx.data.Extraction(extraction_class="ms_type", extraction_text="UHPLC-Q-TOF-MS/MS"),
#             lx.data.Extraction(extraction_class="extraction_form", extraction_text="Herbal Decoction"),
#             lx.data.Extraction(extraction_class="reference", extraction_text="DOI: 10.1016/j.heliyon.2024.e36178"),
#             lx.data.Extraction(extraction_class="reference_title", extraction_text="The therapeutic effect of Yinqiaosan decoction against influenza A virus infection by regulating T cell receptor signaling pathway"),
#             lx.data.Extraction(extraction_class="compound_concentration", extraction_text="N/A"),
#         ]
#     ),
#     lx.data.ExampleData(
#         text=(
#             "Name: Valine\n"
#             "MF: C5H11NO2  MW: 117.15 g/mol\n"
#             "SMILES: CC(C)[C@@H](C(=O)O)NC(=O)OCC1=CC=CC=C1\n"
#             "CAS Number: 72-18-4\n"
#             "草药来源（Source）: Mentha haplocalyx Briq.（薄荷）\n"
#             "MS Type: UHPLC-Q-TOF-MS/MS\n"
#             "提取方式: Herbal Decoction\n"
#             "Reference: DOI: 10.1016/j.heliyon.2024.e36178\n"
#             "Reference Title: The therapeutic effect of Yinqiaosan decoction against influenza A virus infection by regulating T cell receptor signaling pathway\n"
#             "Concentration: N/A"
#         ),
#         extractions=[
#             lx.data.Extraction(extraction_class="compound_name", extraction_text="Valine"),
#             lx.data.Extraction(extraction_class="molecular_formula", extraction_text="C5H11NO2"),
#             lx.data.Extraction(extraction_class="molecular_weight", extraction_text="117.15 g/mol"),
#             lx.data.Extraction(extraction_class="smiles", extraction_text="CC(C)[C@@H](C(=O)O)NC(=O)OCC1=CC=CC=C1"),
#             lx.data.Extraction(extraction_class="cas_number", extraction_text="72-18-4"),
#             lx.data.Extraction(extraction_class="source_herb", extraction_text="Mentha haplocalyx Briq.（薄荷）"),
#             lx.data.Extraction(extraction_class="ms_type", extraction_text="UHPLC-Q-TOF-MS/MS"),
#             lx.data.Extraction(extraction_class="extraction_form", extraction_text="Herbal Decoction"),
#             lx.data.Extraction(extraction_class="reference", extraction_text="DOI: 10.1016/j.heliyon.2024.e36178"),
#             lx.data.Extraction(extraction_class="reference_title", extraction_text="The therapeutic effect of Yinqiaosan decoction against influenza A virus infection by regulating T cell receptor signaling pathway"),
#             lx.data.Extraction(extraction_class="compound_concentration", extraction_text="N/A"),
#         ]
#     ),
# ]


# -----------------------------
# Prompts and extraction runs
# -----------------------------
prompt_description = "Extract the Traditional Chinese Medicine (TCM) formulas including name, source_herbs, ms_type, extraction_form, reference, reference_title in the order they appear in the text."


all_examples = tcm_examples  # + compound_examples


def extract_compounds(text: str):
    return lx.extract(
        text_or_documents=text,
        prompt_description=prompt_description,
        examples=all_examples,
        model_id="gpt-4o",
        api_key=os.getenv("LANGEXTRACT_API_KEY"),
        model_url=os.getenv("LANGEXTRACT_MODEL_URL"),
        extraction_passes=3,  # Multiple passes for improved recall
        max_workers=10,  # Parallel processing for speed
        max_char_buffer=1000,
    )


if __name__ == "__main__":
    # Run extraction on a chinese medicine article
    source_text = read_markdown_file("./hku/full.md")
    result = extract_compounds(source_text)

    print(result)

    # Save and visualize both runs
    lx.io.save_annotated_documents(
        [result], output_name="chem_extraction.jsonl", output_dir="."
    )

    chem_html = lx.visualize("chem_extraction.jsonl")
    with open("chem_extraction.html", "w", encoding="utf-8") as f:
        f.write(chem_html)

    print("Saved: chem_extraction.jsonl/html")
