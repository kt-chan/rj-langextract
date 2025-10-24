import re
import sys
import os
import langextract as lx
from langextract import data
from langextract.providers import registry
from langextract_glmprovider.provider import GLMProviderLanguageModel, GLMProviderSchema
from dotenv import load_dotenv


lx.providers.load_plugins_once()

PROVIDER_CLS_NAME = "GLMProviderLanguageModel"
PATTERNS = ["^glm"]

# Get API key from environment variables
load_dotenv()
api_key = os.getenv("glm_api_key")
model_id = os.getenv("glm_model_id")
base_url = os.getenv("glm_base_url")


def _example_id(pattern: str) -> str:
    """Generate test model ID from pattern."""
    base = re.sub(r"^\^", "", pattern)
    m = re.match(r"[A-Za-z0-9._-]+", base)
    base = m.group(0) if m else (base or "model")
    return model_id if model_id else f"{base}"


sample_ids = [_example_id(p) for p in PATTERNS]
sample_ids.append("unknown-model")

print("Testing glmProvider Provider - Step 5 Checklist:")
print("-" * 50)


if not api_key:
    print(
        "ERROR: glm_api_key not found in environment variables. Please check your .env file."
    )
    sys.exit(1)

# 1 & 2. Provider registration + pattern matching via resolve()
print("1 and 2. Provider registration & pattern matching")
for model_id in sample_ids:
    try:
        provider_class = registry.resolve(model_id)

        ok = provider_class.__name__ == PROVIDER_CLS_NAME
        status = "Passed" if (ok or model_id == "unknown-model") else "Failed"
        note = (
            "expected"
            if ok
            else (
                "expected (no provider)"
                if model_id == "unknown-model"
                else "unexpected provider"
            )
        )
        print(
            f"   {status} {model_id} -> {provider_class.__name__ if ok else 'resolved'} {note}"
        )
    except Exception as e:
        if model_id == "unknown-model":
            print(f"   Passed {model_id}: No provider found (expected)")
        else:
            print(f"   Failed {model_id}: resolve() failed: {e}")

# 3. Inference sanity check
print("\n3. Test inference with sample prompts")
try:
    model_id = (
        sample_ids[0]
        if sample_ids[0] != "unknown-model"
        else (_example_id(PATTERNS[0]) if PATTERNS else "test-model")
    )
    provider = GLMProviderLanguageModel(
        model_id=model_id, api_key=api_key, base_url=base_url, verify_ssl=False
    )
    prompts = ["Who are you?", "What can you do?"]
    results = list(provider.infer(prompts))
    print(f"   Passed Inference returned {len(results)} results")
    for i, result in enumerate(results):
        try:
            out = result[0].output if result and result[0] else None
            print(f"   Passed Result {i+1}: {(out or '')[:60]}...")
        except Exception:
            print(f"   Failed Result {i+1}: Unexpected result shape: {result}")
except Exception as e:
    print(f"   Failed ERROR: {e}")

# 4. Test schema creation and application
print("\n4. Test schema creation and application")
try:

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

    examples = [
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
                    attributes={
                        "score": "0",
                        "result": "阴性",
                        "description": "无染色",
                    },
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
    ]

    schema = GLMProviderSchema.from_examples(examples)
    print(f"   Passed Schema created (keys={list(schema.schema_dict.keys())})")

    schema_class = GLMProviderLanguageModel.get_schema_class()
    print(f"   Passed Provider schema class: {schema_class.__name__}")

    provider = GLMProviderLanguageModel(
        model_id=_example_id(PATTERNS[0]) if PATTERNS else "test-model",
        api_key=api_key,
        base_url=base_url,
        verify_ssl=False,
    )
    provider.apply_schema(schema)
    print(
        f"   Passed Schema applied: response_schema={provider.response_schema is not None} structured={getattr(provider, 'structured_output', False)}"
    )
    prompts = [prompt]
    results = list(provider.infer(prompts))
    print(f"   Passed Inference returned {len(results)} results")
    
except Exception as e:
    print(f"   Failed ERROR: {e}")

# 5. Test factory integration
print("\n5. Test factory integration")
try:
    from langextract import factory

    config = factory.ModelConfig(
        model_id=_example_id(PATTERNS[0]) if PATTERNS else "test-model",
        provider="GLMProviderLanguageModel",
        provider_kwargs={"base_url": base_url, "api_key": api_key, "verify_ssl": False},
    )
    model = factory.create_model(config)
    print(f"   Passed Factory created: {type(model).__name__}")
except Exception as e:
    print(f"   Failed ERROR: {e}")

print("\n" + "-" * 50)
print("Passed! Testing complete!")
