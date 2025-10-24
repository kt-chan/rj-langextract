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
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langextract_glmprovider.provider import GLMProviderLanguageModel, GLMProviderSchema

LLM_API_KEY = None
LLM_BASE_URL = None
LLM_MODEL_ID = None

# Add this line to check the effective level of the root logger
print(f"Root logger level before basicConfig: {logging.getLogger().getEffectiveLevel()}") 


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # <-- ADD THIS LINE
    force=True 
)
logger = logging.getLogger(__name__)
logger.info("test A ...")
print("test B .... ")


def main():
    """主执行函数"""
    logger.info("*" * 64)
    logger.info("Job Starting ...")
    logger.info("*" * 64)


if __name__ == "__main__":
    main()
