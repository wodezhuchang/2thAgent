# -*- coding: utf-8 -*-
"""全局配置：路径、LLM、RAG、记忆、评价指标。

对应开发前文件 §3.6 六层技术栈选型与 §3.7 混合路线。
L1 模型层：DeepSeek（OpenAI 兼容）作为推理/评价模型。
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parents[3]          # D:\2thAgent
BACKEND_DIR = BASE_DIR / "backend"
APP_DIR = BACKEND_DIR / "app"
DATA_DIR = BACKEND_DIR / "data"
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
DOCS_DIR = BASE_DIR / "docs"

# 从 backend/.env 加载密钥等环境变量（key 不写死在源码）
_ENV_FILE = BACKEND_DIR / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)

# 向量库（阶段1 已建成，kb_manager 使用同一路径）
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "trial_evaluation_rules"

# 长期记忆 SQLite（阶段2 新增）
MEMORY_DB_PATH = DATA_DIR / "memory" / "agent_memory.db"
TRACE_DIR = DATA_DIR / "traces"          # 轨迹日志（append-only）

# ---------- LLM（L1 模型层）----------
# 从 .env / 环境变量读取（key 不硬编码）。可用 LLM_BASE_URL/LLM_MODEL 覆盖默认。
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")   # 推理模型（阶段4 可切 R1）
LLM_API_KEY_ENV = "DEEPSEEK_API_KEY"
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# ---------- RAG（阶段1 已建成，此处仅暴露检索参数）----------
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
RAG_TOP_K = 4                    # 每个检索召回片段数

# ---------- 评价指标（对齐《标准化评价指标体系_v1.0.md》）----------
# 四维度权重（百分制合成，见指标体系 §6）
DIM_WEIGHTS = {
    "D1": 0.25,   # 教学设计
    "D2": 0.30,   # 讲授语言
    "D3": 0.20,   # 板书
    "D4": 0.25,   # 教姿教态
}
# 各维度二级指标键（按体系文档）
DIM_INDICATORS = {
    "D1": ["D1.1", "D1.2", "D1.3", "D1.4"],
    "D2": ["D2.1", "D2.2", "D2.3", "D2.4", "D2.5"],
    "D3": ["D3.1", "D3.2", "D3.3", "D3.4"],
    "D4": ["D4.1", "D4.2", "D4.3", "D4.4", "D4.5"],
}
INDICATOR_NAMES = {
    "D1.1": "目标定位", "D1.2": "重难点把握", "D1.3": "环节结构", "D1.4": "内容组织",
    "D2.1": "语言准确性", "D2.2": "表达流畅度", "D2.3": "语速与节奏", "D2.4": "音量与感染力", "D2.5": "互动提问",
    "D3.1": "字体规范", "D3.2": "书写规整", "D3.3": "布局均衡", "D3.4": "内容呼应",
    "D4.1": "站姿", "D4.2": "手势", "D4.3": "眼神交流", "D4.4": "表情应用", "D4.5": "肢体移动",
}
# 免责提示（固定置于报告开头）
DISCLAIMER = "本评价为 AI 辅助练习建议，不作为课程最终考核结果，专业评判以任课教师为准。"
# 指标/Prompt 版本号（可复现，写入记忆 evaluation.prompt_version）
INDICATOR_VERSION = "v1.0"
PROMPT_VERSION = "v1.0"
