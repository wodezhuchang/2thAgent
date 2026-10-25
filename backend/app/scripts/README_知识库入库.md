# 知识库入库脚本说明（阶段 1 / RAG 链路）

> 本文档说明知识库入库脚本的用法。**当前主脚本为 `kb_manager.py`**（多源增量 + 扫描件 OCR），`build_kb.py` 为早期单文件版本，已被取代，仅作参考。

## 环境

- Python 3.13，虚拟环境：`backend\.venv`
- 依赖：`chromadb`、`sentence-transformers`、`numpy`、`pypdf`、`PyMuPDF`、`rapidocr-onnxruntime`、`fontTools`
- 模型：`BAAI/bge-small-zh-v1.5`（首次自动下载，需设置镜像见下）

## 模型下载（首次）

```powershell
# 如无法直连 huggingface，先设镜像
$env:HF_ENDPOINT="https://hf-mirror.com"
```

## 常用命令（主脚本 kb_manager.py）

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"

# 全量入库 / 增量更新（新增、修改的入库，未变的自跳过）
& "backend\.venv\Scripts\python.exe" "backend\app\scripts\kb_manager.py" --update

# 查看知识库状态（文档数、片段数、按类型分布）
& "backend\.venv\Scripts\python.exe" "backend\app\scripts\kb_manager.py" --status

# 检索（可过滤类型/学科）
& "backend\.venv\Scripts\python.exe" "backend\app\scripts\kb_manager.py" --search "义务教育数学课程标准对核心素养的要求" --subject 数学 --top 3
```

完整说明见 `D:\2thAgent\docs\知识库管理系统_使用与复现指南.md`。

## 扫描型 PDF 的 OCR 自动回退

- 先 `pypdf` 提取文本；若提取过少（<200 字，判定为扫描件），自动用 **PyMuPDF 渲染每页 + RapidOCR 识别**后入库。
- 因此文本型与扫描型 PDF 都能处理，无需手工区分。
- 内置 16 个扫描型义务教育课标（2022版）经此路径全部入库，OCR 质量已抽检达标（见 `docs\评审记录_E1-E5.md` 的 E1）。

## 产物

- 向量库持久化于 `backend\data\chroma`
- 集合 `trial_evaluation_rules`，当前 **49 文档 / 12,423 片段**（课标 31 + 量表 4 + 案例 14 + 评课细则 1）
