# 知识库向量化入库脚本（阶段 1 / RAG 链路骨架）

把 `knowledge_base/*.md`（评课细则语料）切块、用 BGE embedding 向量化、写入 Chroma，并提供检索 demo。

## 环境

- Python 3.13，虚拟环境：`backend\.venv`
- 依赖：`chromadb`、`sentence-transformers`、`numpy`
- 模型：`BAAI/bge-small-zh-v1.5`（首次自动下载，需设置镜像见下）

## 首次构建

```powershell
# 如无法直连 huggingface，先设镜像
$env:HF_ENDPOINT="https://hf-mirror.com"

# 全量构建（解析语料 -> chunk -> embedding -> 写 Chroma）
& "backend\.venv\Scripts\python.exe" "backend\app\scripts\build_kb.py"
```

## 检索 demo

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
& "backend\.venv\Scripts\python.exe" "backend\app\scripts\build_kb.py" --search "导入环节时间太长怎么评价"
```

## 其他

- `--clean`：清空重建
- `--model`：更换 embedding 模型
- `--collection`：更换集合名

## 产物

- 向量库持久化于 `backend\data\chroma`
- 集合 `trial_evaluation_rules`，当前 33 个检索片段（来自评课细则语料）
