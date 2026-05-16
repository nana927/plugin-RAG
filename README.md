#  RAG 模块说明

一个即插即用的RAG 检索模块。它负责三件事：

1. 把文件切成文本块。
2. 把文本块转成向量并存入向量库。
3. 根据用户问题召回相关文本块。

默认模式是 `FAISS + hash embedding + simple chunker`，适合本地快速跑通流程。
如果要解析 PDF、扫描件、复杂表格，可以切换到 `deepdoc`。

## 项目结构

```text
RAG/
  __init__.py
  README.md
  .env.example

  config.py                  # 读取 .env / 环境变量，生成 RagConfig
  types.py                   # ChunkRecord / SearchResult 数据结构
  embeddings.py              # hash / dashscope / local embedding
  chunking.py                # simple / deepdoc 文档切块入口
  ingestion.py               # 文件入库：file -> chunks -> records -> RagService
  rag_service.py             # RAG 主服务：add_documents / retrieve / clear
  intent_router.py           # 意图识别与 temperature 策略

  pre_process/               # 预处理模块
    __init__.py
    qa_generator.py          # 入库前生成固定 QA records
    query_decomposer.py      # 检索前复合问题拆解
    table_utils.py           # HTML 表格结构化、合并单元格补齐

  stores/                    # 检索存储层
    __init__.py
    base.py                  # 向量库抽象接口
    faiss_store.py           # 本地 FAISS 存储
    milvus_store.py          # Milvus 存储
    bm25_store.py            # BM25 倒排索引

  examples/                  # 可运行脚本
    faiss_local_cli.py       # 命令行入库 + 交互式问答 demo
    ingest_file_demo.py      # 文件入库示例
    evaluate_rag_qa.py       # RAG 问答评测脚本

  parser/                    # RAGFlow/DeepDoc 迁移解析入口
    manual.py                # DeepDoc 手动解析入口，DeepDocChunker 会调用
    nlp/                     # 分词、query、term weight、旧搜索工具
    utils/                   # 文件路径、ES 连接等兼容工具

  deepdoc/                   # DeepDoc 解析链路
    parser/                  # PDF/DOCX/Excel/PPT/HTML/TXT 等解析器
    vision/                  # OCR、版面识别、表格结构识别
    parser/resume/           # 简历结构化解析附加能力，当前主链路不依赖

  data/
    files/                   # 本地测试文档；公开仓库建议忽略
    testcase/                # 评测测试集
    eval_outputs*/           # 评测输出；运行产物，建议忽略

  storage/                   # FAISS / BM25 本地索引；运行产物，建议忽略

  requirements-rag.txt              # 基础 FAISS + BM25 依赖
  requirements-deepdoc.txt          # DeepDoc PDF/OCR/表格解析依赖
  requirements-milvus.txt           # Milvus 后端依赖
  requirements-dashscope.txt        # DashScope/OpenAI-compatible 调用依赖
  requirements-local-embedding.txt  # 本地 sentence-transformers embedding 依赖
  requirements-eval.txt             # 评测脚本依赖
  requirements-all.txt              # 全量依赖
```

运行产物说明：

```text
__pycache__/
*.pyc
storage/
RAG/storage/
data/eval_outputs*/

这些目录或文件是运行缓存、索引和评测结果，不建议提交到 GitHub。
```



## 支持的文件格式

当前文件解析由 `RAG_CHUNKER` 决定：

```text
RAG_CHUNKER=simple
  适合普通文本类文件。
  解析方式：直接读取文本内容，再按 RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP 切块。

RAG_CHUNKER=deepdoc
  适合 PDF、Office 文档、复杂版面和表格。
  解析方式：走 DeepDoc / RAGFlow 风格解析链路，包含 PDF 解析、OCR、版面识别、表格识别等。
```

### simple chunker

适合：

```text
.txt
.md
.json
.jsonl
.csv
.log
其他可以按 UTF-8 文本读取的文件
```

不适合：

```text
.pdf
.docx
.pptx
.xlsx
扫描件
复杂表格文档
```

### deepdoc chunker

适合：

```text
.pdf
.docx
.doc
.pptx
.ppt
.xlsx
.xls
.html
.htm
.json
.md
.txt
```

其中：

```text
PDF
  支持普通文本 PDF，也支持需要 OCR / 版面识别 / 表格识别的复杂 PDF。

DOCX / PPTX / XLSX
  依赖 python-docx / python-pptx / openpyxl 等解析库。

HTML / Markdown / TXT / JSON
  可以通过 DeepDoc parser 统一解析，但如果只是普通文本，simple chunker 更轻。
```

注意：

```text
1. 使用 deepdoc 前需要安装 requirements-deepdoc.txt。
2. 扫描件 PDF 的 OCR 效果依赖 DeepDoc 模型和图片质量。
3. 表格会优先保留结构，HTML-like 的 <table>/<tr>/<td>/<th> 会在入库前转成结构化文本和 metadata。
4. 当前不建议把音频、视频、图片文件直接入库；如需支持，需要先增加专门的 OCR/ASR/图像理解预处理链路。
```

## 



## 代码调用关系

### 文件入库

```text
examples/faiss_local_cli.py
  命令行入口，只读取 --file。
  backend、chunker、knowledge base、hybrid、debug 等配置都从 .env 读取。
  路径：RAG/examples/faiss_local_cli.py
  -> ingestion.build_file_service()
     创建 RagService，并组织文件入库流程。
     路径：RAG/ingestion.py
  -> ingestion.build_records_from_file()
     调用切块器解析文件，并把结果转换成待入库的 ChunkRecord 列表。
     路径：RAG/ingestion.py
  -> chunking.build_chunker()
     根据 config.chunker 选择 simple 或 deepdoc。
     路径：RAG/chunking.py
     - simple: 读取普通文本，按 chunk_size / overlap 切成 chunks。
       路径：RAG/chunking.py，SimpleChunker.chunk_file()
     - deepdoc: 调用 parser.manual.chunk()，内部执行 PDF/DOCX 解析、OCR、版面识别和表格识别。
       路径：RAG/chunking.py，DeepDocChunker.chunk_file()
       路径：RAG/parser/manual.py，chunk()
       路径：RAG/deepdoc/
  -> ingestion.chunks_to_records()
     把 chunk 字典转换成统一的 ChunkRecord 数据结构。
     路径：RAG/ingestion.py
  -> RagService.add_documents()
     接收入库 records，准备做 embedding 和向量库写入。
     路径：RAG/rag_service.py
  -> embeddings.*.embed_texts()
     根据 RAG_EMBEDDING 生成向量，支持 hash / dashscope / local。
     路径：RAG/embeddings.py
  -> stores.*.upsert()
     根据 RAG_BACKEND 写入 FAISS 或 Milvus。
     路径：RAG/stores/faiss_store.py，FaissStore.upsert()
     路径：RAG/stores/milvus_store.py，MilvusStore.upsert()
```

### 检索

```text
用户问题
  用户在命令行或业务接口中输入的问题文本。
  -> RagService.retrieve()
     RAG 检索入口，接收 query、knowledge_base 和 top_k。
  -> embedding.embed_query()
     把用户问题转换成查询向量。
  -> vector store search
     在 FAISS 或 Milvus 中按向量相似度召回候选 chunks。
  -> 如果 hybrid_enabled=true，再走 BM25 召回
     从 RAG/stores/bm25_store.py 维护的倒排索引里按 BM25 分数召回候选 chunks。
  -> 合并两路候选并重排
     按 chunk_id 去重，分别归一化 vector_score 和 bm25_score，再加权融合。
  -> 返回 SearchResult 列表
     返回命中的文本块、文件名、分数和 metadata。
```

当前混合检索是完整的双路召回：

```text
用户问题
  -> 向量召回 candidate_k
     路径：RAG/rag_service.py，RagService.retrieve()
     路径：RAG/stores/faiss_store.py 或 RAG/stores/milvus_store.py
  -> BM25 召回 candidate_k
     路径：RAG/stores/bm25_store.py，Bm25Store.search()
  -> 合并 chunk_id 去重
     路径：RAG/rag_service.py，RagService._merge_hybrid_results()
  -> 分别归一化 vector_score 和 bm25_score
  -> final_score = RAG_HYBRID_VECTOR_WEIGHT * normalized_vector_score
                  + (1 - RAG_HYBRID_VECTOR_WEIGHT) * normalized_bm25_score
  -> 返回最终 top_k
```

BM25 索引在入库时同步维护：

```text
RagService.add_documents()
  -> stores.Bm25Store.upsert()
     路径：RAG/stores/bm25_store.py
```

删除或清空知识库时也会同步维护 BM25：

```text
RagService.delete_document()
  -> Bm25Store.delete_document()

RagService.clear()
  -> Bm25Store.clear()
```

相关配置：

```text
RAG_HYBRID_ENABLED=true
RAG_HYBRID_VECTOR_WEIGHT=0.7
RAG_HYBRID_CANDIDATE_K=20
RAG_BM25_K1=1.5
RAG_BM25_B=0.75
```

## 意图识别与 temperature

评测脚本会先识别用户问题的意图，再根据意图选择生成答案时的大模型 `temperature`。

实现路径：

```text
RAG/intent_router.py
  -> detect_intent()
     规则优先识别意图，必要时可用 LLM 兜底。
  -> IntentProfile
     保存意图含义、是否严格依赖文档、推荐 temperature。

RAG/examples/evaluate_rag_qa.py
  -> detect_intent()
  -> generate_answer(..., intent_profile=...)
  -> llm.chat(..., temperature=intent_profile.temperature)
  -> detail/log 写入 predicted_intent、intent_temperature、document_dependency
```

当前意图表：

```text
qa                 知识问答        严格依赖文档      temperature=0.0
extract            信息抽取        严格依赖文档      temperature=0.0
compare            对比分析        严格依赖文档      temperature=0.1
summary            文档总结        严格依赖文档      temperature=0.2
reason_analysis    原因分析        半依赖文档        temperature=0.3
plan_generation    方案生成        半依赖文档        temperature=0.5
tool_call          工具调用        严格依赖文档      temperature=0.0
clarification      澄清反问        不严格依赖文档    temperature=0.3
privacy_rejection  文档防泄露拒答  严格依赖文档      temperature=0.0
context_followup   多轮追问        视情况            temperature=0.2
chat               普通闲聊        不严格依赖文档    temperature=0.7
```

默认使用规则识别：

```text
RAG_INTENT_USE_LLM=false
```

如果规则没有识别出来，并且希望用大模型兜底分类，可以开启：

```text
RAG_INTENT_USE_LLM=true
```

## 入库预生成 QA 增强

可以在文档入库时先用较强模型生成一批固定 QA，并把这些 QA 作为额外 records 写入同一个向量库。
召回时系统会先额外找 QA records：如果用户问题和某条预生成 Q 匹配，就把对应 A 放到上下文前面。

整体链路：

```text
文件入库
  -> ingestion.build_records_from_file()
     先得到原始文档 chunks。
  -> pre_process.qa_generator.generate_qa_records()
     用强模型基于文档内容生成固定 QA。
  -> RagService.add_documents()
     原始 chunks 和 QA records 一起写入向量库与 BM25。

用户问题
  -> RagService.retrieve()
     原始 chunks 和 QA records 在同一个向量库 / BM25 索引里一起召回。
  -> 混合检索排序
     QA records 不再特殊前置，而是和普通 chunks 一起按最终分数排序。
  -> build_context()
     排进 top_k 的 QA records 会作为普通上下文参与回答。
```

QA record 的文本格式类似：

```text
【预生成QA】
问题：稳定性测试要求是什么？
答案：稳定性测试要求是在 80% 最佳并发数下运行 3*24h。
```

相关配置：

```text
RAG_QA_GENERATION_ENABLED=false
RAG_QA_GENERATION_MODEL=qwen-max
RAG_QA_GENERATION_MAX_CONTEXT_CHARS=8000
RAG_QA_GENERATION_MAX_PAIRS=20
```

第一次建议这样试：

```text
RAG_QA_GENERATION_ENABLED=true
RAG_QA_GENERATION_MODEL=qwen-max
RAG_QA_GENERATION_MAX_PAIRS=20
```

然后重新入库一次。QA records 是入库时生成的，旧文档不重新入库不会自动补这层索引。

## 显式导入方式

不要从 `RAG` 根包直接导入业务对象。

推荐：

```python
from RAG.config import RagConfig
from RAG.rag_service import RagService
from RAG.ingestion import build_file_service
from RAG.types import ChunkRecord
```

不推荐：

```python
from RAG import RagConfig, RagService
```

这样写是为了让调用方一眼看出对象来自哪个模块。

## 运行方式

先进入 RAG上一层级 目录：

```powershell
cd RAG上一层级
```

### 1. 跑内置 demo

```powershell
python -m RAG.examples.faiss_local_cli (--file XXX) (-y)
```

启动后输入问题：

```text
Question> metric
Question> rag
Question> q
```

`q`、`quit`、`exit` 都可以退出。

### 2. 把普通文本文件放进知识库

```powershell
python -m RAG.examples.faiss_local_cli --file filepath
```

参数说明：

```text
--file      要导入的文件
其他配置从 RAG/.env 读取，例如 RAG_KNOWLEDGE_BASE、RAG_CHUNKER。
```

如果不想遇到重复文件时确认，可以加 `-y`：

```powershell
python -m RAG.examples.faiss_local_cli --file filepath -y
```

开启“向量 + 关键词重排”：

```powershell
RAG_HYBRID_ENABLED=true
```

查看关键词抽取和候选块打分日志：

```powershell
RAG_RETRIEVAL_DEBUG=true
```

调试日志会打印：

```text
[retrieval-debug] query terms
[retrieval-debug] candidate vector / term score
[retrieval-debug] final score
```

### 3. 使用 Milvus 向量库

默认示例使用本地 FAISS。如果要走 Milvus，需要先准备两件事：

```text
1. Python 环境安装 pymilvus。
2. 本机或远程启动 Milvus 服务，并保证 19530 端口可访问。
```

#### 3.1 安装 pymilvus

在 `test_agent` 环境里安装 Milvus Python SDK：

```powershell
cd RAG的上层目录
conda activate 你的环境
pip install -r RAG\requirements-milvus.txt
```

验证是否安装成功：

```powershell
python -c "import pymilvus; print(pymilvus.__version__)"
```

#### 3.2 安装并启动 Milvus 服务

Milvus 官方 Docker Compose 文档：

```text
https://milvus.io/docs/install_standalone-docker-compose.md

具体步骤如下：
```

先安装并启动 Docker Desktop：

```text
https://www.docker.com/products/docker-desktop/



```

确认 Docker 可用：

```powershell
docker --version
docker compose version
docker ps
```

如果 `docker ps` 报 `failed to connect to the docker API`，说明 Docker Desktop 还没启动。

创建一个专门放 Milvus compose 文件的目录：

```powershell
mkdir milvus-standalone
cd milvus-standalone
```

下载官方 standalone `docker-compose.yml`。

PowerShell 可以用：

```powershell
Invoke-WebRequest `
  -Uri "https://github.com/milvus-io/milvus/releases/download/v2.4.23/milvus-standalone-docker-compose.yml" `
  -OutFile "docker-compose.yml"
```

如果你装了 `curl`，也可以用：

```powershell
curl -L "https://github.com/milvus-io/milvus/releases/download/v2.4.23/milvus-standalone-docker-compose.yml" -o docker-compose.yml
```

确认docker desktop打开，启动 Milvus：

```powershell
docker compose up -d
```

查看容器状态：

```powershell
docker compose ps
```

正常会看到类似这些容器：

```text
milvus-etcd
milvus-minio
milvus-standalone
```

其中 `milvus-standalone` 会监听本机 `19530` 端口。

确认端口可访问：

```powershell
Test-NetConnection localhost -Port 19530
```

如果返回：

```text
TcpTestSucceeded : True
```

说明 Milvus 服务可以连接。

Milvus Web UI 通常可以打开：

```text
http://127.0.0.1:9091/webui/
```

停止 Milvus：

```powershell
docker compose down
```

如果要连数据一起删除，在 `milvus-standalone` 目录下删除 `volumes`：

```powershell
Remove-Item -Recurse -Force .\volumes
```

#### 3.3 让 RAG 走 Milvus

设置 RAG 使用 Milvus：

```powershell
$env:RAG_BACKEND="milvus"
$env:MILVUS_HOST="localhost"
$env:MILVUS_PORT="19530"
$env:MILVUS_COLLECTION="rag_chunks"
```

然后另开一个窗口，cd 到RAG的上层目录，运行文件入库和检索：

```powershell
python -m RAG.examples.faiss_local_cli --file RAG\README.md -y
```

也可以在代码里显式配置：

```python
from RAG.config import RagConfig
from RAG.ingestion import build_file_service

config = RagConfig(
    backend="milvus",
    chunker="simple",
    milvus_host="localhost",
    milvus_port="19530",
    milvus_collection="rag_chunks",
)

service = build_file_service(
    "RAG/README.md",
    knowledge_base="readme_milvus_kb",
    config=config,
    ask_on_existing=False,
)
```

注意：当前 Milvus 通路主要是基础向量检索。FAISS 通路里的关键词 hybrid 重排和
`--debug` 候选分数日志目前主要针对 FAISS 实现，Milvus 通路还可以后续补齐。

## 代码示例

### 手动写入 records

```python
from RAG.config import RagConfig
from RAG.rag_service import RagService
from RAG.types import ChunkRecord

service = RagService(RagConfig())

records = [
    ChunkRecord(
        chunk_id="chunk_1",
        doc_id="doc_1",
        doc_name="example.txt",
        text="This is a RAG chunk.",
        metadata={"source": "demo"},
    )
]

service.add_documents(records, knowledge_base="demo")
results = service.retrieve("RAG", knowledge_base="demo", top_k=3)
```

### 从文件入库

```python
from RAG.ingestion import build_file_service

service = build_file_service(
    file_path,
    knowledge_base="resume_kb",
    chunker="deepdoc",
)
```

### 只把文件转成 records

```python
from RAG.config import RagConfig
from RAG.ingestion import build_records_from_file

config = RagConfig(chunker="simple")
records = build_records_from_file("RAG/README.md", "readme_kb", config)
```

## 配置文件

所有可配置参数写在：

```text
RAG/.env.example
```

主要分为几类：

```text
Storage
  RAG_STORAGE_DIR
  RAG_KNOWLEDGE_BASE
  RAG_DEMO_KNOWLEDGE_BASE

Vector store
  RAG_BACKEND
  MILVUS_HOST
  MILVUS_PORT
  MILVUS_COLLECTION

Embedding
  RAG_EMBEDDING
  RAG_EMBEDDING_MODEL
  RAG_EMBEDDING_DIM
  DASHSCOPE_API_KEY
  DASHSCOPE_BASE_URL
  OPENAI_API_KEY
  RAG_LOCAL_EMBEDDING_MODEL

Chunking
  RAG_CHUNKER
  RAG_CHUNK_SIZE
  RAG_CHUNK_OVERLAP

Hybrid retrieval
  RAG_HYBRID_ENABLED
  RAG_HYBRID_VECTOR_WEIGHT
  RAG_HYBRID_CANDIDATE_K
  RAG_BM25_K1
  RAG_BM25_B
  RAG_RETRIEVAL_DEBUG

Query decomposition
  RAG_QUERY_DECOMPOSITION_ENABLED
  RAG_QUERY_DECOMPOSITION_API_KEY
  RAG_QUERY_DECOMPOSITION_BASE_URL
  RAG_QUERY_DECOMPOSITION_MODEL
  RAG_QUERY_DECOMPOSITION_MAX_SUBQUERIES

Intent recognition
  RAG_INTENT_USE_LLM

Generated QA
  RAG_QA_GENERATION_ENABLED
  RAG_QA_GENERATION_API_KEY
  RAG_QA_GENERATION_BASE_URL
  RAG_QA_GENERATION_MODEL
  RAG_QA_GENERATION_TEMPERATURE
  RAG_QA_GENERATION_MAX_CONTEXT_CHARS
  RAG_QA_GENERATION_MAX_PAIRS
  RAG_QA_PDF_PREVIEW_PAGES
  RAG_QA_PDF_PREVIEW_SUMMARY_ENABLED
  RAG_QA_PDF_PREVIEW_MAX_CHARS
  RAG_QA_PDF_PREVIEW_SUMMARY_MAX_CHARS

Keyword extraction
  RAG_KEYWORD_API_KEY
  RAG_KEYWORD_BASE_URL
  RAG_KEYWORD_MODEL

DeepDoc / RAGFlow compatibility
  RAG_PROJECT_BASE
  RAG_DEPLOY_BASE
  ES_URL

RAG QA evaluation
  RAG_EVAL_K
  RAG_EVAL_TOP_K
  RAG_EVAL_MAX_CONTEXT_CHARS
  RAG_EVAL_OUTPUT_DIR
  RAG_EVAL_LOW_SCORE_THRESHOLD
  RAG_EVAL_RESUME
  RAG_EVAL_RETRY_FAILED
  RAG_EVAL_RUN_ID
  RAG_EVAL_ANSWER_MODE
  RAG_EVAL_API_KEY
  RAG_EVAL_BASE_URL
  RAG_EVAL_MODEL
  RAG_EVAL_LLM_TIMEOUT
  RAG_EVAL_LLM_RETRIES
  RAG_EVAL_LLM_RETRY_DELAY
  RAG_EVAL_ALLOW_MISSING_LLM
```

## 依赖安装

最小本地运行：

```powershell
pip install -r RAG/requirements-rag.txt
```

Milvus 向量库：

```powershell
pip install -r RAG/requirements-milvus.txt
```

DashScope / OpenAI-compatible embedding：

```powershell
pip install -r RAG/requirements-dashscope.txt
```

本地语义 embedding：

```powershell
pip install -r RAG/requirements-local-embedding.txt
```

DeepDoc PDF 解析：

```powershell
pip install -r RAG/requirements-deepdoc.txt
```

评测脚本：

```powershell
pip install -r RAG/requirements-eval.txt
```

全量安装：

```powershell
pip install -r RAG/requirements-all.txt
```

Windows 下 `datrie` 用 pip 可能失败，推荐：

```powershell
conda install -n test_agent -c conda-forge datrie -y
```

DeepDoc 需要 NLTK 数据：

```powershell
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('wordnet')"
```

DeepDoc 自带的 XGBoost 模型是旧格式，所以需要：

```powershell
pip install "xgboost>=3.0.2,<3.1"
```

`torch` 没有放进 DeepDoc 默认依赖。它只用于判断 CUDA 是否可用，缺失时会自动走 CPU；
如果你后续要强制使用 GPU，再按自己的 CUDA 版本单独安装 PyTorch。



## RAG 问答评测脚本

脚本位置：

```text
RAG/examples/evaluate_rag_qa.py
```

作用：

```text
1. 读取 JSON / JSONL / CSV 测试集。
2. 对每个问题调用当前 .env 配置下的 RagService.retrieve()。
3. 根据召回内容生成 RAG 输出：
   - RAG_EVAL_ANSWER_MODE=llm 时，用大模型基于召回内容生成答案。
   - RAG_EVAL_ANSWER_MODE=context 时，直接把召回内容当作 RAG 输出，适合只测召回。
4. 用大模型对比 RAG 输出和 standard_answer，得到 llm_accuracy。
5. 统计 answer_keywords 在 RAG 输出中的覆盖率，得到 keyword_coverage。
6. 按公式计算 final_accuracy：
   final_accuracy = RAG_EVAL_K * llm_accuracy + (1 - RAG_EVAL_K) * keyword_coverage
7. 输出整体平均指标、按 intent 分类指标、按 intent/sub_intent 分类指标。
```

测试集字段支持：

```text
case_id
question
standard_answer
answer_keywords
intent
sub_intent
```

如果后续上传的新文件字段名不同，脚本也兼容常见别名，例如 `query`、`问题`、`标准答案`、`keywords`、`类别`。

运行方式：

```powershell
cd RAG的上级目录
python -m RAG.examples.evaluate_rag_qa "测试集路径.json"
```

只测试前 N 条：

```powershell
python -m RAG.examples.evaluate_rag_qa "测试集路径.json" --limit 20
```

评测相关 `.env` 参数：

```text
RAG_EVAL_K=0.7
RAG_EVAL_TOP_K=5
RAG_EVAL_MAX_CONTEXT_CHARS=6000
RAG_EVAL_OUTPUT_DIR=RAG/data/eval_outputs
RAG_EVAL_LOW_SCORE_THRESHOLD=0.6
RAG_EVAL_RESUME=false
RAG_EVAL_RETRY_FAILED=true
RAG_EVAL_RUN_ID=
RAG_EVAL_ANSWER_MODE=llm
RAG_EVAL_API_KEY=
RAG_EVAL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RAG_EVAL_MODEL=qwen-plus
RAG_EVAL_LLM_TIMEOUT=60
RAG_EVAL_LLM_RETRIES=3
RAG_EVAL_LLM_RETRY_DELAY=1.0
RAG_EVAL_ALLOW_MISSING_LLM=false
```

输出文件：

```text
RAG/data/eval_outputs/eval_details_时间戳.jsonl
  每一题的明细：问题、标准答案、RAG 输出、召回片段、关键词覆盖、大模型评分、融合准确率、评分原因。

RAG/data/eval_outputs/eval_low_scores_时间戳.jsonl
  低分样本明细：只记录成功运行但 final_accuracy 低于 RAG_EVAL_LOW_SCORE_THRESHOLD 的题目。

RAG/data/eval_outputs/eval_summary_时间戳.json
  汇总指标：整体平均、按 intent 分类、按 intent/sub_intent 分类。

RAG/data/eval_outputs/eval_时间戳.log
  必要运行日志：每题耗时、分数、命中关键词、异常信息。
```

断点续跑：

```text
RAG_EVAL_RESUME=true
```

开启后，脚本会读取同一个 run_id 的 `eval_details_时间戳.jsonl`，跳过已经成功完成的 case，并继续追加写入明细、低分样本和汇总文件。

如果不指定 `RAG_EVAL_RUN_ID`，脚本会自动续跑输出目录中最近一次 `eval_details_*.jsonl` 对应的 run_id。

也可以手动指定某次运行：

```text
RAG_EVAL_RESUME=true
RAG_EVAL_RUN_ID=20260514_101500
```

注意：

```text
1. 正式使用大模型生成答案和评分时，需要安装评测依赖：
   pip install -r RAG/requirements-eval.txt
2. 如果只想先验证检索和关键词覆盖率，可以临时设置：
   RAG_EVAL_ANSWER_MODE=context
   RAG_EVAL_ALLOW_MISSING_LLM=true
3. 评测前要先把被测文档入库到 RAG_KNOWLEDGE_BASE，否则测试集问题和知识库内容不匹配，指标会很低。
```



# 实测效果

入库文件为pdf格式，包含跨页表格等复杂逻辑。
使用1000条测试用例进行测试，测试效果如下：

| 意图              | 数据量 | 效果[0-1] |
| ----------------- | ------ | --------- |
| 整体效果          | 1000   | 0.66      |
| clarification     | 4      | 0.45      |
| compare           | 131    | 0.27      |
| context_followup  | 3      | 0.47      |
| extract           | 178    | 0.63      |
| plan_generation   | 39     | 0.79      |
| privacy_rejection | 3      | 0.33      |
| qa                | 222    | 0.60      |
| reason_analysis   | 159    | 0.77      |
| summary           | 130    | **0.90**  |
| tool_call         | 131    | 0.80      |

# 项目持续优化中...后续会做：

1、对pdf、doc以外的文件格式更好地的支持；

2、效果的提升；

3、并加入文档中图片信息的提取。