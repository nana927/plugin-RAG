

## FAQ

### 为什么要用 python -m 运行？

推荐从 backend 目录运行：

```powershell
python -m RAG.examples.faiss_local_cli
```

不要进入 `RAG/` 目录直接运行脚本，因为 `RAG/types.py` 可能和 Python 标准库
`types` 模块重名，导致导入异常。

### DeepDoc 报 NLTK 数据缺失

执行：

```powershell
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('wordnet')"
```

### DeepDoc 报 xgboost 模型加载失败

执行：

```powershell
pip install "xgboost>=3.0.2,<3.1"
```

### datrie 安装失败

Windows 下执行：

```powershell
conda install -n test_agent -c conda-forge datrie -y
```

### Milvus 连接失败

如果看到类似：

```text
failed to connect to Milvus
connection refused
```

先检查端口：

```powershell
Test-NetConnection localhost -Port 19530
```

如果 `TcpTestSucceeded` 是 `False`，说明 Milvus 服务没启动，或者端口不是 19530。

如果 Docker 命令报：

```text
failed to connect to the docker API
```

说明 Docker Desktop 没启动。先启动 Docker Desktop，再执行：

```powershell
docker ps
docker compose up -d
```

