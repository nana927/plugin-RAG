from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from RAG.config import RagConfig
from RAG.intent_router import IntentProfile, detect_intent
from RAG.rag_service import RagService
from RAG.types import SearchResult


FIELD_ALIASES = {
    "case_id": ["case_id", "id", "编号"],
    "question": ["question", "query", "问题", "用户问题"],
    "standard_answer": ["standard_answer", "answer", "expected_answer", "标准答案"],
    "answer_keywords": ["answer_keywords", "keywords", "expected_keywords", "标准答案关键词", "答案关键词"],
    "intent": ["intent", "意图", "category", "类别"],
    "sub_intent": ["sub_intent", "子意图", "sub_category"],
}

REQUIRED_SCHEMA_FIELDS = ["question", "standard_answer", "intent"]


@dataclass
class TestCase:
    case_id: str
    question: str
    standard_answer: str
    answer_keywords: List[str]
    intent: str
    sub_intent: str = ""


@dataclass
class CaseResult:
    case_id: str
    intent: str
    sub_intent: str
    question: str
    standard_answer: str
    rag_answer: str
    answer_keywords: List[str]
    matched_keywords: List[str]
    keyword_coverage: Optional[float]
    llm_accuracy: float
    final_accuracy: float
    judge_reason: str
    retrieved: List[Dict[str, Any]]
    elapsed_ms: int
    predicted_intent: str = ""
    intent_temperature: float = 0.0
    document_dependency: str = ""


def env_float(name: str, default: float) -> float:
    """
    作用：
        从环境变量中读取浮点数配置。
    入参：
        name: 环境变量名称。
        default: 环境变量不存在或为空时使用的默认值。
    出参：
        float 类型的配置值。
    主要逻辑：
        先读取 os.getenv(name)，如果为空则返回 default；
        否则把字符串转换成 float。
    """
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def env_int(name: str, default: int) -> int:
    """
    作用：
        从环境变量中读取整数配置。
    入参：
        name: 环境变量名称。
        default: 环境变量不存在或为空时使用的默认值。
    出参：
        int 类型的配置值。
    主要逻辑：
        先读取 os.getenv(name)，如果为空则返回 default；
        否则把字符串转换成 int。
    """
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def env_bool(name: str, default: bool = False) -> bool:
    """
    作用：
        从环境变量中读取布尔配置。
    入参：
        name: 环境变量名称。
        default: 环境变量不存在或为空时使用的默认值。
    出参：
        bool 类型的配置值。
    主要逻辑：
        先读取 os.getenv(name)，如果为空则返回 default；
        否则把 1/true/yes/y 识别为 True，其它值识别为 False。
    """
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y"}


def first_present(row: Dict[str, Any], names: Iterable[str], default: Any = "") -> Any:
    """
    作用：
        从一行测试用例数据中，按多个候选字段名取第一个有效值。
    入参：
        row: 测试用例原始字典。
        names: 候选字段名列表，例如 question/query/问题。
        default: 所有候选字段都不存在或为空时返回的默认值。
    出参：
        第一个非空字段值；如果没有命中则返回 default。
    主要逻辑：
        依次遍历候选字段名；
        跳过不存在、None、空字符串字段；
        一旦找到有效字段就立即返回。
    """
    for name in names:
        if name not in row:
            continue
        value = row[name]
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return default


def schema_field_names(schema: Any) -> set[str]:
    """
    作用：
        从测试集顶层 schema 中提取字段名集合。
    入参：
        schema: 测试集里的 schema 定义，兼容普通 dict、JSON Schema properties、字段列表。
    出参：
        set[str]，schema 声明过的字段名；没有 schema 或无法识别时返回空集合。
    主要逻辑：
        如果 schema 是 dict 且包含 properties，就读取 properties 的 key；
        如果 schema 是普通 dict，就读取 dict 自身的 key；
        如果 schema 是 list，就兼容字符串字段名或包含 name/key/field 的字段描述对象。
    """
    if not schema:
        return set()
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            return {str(key) for key in properties.keys()}
        return {str(key) for key in schema.keys()}
    if isinstance(schema, list):
        fields = set()
        for item in schema:
            if isinstance(item, str):
                fields.add(item)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("key") or item.get("field")
                if name:
                    fields.add(str(name))
        return fields
    return set()


def field_candidates(canonical_name: str, schema_fields: set[str]) -> List[str]:
    """
    作用：
        根据标准字段名生成候选字段名，并让 schema 中声明的字段优先匹配。
    入参：
        canonical_name: 脚本内部的标准字段名，例如 question、standard_answer。
        schema_fields: schema_field_names() 提取出来的字段集合。
    出参：
        List[str]，按优先级排列的候选字段名。
    主要逻辑：
        先从 FIELD_ALIASES 获取该字段的所有别名；
        把出现在 schema_fields 中的别名排在前面；
        再追加其它常见别名，保证没有 schema 时也能兼容旧文件。
    """
    aliases = FIELD_ALIASES.get(canonical_name, [canonical_name])
    schema_first = [name for name in aliases if name in schema_fields]
    fallback = [name for name in aliases if name not in schema_first]
    return schema_first + fallback


def log_schema_usage(schema_fields: set[str]) -> None:
    """
    作用：
        输出 schema 使用情况，并检查核心逻辑字段是否在 schema 中声明。
    入参：
        schema_fields: schema 中声明的字段名集合。
    出参：
        无返回值。
    主要逻辑：
        如果没有 schema，记录当前会使用字段别名自动兼容；
        如果存在 schema，打印字段数量和字段名；
        再检查 question、standard_answer、intent 是否能通过 schema 或别名找到。
    """
    if not schema_fields:
        logging.info("No testcase schema found. Field aliases will be used automatically.")
        return

    logging.info("Loaded testcase schema with %s fields: %s", len(schema_fields), sorted(schema_fields))
    missing = [
        name
        for name in REQUIRED_SCHEMA_FIELDS
        if not any(alias in schema_fields for alias in FIELD_ALIASES.get(name, [name]))
    ]
    if missing:
        logging.warning("Schema is missing required logical fields: %s", missing)


def load_cases(path: Path) -> List[TestCase]:
    """
    作用：
        读取测试集文件，并统一转换成 TestCase 列表。
    入参：
        path: 测试集文件路径，支持 .json / .jsonl / .csv。
    出参：
        List[TestCase]，每个元素是一条标准化后的测试用例。
    主要逻辑：
        先根据文件后缀选择读取方式；
        JSON 文件兼容 list 或包含 cases/data/items/testcases 的 dict；
        CSV 使用 DictReader 读取；
        然后兼容多种字段别名，抽取问题、标准答案、关键词、意图等字段；
        最后过滤掉没有 question 的无效用例。
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict):
            rows = (
                raw.get("cases")
                or raw.get("data")
                or raw.get("items")
                or raw.get("testcases")
                or []
            )
        else:
            rows = []
    elif suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported testcase file type: {path.suffix}")

    cases: List[TestCase] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        keywords = first_present(
            row,
            ["answer_keywords", "keywords", "expected_keywords", "标准答案关键词", "答案关键词"],
            [],
        )
        if isinstance(keywords, str):
            keywords = [item.strip() for item in re.split(r"[,，;；|]", keywords) if item.strip()]

        case = TestCase(
            case_id=str(first_present(row, ["case_id", "id", "编号"], f"case_{idx:04d}")),
            question=str(first_present(row, ["question", "query", "问题", "用户问题"])),
            standard_answer=str(
                first_present(row, ["standard_answer", "answer", "expected_answer", "标准答案"])
            ),
            answer_keywords=[str(item).strip() for item in keywords if str(item).strip()],
            intent=str(first_present(row, ["intent", "意图", "category", "类别"], "unknown")),
            sub_intent=str(first_present(row, ["sub_intent", "子意图", "sub_category"], "")),
        )
        if case.question:
            cases.append(case)
    return cases


def load_cases_with_schema(path: Path) -> List[TestCase]:
    """
    作用：
        读取测试集文件，并优先按照顶层 schema 定义的数据结构解析用例。
    入参：
        path: 测试集文件路径，支持 .json / .jsonl / .csv。
    出参：
        List[TestCase]，每个元素是一条标准化后的测试用例。
    主要逻辑：
        1. JSON 文件如果包含顶层 schema，则先提取 schema 字段集合。
        2. 字段解析时先匹配 schema 中声明过的字段名，再回退到常见别名。
        3. 对 question、standard_answer、intent 等核心字段做 schema 层面的提示日志。
        4. 将每一行原始数据转换为 TestCase，并过滤没有 question 的无效数据。
    """
    suffix = path.suffix.lower()
    schema_fields: set[str] = set()

    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict):
            schema_fields = schema_field_names(raw.get("schema"))
            rows = (
                raw.get("cases")
                or raw.get("data")
                or raw.get("items")
                or raw.get("testcases")
                or []
            )
        else:
            rows = []
    elif suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported testcase file type: {path.suffix}")

    log_schema_usage(schema_fields)

    cases: List[TestCase] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue

        keywords = first_present(
            row,
            field_candidates("answer_keywords", schema_fields),
            [],
        )
        if isinstance(keywords, str):
            keywords = [item.strip() for item in re.split(r"[,，;；|]", keywords) if item.strip()]

        case = TestCase(
            case_id=str(first_present(row, field_candidates("case_id", schema_fields), f"case_{idx:04d}")),
            question=str(first_present(row, field_candidates("question", schema_fields))),
            standard_answer=str(first_present(row, field_candidates("standard_answer", schema_fields))),
            answer_keywords=[str(item).strip() for item in keywords if str(item).strip()],
            intent=str(first_present(row, field_candidates("intent", schema_fields), "unknown")),
            sub_intent=str(first_present(row, field_candidates("sub_intent", schema_fields), "")),
        )
        if case.question:
            cases.append(case)

    return cases


def normalize_text(text: str) -> str:
    """
    作用：
        对文本做关键词匹配前的轻量归一化。
    入参：
        text: 原始文本。
    出参：
        去掉空白并转成小写后的字符串。
    主要逻辑：
        使用正则删除所有连续空白字符；
        再调用 lower()，减少大小写差异对匹配的影响。
    """
    return re.sub(r"\s+", "", text or "").lower()


def keyword_coverage(answer: str, keywords: List[str]) -> tuple[Optional[float], List[str]]:
    """
    作用：
        计算 RAG 输出对标准答案关键词的覆盖率。
    入参：
        answer: RAG 输出文本，可以是生成答案，也可以是召回上下文。
        keywords: 标准答案关键词列表。
    出参：
        (coverage, matched_keywords)：
        coverage 为关键词覆盖率；当 keywords 为空时返回 None；
        matched_keywords 为实际命中的关键词列表。
    主要逻辑：
        如果没有关键词，说明关键词指标不参与该题评分，返回 None；
        否则对 answer 和每个 keyword 做 normalize_text；
        用子串包含关系判断关键词是否出现在 answer 中；
        最后用命中数 / 关键词总数得到覆盖率。
    """
    if not keywords:
        return None, []
    normalized_answer = normalize_text(answer)
    matched = [
        keyword
        for keyword in keywords
        if normalize_text(keyword) and normalize_text(keyword) in normalized_answer
    ]
    return len(matched) / len(keywords), matched


def final_accuracy(llm_accuracy: float, coverage: Optional[float], k_weight: float) -> float:
    """
    作用：
        按配置权重融合大模型语义评分和关键词覆盖率。
    入参：
        llm_accuracy: 大模型裁判给出的语义准确率，范围通常为 0~1。
        coverage: 关键词覆盖率；如果该题没有关键词，则为 None。
        k_weight: 语义评分权重 K。
    出参：
        融合后的最终准确率。
    主要逻辑：
        如果 coverage 为 None，说明没有关键词指标，语义评分承担全部分数；
        否则使用公式 K * llm_accuracy + (1 - K) * coverage。
    """
    if coverage is None:
        return llm_accuracy
    return k_weight * llm_accuracy + (1.0 - k_weight) * coverage


def result_to_dict(result: SearchResult) -> Dict[str, Any]:
    """
    作用：
        把 SearchResult 对象转换成可 JSON 序列化的字典。
    入参：
        result: RAG 检索返回的一条 SearchResult。
    出参：
        Dict[str, Any]，包含 chunk_id、doc_id、文件名、分数和文本等字段。
    主要逻辑：
        逐字段读取 SearchResult 的属性；
        保留 vector_score 和 term_score，方便后续分析混合检索表现。
    """
    return {
        "chunk_id": result.chunk_id,
        "doc_id": result.doc_id,
        "file_name": result.file_name,
        "score": result.score,
        "vector_score": result.vector_score,
        "term_score": result.term_score,
        "text": result.text,
    }


def build_context(results: List[SearchResult], max_chars: int) -> str:
    """
    作用：
        把 top_k 检索结果拼接成提供给大模型的上下文文本。
    入参：
        results: RagService.retrieve() 返回的检索结果列表。
        max_chars: 上下文最大字符数，避免 prompt 过长。
    出参：
        拼接后的上下文字符串。
    主要逻辑：
        按检索结果顺序遍历；
        给每个片段加上序号、文件名和 score；
        累加字符数，超过 max_chars 时截断当前片段并停止继续拼接。
    """
    pieces = []
    total = 0
    for idx, item in enumerate(results, start=1):
        text = item.text or ""
        header = f"[{idx}] {item.file_name} score={item.score:.4f}\n"
        piece = header + text.strip()
        if total + len(piece) > max_chars:
            piece = piece[: max(0, max_chars - total)]
        if piece:
            pieces.append(piece)
            total += len(piece)
        if total >= max_chars:
            break
    return "\n\n".join(pieces)


class LlmClient:
    def __init__(self) -> None:
        """
        作用：
            初始化大模型客户端配置，但不立即创建真实 SDK client。
        入参：
            无显式入参，配置全部从环境变量读取。
        出参：
            无返回值；初始化实例属性。
        主要逻辑：
            依次读取 RAG_EVAL_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY；
            读取 base_url、model、timeout；
            将 _client 设为 None，等第一次调用时再懒加载 OpenAI 客户端。
        """
        self.api_key = (
            os.getenv("RAG_EVAL_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = os.getenv(
            "RAG_EVAL_BASE_URL",
            os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self.model = os.getenv("RAG_EVAL_MODEL", os.getenv("RAG_KEYWORD_MODEL", "qwen-plus"))
        self.timeout = env_int("RAG_EVAL_LLM_TIMEOUT", 60)
        self.retries = max(1, env_int("RAG_EVAL_LLM_RETRIES", 3))
        self.retry_delay = max(0.0, env_float("RAG_EVAL_LLM_RETRY_DELAY", 1.0))
        self._client = None

    @property
    def enabled(self) -> bool:
        """
        作用：
            判断当前是否具备调用大模型的基本条件。
        入参：
            无。
        出参：
            bool；有 API Key 返回 True，否则返回 False。
        主要逻辑：
            检查 self.api_key 是否为非空字符串。
        """
        return bool(self.api_key)

    def _get_client(self):
        """
        作用：
            获取 OpenAI 兼容的大模型客户端。
        入参：
            无。
        出参：
            OpenAI SDK client 实例。
        主要逻辑：
            如果没有 API Key，直接抛错；
            如果 _client 尚未创建，则导入 openai.OpenAI 并创建客户端；
            如果已经创建过，则复用同一个客户端。
        """
        if not self.enabled:
            raise RuntimeError("Missing RAG_EVAL_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY")
        self.validate()
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    def validate(self) -> None:
        """
        作用：
            在真正调用大模型前校验 OpenAI-compatible 配置。
        入参：
            无。
        出参：
            无返回值；配置不合法时抛出 RuntimeError。
        主要逻辑：
            API Key 会进入 HTTP Authorization header，必须是 ASCII 字符；
            如果用户把“你的API_KEY”这类中文占位符写进 .env，会提前给出明确错误。
        """
        if not self.enabled:
            return
        try:
            self.api_key.encode("ascii")
        except UnicodeEncodeError as exc:
            raise RuntimeError(
                "RAG_EVAL_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY contains non-ASCII "
                "characters. Please replace placeholders like '你的API_KEY' with a real API key."
            ) from exc

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """
        作用：
            调用 OpenAI 兼容 chat completions 接口。
        入参：
            messages: Chat messages 列表，包含 system/user 等消息。
            temperature: 采样温度，评测默认使用 0 保持稳定。
        出参：
            模型返回的文本内容；如果为空则返回空字符串。
        主要逻辑：
            先通过 _get_client() 获取客户端；
            调用 client.chat.completions.create；
            返回第一条 choice 的 message.content。
        """
        client = self._get_client()
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=messages,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                if attempt >= self.retries:
                    break
                sleep_seconds = self.retry_delay * attempt
                logging.warning(
                    "LLM call failed, retrying %s/%s after %.1fs: %s",
                    attempt,
                    self.retries,
                    sleep_seconds,
                    exc,
                )
                time.sleep(sleep_seconds)
        raise last_exc or RuntimeError("LLM call failed.")


def generate_answer(
    llm: LlmClient,
    question: str,
    context: str,
    answer_mode: str,
    allow_missing_llm: bool,
    intent_profile: IntentProfile,
) -> str:
    """
    作用：
        根据召回上下文生成本题的 RAG 输出。
    入参：
        llm: 大模型客户端封装。
        question: 用户问题。
        context: 检索结果拼接后的上下文。
        answer_mode: 输出模式，llm 表示让大模型生成答案，context 表示直接使用上下文。
        allow_missing_llm: 大模型不可用时是否允许降级继续跑。
        intent_profile: 意图识别结果，用于选择 prompt 约束和 temperature。
    出参：
        RAG 输出文本。
    主要逻辑：
        如果 answer_mode=context，或者没有配置大模型 API Key，直接返回 context；
        否则根据意图构造 prompt，并用意图对应 temperature 调用大模型；
        如果调用失败且允许降级，则记录 warning 并返回 context。
    """
    if answer_mode == "context" or not llm.enabled:
        return context

    try:
        if intent_profile.document_dependency == "open":
            system_prompt = (
                "你是一个友好的助手。用户问题不要求严格依赖文档时，可以自然回答；"
                "如果检索内容有帮助，可以参考检索内容，但不要编造文档事实。"
            )
        elif intent_profile.document_dependency == "semi_strict":
            system_prompt = (
                "你是一个 RAG 分析助手。优先依据给定检索内容回答；"
                "可以做少量合理分析或方案组织，但涉及事实、数字、名称、版本号时必须来自检索内容。"
            )
        else:
            system_prompt = (
                "你是一个严谨的 RAG 问答助手。只能依据给定检索内容回答；"
                "如果检索内容不足以回答，就说“未在知识库中找到明确依据”。"
                "回答要简洁，保留关键数字、名称、版本号和限制条件。"
            )
        return llm.chat(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        f"识别意图：{intent_profile.name}（{intent_profile.description}）\n"
                        f"问题：{question}\n\n检索内容：\n{context}\n\n请给出答案："
                    ),
                },
            ],
            temperature=intent_profile.temperature,
        ).strip()
    except Exception:
        if allow_missing_llm:
            logging.warning("LLM answer generation failed, fallback to retrieved context.", exc_info=True)
            return context
        raise


def parse_json_object(text: str) -> Dict[str, Any]:
    """
    作用：
        从大模型返回文本中解析 JSON 对象。
    入参：
        text: 大模型原始输出。
    出参：
        解析后的字典。
    主要逻辑：
        先直接 json.loads；
        如果失败，再用正则从文本中截取第一个 {...} 片段；
        如果仍然找不到 JSON 对象，则继续抛出解析异常。
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def judge_answer(
    llm: LlmClient,
    question: str,
    standard_answer: str,
    rag_answer: str,
    allow_missing_llm: bool,
) -> tuple[float, str]:
    """
    作用：
        使用大模型裁判评估 RAG 输出和标准答案的语义一致性。
    入参：
        llm: 大模型客户端封装。
        question: 测试问题。
        standard_answer: 标准答案。
        rag_answer: RAG 实际输出。
        allow_missing_llm: 大模型不可用时是否允许按 0 分继续跑。
    出参：
        (llm_accuracy, judge_reason)：
        llm_accuracy 是 0~1 的语义准确率；
        judge_reason 是大模型给出的简短评分原因。
    主要逻辑：
        如果没有 API Key 且允许降级，返回 0 分和说明；
        否则构造裁判 prompt，要求模型只输出 JSON；
        解析 JSON 中的 score 和 reason；
        将 score 限制在 0~1；
        如果调用或解析失败且允许降级，则返回 0 分并记录原因。
    """
    if not llm.enabled:
        if allow_missing_llm:
            return 0.0, "未配置大模型 API Key，llm_accuracy 按 0 记录。"
        raise RuntimeError("Missing LLM API key for answer judging.")

    try:
        content = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是问答评测裁判。请比较 RAG 输出和标准答案是否语义一致。"
                        "只输出 JSON，不要输出 Markdown。JSON 格式："
                        "{\"score\": 0到1之间的小数, \"reason\": \"简短原因\"}。"
                        "评分标准：1 表示完全正确；0.7 表示主要事实正确但有遗漏；"
                        "0.4 表示只命中少量事实；0 表示错误、无依据或答非所问。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"问题：{question}\n\n"
                        f"标准答案：{standard_answer}\n\n"
                        f"RAG 输出：{rag_answer}\n\n"
                        "请给出 JSON 评分："
                    ),
                },
            ]
        )
        data = parse_json_object(content)
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return score, str(data.get("reason", "")).strip()
    except Exception as exc:
        if allow_missing_llm:
            logging.warning("LLM judging failed, llm_accuracy is recorded as 0: %s", exc)
            return 0.0, f"大模型评分失败，llm_accuracy 按 0 记录：{exc}"
        raise


def summarize(results: List[CaseResult], k_weight: float) -> Dict[str, Any]:
    """
    作用：
        汇总所有测试结果，生成整体和分类指标。
    入参：
        results: 每道题的评测结果列表。
        k_weight: 当前使用的语义评分权重 K，会写入汇总结果中。
    出参：
        汇总字典，包含 total、by_intent、by_sub_intent。
    主要逻辑：
        先定义 aggregate() 计算一组样本的平均指标；
        再按 intent 和 intent/sub_intent 分组；
        分别对整体、意图组、子意图组调用 aggregate()。
    """
    def aggregate(items: List[CaseResult]) -> Dict[str, Any]:
        """
        作用：
            计算某一组 CaseResult 的平均指标。
        入参：
            items: 同一统计口径下的评测结果列表。
        出参：
            包含 count、final_accuracy、keyword_coverage、keyword_case_count、
            llm_accuracy 的指标字典。
        主要逻辑：
            空列表返回 0 或 None；
            final_accuracy 和 llm_accuracy 对组内全部样本取平均；
            keyword_coverage 只对存在关键词的样本取平均；
            keyword_case_count 记录参与关键词覆盖率统计的样本数量。
        """
        count = len(items)
        if count == 0:
            return {
                "count": 0,
                "final_accuracy": 0.0,
                "keyword_coverage": None,
                "keyword_case_count": 0,
                "llm_accuracy": 0.0,
            }
        keyword_items = [item for item in items if item.keyword_coverage is not None]
        return {
            "count": count,
            "final_accuracy": sum(item.final_accuracy for item in items) / count,
            "keyword_coverage": (
                sum(item.keyword_coverage for item in keyword_items if item.keyword_coverage is not None)
                / len(keyword_items)
                if keyword_items
                else None
            ),
            "keyword_case_count": len(keyword_items),
            "llm_accuracy": sum(item.llm_accuracy for item in items) / count,
        }

    by_intent: Dict[str, List[CaseResult]] = defaultdict(list)
    by_sub_intent: Dict[str, List[CaseResult]] = defaultdict(list)
    for item in results:
        by_intent[item.intent].append(item)
        sub_key = f"{item.intent}/{item.sub_intent or 'unknown'}"
        by_sub_intent[sub_key].append(item)

    return {
        "k_weight": k_weight,
        "total": aggregate(results),
        "by_intent": {key: aggregate(value) for key, value in sorted(by_intent.items())},
        "by_sub_intent": {key: aggregate(value) for key, value in sorted(by_sub_intent.items())},
    }


def write_json(path: Path, data: Any) -> None:
    """
    作用：
        将 Python 对象写成格式化 JSON 文件。
    入参：
        path: 输出文件路径。
        data: 需要写入的 Python 对象，通常是 dict。
    出参：
        无返回值。
    主要逻辑：
        先确保父目录存在；
        再用 ensure_ascii=False 保留中文；
        使用 indent=2 方便人工阅读。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    """
    作用：
        向 JSONL 明细文件追加一条评测记录。
    入参：
        path: JSONL 输出文件路径。
        data: 单条评测明细字典。
    出参：
        无返回值。
    主要逻辑：
        先确保父目录存在；
        以追加模式打开文件；
        将 data 序列化为一行 JSON 并写入换行符。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def find_latest_run_id(output_dir: Path) -> Optional[str]:
    """
    作用：
        在评测输出目录中查找最近一次运行的 run_id。
    入参：
        output_dir: 评测输出目录。
    出参：
        最近的 run_id；如果没有历史明细文件则返回 None。
    主要逻辑：
        查找 eval_details_*.jsonl；
        按文件修改时间倒序排序；
        从文件名中截取时间戳部分作为 run_id。
    """
    files = sorted(
        output_dir.glob("eval_details_*.jsonl"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return None
    name = files[0].name
    return name.removeprefix("eval_details_").removesuffix(".jsonl")


def load_existing_results(detail_path: Path) -> List[CaseResult]:
    """
    作用：
        从已有评测明细中恢复已经成功完成的样本结果。
    入参：
        detail_path: eval_details_时间戳.jsonl 文件路径。
    出参：
        List[CaseResult]，每个 case_id 只保留最后一条成功完成且带 final_accuracy 的记录。
    主要逻辑：
        逐行读取 JSONL；
        跳过空行、解析失败行、包含 error 的失败记录；
        按 case_id 覆盖旧成功记录，避免同一题多次重跑成功后被重复计分；
        将成功记录还原为 CaseResult，用于续跑时跳过和最终汇总。
    """
    if not detail_path.exists():
        return []

    results_by_case_id: Dict[str, CaseResult] = {}
    with detail_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("error") or "final_accuracy" not in item:
                continue
            case_id = str(item.get("case_id", ""))
            if not case_id:
                continue
            results_by_case_id[case_id] = CaseResult(
                case_id=case_id,
                intent=str(item.get("intent", "unknown")),
                sub_intent=str(item.get("sub_intent", "")),
                question=str(item.get("question", "")),
                standard_answer=str(item.get("standard_answer", "")),
                rag_answer=str(item.get("rag_answer", "")),
                answer_keywords=list(item.get("answer_keywords") or []),
                matched_keywords=list(item.get("matched_keywords") or []),
                keyword_coverage=item.get("keyword_coverage"),
                llm_accuracy=float(item.get("llm_accuracy", 0.0)),
                final_accuracy=float(item.get("final_accuracy", 0.0)),
                judge_reason=str(item.get("judge_reason", "")),
                retrieved=list(item.get("retrieved") or []),
                elapsed_ms=int(item.get("elapsed_ms", 0)),
                predicted_intent=str(item.get("predicted_intent", "")),
                intent_temperature=float(item.get("intent_temperature", 0.0)),
                document_dependency=str(item.get("document_dependency", "")),
            )
    return list(results_by_case_id.values())


def load_failed_case_ids(detail_path: Path) -> set[str]:
    """
    作用：
        从已有评测明细中读取失败样本的 case_id。
    入参：
        detail_path: eval_details_时间戳.jsonl 文件路径。
    出参：
        set[str]，包含已有失败记录的 case_id。
    主要逻辑：
        逐行读取 JSONL；
        只收集带 error 字段的记录；
        用于续跑时根据 RAG_EVAL_RETRY_FAILED 决定是否跳过失败样本。
    """
    if not detail_path.exists():
        return set()

    failed_case_ids: set[str] = set()
    with detail_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("error") and item.get("case_id"):
                failed_case_ids.add(str(item["case_id"]))
    return failed_case_ids


def setup_logging(output_dir: Path, run_id: str) -> Path:
    """
    作用：
        初始化评测脚本日志输出。
    入参：
        output_dir: 评测输出目录。
        run_id: 本次运行 ID，一般是时间戳。
    出参：
        log_path: 本次运行的日志文件路径。
    主要逻辑：
        创建输出目录；
        按 run_id 生成 eval_时间戳.log；
        配置 logging 同时输出到控制台和日志文件。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"eval_{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


def parse_args() -> argparse.Namespace:
    """
    作用：
        解析命令行参数。
    入参：
        无显式入参，argparse 会从 sys.argv 读取。
    出参：
        argparse.Namespace，包含 testcase_file 和可选 limit。
    主要逻辑：
        定义必填参数 testcase_file；
        定义可选参数 --limit，用于只评测前 N 条；
        调用 parser.parse_args() 返回解析结果。
    """
    parser = argparse.ArgumentParser(
        description="Evaluate RAG QA quality with LLM judging and keyword coverage."
    )
    parser.add_argument("testcase_file", help="JSON/JSONL/CSV testcase file.")
    parser.add_argument("--limit", type=int, help="Only evaluate first N cases.")
    return parser.parse_args()


def main() -> None:
    """
    作用：
        RAG 问答评测脚本主入口。
    入参：
        无显式入参，配置从 .env / 环境变量 / 命令行参数读取。
    出参：
        无返回值；运行结束时会打印 summary，并写入日志和结果文件。
    主要逻辑：
        1. 解析命令行参数和 RagConfig。
        2. 读取评测相关环境变量，例如 K、top_k、answer_mode、输出目录。
        3. 初始化日志、测试集、RagService、大模型客户端和输出文件路径。
        4. 遍历每条测试用例：
           - 调用 RAG 检索。
           - 拼接检索上下文。
           - 生成 RAG 输出。
           - 计算关键词覆盖率。
           - 调用大模型裁判得到语义准确率。
           - 融合得到最终准确率。
           - 写入逐题 JSONL 明细和运行日志。
        5. 汇总整体、按 intent、按 sub_intent 的指标。
        6. 写入 summary JSON，并在控制台打印。
    """
    args = parse_args()
    config = RagConfig()
    k_weight = env_float("RAG_EVAL_K", 0.7)
    top_k = env_int("RAG_EVAL_TOP_K", 5)
    max_context_chars = env_int("RAG_EVAL_MAX_CONTEXT_CHARS", 6000)
    answer_mode = os.getenv("RAG_EVAL_ANSWER_MODE", "llm").lower()
    allow_missing_llm = env_bool("RAG_EVAL_ALLOW_MISSING_LLM", False)
    intent_use_llm = env_bool("RAG_INTENT_USE_LLM", False)
    low_score_threshold = env_float("RAG_EVAL_LOW_SCORE_THRESHOLD", 0.6)
    resume_enabled = env_bool("RAG_EVAL_RESUME", False)
    retry_failed = env_bool("RAG_EVAL_RETRY_FAILED", True)
    output_dir = Path(os.getenv("RAG_EVAL_OUTPUT_DIR", "RAG/data/eval_outputs"))
    requested_run_id = os.getenv("RAG_EVAL_RUN_ID", "").strip()
    if requested_run_id:
        run_id = requested_run_id
    elif resume_enabled:
        run_id = find_latest_run_id(output_dir) or datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(output_dir, run_id)

    cases = load_cases_with_schema(Path(args.testcase_file))
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise RuntimeError("No test cases loaded.")

    service = RagService(config)
    llm = LlmClient()
    if not allow_missing_llm:
        llm.validate()
    detail_path = output_dir / f"eval_details_{run_id}.jsonl"
    low_score_path = output_dir / f"eval_low_scores_{run_id}.jsonl"
    summary_path = output_dir / f"eval_summary_{run_id}.json"
    previous_results = load_existing_results(detail_path) if resume_enabled else []
    previous_failed_case_ids = load_failed_case_ids(detail_path) if resume_enabled else set()
    completed_case_ids = {item.case_id for item in previous_results}
    if resume_enabled and not retry_failed:
        completed_case_ids.update(previous_failed_case_ids)

    logging.info(
        "Start eval: cases=%s backend=%s kb=%s top_k=%s answer_mode=%s intent_use_llm=%s k=%s low_score_threshold=%s resume=%s retry_failed=%s completed=%s previous_failed=%s output=%s",
        len(cases),
        config.backend,
        config.knowledge_base,
        top_k,
        answer_mode,
        intent_use_llm,
        k_weight,
        low_score_threshold,
        resume_enabled,
        retry_failed,
        len(completed_case_ids),
        len(previous_failed_case_ids),
        output_dir,
    )

    results: List[CaseResult] = list(previous_results)
    for idx, case in enumerate(cases, start=1):
        if case.case_id in completed_case_ids:
            logging.info("[%s/%s] %s skipped by resume checkpoint.", idx, len(cases), case.case_id)
            continue
        started = time.time()
        try:
            intent_profile = detect_intent(case.question, llm=llm, use_llm=intent_use_llm)
            retrieved = service.retrieve(case.question, config.knowledge_base, top_k=top_k)
            context = build_context(retrieved, max_context_chars)
            rag_answer = generate_answer(
                llm,
                case.question,
                context,
                answer_mode,
                allow_missing_llm=allow_missing_llm,
                intent_profile=intent_profile,
            )
            coverage, matched_keywords = keyword_coverage(rag_answer, case.answer_keywords)
            llm_accuracy, judge_reason = judge_answer(
                llm,
                case.question,
                case.standard_answer,
                rag_answer,
                allow_missing_llm=allow_missing_llm,
            )
            accuracy = final_accuracy(llm_accuracy, coverage, k_weight)
            elapsed_ms = int((time.time() - started) * 1000)

            result = CaseResult(
                case_id=case.case_id,
                intent=case.intent,
                sub_intent=case.sub_intent,
                question=case.question,
                standard_answer=case.standard_answer,
                rag_answer=rag_answer,
                answer_keywords=case.answer_keywords,
                matched_keywords=matched_keywords,
                keyword_coverage=coverage,
                llm_accuracy=llm_accuracy,
                final_accuracy=accuracy,
                judge_reason=judge_reason,
                retrieved=[result_to_dict(item) for item in retrieved],
                elapsed_ms=elapsed_ms,
                predicted_intent=intent_profile.name,
                intent_temperature=intent_profile.temperature,
                document_dependency=intent_profile.document_dependency,
            )
            results.append(result)
            completed_case_ids.add(case.case_id)
            append_jsonl(detail_path, result.__dict__)
            keyword_log = f"{coverage:.4f}" if coverage is not None else "NA"
            logging.info(
                "[%s/%s] %s intent=%s predicted_intent=%s temperature=%.2f final_accuracy=%.4f semantic_score=%.4f keyword_score=%s matched_keywords=%s elapsed=%sms",
                idx,
                len(cases),
                case.case_id,
                case.intent,
                intent_profile.name,
                intent_profile.temperature,
                accuracy,
                llm_accuracy,
                keyword_log,
                matched_keywords,
                elapsed_ms,
            )
            if accuracy < low_score_threshold:
                append_jsonl(low_score_path, result.__dict__)
                logging.warning(
                    "[low-score] %s final_accuracy=%.4f semantic_score=%.4f keyword_score=%s threshold=%.4f",
                    case.case_id,
                    accuracy,
                    llm_accuracy,
                    keyword_log,
                    low_score_threshold,
                )
        except Exception as exc:
            elapsed_ms = int((time.time() - started) * 1000)
            logging.error("[%s/%s] %s failed after %sms: %s", idx, len(cases), case.case_id, elapsed_ms, exc)
            failed = {
                "case_id": case.case_id,
                "intent": case.intent,
                "sub_intent": case.sub_intent,
                "question": case.question,
                "standard_answer": case.standard_answer,
                "error": str(exc),
                "elapsed_ms": elapsed_ms,
            }
            append_jsonl(detail_path, failed)

    summary = summarize(results, k_weight)
    summary.update(
        {
            "run_id": run_id,
            "testcase_file": str(Path(args.testcase_file).resolve()),
            "detail_path": str(detail_path.resolve()),
            "low_score_path": str(low_score_path.resolve()),
            "log_path": str(log_path.resolve()),
            "config": {
                "backend": config.backend,
                "knowledge_base": config.knowledge_base,
                "embedding": config.embedding,
                "chunker": config.chunker,
                "hybrid_enabled": config.hybrid_enabled,
                "top_k": top_k,
                "answer_mode": answer_mode,
                "intent_use_llm": intent_use_llm,
                "llm_model": llm.model,
                "low_score_threshold": low_score_threshold,
                "resume_enabled": resume_enabled,
                "retry_failed": retry_failed,
                "completed_before_run": len(previous_results),
                "failed_before_run": len(previous_failed_case_ids),
            },
        }
    )
    write_json(summary_path, summary)
    logging.info("Finished eval. summary=%s details=%s log=%s", summary_path, detail_path, log_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
