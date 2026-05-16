from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class IntentProfile:
    name: str
    description: str
    document_dependency: str
    temperature: float
    keywords: List[str]


INTENT_PROFILES: Dict[str, IntentProfile] = {
    "qa": IntentProfile("qa", "知识问答", "strict", 0.0, ["什么", "多少", "是否", "有没有", "要求", "规定", "支持", "包含"]),
    "extract": IntentProfile("extract", "信息抽取", "strict", 0.0, ["抽取", "提取", "字段", "列出", "名称", "指标", "参数"]),
    "compare": IntentProfile("compare", "对比分析", "strict", 0.1, ["对比", "比较", "区别", "差异", "相同", "不同"]),
    "summary": IntentProfile("summary", "文档总结", "strict", 0.2, ["总结", "概括", "归纳", "梳理", "摘要"]),
    "reason_analysis": IntentProfile("reason_analysis", "原因分析", "semi_strict", 0.3, ["原因", "为什么", "分析", "影响", "风险"]),
    "plan_generation": IntentProfile("plan_generation", "方案生成", "semi_strict", 0.5, ["方案", "计划", "步骤", "建议", "怎么做", "如何实现"]),
    "tool_call": IntentProfile("tool_call", "工具调用", "strict", 0.0, ["调用", "查询", "检索", "运行", "执行", "工具"]),
    "clarification": IntentProfile("clarification", "澄清反问", "open", 0.3, ["什么意思", "不清楚", "怎么理解", "需要哪些信息", "澄清"]),
    "privacy_rejection": IntentProfile("privacy_rejection", "文档防泄露拒答", "strict", 0.0, ["泄露", "密钥", "密码", "隐私", "身份证", "手机号", "全部原文"]),
    "context_followup": IntentProfile("context_followup", "多轮追问", "depends", 0.2, ["继续", "上面", "刚才", "这个", "它", "前面"]),
    "chat": IntentProfile("chat", "普通闲聊", "open", 0.7, ["你好", "谢谢", "你是谁", "聊天", "讲个笑话"]),
}

DEFAULT_INTENT = "qa"


def get_intent_profile(intent: str) -> IntentProfile:
    return INTENT_PROFILES.get(intent, INTENT_PROFILES[DEFAULT_INTENT])


def intent_temperature(intent: str) -> float:
    return get_intent_profile(intent).temperature


def detect_intent_by_rules(question: str) -> Optional[str]:
    text = question.strip().lower()
    if not text:
        return "clarification"

    scores: Dict[str, int] = {}
    for intent, profile in INTENT_PROFILES.items():
        score = 0
        for keyword in profile.keywords:
            if keyword.lower() in text:
                score += 1
        if score:
            scores[intent] = score

    if re.search(r"(抽取|提取|字段|列出).{0,12}[:：]", text):
        scores["extract"] = scores.get("extract", 0) + 3
    if re.search(r"(对比|比较).*(和|与|及|以及)", text):
        scores["compare"] = scores.get("compare", 0) + 3
    if re.search(r"(总结|概括|摘要|归纳)", text):
        scores["summary"] = scores.get("summary", 0) + 2
    if re.search(r"(原因|为什么|为何)", text):
        scores["reason_analysis"] = scores.get("reason_analysis", 0) + 2
    if re.search(r"(方案|计划|建议|步骤|怎么做|如何)", text):
        scores["plan_generation"] = scores.get("plan_generation", 0) + 2
    if re.search(r"(泄露|密码|密钥|身份证|隐私|手机号|全部原文)", text):
        scores["privacy_rejection"] = scores.get("privacy_rejection", 0) + 4
    if re.search(r"(继续|上面|刚才|前面|这个|它)", text):
        scores["context_followup"] = scores.get("context_followup", 0) + 1

    if not scores:
        return None
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]


def parse_intent_response(text: str) -> Optional[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    intent = str(data.get("intent", "")).strip()
    return intent if intent in INTENT_PROFILES else None


def detect_intent(question: str, llm: Optional[Any] = None, use_llm: bool = False) -> IntentProfile:
    rule_intent = detect_intent_by_rules(question)
    if rule_intent:
        return get_intent_profile(rule_intent)

    if use_llm and llm and llm.enabled:
        labels = "\n".join(
            f"- {profile.name}: {profile.description}, {profile.document_dependency}, temperature={profile.temperature}"
            for profile in INTENT_PROFILES.values()
        )
        response = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是 RAG 意图识别器。只能从给定 intent 列表中选择一个，"
                        "输出 JSON：{\"intent\":\"...\"}，不要输出其他内容。"
                    ),
                },
                {"role": "user", "content": f"intent 列表：\n{labels}\n\n用户问题：{question}"},
            ],
            temperature=0.0,
        )
        llm_intent = parse_intent_response(response)
        if llm_intent:
            return get_intent_profile(llm_intent)

    return get_intent_profile(DEFAULT_INTENT)
