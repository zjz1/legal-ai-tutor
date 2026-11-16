# -*- coding: utf-8 -*-
"""提示词注册表：带版本号，便于实验对比与后续 A/B。

占位符使用 $name 形式（string.Template），避免用户文本里的花括号冲突。
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template
from typing import Any


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    system: str
    user: str

    def render(self, **kwargs: Any) -> tuple[str, str]:
        return self.system, Template(self.user).safe_substitute(**kwargs)


PROMPTS: dict[str, dict[str, Prompt]] = {
    "local_answer": {
        "v1": Prompt(
            name="local_answer",
            version="v1",
            system=(
                "你是一位执业多年的中国律师，熟悉民商事、刑事与行政法律实务。"
                "回答要准确、具体，必要时引用现行有效的法条；无法确定时明确说明，不得臆造条号。"
            ),
            user="$question",
        )
    },
    "extract": {
        "v1": Prompt(
            name="extract",
            version="v1",
            system="你是法律文书信息抽取助手，只做抽取，不做法律分析。严格按要求输出 JSON。",
            user="""请从下面的案情描述中抽取三要素，只输出如下 JSON：

{
  "legal_subjects": ["法律主体，例如：原告张某、被告某公司"],
  "dispute_focus": ["争议焦点，例如：借款合同是否成立并生效"],
  "legal_articles": ["涉及法条，例如：《中华人民共和国民法典》第六百六十七条"]
}

要求：
1. 只输出 JSON，不要 Markdown 代码块，不要任何解释；
2. 信息不足时对应字段填 "未提及"，不得编造当事人姓名或法条条号；
3. 无法确定具体条号时，只写法律名称。

案情描述：
$question""",
        )
    },
    "analyze": {
        "v1": Prompt(
            name="analyze",
            version="v1",
            system=(
                "你是一名资深法律专家，擅长基于争议焦点和现行法条出具专业分析。"
                "请分点论述：① 法律关系认定；② 争议焦点逐项分析；③ 可行建议与风险提示。"
                "只能引用【候选法条】中给出的条文；若候选不足，明确说明无法确定，不得自行编造条号。"
            ),
            user="""请基于以下要素与候选法条，生成专业法律分析意见。

【已抽取要素】
$elements

【候选法条】
$context

【原始案情】
$question""",
        )
    },
    "polish": {
        "v1": Prompt(
            name="polish",
            version="v1",
            system=(
                "你是法院文书的文字校对专家。把草稿改写成规范的书面法律表达："
                "术语准确、语句严谨、结构分点、去除口语化；不得改变原意，不得新增事实或结论。"
            ),
            user="请对以下草稿进行法言法语润色，直接输出润色后的正文：\n\n$draft",
        )
    },
    "learning_article": {
        "v1": Prompt(
            name="learning_article",
            version="v1",
            system="你是法学教师，负责把法条拆解成学生能理解的结构化讲义。只输出 JSON。",
            user="""请拆解下列法条，输出 JSON：

{
  "elements": ["构成要件"],
  "exceptions": ["例外情形"],
  "burden": ["举证责任要点"],
  "example": "一个简短示例"
}

法条：$article_text""",
        )
    },
    "learning_quiz": {
        "v1": Prompt(
            name="learning_quiz",
            version="v1",
            system="你是法考命题老师。只输出 JSON，不要解释。",
            user="""针对下列法条出 $count 道自测题，输出 JSON：

{
  "questions": [
    {"type": "single_choice", "stem": "题干", "options": ["A选项", "B选项", "C选项", "D选项"],
     "answer": "A", "explanation": "解析"}
  ]
}

法条：$article_text""",
        )
    },
    "learning_grade": {
        "v1": Prompt(
            name="learning_grade",
            version="v1",
            system="你是法学阅卷老师，按要点给分。只输出 JSON。",
            user="""请给学生的作答评分，输出 JSON：

{"score": 0, "hits": ["答对的要点"], "misses": ["遗漏的要点"], "comment": "简短评语"}

题目：$question
参考答案：$reference
学生作答：$answer""",
        )
    },
}

ELEMENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "legal_subjects": {"type": "array", "items": {"type": "string"}},
        "dispute_focus": {"type": "array", "items": {"type": "string"}},
        "legal_articles": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["legal_subjects", "dispute_focus", "legal_articles"],
}

ARTICLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "elements": {"type": "array", "items": {"type": "string"}},
        "exceptions": {"type": "array", "items": {"type": "string"}},
        "burden": {"type": "array", "items": {"type": "string"}},
        "example": {"type": "string"},
    },
    "required": ["elements", "exceptions", "burden", "example"],
}

QUIZ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "stem": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "answer": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["type", "stem", "answer"],
            },
        }
    },
    "required": ["questions"],
}

GRADE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "hits": {"type": "array", "items": {"type": "string"}},
        "misses": {"type": "array", "items": {"type": "string"}},
        "comment": {"type": "string"},
    },
    "required": ["score"],
}


def get_prompt(name: str, version: str = "v1") -> Prompt:
    versions = PROMPTS.get(name)
    if not versions:
        raise KeyError(f"提示词不存在：{name}")
    if version not in versions:
        raise KeyError(f"提示词 {name} 没有版本 {version}")
    return versions[version]
