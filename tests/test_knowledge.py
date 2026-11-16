# -*- coding: utf-8 -*-
"""法条语料、检索与引用校验。"""

from __future__ import annotations

from legal_ai_tutor.knowledge import (
    HybridRetriever,
    LawCorpus,
    extract_citations,
    normalize_article,
    tokenize,
    validate_text,
)


def test_normalize_article_chinese_numerals():
    assert normalize_article("第六百六十七条") == "667"
    assert normalize_article("第十一条") == "11"
    assert normalize_article("第六百八十条") == "680"
    assert normalize_article("667") == "667"
    assert normalize_article("无关文本") is None


def test_corpus_lookup_with_alias(corpus: LawCorpus):
    assert len(corpus) > 0
    found = corpus.get("民法典", "667")
    assert found is not None
    assert "借款合同" in found.text
    assert corpus.get("中华人民共和国民法典", "第六百六十七条") is found


def test_repealed_article_is_flagged(corpus: LawCorpus):
    checks = validate_text("依据《中华人民共和国合同法》第二百一十条，合同成立。", corpus)
    assert len(checks) == 1
    assert checks[0].valid is False
    assert "废止" in checks[0].reason


def test_unknown_law_is_flagged(corpus: LawCorpus):
    checks = validate_text("依据《中华人民共和国外星法》第一条。", corpus)
    assert checks[0].valid is False


def test_valid_citation_passes(corpus: LawCorpus):
    checks = validate_text("依据《中华人民共和国民法典》第六百七十九条，借款合同成立。", corpus)
    assert checks[0].valid is True
    assert checks[0].effective_date == "2021-01-01"


def test_extract_citations_deduplicates():
    text = "《民法典》第六百七十九条与《民法典》第六百七十九条相同"
    assert extract_citations(text) == [("民法典", "679")]


def test_tokenize_mixes_cjk_and_latin():
    tokens = tokenize("民法典 Article 667")
    assert "民" in tokens
    assert "民法" in tokens
    assert "article" in tokens


def test_retriever_finds_lending_articles(retriever: HybridRetriever):
    results = retriever.search("借款 利息 逾期", top_k=3)
    assert results
    numbers = {item.article.article for item in results}
    assert numbers & {"667", "676", "680"}
    assert retriever.as_context(results)
