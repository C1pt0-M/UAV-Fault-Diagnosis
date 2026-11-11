from flask import Flask

from tests.test_knowledge_retrieval_context import make_service


def test_interpretation_retrieval_keeps_only_high_relevance_items(tmp_path):
    documents = [
        ("one", "doc-1", "电机故障排查手册", "motor 电机故障通常表现为振动、异响和温升异常。", 0.92),
        ("two", "doc-2", "无人机任务概览", "motor 这是一个泛化的任务背景介绍。", 0.32),
        ("three", "doc-3", "乱码文档", "motor 刷直预芯 摘 要 枪业远系统要亩执胁浒踩跋低?", 0.88),
    ]
    service, _, _, _ = make_service(tmp_path, "semantic", documents)
    with Flask("knowledge-relevance").app_context():
        result = service.retrieve_for_interpretation("motor", candidate_limit=5, max_items=3)
    assert len(result.hits) == 1
    assert result.hits[0]["source_id"] == "doc-1"
