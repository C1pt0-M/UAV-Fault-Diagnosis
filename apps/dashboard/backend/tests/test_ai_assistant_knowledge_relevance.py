from tests.test_ai_assistant import app, auth_header, seeded_records  # noqa: F401

from uav_dashboard_api.services.ai_assistant import ai_assistant_service


def test_select_knowledge_hits_keeps_only_high_relevance_items(app):
    with app.app_context():
        hits = [
            {
                "source_type": "knowledge_base",
                "source_id": "doc-1",
                "title": "电机故障排查手册",
                "snippet": "电机故障通常表现为振动、异响和温升异常。",
                "score": 0.92,
            },
            {
                "source_type": "knowledge_base",
                "source_id": "doc-2",
                "title": "无人机任务概览",
                "snippet": "这是一个泛化的任务背景介绍。",
                "score": 0.32,
            },
            {
                "source_type": "knowledge_base",
                "source_id": "doc-3",
                "title": "乱码文档",
                "snippet": "刷直预芯 摘 要 枪业远系统要亩执胁浒踩跋低?",
                "score": 0.88,
            },
        ]

        filtered = ai_assistant_service._select_knowledge_hits(
            query="电机故障的表现有哪些",
            hits=hits,
            max_items=3,
        )

        assert len(filtered) == 1
        assert filtered[0]["source_id"] == "doc-1"
