from tests.test_ai_assistant import app  # noqa: F401

from uav_dashboard_api.services.knowledge_base import knowledge_base


def test_text_quality_filter_flags_garbled_content(app):
    with app.app_context():
        assert knowledge_base._is_text_quality_acceptable("电机故障通常表现为振动、异响和温升异常。") is True
        assert knowledge_base._is_text_quality_acceptable("刷直预芯 摘 要 枪业远系统要亩执胁浒踩跋低?") is False
