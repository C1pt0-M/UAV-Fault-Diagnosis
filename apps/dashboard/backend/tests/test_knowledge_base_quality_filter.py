from dashboard.backend.tests.test_ai_assistant import app  # noqa: F401

from uav_dashboard_api.services.knowledge_base import knowledge_base


def test_text_quality_filter_flags_garbled_content(app):
    with app.app_context():
        assert knowledge_base._is_text_quality_acceptable("鐢垫満鏁呴殰閫氬父琛ㄧ幇涓烘尟鍔ㄣ€佸紓鍝嶅拰娓╁崌寮傚父銆?) is True
        assert knowledge_base._is_text_quality_acceptable("刷直预芯 摘 要 枪业远系统要亩执胁浒踩跋低?) is False
