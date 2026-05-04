from dashboard.backend.tests.test_ai_assistant import app, auth_header, seeded_records  # noqa: F401

from uav_dashboard_api.services.ai_assistant import ai_assistant_service


def test_select_knowledge_hits_keeps_only_high_relevance_items(app):
    with app.app_context():
        hits = [
            {
                "source_type": "knowledge_base",
                "source_id": "doc-1",
                "title": "鐢垫満鏁呴殰鎺掓煡鎵嬪唽",
                "snippet": "鐢垫満鏁呴殰閫氬父琛ㄧ幇涓烘尟鍔ㄣ€佸紓鍝嶅拰娓╁崌寮傚父銆?,
                "score": 0.92,
            },
            {
                "source_type": "knowledge_base",
                "source_id": "doc-2",
                "title": "鏃犱汉鏈轰换鍔℃瑙?,
                "snippet": "杩欐槸涓€涓硾鍖栫殑浠诲姟鑳屾櫙浠嬬粛銆?,
                "score": 0.32,
            },
            {
                "source_type": "knowledge_base",
                "source_id": "doc-3",
                "title": "涔辩爜鏂囨。",
                "snippet": "刷直预芯 摘 要 枪业远系统要亩执胁浒踩跋低?,
                "score": 0.88,
            },
        ]

        filtered = ai_assistant_service._select_knowledge_hits(
            query="鐢垫満鏁呴殰鐨勮〃鐜版湁鍝簺",
            hits=hits,
            max_items=3,
        )

        assert len(filtered) == 1
        assert filtered[0]["source_id"] == "doc-1"
