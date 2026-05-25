from pathlib import Path

from uav_dashboard_api.services.knowledge_base import knowledge_base


def test_knowledge_base_token_regex_uses_direct_chinese_range_in_source():
    source_path = Path(__file__).resolve().parents[1] / 'uav_dashboard_api' / 'services' / 'knowledge_base.py'
    source_text = source_path.read_text(encoding='utf-8')
    legacy_pattern = '[' + '\\\\' + 'u4e00-' + '\\\\' + 'u9fff]'

    assert '[一-龥]' in source_text
    assert legacy_pattern not in source_text


def test_tokenize_keeps_english_words_and_chinese_characters():
    assert knowledge_base._tokenize('电机 bearing 温度 fault') == [
        '电',
        '机',
        'bearing',
        '温',
        '度',
        'fault',
    ]
