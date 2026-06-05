from flask import Blueprint, jsonify, request

from uav_dashboard_api.services.ai_assistant import ai_assistant_service
from uav_dashboard_api.services.knowledge_base import knowledge_base
from uav_dashboard_api.utils.jwt_utils import token_required

bp = Blueprint('ai', __name__)


@bp.route('/chat', methods=['POST'])
@token_required
def chat(current_user):
    payload = request.get_json(silent=True) or {}
    try:
        data = ai_assistant_service.chat(
            user_id=getattr(current_user, 'id', None),
            message=payload.get('message', ''),
            scene=payload.get('scene'),
            conversation_id=payload.get('conversation_id'),
            report_id=payload.get('report_id'),
            run_id=payload.get('run_id'),
        )
        return jsonify({'success': True, 'data': data})
    except ValueError as exc:
        return jsonify({'success': False, 'message': str(exc)}), 400
    except Exception as exc:
        return jsonify({'success': False, 'message': f'AI 对话失败: {exc}'}), 500


@bp.route('/analyze-report', methods=['POST'])
@token_required
def analyze_report(current_user):
    payload = request.get_json(silent=True) or {}
    report_id = payload.get('report_id')
    if report_id is None:
        return jsonify({'success': False, 'message': 'report_id 为必填'}), 400
    try:
        data = ai_assistant_service.analyze_report(
            user_id=getattr(current_user, 'id', None),
            report_id=int(report_id),
            question=payload.get('question'),
            conversation_id=payload.get('conversation_id'),
        )
        return jsonify({'success': True, 'data': data})
    except ValueError as exc:
        return jsonify({'success': False, 'message': str(exc)}), 400
    except Exception as exc:
        return jsonify({'success': False, 'message': f'AI 报告分析失败: {exc}'}), 500

@bp.route('/conversations', methods=['GET'])
@token_required
def list_conversations(current_user):
    limit = request.args.get('limit', default=20, type=int)
    data = ai_assistant_service.list_conversations(
        user_id=getattr(current_user, 'id', None),
        limit=limit,
    )
    return jsonify({'success': True, 'data': data})


@bp.route('/conversations/<int:conversation_id>', methods=['GET'])
@token_required
def get_conversation(current_user, conversation_id: int):
    try:
        data = ai_assistant_service.get_conversation_detail(
            conversation_id=conversation_id,
            user_id=getattr(current_user, 'id', None),
        )
        return jsonify({'success': True, 'data': data})
    except ValueError as exc:
        return jsonify({'success': False, 'message': str(exc)}), 404


@bp.route('/knowledge/sources', methods=['GET'])
@token_required
def knowledge_sources(current_user):
    data = knowledge_base.list_sources()
    return jsonify({'success': True, 'data': data})


@bp.route('/knowledge/reindex', methods=['POST'])
@token_required
def knowledge_reindex(current_user):
    data = knowledge_base.rebuild_index()
    return jsonify({'success': True, 'message': '知识库索引已重建', 'data': data})
