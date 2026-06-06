from flask import Blueprint, jsonify, request

from uav_dashboard_api import db
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.services.diagnosis_report import create_or_get_diagnosis_report
from uav_dashboard_api.utils.jwt_utils import token_required

bp = Blueprint('diagnosis_reports', __name__)

OFFLINE_MULTICLASS_TASK = 'offline_multiclass'


@bp.route('/', methods=['POST'], strict_slashes=False)
@token_required
def create_diagnosis_report(current_user):
    payload = request.get_json() or {}
    run_id = payload.get('run_id')
    if run_id is None:
        return jsonify({'success': False, 'message': 'run_id 为必填'}), 400

    try:
        run = db.session.get(RunRecord, int(run_id))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'message': 'run_id 格式错误'}), 400
    if run is None:
        return jsonify({'success': False, 'message': '任务不存在'}), 404
    if run.task != OFFLINE_MULTICLASS_TASK:
        return jsonify({'success': False, 'message': '仅支持离线分类诊断任务'}), 400
    if run.status != 'success':
        return jsonify({'success': False, 'message': '仅可为成功任务生成报告'}), 409

    try:
        report, created = create_or_get_diagnosis_report(
            run=run,
            created_by=getattr(current_user, 'id', None),
        )
    except ValueError as exc:
        return jsonify({'success': False, 'message': str(exc)}), 400
    except Exception as exc:
        return jsonify({'success': False, 'message': f'生成报告失败: {exc}'}), 500

    return jsonify({
        'success': True,
        'message': '报告已生成' if created else '报告已存在',
        'created': created,
        'data': report.to_dict(),
    }), 201 if created else 200


@bp.route('/', methods=['GET'], strict_slashes=False)
@token_required
def list_diagnosis_reports(current_user):
    page = request.args.get('page', default=1, type=int)
    per_page = request.args.get('per_page', default=10, type=int)
    asset_id = request.args.get('asset_id', type=int)
    label = request.args.get('label', default='', type=str).strip()

    query = DiagnosisReport.query
    if asset_id:
        query = query.filter_by(asset_id=asset_id)
    if label:
        query = query.filter(DiagnosisReport.prediction_label == label)

    pagination = query.order_by(DiagnosisReport.created_at.desc()).paginate(
        page=max(1, page),
        per_page=max(1, min(per_page, 100)),
        error_out=False,
    )
    return jsonify({
        'success': True,
        'data': [item.to_dict() for item in pagination.items],
        'pagination': {
            'total': pagination.total,
            'page': pagination.page,
            'per_page': pagination.per_page,
            'pages': pagination.pages,
        },
    })


@bp.route('/<int:report_id>', methods=['GET'])
@token_required
def get_diagnosis_report(current_user, report_id: int):
    report = db.session.get(DiagnosisReport, report_id)
    if report is None:
        return jsonify({'success': False, 'message': '报告不存在'}), 404
    return jsonify({'success': True, 'data': report.to_dict()})


@bp.route('/<int:report_id>', methods=['DELETE'])
@token_required
def delete_diagnosis_report(current_user, report_id: int):
    report = db.session.get(DiagnosisReport, report_id)
    if report is None:
        return jsonify({'success': False, 'message': '报告不存在'}), 404

    db.session.delete(report)
    db.session.commit()

    return jsonify({'success': True, 'message': '报告已删除'})
