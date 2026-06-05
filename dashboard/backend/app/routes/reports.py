# 报告管理相关API
from flask import Blueprint, request, jsonify, abort
from datetime import datetime
from app import db
from app.models.report import Report
from app.utils.jwt_utils import token_required
import random

# 创建蓝图
bp = Blueprint('reports', __name__)


def generate_id(prefix='RPT'):
    """生成ID"""
    return f'{prefix}-{random.randint(100000, 999999)}'

def _get_report_or_404(report_id):
    report = db.session.get(Report, report_id)
    if report is None:
        abort(404)
    return report


@bp.route('/', methods=['GET'])
@token_required
def get_reports(current_user):
    """获取所有报告"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    report_type = request.args.get('type')
    status = request.args.get('status')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    query = Report.query
    if report_type:
        query = query.filter_by(type=report_type)
    if status:
        query = query.filter_by(status=status)
    
    if start_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            query = query.filter(Report.created_at >= start)
        except ValueError:
            pass
    
    if end_date:
        try:
            end = datetime.strptime(end_date, '%Y-%m-%d')
            end = end.replace(hour=23, minute=59, second=59)
            query = query.filter(Report.created_at <= end)
        except ValueError:
            pass

    pagination = query.order_by(Report.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        'success': True,
        'data': [report.to_dict() for report in pagination.items],
        'pagination': {
            'total': pagination.total,
            'page': page,
            'per_page': per_page,
            'pages': pagination.pages
        }
    })

@bp.route('/<string:report_id>', methods=['GET'])
@token_required
def get_report(current_user, report_id):
    """获取单个报告详情"""
    report = _get_report_or_404(report_id)
    return jsonify({
        'success': True,
        'data': report.to_dict()
    })

@bp.route('/', methods=['POST'])
@token_required
def create_report(current_user):
    """创建新报告"""
    data = request.get_json()
    report = Report(
        id=generate_id(),
        title=data.get('title'),
        type=data.get('type'),
        drone_id=data.get('drone_id'),
        creator=data.get('creator'),
        status=data.get('status', '草稿'),
        summary=data.get('summary'),
        content=data.get('content')
    )
    db.session.add(report)
    db.session.commit()
    return jsonify({
        'success': True,
        'message': '报告创建成功',
        'data': report.to_dict()
    })

@bp.route('/<string:report_id>', methods=['PUT'])
@token_required
def update_report(current_user, report_id):
    """更新报告"""
    report = _get_report_or_404(report_id)
    data = request.get_json()

    if 'status' in data:
        report.status = data['status']
    if 'summary' in data:
        report.summary = data['summary']
    if 'content' in data:
        report.content = data['content']

    db.session.commit()
    return jsonify({
        'success': True,
        'message': '报告更新成功',
        'data': report.to_dict()
    })

@bp.route('/<string:report_id>', methods=['DELETE'])
@token_required
def delete_report(current_user, report_id):
    """删除报告"""
    report = _get_report_or_404(report_id)
    db.session.delete(report)
    db.session.commit()
    return jsonify({
        'success': True,
        'message': '报告删除成功'
    })
