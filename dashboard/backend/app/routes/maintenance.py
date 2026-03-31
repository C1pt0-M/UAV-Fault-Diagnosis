# 维修方案相关API
from flask import Blueprint, request, jsonify, abort
import json
from app import db
from app.models.maintenance import Maintenance
from app.utils.jwt_utils import token_required
import random

# 创建蓝图
bp = Blueprint('maintenance', __name__)


def generate_id(prefix='MNT'):
    """生成ID"""
    return f'{prefix}-{random.randint(100000, 999999)}'

def _get_task_or_404(task_id):
    task = db.session.get(Maintenance, task_id)
    if task is None:
        abort(404)
    return task


@bp.route('/', methods=['GET'])
@token_required
def get_maintenance_tasks(current_user):
    """获取所有维修任务"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    status = request.args.get('status')

    query = Maintenance.query
    if status:
        query = query.filter_by(status=status)

    pagination = query.order_by(Maintenance.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        'success': True,
        'data': [task.to_dict() for task in pagination.items],
        'total': pagination.total,
        'page': page,
        'per_page': per_page
    })

@bp.route('/<string:task_id>', methods=['GET'])
@token_required
def get_maintenance_task(current_user, task_id):
    """获取单个维修任务详情"""
    task = _get_task_or_404(task_id)
    return jsonify({
        'success': True,
        'data': task.to_dict()
    })

@bp.route('/', methods=['POST'])
@token_required
def create_maintenance_task(current_user):
    """创建维修任务"""
    data = request.get_json()
    task = Maintenance(
        id=generate_id(),
        drone_id=data.get('drone_id'),
        title=data.get('title'),
        fault_type=data.get('fault_type'),
        description=data.get('description'),
        priority=data.get('priority', '中'),
        status=data.get('status', '待处理'),
        assignee=data.get('assignee'),
        steps=json.dumps(data.get('steps', []), ensure_ascii=False),
        parts=json.dumps(data.get('parts', []), ensure_ascii=False),
        records=json.dumps([], ensure_ascii=False)
    )
    db.session.add(task)
    db.session.commit()
    return jsonify({
        'success': True,
        'message': '维修任务创建成功',
        'data': task.to_dict()
    })

@bp.route('/<string:task_id>', methods=['PUT'])
@token_required
def update_maintenance_task(current_user, task_id):
    """更新维修任务"""
    task = _get_task_or_404(task_id)
    data = request.get_json()

    if 'status' in data:
        task.status = data['status']
    if 'assignee' in data:
        task.assignee = data['assignee']
    if 'records' in data:
        task.records = json.dumps(data['records'], ensure_ascii=False)

    db.session.commit()
    return jsonify({
        'success': True,
        'message': '维修任务更新成功',
        'data': task.to_dict()
    })

@bp.route('/<string:task_id>/records', methods=['POST'])
@token_required
def add_maintenance_record(current_user, task_id):
    """添加维修记录"""
    task = _get_task_or_404(task_id)
    data = request.get_json()

    records = json.loads(task.records) if task.records else []
    records.append({
        'time': data.get('time'),
        'content': data.get('content')
    })

    task.records = json.dumps(records, ensure_ascii=False)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': '维修记录添加成功',
        'data': task.to_dict()
    })

@bp.route('/<string:task_id>', methods=['DELETE'])
@token_required
def delete_maintenance_task(current_user, task_id):
    """删除维修任务"""
    task = _get_task_or_404(task_id)
    db.session.delete(task)
    db.session.commit()
    return jsonify({
        'success': True,
        'message': '维修任务删除成功'
    })
