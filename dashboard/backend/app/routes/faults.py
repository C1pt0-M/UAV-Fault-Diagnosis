# 故障检测相关API
from flask import Blueprint, request, jsonify
from datetime import datetime
from app import db
from app.models.fault import Fault
from app.utils.validators import Validators, sanitize_input, validate_required_fields
from app.utils.jwt_utils import token_required, admin_required
import random

# 创建蓝图
bp = Blueprint('faults', __name__)

def generate_id(prefix='FLT'):
    """生成ID"""
    return f'{prefix}-{random.randint(100000, 999999)}'

@bp.route('/', methods=['GET'])
@token_required
def get_faults(current_user):
    """获取所有故障列表"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        status = request.args.get('status')

        query = Fault.query
        if status:
            query = query.filter_by(status=status)

        pagination = query.order_by(Fault.detect_time.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )

        return jsonify({
            'success': True,
            'data': [fault.to_dict() for fault in pagination.items],
            'pagination': {
                'total': pagination.total,
                'page': page,
                'per_page': per_page,
                'pages': pagination.pages
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取故障列表失败: {str(e)}'
        }), 500

@bp.route('/<string:fault_id>', methods=['GET'])
@token_required
def get_fault(current_user, fault_id):
    """获取单个故障详情"""
    try:
        fault = db.session.get(Fault, fault_id)
        if not fault:
            return jsonify({
                'success': False,
                'message': '故障记录不存在'
            }), 404

        return jsonify({
            'success': True,
            'data': fault.to_dict()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取故障详情失败: {str(e)}'
        }), 500

@bp.route('/', methods=['POST'])
@token_required
def create_fault(current_user):
    """上报新故障"""
    try:
        data = request.get_json()

        # 验证必填字段
        is_valid, error_msg = validate_required_fields(data, ['drone_id', 'fault_type'])
        if not is_valid:
            return jsonify({'success': False, 'message': error_msg}), 400

        # 清理输入数据
        sanitized_data = sanitize_input(data)

        # 验证故障级别
        if 'level' in sanitized_data:
            is_valid, error_msg = Validators.validate_fault_level(sanitized_data['level'])
            if not is_valid:
                return jsonify({'success': False, 'message': error_msg}), 400

        # 验证故障状态
        if 'status' in sanitized_data:
            is_valid, error_msg = Validators.validate_fault_status(sanitized_data['status'])
            if not is_valid:
                return jsonify({'success': False, 'message': error_msg}), 400

        fault = Fault(
            id=generate_id(),
            drone_id=sanitized_data.get('drone_id'),
            fault_type=sanitized_data.get('fault_type'),
            description=sanitized_data.get('description'),
            level=sanitized_data.get('level', '信息'),
            status=sanitized_data.get('status', '待处理')
        )
        db.session.add(fault)
        db.session.commit()
        return jsonify({
            'success': True,
            'message': '故障上报成功',
            'data': fault.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'上报失败: {str(e)}'}), 500

@bp.route('/<string:fault_id>/status', methods=['PUT'])
@token_required
def update_fault_status(current_user, fault_id):
    """更新故障状态"""
    try:
        fault = db.session.get(Fault, fault_id)
        if not fault:
            return jsonify({
                'success': False,
                'message': '故障记录不存在'
            }), 404

        data = request.get_json()
        if not data or 'status' not in data:
            return jsonify({
                'success': False,
                'message': '缺少状态参数'
            }), 400

        # 验证状态值
        valid_statuses = ['待处理', '处理中', '已解决']
        if data['status'] not in valid_statuses:
            return jsonify({
                'success': False,
                'message': f'无效的状态值，必须是: {", ".join(valid_statuses)}'
            }), 400

        fault.status = data['status']
        if data['status'] == '已解决':
            fault.resolved_time = datetime.now()

        db.session.commit()
        return jsonify({
            'success': True,
            'message': '故障状态更新成功',
            'data': fault.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'更新故障状态失败: {str(e)}'
        }), 500

@bp.route('/<string:fault_id>', methods=['DELETE'])
@token_required
@admin_required
def delete_fault(current_user, fault_id):
    """删除故障记录"""
    try:
        fault = db.session.get(Fault, fault_id)
        if not fault:
            return jsonify({
                'success': False,
                'message': '故障记录不存在'
            }), 404

        db.session.delete(fault)
        db.session.commit()
        return jsonify({
            'success': True,
            'message': '故障记录删除成功'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'删除故障记录失败: {str(e)}'
        }), 500

@bp.route('/stats', methods=['GET'])
@token_required
def get_fault_stats(current_user):
    """获取故障统计信息"""
    try:
        from sqlalchemy import func

        stats = db.session.query(
            Fault.status,
            func.count(Fault.id)
        ).group_by(Fault.status).all()

        total = sum(s[1] for s in stats)
        pending = next((s[1] for s in stats if s[0] == '待处理'), 0)
        processing = next((s[1] for s in stats if s[0] == '处理中'), 0)
        resolved = next((s[1] for s in stats if s[0] == '已解决'), 0)

        critical = Fault.query.filter_by(level='严重').count()
        warning = Fault.query.filter_by(level='警告').count()
        info = Fault.query.filter_by(level='信息').count()

        return jsonify({
            'success': True,
            'data': {
                'total': total,
                'pending': pending,
                'processing': processing,
                'resolved': resolved,
                'critical': critical,
                'warning': warning,
                'info': info
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取故障统计信息失败: {str(e)}'
        }), 500
