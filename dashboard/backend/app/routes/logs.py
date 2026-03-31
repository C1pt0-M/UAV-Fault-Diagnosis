# 日志管理相关API
from flask import Blueprint, request, jsonify
from datetime import datetime
from app import db
from app.models.log import Log
from app.utils.jwt_utils import token_required
import random

# 创建蓝图
bp = Blueprint('logs', __name__)

def generate_id(prefix='LOG'):
    """生成ID"""
    return f'{prefix}-{random.randint(100000, 999999)}'

@bp.route('/', methods=['GET'])
@token_required
def get_logs(current_user):
    """获取所有日志列表"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        log_type = request.args.get('type')
        drone_id = request.args.get('drone_id')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        query = Log.query

        # 按类型筛选
        if log_type:
            query = query.filter_by(type=log_type)

        # 按无人机ID筛选
        if drone_id:
            query = query.filter_by(drone_id=drone_id)

        # 按日期范围筛选
        if start_date:
            start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
            query = query.filter(Log.created_at >= start_datetime)

        if end_date:
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
            # 结束日期包含当天，所以加一天
            end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
            query = query.filter(Log.created_at <= end_datetime)

        pagination = query.order_by(Log.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )

        return jsonify({
            'success': True,
            'data': [log.to_dict() for log in pagination.items],
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
            'message': f'获取日志列表失败: {str(e)}'
        }), 500

@bp.route('/<string:log_id>', methods=['GET'])
@token_required
def get_log(current_user, log_id):
    """获取单个日志详情"""
    try:
        log = db.session.get(Log, log_id)
        if not log:
            return jsonify({
                'success': False,
                'message': '日志记录不存在'
            }), 404

        return jsonify({
            'success': True,
            'data': log.to_dict()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取日志详情失败: {str(e)}'
        }), 500

@bp.route('/', methods=['POST'])
@token_required
def create_log(current_user):
    """创建新日志"""
    try:
        data = request.get_json()
        
        # 验证必填字段
        if not data.get('type'):
            return jsonify({
                'success': False,
                'message': '日志类型不能为空'
            }), 400
            
        if not data.get('content'):
            return jsonify({
                'success': False,
                'message': '日志内容不能为空'
            }), 400

        # 验证日志类型
        valid_types = ['flight', 'fault', 'maintenance', 'system']
        if data['type'] not in valid_types:
            return jsonify({
                'success': False,
                'message': f'无效的日志类型，必须是: {", ".join(valid_types)}'
            }), 400

        log = Log(
            id=generate_id(),
            drone_id=data.get('drone_id'),
            type=data['type'],
            content=data['content'],
            operator=data.get('operator', current_user.username if hasattr(current_user, 'username') else '系统')
        )
        db.session.add(log)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '日志创建成功',
            'data': log.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'创建日志失败: {str(e)}'
        }), 500

@bp.route('/<string:log_id>', methods=['DELETE'])
@token_required
def delete_log(current_user, log_id):
    """删除日志"""
    try:
        log = db.session.get(Log, log_id)
        if not log:
            return jsonify({
                'success': False,
                'message': '日志记录不存在'
            }), 404

        db.session.delete(log)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '日志删除成功'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'删除日志失败: {str(e)}'
        }), 500

