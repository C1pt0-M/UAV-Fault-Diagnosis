# 无人机相关API
from flask import Blueprint, request, jsonify, abort
from app import db
from app.models.drone import Drone
from app.utils.validators import Validators, sanitize_input, validate_required_fields
from app.utils.jwt_utils import token_required, admin_required
import random
import string

# 创建蓝图
bp = Blueprint('drones', __name__)


def generate_id(prefix='DRN'):
    """生成ID"""
    return f'{prefix}-{random.randint(100000, 999999)}'

def _get_drone_or_404(drone_id):
    drone = db.session.get(Drone, drone_id)
    if drone is None:
        abort(404)
    return drone


@bp.route('/', methods=['GET'])
@token_required
def get_drones(current_user):
    """获取所有无人机列表"""
    drones = Drone.query.all()
    return jsonify({
        'success': True,
        'data': [drone.to_dict() for drone in drones]
    })

@bp.route('/<string:drone_id>', methods=['GET'])
@token_required
def get_drone(current_user, drone_id):
    """获取单个无人机详情"""
    drone = _get_drone_or_404(drone_id)
    return jsonify({
        'success': True,
        'data': drone.to_dict()
    })

@bp.route('/', methods=['POST'])
@token_required
@admin_required
def create_drone(current_user):
    """创建新无人机"""
    try:
        data = request.get_json()

        # 清理输入数据
        sanitized_data = sanitize_input(data)

        # 验证电池电量
        if 'battery' in sanitized_data:
            is_valid, error_msg = Validators.validate_battery_level(sanitized_data['battery'])
            if not is_valid:
                return jsonify({'success': False, 'message': error_msg}), 400

        # 验证状态
        if 'status' in sanitized_data:
            is_valid, error_msg = Validators.validate_drone_status(sanitized_data['status'])
            if not is_valid:
                return jsonify({'success': False, 'message': error_msg}), 400

        drone = Drone(
            id=generate_id(),
            name=sanitized_data.get('name', f'无人机-{random.randint(1, 999)}'),
            status=sanitized_data.get('status', 'offline'),
            battery=sanitized_data.get('battery', 100),
            signal=sanitized_data.get('signal', '强')
        )
        db.session.add(drone)
        db.session.commit()
        return jsonify({
            'success': True,
            'message': '无人机创建成功',
            'data': drone.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'创建失败: {str(e)}'}), 500

@bp.route('/<string:drone_id>', methods=['PUT'])
@token_required
@admin_required
def update_drone(current_user, drone_id):
    """更新无人机信息"""
    drone = _get_drone_or_404(drone_id)
    data = request.get_json()

    if 'status' in data:
        drone.status = data['status']
    if 'battery' in data:
        drone.battery = data['battery']
    if 'signal' in data:
        drone.signal = data['signal']
    if 'height' in data:
        drone.height = data['height']
    if 'speed' in data:
        drone.speed = data['speed']
    if 'distance' in data:
        drone.distance = data['distance']
    if 'duration' in data:
        drone.duration = data['duration']
    if 'task_status' in data:
        drone.task_status = data['task_status']

    db.session.commit()
    return jsonify({
        'success': True,
        'message': '无人机信息更新成功',
        'data': drone.to_dict()
    })

@bp.route('/<string:drone_id>', methods=['DELETE'])
@token_required
@admin_required
def delete_drone(current_user, drone_id):
    """删除无人机"""
    drone = _get_drone_or_404(drone_id)
    db.session.delete(drone)
    db.session.commit()
    return jsonify({
        'success': True,
        'message': '无人机删除成功'
    })

@bp.route('/stats', methods=['GET'])
@token_required
def get_drone_stats(current_user):
    """获取无人机统计信息"""
    from sqlalchemy import func

    stats = db.session.query(
        Drone.status,
        func.count(Drone.id)
    ).group_by(Drone.status).all()

    total = sum(s[1] for s in stats)
    online = next((s[1] for s in stats if s[0] == 'online'), 0)
    offline = next((s[1] for s in stats if s[0] == 'offline'), 0)

    return jsonify({
        'success': True,
        'data': {
            'total': total,
            'online': online,
            'offline': offline
        }
    })
