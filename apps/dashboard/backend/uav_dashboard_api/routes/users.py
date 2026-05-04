# 用户管理路由
from flask import Blueprint, request, jsonify
from uav_dashboard_api import db
from uav_dashboard_api.models.user import User
from uav_dashboard_api.routes.auth import hash_password
from uav_dashboard_api.utils.jwt_utils import token_required, admin_required
from uav_dashboard_api.utils.logger import get_logger
from uav_dashboard_api.utils.error_logger import error_logger
from datetime import datetime

bp = Blueprint('users', __name__)
logger = get_logger(__name__)

@bp.route('/', methods=['GET'])
@token_required
def get_users(current_user):
    """获取用户列表"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        search = request.args.get('search', '')
        role = request.args.get('role', '')
        status = request.args.get('status', '')

        # 构建查询
        query = User.query

        # 搜索条件
        if search:
            query = query.filter(
                (User.username.like(f'%{search}%')) |
                (User.real_name.like(f'%{search}%')) |
                (User.email.like(f'%{search}%'))
            )

        # 角色筛选
        if role:
            query = query.filter(User.role == role)

        # 状态筛选
        if status:
            query = query.filter(User.status == status)

        # 分页
        pagination = query.order_by(User.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )

        return jsonify({
            'success': True,
            'data': [user.to_dict() for user in pagination.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages
            }
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'message': f'获取用户列表失败: {str(e)}'}), 500

@bp.route('/<int:user_id>', methods=['GET'])
@token_required
def get_user(current_user, user_id):
    """获取用户详情"""
    try:
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'success': False, 'message': '用户不存在'}), 404

        return jsonify({
            'success': True,
            'data': user.to_dict()
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'message': f'获取用户详情失败: {str(e)}'}), 500

@bp.route('/', methods=['POST'])
@token_required
@admin_required
def create_user(current_user):
    """创建用户"""
    try:
        data = request.get_json()

        # 验证必填字段
        if not data.get('username') or not data.get('password'):
            return jsonify({'success': False, 'message': '用户名和密码不能为空'}), 400

        # 检查用户名是否已存在
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'success': False, 'message': '用户名已存在'}), 400

        # 检查邮箱是否已存在
        if data.get('email') and User.query.filter_by(email=data['email']).first():
            return jsonify({'success': False, 'message': '邮箱已被注册'}), 400

        # 创建用户
        user = User(
            username=data['username'],
            password=hash_password(data['password']),
            real_name=data.get('real_name', ''),
            email=data.get('email', ''),
            phone=data.get('phone', ''),
            department=data.get('department', ''),
            role=data.get('role', 'user'),
            status=data.get('status', 'active'),
            avatar=data.get('avatar', '')
        )

        db.session.add(user)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '用户创建成功',
            'data': user.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'创建用户失败: {str(e)}'}), 500

@bp.route('/<int:user_id>', methods=['PUT'])
@token_required
@admin_required
def update_user(current_user, user_id):
    """更新用户"""
    try:
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'success': False, 'message': '用户不存在'}), 404

        data = request.get_json()

        # 更新字段
        if 'real_name' in data:
            user.real_name = data['real_name']
        if 'email' in data:
            # 检查邮箱是否已被其他用户使用
            existing_user = User.query.filter(
                User.email == data['email'],
                User.id != user_id
            ).first()
            if existing_user:
                return jsonify({'success': False, 'message': '邮箱已被其他用户使用'}), 400
            user.email = data['email']
        if 'phone' in data:
            user.phone = data['phone']
        if 'department' in data:
            user.department = data['department']
        if 'role' in data:
            user.role = data['role']
        if 'status' in data:
            user.status = data['status']
        if 'avatar' in data:
            user.avatar = data['avatar']

        # 如果要更新密码
        if 'password' in data and data['password']:
            user.password = hash_password(data['password'])

        db.session.commit()

        return jsonify({
            'success': True,
            'message': '用户更新成功',
            'data': user.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'更新用户失败: {str(e)}'}), 500

@bp.route('/<int:user_id>', methods=['DELETE'])
@token_required
@admin_required
def delete_user(current_user, user_id):
    """删除用户"""
    try:
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'success': False, 'message': '用户不存在'}), 404

        # 不允许删除管理员
        if user.role == 'admin':
            return jsonify({'success': False, 'message': '不能删除管理员账号'}), 403

        db.session.delete(user)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '用户删除成功'
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'删除用户失败: {str(e)}'}), 500

@bp.route('/stats', methods=['GET'])
@token_required
def get_user_stats(current_user):
    """获取用户统计信息"""
    try:
        total = User.query.count()
        active = User.query.filter_by(status='active').count()
        inactive = User.query.filter_by(status='inactive').count()
        admin = User.query.filter_by(role='admin').count()
        user_count = User.query.filter_by(role='user').count()

        return jsonify({
            'success': True,
            'data': {
                'total': total,
                'active': active,
                'inactive': inactive,
                'admin': admin,
                'user': user_count
            }
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'message': f'获取统计信息失败: {str(e)}'}), 500

@bp.route('/<int:user_id>/reset-password', methods=['POST'])
@token_required
@admin_required
def reset_user_password(current_user, user_id):
    """重置用户密码"""
    try:
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'success': False, 'message': '用户不存在'}), 404

        data = request.get_json()
        new_password = data.get('new_password')

        if not new_password:
            return jsonify({'success': False, 'message': '新密码不能为空'}), 400

        # 更新密码
        user.password = hash_password(new_password)
        db.session.commit()

        # 记录审计日志
        error_logger.log_audit(
            user_id=current_user.id,
            action='reset_password',
            resource=f'user:{user_id}',
            details={'reset_by': current_user.username}
        )

        logger.info(f'管理员 {current_user.username} 重置了用户 {user.username} 的密码')

        return jsonify({
            'success': True,
            'message': '密码重置成功'
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f'重置密码失败: {str(e)}')
        return jsonify({'success': False, 'message': f'重置密码失败: {str(e)}'}), 500

@bp.route('/<int:user_id>/toggle-status', methods=['POST'])
@token_required
@admin_required
def toggle_user_status(current_user, user_id):
    """切换用户状态（激活/禁用）"""
    try:
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'success': False, 'message': '用户不存在'}), 404

        # 不允许禁用管理员
        if user.role == 'admin':
            return jsonify({'success': False, 'message': '不能禁用管理员账号'}), 403

        # 切换状态
        user.status = 'inactive' if user.status == 'active' else 'active'
        db.session.commit()

        # 记录审计日志
        error_logger.log_audit(
            user_id=current_user.id,
            action='toggle_status',
            resource=f'user:{user_id}',
            details={
                'new_status': user.status,
                'toggled_by': current_user.username
            }
        )

        logger.info(f'管理员 {current_user.username} 将用户 {user.username} 的状态改为 {user.status}')

        return jsonify({
            'success': True,
            'message': f'用户已{"禁用" if user.status == "inactive" else "激活"}',
            'data': user.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f'切换用户状态失败: {str(e)}')
        return jsonify({'success': False, 'message': f'切换用户状态失败: {str(e)}'}), 500

@bp.route('/batch-delete', methods=['POST'])
@token_required
@admin_required
def batch_delete_users(current_user):
    """批量删除用户"""
    try:
        data = request.get_json()
        user_ids = data.get('user_ids', [])

        if not user_ids:
            return jsonify({'success': False, 'message': '请选择要删除的用户'}), 400

        # 查询用户
        users = User.query.filter(User.id.in_(user_ids)).all()

        if not users:
            return jsonify({'success': False, 'message': '未找到要删除的用户'}), 404

        # 检查是否包含管理员
        admin_users = [u for u in users if u.role == 'admin']
        if admin_users:
            return jsonify({'success': False, 'message': '不能删除管理员账号'}), 403

        # 删除用户
        deleted_count = 0
        for user in users:
            db.session.delete(user)
            deleted_count += 1

        db.session.commit()

        # 记录审计日志
        error_logger.log_audit(
            user_id=current_user.id,
            action='batch_delete',
            resource='users',
            details={
                'deleted_count': deleted_count,
                'deleted_by': current_user.username
            }
        )

        logger.info(f'管理员 {current_user.username} 批量删除了 {deleted_count} 个用户')

        return jsonify({
            'success': True,
            'message': f'成功删除 {deleted_count} 个用户'
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f'批量删除用户失败: {str(e)}')
        return jsonify({'success': False, 'message': f'批量删除用户失败: {str(e)}'}), 500

@bp.route('/batch-update-status', methods=['POST'])
@token_required
@admin_required
def batch_update_status(current_user):
    """批量更新用户状态"""
    try:
        data = request.get_json()
        user_ids = data.get('user_ids', [])
        new_status = data.get('status')

        if not user_ids:
            return jsonify({'success': False, 'message': '请选择要更新的用户'}), 400

        if new_status not in ['active', 'inactive']:
            return jsonify({'success': False, 'message': '无效的状态'}), 400

        # 查询用户
        users = User.query.filter(User.id.in_(user_ids)).all()

        if not users:
            return jsonify({'success': False, 'message': '未找到要更新的用户'}), 404

        # 检查是否包含管理员
        admin_users = [u for u in users if u.role == 'admin']
        if admin_users and new_status == 'inactive':
            return jsonify({'success': False, 'message': '不能禁用管理员账号'}), 403

        # 更新状态
        updated_count = 0
        for user in users:
            if user.role != 'admin':  # 不更新管理员状态
                user.status = new_status
                updated_count += 1

        db.session.commit()

        # 记录审计日志
        error_logger.log_audit(
            user_id=current_user.id,
            action='batch_update_status',
            resource='users',
            details={
                'updated_count': updated_count,
                'new_status': new_status,
                'updated_by': current_user.username
            }
        )

        logger.info(f'管理员 {current_user.username} 批量更新了 {updated_count} 个用户的状态为 {new_status}')

        return jsonify({
            'success': True,
            'message': f'成功更新 {updated_count} 个用户的状态'
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f'批量更新用户状态失败: {str(e)}')
        return jsonify({'success': False, 'message': f'批量更新用户状态失败: {str(e)}'}), 500

@bp.route('/profile', methods=['GET'])
@token_required
def get_profile(current_user):
    """获取当前用户信息"""
    try:
        return jsonify({
            'success': True,
            'data': current_user.to_dict()
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'message': f'获取用户信息失败: {str(e)}'}), 500

@bp.route('/profile', methods=['PUT'])
@token_required
def update_profile(current_user):
    """更新当前用户信息"""
    try:
        data = request.get_json()

        # 更新允许用户自己修改的字段
        if 'real_name' in data:
            current_user.real_name = data['real_name']
        if 'email' in data:
            # 检查邮箱是否已被其他用户使用
            existing_user = User.query.filter(
                User.email == data['email'],
                User.id != current_user.id
            ).first()
            if existing_user:
                return jsonify({'success': False, 'message': '邮箱已被其他用户使用'}), 400
            current_user.email = data['email']
        if 'phone' in data:
            current_user.phone = data['phone']
        if 'department' in data:
            current_user.department = data['department']
        if 'avatar' in data:
            current_user.avatar = data['avatar']

        db.session.commit()

        logger.info(f'用户 {current_user.username} 更新了个人信息')

        return jsonify({
            'success': True,
            'message': '个人信息更新成功',
            'data': current_user.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f'更新个人信息失败: {str(e)}')
        return jsonify({'success': False, 'message': f'更新个人信息失败: {str(e)}'}), 500

@bp.route('/change-password', methods=['POST'])
@token_required
def change_password(current_user):
    """修改密码"""
    try:
        data = request.get_json()
        old_password = data.get('old_password')
        new_password = data.get('new_password')

        if not old_password or not new_password:
            return jsonify({'success': False, 'message': '旧密码和新密码不能为空'}), 400

        # 验证旧密码
        from uav_dashboard_api.routes.auth import verify_password
        if not verify_password(old_password, current_user.password):
            return jsonify({'success': False, 'message': '旧密码错误'}), 400

        # 更新密码
        current_user.password = hash_password(new_password)
        db.session.commit()

        # 记录审计日志
        error_logger.log_audit(
            user_id=current_user.id,
            action='change_password',
            resource=f'user:{current_user.id}',
            details={'changed_by': current_user.username}
        )

        logger.info(f'用户 {current_user.username} 修改了密码')

        return jsonify({
            'success': True,
            'message': '密码修改成功'
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f'修改密码失败: {str(e)}')
        return jsonify({'success': False, 'message': f'修改密码失败: {str(e)}'}), 500
