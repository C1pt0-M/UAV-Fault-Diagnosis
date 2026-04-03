# 用户认证路由
from flask import Blueprint, request, jsonify
from datetime import datetime
from app import db
from app.models.user import User
from app.utils.jwt_utils import JWTUtils, token_required
from app.utils.validators import (
    Validators, sanitize_input, validate_required_fields, ValidationError
)
from app.utils.logger import get_logger
import bcrypt

bp = Blueprint('auth', __name__)
logger = get_logger(__name__)

def hash_password(password):
    """
    使用bcrypt加密密码

    Args:
        password: 明文密码

    Returns:
        str: 加密后的密码
    """
    # 生成盐值并加密
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password, hashed_password):
    """
    验证密码

    Args:
        password: 明文密码
        hashed_password: 加密后的密码

    Returns:
        bool: 密码是否正确
    """
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

@bp.route('/login', methods=['POST'])
def login():
    """用户登录"""
    try:
        data = request.get_json()
        logger.info(f'用户登录尝试: username={data.get("username")}')

        # 验证必填字段
        is_valid, error_msg = validate_required_fields(data, ['username', 'password'])
        if not is_valid:
            logger.warning(f'登录验证失败: {error_msg}')
            return jsonify({'success': False, 'message': error_msg}), 400

        username = data.get('username')
        password = data.get('password')

        # 验证用户名格式
        is_valid, error_msg = Validators.validate_username(username)
        if not is_valid:
            logger.warning(f'用户名格式验证失败: {error_msg}')
            return jsonify({'success': False, 'message': error_msg}), 400

        # 验证密码格式
        is_valid, error_msg = Validators.validate_password(password)
        if not is_valid:
            logger.warning(f'密码格式验证失败: {error_msg}')
            return jsonify({'success': False, 'message': error_msg}), 400

        # 查找用户
        user = User.query.filter_by(username=username).first()

        if not user:
            logger.warning(f'登录失败: 用户不存在 - {username}')
            return jsonify({'success': False, 'message': '用户名或密码错误'}), 401

        # 验证密码
        if not verify_password(password, user.password):
            logger.warning(f'登录失败: 密码错误 - {username}')
            return jsonify({'success': False, 'message': '用户名或密码错误'}), 401

        # 检查用户状态
        if user.status != 'active':
            logger.warning(f'登录失败: 账号已被禁用 - {username}')
            return jsonify({'success': False, 'message': '账号已被禁用'}), 403

        # 生成JWT token
        token = JWTUtils.generate_token(user.id)

        # 更新登录信息
        user.last_login = datetime.now()
        user.login_count += 1
        db.session.commit()

        logger.info(f'用户登录成功: {username}')
        return jsonify({
            'success': True,
            'message': '登录成功',
            'data': {
                'token': token,
                'user': user.to_dict()
            }
        }), 200

    except Exception as e:
        logger.error(f'登录异常: {str(e)}', exc_info=True)
        return jsonify({'success': False, 'message': f'登录失败: {str(e)}'}), 500

@bp.route('/register', methods=['POST'])
def register():
    """用户注册"""
    try:
        from flask import current_app

        if not current_app.config.get('ENABLE_PUBLIC_REGISTER', False):
            return jsonify({
                'success': False,
                'message': 'public_registration_disabled'
            }), 403

        data = request.get_json()

        # 验证必填字段
        is_valid, error_msg = validate_required_fields(data, ['username', 'password'])
        if not is_valid:
            return jsonify({'success': False, 'message': error_msg}), 400

        username = data.get('username')
        password = data.get('password')
        real_name = data.get('real_name', '')
        email = data.get('email', '')
        phone = data.get('phone', '')
        department = data.get('department', '')

        # 验证用户名格式
        is_valid, error_msg = Validators.validate_username(username)
        if not is_valid:
            return jsonify({'success': False, 'message': error_msg}), 400

        # 验证密码格式
        is_valid, error_msg = Validators.validate_password(password)
        if not is_valid:
            return jsonify({'success': False, 'message': error_msg}), 400

        # 验证邮箱格式
        is_valid, error_msg = Validators.validate_email(email)
        if not is_valid:
            return jsonify({'success': False, 'message': error_msg}), 400

        # 验证手机号格式
        is_valid, error_msg = Validators.validate_phone(phone)
        if not is_valid:
            return jsonify({'success': False, 'message': error_msg}), 400

        # 检查用户名是否已存在
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': '用户名已存在'}), 400

        # 检查邮箱是否已存在
        if email and User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': '邮箱已被注册'}), 400

        # 清理输入数据，防止XSS攻击
        sanitized_data = sanitize_input({
            'real_name': real_name,
            'department': department
        })

        # 创建用户（密码已经加密）
        user = User(
            username=username,
            password=hash_password(password),
            real_name=sanitized_data['real_name'],
            email=email,
            phone=phone,
            department=sanitized_data['department'],
            role='user',
            status='active'
        )

        db.session.add(user)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '注册成功',
            'data': user.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'注册失败: {str(e)}'}), 500

@bp.route('/logout', methods=['POST'])
def logout():
    """用户登出"""
    return jsonify({
        'success': True,
        'message': '登出成功'
    }), 200

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
    """更新用户信息"""
    try:
        data = request.get_json()

        # 更新字段
        if 'real_name' in data:
            current_user.real_name = data['real_name']
        if 'email' in data:
            current_user.email = data['email']
        if 'phone' in data:
            current_user.phone = data['phone']
        if 'department' in data:
            current_user.department = data['department']
        if 'avatar' in data:
            current_user.avatar = data['avatar']

        # 如果要更新密码
        if 'password' in data and data['password']:
            current_user.password = hash_password(data['password'])

        db.session.commit()

        return jsonify({
            'success': True,
            'message': '更新成功',
            'data': current_user.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'更新失败: {str(e)}'}), 500
