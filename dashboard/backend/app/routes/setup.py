# 系统安装向导路由
from flask import Blueprint, jsonify, request
from app import db
from app.models.user import User
from app.routes.auth import hash_password
from app.utils.logger import get_logger
from app.utils.validators import Validators, validate_required_fields
from datetime import datetime

bp = Blueprint('setup', __name__)
logger = get_logger(__name__)

@bp.route('/status', methods=['GET'])
def setup_status():
    """
    获取系统安装状态
    """
    try:
        # 检查是否有用户
        user_count = User.query.count()

        # 检查是否有管理员
        admin_count = User.query.filter_by(role='admin').count()

        return jsonify({
            'success': True,
            'data': {
                'is_installed': user_count > 0,
                'has_admin': admin_count > 0,
                'user_count': user_count,
                'admin_count': admin_count,
                'needs_setup': user_count == 0
            }
        })
    except Exception as e:
        logger.error(f'获取安装状态失败: {str(e)}')
        return jsonify({
            'success': False,
            'message': f'获取安装状态失败: {str(e)}'
        }), 500

@bp.route('/initialize', methods=['POST'])
def initialize_system():
    """
    初始化系统（创建管理员账户）
    """
    try:
        data = request.get_json()
        logger.info('系统初始化请求')

        # 检查系统是否已安装
        user_count = User.query.count()
        if user_count > 0:
            return jsonify({
                'success': False,
                'message': '系统已安装，无法重复初始化'
            }), 400

        # 验证必填字段
        is_valid, error_msg = validate_required_fields(data, [
            'username', 'password', 'real_name', 'email'
        ])
        if not is_valid:
            return jsonify({
                'success': False,
                'message': error_msg
            }), 400

        # 验证用户名
        is_valid, error_msg = Validators.validate_username(data['username'])
        if not is_valid:
            return jsonify({
                'success': False,
                'message': error_msg
            }), 400

        # 验证密码
        is_valid, error_msg = Validators.validate_password(data['password'])
        if not is_valid:
            return jsonify({
                'success': False,
                'message': error_msg
            }), 400

        # 验证邮箱
        is_valid, error_msg = Validators.validate_email(data['email'])
        if not is_valid:
            return jsonify({
                'success': False,
                'message': error_msg
            }), 400

        # 检查用户名是否已存在
        existing_user = User.query.filter_by(username=data['username']).first()
        if existing_user:
            return jsonify({
                'success': False,
                'message': '用户名已存在'
            }), 400

        # 检查邮箱是否已存在
        existing_email = User.query.filter_by(email=data['email']).first()
        if existing_email:
            return jsonify({
                'success': False,
                'message': '邮箱已存在'
            }), 400

        # 创建管理员账户
        admin = User(
            username=data['username'],
            password=hash_password(data['password']),
            real_name=data['real_name'],
            email=data['email'],
            phone=data.get('phone', ''),
            department=data.get('department', '技术部'),
            role='admin',
            status='active'
        )

        db.session.add(admin)
        db.session.commit()

        logger.info(f'系统初始化成功，创建管理员账户: {data["username"]}')

        return jsonify({
            'success': True,
            'message': '系统初始化成功',
            'data': {
                'username': admin.username,
                'real_name': admin.real_name,
                'email': admin.email,
                'role': admin.role
            }
        })

    except Exception as e:
        logger.error(f'系统初始化失败: {str(e)}')
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'系统初始化失败: {str(e)}'
        }), 500

@bp.route('/check-username', methods=['POST'])
def check_username():
    """
    检查用户名是否可用
    """
    try:
        data = request.get_json()
        username = data.get('username', '')

        if not username:
            return jsonify({
                'success': False,
                'message': '用户名不能为空'
            }), 400

        # 检查用户名格式
        is_valid, error_msg = Validators.validate_username(username)
        if not is_valid:
            return jsonify({
                'success': True,
                'data': {
                    'available': False,
                    'message': error_msg
                }
            })

        # 检查用户名是否已存在
        existing_user = User.query.filter_by(username=username).first()

        return jsonify({
            'success': True,
            'data': {
                'available': existing_user is None,
                'message': '用户名可用' if existing_user is None else '用户名已存在'
            }
        })

    except Exception as e:
        logger.error(f'检查用户名失败: {str(e)}')
        return jsonify({
            'success': False,
            'message': f'检查用户名失败: {str(e)}'
        }), 500

@bp.route('/check-email', methods=['POST'])
def check_email():
    """
    检查邮箱是否可用
    """
    try:
        data = request.get_json()
        email = data.get('email', '')

        if not email:
            return jsonify({
                'success': False,
                'message': '邮箱不能为空'
            }), 400

        # 检查邮箱格式
        is_valid, error_msg = Validators.validate_email(email)
        if not is_valid:
            return jsonify({
                'success': True,
                'data': {
                    'available': False,
                    'message': error_msg
                }
            })

        # 检查邮箱是否已存在
        existing_email = User.query.filter_by(email=email).first()

        return jsonify({
            'success': True,
            'data': {
                'available': existing_email is None,
                'message': '邮箱可用' if existing_email is None else '邮箱已存在'
            }
        })

    except Exception as e:
        logger.error(f'检查邮箱失败: {str(e)}')
        return jsonify({
            'success': False,
            'message': f'检查邮箱失败: {str(e)}'
        }), 500
