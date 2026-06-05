# JWT工具类
import jwt
import datetime
from functools import wraps
from flask import request, jsonify, current_app
from app.models.user import User

class JWTUtils:
    """JWT工具类"""

    @staticmethod
    def generate_token(user_id, expires_in=None):
        """
        生成JWT token

        Args:
            user_id: 用户ID
            expires_in: 过期时间（秒），默认从配置读取

        Returns:
            str: JWT token
        """
        try:
            if expires_in is None:
                expires_in = current_app.config.get('JWT_EXPIRE_HOURS', 24) * 60 * 60

            payload = {
                'user_id': user_id,
                'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=expires_in),
                'iat': datetime.datetime.now(datetime.timezone.utc),
                'type': 'access'
            }
            secret_key = current_app.config.get('JWT_SECRET_KEY', current_app.config['SECRET_KEY'])
            token = jwt.encode(
                payload,
                secret_key,
                algorithm='HS256'
            )
            return token
        except Exception as e:
            raise Exception(f'生成token失败: {str(e)}')

    @staticmethod
    def decode_token(token):
        """
        解码JWT token

        Args:
            token: JWT token

        Returns:
            dict: 解码后的payload

        Raises:
            Exception: token无效或过期
        """
        try:
            # 使用JWT专用密钥，如果不存在则使用SECRET_KEY
            secret_key = current_app.config.get('JWT_SECRET_KEY', current_app.config['SECRET_KEY'])
            payload = jwt.decode(
                token,
                secret_key,
                algorithms=['HS256']
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception('token已过期')
        except jwt.InvalidTokenError:
            raise Exception('无效的token')

    @staticmethod
    def verify_token(token):
        """
        验证token并返回用户信息

        Args:
            token: JWT token

        Returns:
            User: 用户对象

        Raises:
            Exception: token无效或用户不存在
        """
        try:
            payload = JWTUtils.decode_token(token)
            user_id = payload.get('user_id')

            if not user_id:
                raise Exception('token中缺少用户ID')

            from app import db
            user = db.session.get(User, user_id)
            if not user:
                raise Exception('用户不存在')

            if user.status != 'active':
                raise Exception('账号已被禁用')

            return user
        except Exception as e:
            raise Exception(f'token验证失败: {str(e)}')


def token_required(f):
    """
    JWT认证装饰器

    使用方法:
        @bp.route('/profile', methods=['GET'])
        @token_required
        def get_profile(current_user):
            return jsonify({'data': current_user.to_dict()})
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # 从header中获取token
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                # 支持两种格式: "Bearer <token>" 或直接 "<token>"
                if auth_header.startswith('Bearer '):
                    token = auth_header.split(' ')[1]
                else:
                    token = auth_header
            except IndexError:
                return jsonify({
                    'success': False,
                    'message': 'token格式错误'
                }), 401

        if not token:
            return jsonify({
                'success': False,
                'message': '未提供认证token'
            }), 401

        try:
            current_user = JWTUtils.verify_token(token)
        except Exception as e:
            return jsonify({
                'success': False,
                'message': str(e)
            }), 401

        # 将当前用户传递给路由函数
        return f(current_user, *args, **kwargs)

    return decorated


def admin_required(f):
    """
    管理员权限装饰器

    使用方法:
        @bp.route('/users', methods=['DELETE'])
        @token_required
        @admin_required
        def delete_user(current_user):
            # ...
    """
    @wraps(f)
    def decorated(current_user, *args, **kwargs):
        if current_user.role != 'admin':
            return jsonify({
                'success': False,
                'message': '需要管理员权限'
            }), 403
        return f(current_user, *args, **kwargs)

    return decorated
