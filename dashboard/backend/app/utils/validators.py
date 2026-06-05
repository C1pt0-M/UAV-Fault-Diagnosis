# 输入验证和XSS防护工具
import re
import html
from functools import wraps
from flask import request, jsonify
from app.utils.error_handler import ValidationError

class Validators:
    """输入验证器"""

    @staticmethod
    def validate_username(username):
        """
        验证用户名

        Args:
            username: 用户名

        Returns:
            tuple: (is_valid, error_message)
        """
        if not username:
            return False, '用户名不能为空'

        if len(username) < 3:
            return False, '用户名长度不能少于3个字符'

        if len(username) > 20:
            return False, '用户名长度不能超过20个字符'

        # 只允许字母、数字、下划线
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, '用户名只能包含字母、数字和下划线'

        return True, None

    @staticmethod
    def validate_password(password):
        """
        验证密码

        Args:
            password: 密码

        Returns:
            tuple: (is_valid, error_message)
        """
        if not password:
            return False, '密码不能为空'

        if len(password) < 6:
            return False, '密码长度不能少于6个字符'

        if len(password) > 20:
            return False, '密码长度不能超过20个字符'

        # 可选：要求包含字母和数字
        # if not re.search(r'[a-zA-Z]', password):
        #     return False, '密码必须包含字母'
        # if not re.search(r'\d', password):
        #     return False, '密码必须包含数字'

        return True, None

    @staticmethod
    def validate_email(email):
        """
        验证邮箱格式

        Args:
            email: 邮箱地址

        Returns:
            tuple: (is_valid, error_message)
        """
        if not email:
            return True, None  # 邮箱是可选的

        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False, '邮箱格式不正确'

        return True, None

    @staticmethod
    def validate_phone(phone):
        """
        验证手机号格式（中国大陆）

        Args:
            phone: 手机号

        Returns:
            tuple: (is_valid, error_message)
        """
        if not phone:
            return True, None  # 手机号是可选的

        # 中国大陆手机号：1开头，11位数字
        pattern = r'^1[3-9]\d{9}$'
        if not re.match(pattern, phone):
            return False, '手机号格式不正确'

        return True, None

    @staticmethod
    def validate_drone_status(status):
        """
        验证无人机状态

        Args:
            status: 状态值

        Returns:
            tuple: (is_valid, error_message)
        """
        valid_statuses = ['online', 'offline', 'maintenance']
        if status not in valid_statuses:
            return False, f'状态值必须是以下之一: {", ".join(valid_statuses)}'
        return True, None

    @staticmethod
    def validate_battery_level(battery):
        """
        验证电池电量

        Args:
            battery: 电池电量

        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            battery_int = int(battery)
            if battery_int < 0 or battery_int > 100:
                return False, '电池电量必须在0-100之间'
            return True, None
        except (ValueError, TypeError):
            return False, '电池电量必须是数字'

    @staticmethod
    def validate_fault_level(level):
        """
        验证故障级别

        Args:
            level: 故障级别

        Returns:
            tuple: (is_valid, error_message)
        """
        valid_levels = ['严重', '警告', '信息']
        if level not in valid_levels:
            return False, f'故障级别必须是以下之一: {", ".join(valid_levels)}'
        return True, None

    @staticmethod
    def validate_fault_status(status):
        """
        验证故障状态

        Args:
            status: 故障状态

        Returns:
            tuple: (is_valid, error_message)
        """
        valid_statuses = ['待处理', '处理中', '已解决']
        if status not in valid_statuses:
            return False, f'故障状态必须是以下之一: {", ".join(valid_statuses)}'
        return True, None


def sanitize_html(text):
    """
    清理HTML，防止XSS攻击

    Args:
        text: 输入文本

    Returns:
        str: 清理后的文本
    """
    if not text:
        return text

    # 转义HTML特殊字符
    sanitized = html.escape(str(text))

    # 可选：移除危险的HTML标签和属性
    # 这里使用简单的转义，如果需要更复杂的处理，可以使用 bleach 库
    # import bleach
    # sanitized = bleach.clean(text, tags=[], attributes={}, strip=True)

    return sanitized


def sanitize_input(data):
    """
    清理输入数据中的所有字符串字段

    Args:
        data: 输入数据（字典或列表）

    Returns:
        清理后的数据
    """
    if isinstance(data, dict):
        return {key: sanitize_input(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_input(item) for item in data]
    elif isinstance(data, str):
        return sanitize_html(data)
    else:
        return data


def validate_required_fields(data, required_fields):
    """
    验证必填字段

    Args:
        data: 输入数据
        required_fields: 必填字段列表

    Returns:
        tuple: (is_valid, error_message)
    """
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f'字段 {field} 是必填的'
    return True, None


def validation_error_handler(f):
    """
    验证错误处理装饰器

    使用方法:
        @validation_error_handler
        def create_drone():
            # 验证逻辑
            if not is_valid:
                raise ValidationError('验证失败')
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValidationError as e:
            return jsonify({
                'success': False,
                'message': str(e)
            }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'请求处理失败: {str(e)}'
            }), 500

    return decorated
