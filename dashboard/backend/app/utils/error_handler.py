# 全局异常处理器
from flask import jsonify
from werkzeug.exceptions import HTTPException
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import traceback
from datetime import datetime
from app.utils.logger import get_logger

logger = get_logger(__name__)


class APIError(Exception):
    """自定义API异常基类"""
    def __init__(self, message, status_code=500, payload=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        rv['status'] = 'error'
        rv['timestamp'] = datetime.now().isoformat()
        return rv


class ValidationError(APIError):
    """验证错误"""
    def __init__(self, message, payload=None):
        super().__init__(message, 400, payload)


class AuthenticationError(APIError):
    """认证错误"""
    def __init__(self, message, payload=None):
        super().__init__(message, 401, payload)


class AuthorizationError(APIError):
    """授权错误"""
    def __init__(self, message, payload=None):
        super().__init__(message, 403, payload)


class NotFoundError(APIError):
    """资源未找到错误"""
    def __init__(self, message, payload=None):
        super().__init__(message, 404, payload)


class ConflictError(APIError):
    """资源冲突错误"""
    def __init__(self, message, payload=None):
        super().__init__(message, 409, payload)


class RateLimitError(APIError):
    """限流错误"""
    def __init__(self, message, payload=None):
        super().__init__(message, 429, payload)


class DatabaseError(APIError):
    """数据库错误"""
    def __init__(self, message, payload=None):
        super().__init__(message, 500, payload)


def handle_api_error(error):
    """处理自定义API错误"""
    logger.warning(f'API Error: {error.message}', extra={
        'status_code': error.status_code,
        'payload': error.payload
    })
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def handle_http_exception(error):
    """处理HTTP异常"""
    logger.warning(f'HTTP Exception: {error.code} - {error.description}')
    response = jsonify({
        'status': 'error',
        'message': error.description,
        'status_code': error.code,
        'timestamp': datetime.now().isoformat()
    })
    response.status_code = error.code
    return response


def handle_sqlalchemy_error(error):
    """处理SQLAlchemy数据库错误"""
    logger.error(f'Database Error: {str(error)}', exc_info=True)

    # 处理完整性约束错误
    if isinstance(error, IntegrityError):
        if 'FOREIGN KEY constraint failed' in str(error):
            return jsonify({
                'status': 'error',
                'message': '关联数据不存在，操作失败',
                'error_type': 'foreign_key_constraint',
                'timestamp': datetime.now().isoformat()
            }), 400
        elif 'UNIQUE constraint failed' in str(error):
            return jsonify({
                'status': 'error',
                'message': '数据已存在，请勿重复添加',
                'error_type': 'unique_constraint',
                'timestamp': datetime.now().isoformat()
            }), 409
        elif 'NOT NULL constraint failed' in str(error):
            return jsonify({
                'status': 'error',
                'message': '必填字段不能为空',
                'error_type': 'not_null_constraint',
                'timestamp': datetime.now().isoformat()
            }), 400

    # 处理其他数据库错误
    return jsonify({
        'status': 'error',
        'message': '数据库操作失败，请稍后重试',
        'error_type': 'database_error',
        'timestamp': datetime.now().isoformat()
    }), 500


def handle_generic_error(error):
    """处理未捕获的通用错误"""
    logger.error(f'Unhandled Exception: {str(error)}', exc_info=True)

    # 检查是否是SQLAlchemy错误
    if isinstance(error, SQLAlchemyError):
        return handle_sqlalchemy_error(error)

    # 检查是否是HTTP异常
    if isinstance(error, HTTPException):
        return handle_http_exception(error)

    # 检查是否是自定义API错误
    if isinstance(error, APIError):
        return handle_api_error(error)

    # 其他未知错误
    error_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{id(error)}"
    logger.error(f'Error ID: {error_id}, Traceback:\n{traceback.format_exc()}')

    # 开发环境返回详细错误信息
    import os
    if os.getenv('FLASK_ENV') == 'development':
        return jsonify({
            'status': 'error',
            'message': str(error),
            'error_type': 'internal_error',
            'error_id': error_id,
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }), 500

    # 生产环境只返回友好提示
    return jsonify({
        'status': 'error',
        'message': '服务器内部错误，请联系管理员',
        'error_type': 'internal_error',
        'error_id': error_id,
        'timestamp': datetime.now().isoformat()
    }), 500


def register_error_handlers(app):
    """注册所有错误处理器"""
    # 注册自定义API错误处理器
    app.register_error_handler(APIError, handle_api_error)

    # 注册HTTP异常处理器
    app.register_error_handler(HTTPException, handle_http_exception)

    # 注册SQLAlchemy错误处理器
    app.register_error_handler(SQLAlchemyError, handle_sqlalchemy_error)

    # 注册通用错误处理器
    app.register_error_handler(Exception, handle_generic_error)

    logger.info('错误处理器注册完成')
