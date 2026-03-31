# 增强的错误日志记录系统
import logging
import os
import json
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from functools import wraps
from flask import request, g
import time

class ErrorLogger:
    """错误日志记录器"""

    def __init__(self, app=None):
        self.app = app
        self.error_log_dir = None
        self.audit_log_dir = None

        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """初始化错误日志系统"""
        # 创建日志目录
        log_dir = os.path.join(os.path.dirname(__file__), '../../logs')
        self.error_log_dir = os.path.join(log_dir, 'errors')
        self.audit_log_dir = os.path.join(log_dir, 'audit')

        for dir_path in [self.error_log_dir, self.audit_log_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # 配置错误日志处理器
        self.setup_error_handlers(app)

        # 配置审计日志处理器
        self.setup_audit_logger(app)

    def setup_error_handlers(self, app):
        """配置错误日志处理器"""
        # 错误日志格式
        error_format = {
            'timestamp': '%(asctime)s',
            'level': '%(levelname)s',
            'module': '%(module)s',
            'function': '%(funcName)s',
            'line': '%(lineno)d',
            'message': '%(message)s',
            'request_id': '%(request_id)s',
            'user_id': '%(user_id)s',
            'ip_address': '%(ip_address)s',
            'user_agent': '%(user_agent)s',
            'request_path': '%(request_path)s',
            'request_method': '%(request_method)s',
            'stack_trace': '%(stack_trace)s'
        }

        # 创建错误日志处理器
        error_handler = RotatingFileHandler(
            os.path.join(self.error_log_dir, 'error.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=20,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)

        # 自定义格式化器
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {}
                for key, value in error_format.items():
                    try:
                        if key == 'stack_trace' and hasattr(record, 'stack_trace'):
                            log_data[key] = record.stack_trace
                        elif key in ['request_id', 'user_id', 'ip_address', 'user_agent',
                                    'request_path', 'request_method']:
                            log_data[key] = getattr(record, key, 'N/A')
                        else:
                            log_data[key] = self.formatTime(record, self.datefmt) if key == 'timestamp' else value
                            if key != 'timestamp':
                                log_data[key] = getattr(record, value.strip('%'), 'N/A')
                    except:
                        log_data[key] = 'N/A'

                return json.dumps(log_data, ensure_ascii=False)

        error_handler.setFormatter(JSONFormatter())
        app.logger.addHandler(error_handler)

    def setup_audit_logger(self, app):
        """配置审计日志处理器"""
        # 审计日志格式
        audit_format = '%(asctime)s | %(levelname)s | %(user_id)s | %(action)s | %(resource)s | %(details)s'

        # 创建审计日志处理器
        audit_handler = RotatingFileHandler(
            os.path.join(self.audit_log_dir, 'audit.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=20,
            encoding='utf-8'
        )
        audit_handler.setLevel(logging.INFO)

        class AuditFormatter(logging.Formatter):
            def format(self, record):
                record.user_id = getattr(record, 'user_id', 'anonymous')
                record.action = getattr(record, 'action', 'unknown')
                record.resource = getattr(record, 'resource', 'unknown')
                record.details = getattr(record, 'details', '{}')
                return super().format(record)

        audit_handler.setFormatter(AuditFormatter(audit_format, '%Y-%m-%d %H:%M:%S'))

        # 创建审计日志器
        self.audit_logger = logging.getLogger('audit')
        self.audit_logger.addHandler(audit_handler)
        self.audit_logger.setLevel(logging.INFO)

    def log_error(self, error, request_info=None, user_info=None):
        """
        记录错误日志

        Args:
            error: 异常对象
            request_info: 请求信息
            user_info: 用户信息
        """
        try:
            # 获取请求信息
            req_info = request_info or {}
            if request:
                req_info.update({
                    'path': request.path,
                    'method': request.method,
                    'ip': request.remote_addr,
                    'user_agent': request.headers.get('User-Agent', 'Unknown')
                })

            # 获取用户信息
            usr_info = user_info or {}
            if hasattr(g, 'user_id'):
                usr_info['user_id'] = g.user_id

            # 记录错误
            logger = logging.getLogger(__name__)
            logger.error(
                str(error),
                extra={
                    'request_id': req_info.get('request_id', 'N/A'),
                    'user_id': usr_info.get('user_id', 'anonymous'),
                    'ip_address': req_info.get('ip', 'N/A'),
                    'user_agent': req_info.get('user_agent', 'N/A'),
                    'request_path': req_info.get('path', 'N/A'),
                    'request_method': req_info.get('method', 'N/A'),
                    'stack_trace': traceback.format_exc()
                }
            )

        except Exception as e:
            print(f"记录错误日志失败: {str(e)}")

    def log_audit(self, user_id, action, resource, details=None):
        """
        记录审计日志

        Args:
            user_id: 用户ID
            action: 操作类型
            resource: 资源类型
            details: 详细信息
        """
        try:
            self.audit_logger.info(
                f'{action} on {resource}',
                extra={
                    'user_id': user_id,
                    'action': action,
                    'resource': resource,
                    'details': json.dumps(details or {}, ensure_ascii=False)
                }
            )
        except Exception as e:
            print(f"记录审计日志失败: {str(e)}")


def log_errors(f):
    """
    错误日志装饰器

    用于自动记录函数执行中的错误
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            # 获取请求信息
            request_info = {}
            if request:
                request_info = {
                    'request_id': getattr(g, 'request_id', 'N/A'),
                    'path': request.path,
                    'method': request.method,
                    'ip': request.remote_addr,
                    'user_agent': request.headers.get('User-Agent', 'Unknown')
                }

            # 获取用户信息
            user_info = {}
            if hasattr(g, 'user_id'):
                user_info['user_id'] = g.user_id

            # 记录错误
            error_logger = ErrorLogger()
            error_logger.log_error(e, request_info, user_info)

            # 重新抛出异常
            raise

    return decorated_function


def log_performance(f):
    """
    性能日志装饰器

    用于记录函数执行时间
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            execution_time = time.time() - start_time

            # 记录性能日志
            logger = logging.getLogger('performance')
            logger.info(
                f'{f.__name__} executed in {execution_time:.2f}s',
                extra={
                    'function': f.__name__,
                    'execution_time': execution_time,
                    'request_id': getattr(g, 'request_id', 'N/A')
                }
            )

            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger = logging.getLogger('performance')
            logger.error(
                f'{f.__name__} failed after {execution_time:.2f}s: {str(e)}',
                extra={
                    'function': f.__name__,
                    'execution_time': execution_time,
                    'error': str(e),
                    'request_id': getattr(g, 'request_id', 'N/A')
                }
            )
            raise

    return decorated_function


# 全局错误日志记录器实例
error_logger = ErrorLogger()


def init_error_logger(app):
    """
    初始化错误日志记录器

    Args:
        app: Flask应用实例
    """
    error_logger.init_app(app)
    return error_logger
