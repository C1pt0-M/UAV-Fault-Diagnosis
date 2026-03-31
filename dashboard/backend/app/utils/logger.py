# 日志配置工具
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(app):
    """
    配置应用日志系统

    Args:
        app: Flask应用实例
    """
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(__file__), '../../logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 日志格式
    log_format = '[%(asctime)s] %(levelname)s in %(module)s [%(pathname)s:%(lineno)d]: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 配置应用日志
    app_handler = RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=10,
        encoding='utf-8'
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(logging.Formatter(log_format, date_format))

    # 配置错误日志
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=10,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(log_format, date_format))

    # 配置访问日志
    access_handler = RotatingFileHandler(
        os.path.join(log_dir, 'access.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=10,
        encoding='utf-8'
    )
    access_handler.setLevel(logging.INFO)
    access_handler.setFormatter(logging.Formatter(log_format, date_format))

    # 添加处理器
    app.logger.addHandler(app_handler)
    app.logger.addHandler(error_handler)

    # 如果不是开发环境，不输出到控制台
    if not app.debug:
        app.logger.setLevel(logging.INFO)
    else:
        # 开发环境同时输出到控制台
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter(log_format, date_format))
        app.logger.addHandler(stream_handler)

    app.logger.info('应用启动')
    return app_handler, error_handler, access_handler


def get_logger(name):
    """
    获取指定名称的logger

    Args:
        name: logger名称

    Returns:
        logger实例
    """
    return logging.getLogger(name)


class LoggerMixin:
    """日志混入类，用于在类中方便地使用日志"""

    @property
    def logger(self):
        """获取logger实例"""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger


def log_request(response):
    """
    记录请求日志

    Args:
        response: Flask响应对象

    Returns:
        response
    """
    from flask import request
    access_logger = logging.getLogger('access')

    # 记录请求信息
    access_logger.info(
        f'{request.remote_addr} - {request.method} {request.path} - '
        f'Status: {response.status_code} - '
        f'User-Agent: {request.user_agent}'
    )

    return response


def log_exception(sender, exception, **extra):
    """
    记录异常日志

    Args:
        sender: 异常发送者
        exception: 异常对象
        **extra: 额外参数
    """
    error_logger = logging.getLogger('error')
    error_logger.error(
        f'Exception occurred: {type(exception).__name__}: {str(exception)}',
        exc_info=True
    )
