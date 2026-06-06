# 日志配置工具
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime


def _resolve_log_dir(app):
    configured = app.config.get('LOG_DIR')
    if configured:
        return os.fspath(configured)
    if app.config.get('TESTING') and app.config.get('RUNTIME_DIR'):
        return os.path.join(os.fspath(app.config['RUNTIME_DIR']), 'logs')
    return os.path.join(os.path.dirname(__file__), '../../logs')


def _close_managed_handlers(logger):
    for handler in list(logger.handlers):
        if not getattr(handler, '_uav_dashboard_managed', False):
            continue
        logger.removeHandler(handler)
        handler.close()


def _build_file_handler(path, level, formatter, max_bytes, backup_count):
    handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler._uav_dashboard_managed = True
    return handler


def setup_logger(app):
    """
    配置应用日志系统

    Args:
        app: Flask应用实例
    """
    # 创建日志目录。测试环境使用各自 runtime，避免 Windows 下共享 app.log 文件锁。
    log_dir = _resolve_log_dir(app)
    os.makedirs(log_dir, exist_ok=True)
    _close_managed_handlers(app.logger)

    # 日志格式
    log_format = '[%(asctime)s] %(levelname)s in %(module)s [%(pathname)s:%(lineno)d]: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    log_level = getattr(logging, str(app.config.get('LOG_LEVEL', 'INFO')).upper(), logging.INFO)
    max_bytes = int(app.config.get('LOG_MAX_BYTES', 10 * 1024 * 1024))
    backup_count = int(app.config.get('LOG_BACKUP_COUNT', 10))
    formatter = logging.Formatter(log_format, date_format)

    # 配置应用日志
    app_handler = _build_file_handler(
        os.path.join(log_dir, 'app.log'),
        log_level,
        formatter,
        max_bytes,
        backup_count,
    )

    # 配置错误日志
    error_handler = _build_file_handler(
        os.path.join(log_dir, 'error.log'),
        logging.ERROR,
        formatter,
        max_bytes,
        backup_count,
    )

    # 配置访问日志
    access_handler = _build_file_handler(
        os.path.join(log_dir, 'access.log'),
        log_level,
        formatter,
        max_bytes,
        backup_count,
    )

    # 添加处理器
    app.logger.addHandler(app_handler)
    app.logger.addHandler(error_handler)

    # 如果不是开发环境，不输出到控制台
    if not app.debug:
        app.logger.setLevel(log_level)
    else:
        # 开发环境同时输出到控制台
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        stream_handler._uav_dashboard_managed = True
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
