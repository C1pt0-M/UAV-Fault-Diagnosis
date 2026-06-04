from flask import current_app

from uav_dashboard_api import db
from uav_dashboard_api.models.user import User
from uav_dashboard_api.routes.auth import hash_password
from uav_dashboard_api.utils.logger import get_logger

logger = get_logger(__name__)


def initialize_if_needed():
    """按显式配置执行管理员账号自动初始化。"""
    try:
        if not current_app.config.get('ALLOW_AUTO_BOOTSTRAP_ADMIN', False):
            logger.info('已禁用管理员自动创建，跳过自动初始化')
            return False
        if User.query.count() > 0:
            logger.info('数据库已初始化，跳过管理员自动创建')
            return False

        username = current_app.config.get('BOOTSTRAP_ADMIN_USERNAME', '')
        password = current_app.config.get('BOOTSTRAP_ADMIN_PASSWORD', '')
        real_name = current_app.config.get('BOOTSTRAP_ADMIN_REAL_NAME', '')
        email = current_app.config.get('BOOTSTRAP_ADMIN_EMAIL', '')
        department = current_app.config.get('BOOTSTRAP_ADMIN_DEPARTMENT', '平台组')
        if not all([username, password, real_name, email]):
            logger.warning('已开启管理员自动创建，但缺少显式管理员凭据配置，跳过自动初始化')
            return False

        admin = User(
            username=username,
            password=hash_password(password),
            real_name=real_name,
            email=email,
            phone='',
            department=department,
            role='admin',
            status='active',
        )
        db.session.add(admin)
        db.session.commit()
        logger.info(f'已通过显式配置创建管理员账号: {username}')
        return True
    except Exception as exc:
        db.session.rollback()
        logger.error(f'自动初始化失败: {exc}')
        return False
