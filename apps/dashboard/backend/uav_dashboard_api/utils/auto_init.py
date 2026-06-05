from flask import current_app

from uav_dashboard_api import db
from uav_dashboard_api.models.user import User
from uav_dashboard_api.routes.auth import hash_password
from uav_dashboard_api.utils.logger import get_logger

logger = get_logger(__name__)


def initialize_if_needed():
    """首次启动自动创建默认管理员账号。"""
    try:
        if not current_app.config.get('ALLOW_AUTO_BOOTSTRAP_ADMIN', False):
            logger.info('已禁用默认管理员自动创建，跳过自动初始化')
            return False
        if User.query.count() > 0:
            logger.info('数据库已初始化，跳过默认账号创建')
            return False

        admin = User(
            username='admin',
            password=hash_password('admin123'),
            real_name='系统管理员',
            email='admin@uav.local',
            phone='',
            department='平台组',
            role='admin',
            status='active',
        )
        db.session.add(admin)
        db.session.commit()
        logger.info('已创建默认管理员: admin / admin123')
        return True
    except Exception as exc:
        db.session.rollback()
        logger.error(f'自动初始化失败: {exc}')
        return False
