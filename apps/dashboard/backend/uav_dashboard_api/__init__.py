"""Flask application factory for the UAV fault diagnosis dashboard API."""

from flask import Flask, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine

from uav_dashboard_api.config import resolve_config_class

db = SQLAlchemy()


_INSECURE_SECRET_VALUES = {
    '',
    'change-me',
    'change-me-too',
    'dev-secret-key-change-in-production',
}


def _validate_runtime_config(app):
    if app.config.get('APP_ENV') != 'production':
        return

    secret_key = str(app.config.get('SECRET_KEY') or '').strip()
    jwt_secret_key = str(app.config.get('JWT_SECRET_KEY') or '').strip()
    if secret_key in _INSECURE_SECRET_VALUES or len(secret_key) < 32:
        raise ValueError('production SECRET_KEY 必须设置为至少 32 字符的非默认值')
    if jwt_secret_key in _INSECURE_SECRET_VALUES or len(jwt_secret_key) < 32:
        raise ValueError('production JWT_SECRET_KEY 必须设置为至少 32 字符的非默认值')
    if secret_key == jwt_secret_key:
        raise ValueError('production SECRET_KEY 与 JWT_SECRET_KEY 必须使用不同值')


def _configure_sqlite_engine(app):
    if app.config.get('DB_TYPE') != 'sqlite':
        return

    @event.listens_for(Engine, 'connect')
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f"PRAGMA busy_timeout = {int(app.config.get('SQLITE_BUSY_TIMEOUT', 30)) * 1000}")
            if app.config.get('SQLITE_ENABLE_WAL', True):
                cursor.execute('PRAGMA journal_mode=WAL')
            synchronous = str(app.config.get('SQLITE_SYNCHRONOUS', 'NORMAL')).upper()
            cursor.execute(f'PRAGMA synchronous={synchronous}')
        finally:
            cursor.close()


def create_app(config_class=None):
    """应用工厂函数"""
    app = Flask(__name__)
    selected_config = (
        resolve_config_class(config_class)
        if isinstance(config_class, str)
        else config_class or resolve_config_class()
    )
    app.config.from_object(selected_config)
    _validate_runtime_config(app)

    from uav_dashboard_api.utils.persistence_guard import audit_runtime_persistence, backup_sqlite_database

    _configure_sqlite_engine(app)
    db.init_app(app)

    from uav_dashboard_api.utils.logger import log_request, setup_logger
    setup_logger(app)

    from uav_dashboard_api.utils.error_logger import init_error_logger
    init_error_logger(app)

    from uav_dashboard_api.utils.error_handler import register_error_handlers
    register_error_handlers(app)

    cors_origins = list(app.config.get('CORS_ORIGIN_LIST') or [])
    CORS(
        app,
        resources={
            r"/*": {
                "origins": cors_origins or [],
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
                "allow_headers": [
                    "Content-Type",
                    "Authorization",
                    "X-Request-ID",
                    "Accept",
                    "Origin",
                ],
                "expose_headers": ["Content-Length", "Content-Range"],
                "supports_credentials": True,
                "max_age": 3600,
            }
        },
    )

    app.after_request(log_request)
    from uav_dashboard_api.routes import ai, auth, dashboard, diagnosis_reports, health, settings, setup, users
    from uav_dashboard_api.routes import telemetry

    api_prefix = app.config['API_PREFIX']

    app.register_blueprint(auth.bp, url_prefix=f'{api_prefix}/auth')
    app.register_blueprint(users.bp, url_prefix=f'{api_prefix}/users')
    if app.config.get('ENABLE_SETUP_API', False):
        app.register_blueprint(setup.bp, url_prefix=f'{api_prefix}/setup')
    app.register_blueprint(settings.bp, url_prefix=f'{api_prefix}/settings')
    app.register_blueprint(dashboard.bp, url_prefix=f'{api_prefix}')
    app.register_blueprint(
        diagnosis_reports.bp, url_prefix=f'{api_prefix}/reports/diagnosis'
    )
    if app.config.get('ENABLE_AI', True):
        app.register_blueprint(ai.bp, url_prefix=f'{api_prefix}/ai')
    app.register_blueprint(health.bp)
    if app.config.get('ENABLE_TELEMETRY', True):
        app.register_blueprint(telemetry.bp, url_prefix=f'{api_prefix}/telemetry')

    @app.route('/')
    def index():
        payload = {
            'message': '基于数据驱动的无人机故障智能诊断系统后端API',
            'version': '2.0.0',
            'status': 'running',
            'api_prefix': api_prefix,
            'endpoints': {
                'health': '/health',
                'auth': '/api/auth/*',
                'overview': '/api/overview',
                'models': '/api/models',
                'assets': '/api/assets',
                'runs': '/api/runs',
                'analysis': '/api/analysis/*',
                'diagnosis_reports': '/api/reports/diagnosis/*',
            },
        }
        if app.config.get('ENABLE_SETUP_API', False):
            payload['endpoints']['setup'] = '/api/setup/*'
        if app.config.get('ENABLE_AI', True):
            payload['endpoints']['ai'] = '/api/ai/*'
        if app.config.get('ENABLE_TELEMETRY', True):
            payload['endpoints']['telemetry'] = '/api/telemetry/*'
        return jsonify(payload)

    backup_path = backup_sqlite_database(app.config)
    if backup_path is not None:
        app.logger.info(f'SQLite 启动备份已创建: {backup_path}')

    with app.app_context():
        db.create_all()
        from uav_dashboard_api.utils.auto_init import initialize_if_needed
        initialize_if_needed()
        persistence_snapshot = audit_runtime_persistence(app.config)
        for warning in persistence_snapshot.get('warnings', []):
            app.logger.warning(warning)
        if app.config.get('ENABLE_TELEMETRY', True):
            from uav_dashboard_api.services.telemetry_manager import telemetry_manager

            telemetry_manager.init_app(app)
        if app.config.get('ENABLE_AI', True):
            from uav_dashboard_api.services.ai_assistant import ai_assistant_service

            ai_assistant_service.init_app(app)

    app.logger.info('应用初始化完成')
    return app
