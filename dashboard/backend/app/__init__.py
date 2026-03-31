# Flask应用初始化文件
import signal

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine

from config import Config

db = SQLAlchemy()


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


def create_app(config_class=Config):
    """应用工厂函数"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    from app.utils.persistence_guard import audit_runtime_persistence, backup_sqlite_database

    _configure_sqlite_engine(app)
    db.init_app(app)

    from app.utils.logger import log_request, setup_logger
    setup_logger(app)

    from app.utils.error_logger import init_error_logger
    init_error_logger(app)

    from app.utils.rate_limiter import init_circuit_breakers, init_rate_limiters
    init_rate_limiters()
    init_circuit_breakers()

    from app.utils.error_handler import register_error_handlers
    register_error_handlers(app)

    CORS(
        app,
        resources={
            r"/*": {
                "origins": [
                    "http://localhost:3000",
                    "http://127.0.0.1:3000",
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                    "http://localhost:5174",
                    "http://127.0.0.1:5174",
                ],
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
                "allow_headers": ["Content-Type", "Authorization", "X-Request-ID", "Accept", "Origin"],
                "expose_headers": ["Content-Length", "Content-Range"],
                "supports_credentials": True,
                "max_age": 3600,
            }
        },
    )

    @app.after_request
    def after_request(response):
        origin = request.headers.get('Origin')
        if origin:
            response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

    app.after_request(log_request)
    signal.signal(signal.SIGINT, lambda sig, frame: app.logger.info('应用关闭'))

    from app.routes import ai, auth, dashboard, diagnosis_reports, health, setup, users
    from app.routes import telemetry

    app.register_blueprint(auth.bp, url_prefix=f'{Config.API_PREFIX}/auth')
    app.register_blueprint(users.bp, url_prefix=f'{Config.API_PREFIX}/users')
    app.register_blueprint(setup.bp, url_prefix=f'{Config.API_PREFIX}/setup')
    app.register_blueprint(dashboard.bp, url_prefix=f'{Config.API_PREFIX}')
    app.register_blueprint(diagnosis_reports.bp, url_prefix=f'{Config.API_PREFIX}/reports/diagnosis')
    app.register_blueprint(ai.bp, url_prefix=f'{Config.API_PREFIX}/ai')
    app.register_blueprint(health.bp)
    app.register_blueprint(telemetry.bp, url_prefix=f'{Config.API_PREFIX}/telemetry')

    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'service': 'uav-fault-dashboard-backend',
            'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        })

    @app.route('/')
    def index():
        return jsonify({
            'message': '无人机故障诊断系统后端API',
            'version': '2.0.0',
            'status': 'running',
            'api_prefix': Config.API_PREFIX,
            'endpoints': {
                'health': '/health',
                'auth': '/api/auth/*',
                'setup': '/api/setup/*',
                'overview': '/api/overview',
                'models': '/api/models',
                'assets': '/api/assets',
                'runs': '/api/runs',
                'analysis': '/api/analysis/*',
                'diagnosis_reports': '/api/reports/diagnosis/*',
                'ai': '/api/ai/*',
                'telemetry': '/api/telemetry/*',
            },
        })

    backup_path = backup_sqlite_database(app.config)
    if backup_path is not None:
        app.logger.info(f'SQLite 启动备份已创建: {backup_path}')

    with app.app_context():
        db.create_all()
        from app.utils.auto_init import initialize_if_needed
        initialize_if_needed()
        persistence_snapshot = audit_runtime_persistence(app.config)
        for warning in persistence_snapshot.get('warnings', []):
            app.logger.warning(warning)
        # 初始化遥测管理器
        from app.services.telemetry_manager import telemetry_manager
        telemetry_manager.init_app(app)
        from app.services.ai_assistant import ai_assistant_service
        ai_assistant_service.init_app(app)

    app.logger.info('应用初始化完成')
    return app
