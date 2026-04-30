# 数据库配置文件
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

# 获取项目目录
BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BASE_DIR.parent
PROJECT_ROOT = DASHBOARD_DIR.parent.resolve()

# 加载环境变量
load_dotenv(BASE_DIR / '.env')


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _get_env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _get_env_list(name: str, default: str | None = None) -> tuple[str, ...]:
    raw_value = _get_env(name, default)
    if raw_value in (None, ''):
        return ()
    return tuple(item.strip() for item in str(raw_value).split(',') if item.strip())


def _resolve_path(raw_value: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _resolve_optional_path(
    name: str,
    default: str | None,
    *,
    base_dir: Path,
) -> str:
    raw_value = _get_env(name, default)
    if raw_value in (None, ''):
        return ''
    return str(_resolve_path(raw_value, base_dir=base_dir))


def _default_python_bin() -> str:
    candidates = [sys.executable, shutil.which('python3'), shutil.which('python')]
    for candidate in candidates:
        candidate_str = str(candidate or '').strip()
        if candidate_str:
            return candidate_str
    return 'python'


RESOLVED_REPO_ROOT = _resolve_path(
    _get_env('REPO_ROOT', str(PROJECT_ROOT)),
    base_dir=PROJECT_ROOT,
)


class Config:
    """应用配置类"""

    # Flask配置
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = _get_env_bool('FLASK_DEBUG', False)
    TESTING = False
    APP_ENV = _get_env('APP_ENV', 'development')

    # API配置
    API_PREFIX = os.getenv('API_PREFIX', '/api')

    # 运行目录
    RUNTIME_DIR = _resolve_path(
        _get_env('RUNTIME_DIR', str(BASE_DIR / 'runtime')),
        base_dir=BASE_DIR,
    )
    UPLOAD_DIR = _resolve_path(
        _get_env('UPLOAD_DIR', str(RUNTIME_DIR / 'uploads')),
        base_dir=BASE_DIR,
    )
    DERIVED_DIR = _resolve_path(
        _get_env('DERIVED_DIR', str(RUNTIME_DIR / 'derived')),
        base_dir=BASE_DIR,
    )
    RUNS_DIR = _resolve_path(
        _get_env('RUNS_DIR', str(RUNTIME_DIR / 'runs')),
        base_dir=BASE_DIR,
    )
    REPORTS_DIR = _resolve_path(
        _get_env('REPORTS_DIR', str(RUNTIME_DIR / 'reports')),
        base_dir=BASE_DIR,
    )
    AI_RUNTIME_DIR = _resolve_path(
        _get_env('AI_RUNTIME_DIR', str(RUNTIME_DIR / 'ai')),
        base_dir=BASE_DIR,
    )
    AI_SETTINGS_PATH = _resolve_path(
        _get_env('AI_SETTINGS_PATH', str(RUNTIME_DIR / 'ai_settings.json')),
        base_dir=BASE_DIR,
    )
    AI_KNOWLEDGE_DIR = _resolve_path(
        _get_env('AI_KNOWLEDGE_DIR', 'knowledge_base'),
        base_dir=RESOLVED_REPO_ROOT,
    )

    for _dir in (RUNTIME_DIR, UPLOAD_DIR, DERIVED_DIR, RUNS_DIR, REPORTS_DIR, AI_RUNTIME_DIR, AI_KNOWLEDGE_DIR):
        _dir.mkdir(parents=True, exist_ok=True)

    # 数据库配置（默认SQLite）
    DB_TYPE = os.getenv('DB_TYPE', 'sqlite')
    if DB_TYPE == 'sqlite':
        db_path = _resolve_path(
            _get_env('DB_PATH', str(RUNTIME_DIR / 'dashboard.db')),
            base_dir=BASE_DIR,
        )
        db_path.parent.mkdir(parents=True, exist_ok=True)
        SQLALCHEMY_DATABASE_URI = f'sqlite:///{db_path}'
        SQLALCHEMY_ENGINE_OPTIONS = {
            'connect_args': {
                'check_same_thread': False,
                'timeout': int(os.getenv('SQLITE_BUSY_TIMEOUT', '30')),
            },
        }
    else:
        DB_HOST = os.getenv('DB_HOST', 'localhost')
        DB_PORT = os.getenv('DB_PORT', '3306')
        DB_USER = os.getenv('DB_USER', 'root')
        DB_PASSWORD = os.getenv('DB_PASSWORD', '123456')
        DB_NAME = os.getenv('DB_NAME', 'drone_management')
        SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
        SQLALCHEMY_ENGINE_OPTIONS = {
            'pool_size': int(os.getenv('DB_POOL_SIZE', '10')),
            'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
            'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '3600')),
            'pool_pre_ping': True,
            'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '20')),
        }

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = DEBUG
    SQLITE_BUSY_TIMEOUT = int(os.getenv('SQLITE_BUSY_TIMEOUT', '30'))
    SQLITE_LOCK_RETRIES = int(os.getenv('SQLITE_LOCK_RETRIES', '2'))
    SQLITE_LOCK_RETRY_DELAY = float(os.getenv('SQLITE_LOCK_RETRY_DELAY', '0.25'))
    SQLITE_ENABLE_WAL = os.getenv('SQLITE_ENABLE_WAL', 'True') == 'True'
    SQLITE_SYNCHRONOUS = os.getenv('SQLITE_SYNCHRONOUS', 'NORMAL')

    # 限流配置
    RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'True') == 'True'
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
    RATE_LIMIT_PER_HOUR = int(os.getenv('RATE_LIMIT_PER_HOUR', '1000'))
    RATE_LIMIT_STORAGE_URL = os.getenv('RATE_LIMIT_STORAGE_URL', 'memory://')

    # 熔断器配置
    CIRCUIT_BREAKER_ENABLED = os.getenv('CIRCUIT_BREAKER_ENABLED', 'True') == 'True'
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv('CIRCUIT_BREAKER_FAILURE_THRESHOLD', '5'))
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT = int(os.getenv('CIRCUIT_BREAKER_RECOVERY_TIMEOUT', '60'))
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION = None

    # JWT配置
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', SECRET_KEY)
    JWT_EXPIRE_HOURS = int(os.getenv('JWT_EXPIRE_HOURS', '24'))

    # 日志配置
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', '10485760'))
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '10'))

    # 安全配置
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', str(128 * 1024 * 1024)))

    # CORS配置
    CORS_ORIGINS = _get_env(
        'CORS_ORIGINS',
        'http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174',
    )
    CORS_ORIGIN_LIST = _get_env_list('CORS_ORIGINS', CORS_ORIGINS)

    # 任务执行配置
    TASK_MAX_WORKERS = int(os.getenv('TASK_MAX_WORKERS', '2'))
    ALLOW_AUTO_BOOTSTRAP_ADMIN = _get_env_bool('ALLOW_AUTO_BOOTSTRAP_ADMIN', DEBUG)
    ENABLE_SETUP_API = _get_env_bool('ENABLE_SETUP_API', DEBUG)
    ENABLE_PUBLIC_REGISTER = _get_env_bool('ENABLE_PUBLIC_REGISTER', DEBUG)
    ENABLE_TELEMETRY = _get_env_bool('ENABLE_TELEMETRY', True)
    ENABLE_BLIND_TEST = _get_env_bool('ENABLE_BLIND_TEST', False)
    TELEMETRY_ALLOWED_BACKENDS = _get_env_list('TELEMETRY_ALLOWED_BACKENDS', 'replay')
    ENABLE_AI = _get_env_bool('ENABLE_AI', True)
    EXPOSE_SERVER_PATHS = _get_env_bool('EXPOSE_SERVER_PATHS', DEBUG)

    # 桥接配置
    REPO_ROOT = str(RESOLVED_REPO_ROOT)
    _REPO_ROOT_PATH = Path(REPO_ROOT)
    MODEL_ROOT = _resolve_optional_path(
        'MODEL_ROOT',
        'models',
        base_dir=_REPO_ROOT_PATH,
    )
    ML_PYTHON_BIN = _get_env('ML_PYTHON_BIN', _default_python_bin())
    TELEMETRY_PYTHON_BIN = _get_env('TELEMETRY_PYTHON_BIN', ML_PYTHON_BIN)
    PX4_ROOT = _resolve_optional_path(
        'PX4_ROOT',
        str(Path.home() / 'PX4-Autopilot'),
        base_dir=_REPO_ROOT_PATH,
    )
    PX4_SYS_AUTOSTART = os.getenv('PX4_SYS_AUTOSTART', '4001')
    PX4_SIM_TARGET = os.getenv('PX4_SIM_TARGET', 'gz_x500_lawn')
    PX4_READY_HOST = os.getenv('PX4_READY_HOST', '127.0.0.1')
    PX4_READY_PORT = int(os.getenv('PX4_READY_PORT', '18570'))
    TELEMETRY_MAVSDK_ADDRESS = os.getenv('TELEMETRY_MAVSDK_ADDRESS', 'udpin://0.0.0.0:14550')
    TELEMETRY_MAVSDK_GRPC_PORT = int(os.getenv('TELEMETRY_MAVSDK_GRPC_PORT', '50052'))
    TELEMETRY_PYMAV_ADDRESS = os.getenv('TELEMETRY_PYMAV_ADDRESS', 'udpin:0.0.0.0:14030')
    TELEMETRY_FORWARD_HOST = os.getenv('TELEMETRY_FORWARD_HOST', '127.0.0.1')
    TELEMETRY_FORWARD_PORT = int(os.getenv('TELEMETRY_FORWARD_PORT', '15550'))
    TELEMETRY_EMIT_RATE_HZ = int(os.getenv('TELEMETRY_EMIT_RATE_HZ', '50'))
    TELEMETRY_ENABLE_PYMAV = _get_env_bool('TELEMETRY_ENABLE_PYMAV', True)
    RFLY_PX4MAVCTRL_PATH = _resolve_optional_path(
        'RFLY_PX4MAVCTRL_PATH',
        '',
        base_dir=_REPO_ROOT_PATH,
    )
    RFLY_BRIDGE_SCRIPT = _resolve_optional_path(
        'RFLY_BRIDGE_SCRIPT',
        '',
        base_dir=_REPO_ROOT_PATH,
    )
    RFLY_FORWARD_HOST = os.getenv('RFLY_FORWARD_HOST', '127.0.0.1')
    RFLY_FORWARD_PORT = int(os.getenv('RFLY_FORWARD_PORT', '15550'))
    RFLY_EMIT_RATE_HZ = int(os.getenv('RFLY_EMIT_RATE_HZ', '50'))
    RFLY_RPM_SCALE = float(os.getenv('RFLY_RPM_SCALE', '10000.0'))
    RFLY_TEST_ID = int(os.getenv('RFLY_TEST_ID', '1'))
    RFLY_UDP_PORT = int(os.getenv('RFLY_UDP_PORT', '20100'))
    RFLY_UDP_MODE = int(os.getenv('RFLY_UDP_MODE', '2'))
    RFLY_MAV_IP = os.getenv('RFLY_MAV_IP', '127.0.0.1')
    RFLY_PYMAV_ADDRESS = os.getenv('RFLY_PYMAV_ADDRESS', 'udpin:0.0.0.0:14030')
    RFLY_ENABLE_PYMAV = _get_env_bool('RFLY_ENABLE_PYMAV', False)
    RFLY_RECEIVER_TIMEOUT_S = float(os.getenv('RFLY_RECEIVER_TIMEOUT_S', '60.0'))
    FAULT_DATASET_ROOT = _resolve_optional_path(
        'FAULT_DATASET_ROOT',
        '',
        base_dir=_REPO_ROOT_PATH,
    )
    ULG_PREPARE_SCRIPT = _resolve_optional_path(
        'ULG_PREPARE_SCRIPT',
        '',
        base_dir=_REPO_ROOT_PATH,
    )
    OFFLINE_NORM_STATS = _resolve_optional_path(
        'OFFLINE_NORM_STATS',
        'models/transfer3_cnnlstm_normalproto/norm_stats.json',
        base_dir=_REPO_ROOT_PATH,
    )
    DEFAULT_OFFLINE_MODEL = os.getenv('DEFAULT_OFFLINE_MODEL', 'transfer3_cnnlstm_normalproto')
    DEFAULT_ONLINE_MODEL = os.getenv('DEFAULT_ONLINE_MODEL', 'online_detector_kg_mstca_rfly')
    ONLINE_MODEL_PRIORITY = tuple(
        item.strip()
        for item in os.getenv(
            'ONLINE_MODEL_PRIORITY',
            'online_detector_kg_mstca_rfly,online_detector_mstca_rfly,online_detector_cnn1d_rfly,online_detector_CNN,online_detector_final',
        ).split(',')
        if item.strip()
    )
    _severity_rules_raw = Path(
        _get_env(
            'SEVERITY_RULES_PATH',
            str(BASE_DIR / 'app' / 'configs' / 'severity_rules_v1.yaml'),
        )
    )
    SEVERITY_RULES_PATH = (
        _severity_rules_raw if _severity_rules_raw.is_absolute() else (BASE_DIR / _severity_rules_raw).resolve()
    )
    MODEL_WHITELIST = tuple(
        item.strip()
        for item in os.getenv('MODEL_WHITELIST', '').split(',')
        if item.strip()
    )
    AI_ENABLE_KNOWLEDGE_BASE = os.getenv('AI_ENABLE_KNOWLEDGE_BASE', 'True') == 'True'
    AI_PROVIDER = os.getenv('AI_PROVIDER', 'openai_compatible')
    AI_BASE_URL = os.getenv('AI_BASE_URL', 'https://api.siliconflow.cn/v1')
    AI_API_KEY = os.getenv('AI_API_KEY', '')
    AI_MODEL = os.getenv('AI_MODEL', 'deepseek-ai/DeepSeek-V3')
    AI_EMBEDDING_BASE_URL = os.getenv('AI_EMBEDDING_BASE_URL', AI_BASE_URL)
    AI_EMBEDDING_API_KEY = os.getenv('AI_EMBEDDING_API_KEY', AI_API_KEY)
    AI_EMBEDDING_MODEL = os.getenv('AI_EMBEDDING_MODEL', '')
    AI_HTTP_TIMEOUT = int(os.getenv('AI_HTTP_TIMEOUT', '45'))
    AI_CHAT_HARD_TIMEOUT = float(os.getenv('AI_CHAT_HARD_TIMEOUT', '60'))
    AI_EMBED_BATCH_SIZE = int(os.getenv('AI_EMBED_BATCH_SIZE', '16'))
    AI_RAG_MODE = os.getenv('AI_RAG_MODE', 'bm25').strip().lower()
    AI_TOP_K = int(os.getenv('AI_TOP_K', '5'))
    AI_MAX_CONTEXT_CHUNKS = int(os.getenv('AI_MAX_CONTEXT_CHUNKS', '6'))
    AI_PROJECT_DOC_PATHS = tuple(
        item.strip()
        for item in os.getenv('AI_PROJECT_DOC_PATHS', '').split(',')
        if item.strip()
    )


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    ALLOW_AUTO_BOOTSTRAP_ADMIN = True
    ENABLE_SETUP_API = True
    ENABLE_PUBLIC_REGISTER = True
    ENABLE_BLIND_TEST = True
    TELEMETRY_ALLOWED_BACKENDS = ('replay', 'rfly')


class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    SQLALCHEMY_ECHO = False
    ALLOW_AUTO_BOOTSTRAP_ADMIN = False
    ENABLE_SETUP_API = False
    ENABLE_PUBLIC_REGISTER = False
    ENABLE_BLIND_TEST = False
    TELEMETRY_ALLOWED_BACKENDS = ('replay',)


class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_ENGINE_OPTIONS = {}
    SQLALCHEMY_ECHO = False


# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
