# 限流和熔断器工具
import time
import threading
from functools import wraps
from collections import deque
from flask import request, jsonify
from app.utils.logger import get_logger
from config import Config

logger = get_logger(__name__)


class RateLimiter:
    """基于令牌桶算法的限流器"""

    def __init__(self, capacity, refill_rate):
        """
        初始化限流器

        Args:
            capacity: 桶容量（最大令牌数）
            refill_rate: 令牌补充速率（每秒补充的令牌数）
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill_time = time.time()
        self.lock = threading.Lock()

    def _refill_tokens(self):
        """补充令牌"""
        now = time.time()
        time_passed = now - self.last_refill_time
        tokens_to_add = time_passed * self.refill_rate

        with self.lock:
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill_time = now

    def acquire(self, tokens=1):
        """
        获取令牌

        Args:
            tokens: 需要获取的令牌数

        Returns:
            bool: 是否成功获取令牌
        """
        self._refill_tokens()

        with self.lock:
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def reset(self):
        """重置限流器"""
        with self.lock:
            self.tokens = self.capacity
            self.last_refill_time = time.time()


class SlidingWindowRateLimiter:
    """基于滑动窗口的限流器"""

    def __init__(self, max_requests, window_seconds):
        """
        初始化滑动窗口限流器

        Args:
            max_requests: 窗口内最大请求数
            window_seconds: 窗口时间（秒）
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.lock = threading.Lock()

    def is_allowed(self):
        """
        检查是否允许请求

        Returns:
            bool: 是否允许请求
        """
        now = time.time()

        with self.lock:
            # 移除窗口外的旧请求
            while self.requests and self.requests[0] <= now - self.window_seconds:
                self.requests.popleft()

            # 检查是否超过限制
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

    def reset(self):
        """重置限流器"""
        with self.lock:
            self.requests.clear()


class CircuitBreaker:
    """熔断器"""

    CLOSED = 'closed'      # 关闭状态：正常工作
    OPEN = 'open'          # 打开状态：熔断中
    HALF_OPEN = 'half_open'  # 半开状态：尝试恢复

    def __init__(self, failure_threshold=5, recovery_timeout=60, expected_exception=None):
        """
        初始化熔断器

        Args:
            failure_threshold: 失败阈值，超过此值后熔断
            recovery_timeout: 恢复超时时间（秒）
            expected_exception: 预期的异常类型
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = self.CLOSED
        self.lock = threading.Lock()

    def call(self, func):
        """
        装饰器：包装函数调用

        Args:
            func: 被包装的函数

        Returns:
            函数执行结果
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.is_allowed():
                raise Exception('熔断器已打开，服务暂时不可用')

            try:
                result = func(*args, **kwargs)
                self.on_success()
                return result
            except Exception as e:
                if self.expected_exception is None or isinstance(e, self.expected_exception):
                    self.on_failure()
                raise

        return wrapper

    def is_allowed(self):
        """
        检查是否允许调用

        Returns:
            bool: 是否允许调用
        """
        with self.lock:
            if self.state == self.CLOSED:
                return True
            elif self.state == self.OPEN:
                # 检查是否可以尝试恢复
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = self.HALF_OPEN
                    logger.info('熔断器进入半开状态')
                    return True
                return False
            else:  # HALF_OPEN
                return True

    def on_success(self):
        """成功回调"""
        with self.lock:
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                self.failure_count = 0
                logger.info('熔断器恢复到关闭状态')
            elif self.state == self.CLOSED:
                self.failure_count = 0

    def on_failure(self):
        """失败回调"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = self.OPEN
                logger.warning(f'熔断器打开，失败次数: {self.failure_count}')

    def reset(self):
        """重置熔断器"""
        with self.lock:
            self.failure_count = 0
            self.last_failure_time = None
            self.state = self.CLOSED
            logger.info('熔断器重置')

    def get_state(self):
        """获取当前状态"""
        return self.state


# 全局限流器实例
minute_limiter = None
hour_limiter = None

# 全局熔断器实例
database_circuit_breaker = None
external_api_circuit_breaker = None


def init_rate_limiters():
    """初始化全局限流器"""
    global minute_limiter, hour_limiter

    # 每分钟限流器（令牌桶）
    minute_limiter = RateLimiter(
        capacity=Config.RATE_LIMIT_PER_MINUTE,
        refill_rate=Config.RATE_LIMIT_PER_MINUTE / 60
    )

    # 每小时限流器（滑动窗口）
    hour_limiter = SlidingWindowRateLimiter(
        max_requests=Config.RATE_LIMIT_PER_HOUR,
        window_seconds=3600
    )

    logger.info('限流器初始化完成')


def init_circuit_breakers():
    """初始化全局熔断器"""
    global database_circuit_breaker, external_api_circuit_breaker

    # 数据库熔断器
    database_circuit_breaker = CircuitBreaker(
        failure_threshold=Config.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout=Config.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        expected_exception=Exception
    )

    # 外部API熔断器
    external_api_circuit_breaker = CircuitBreaker(
        failure_threshold=Config.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout=Config.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        expected_exception=Exception
    )

    logger.info('熔断器初始化完成')


def rate_limit(f):
    """
    限流装饰器

    Args:
        f: 被装饰的函数

    Returns:
        函数执行结果或错误响应
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not Config.RATE_LIMIT_ENABLED:
            return f(*args, **kwargs)

        # 获取客户端IP
        client_ip = request.remote_addr

        # 检查分钟限流
        if not minute_limiter.acquire():
            logger.warning(f'分钟限流触发: {client_ip}')
            return jsonify({
                'status': 'error',
                'message': '请求过于频繁，请稍后再试',
                'error_type': 'rate_limit_exceeded',
                'limit_type': 'per_minute'
            }), 429

        # 检查小时限流
        if not hour_limiter.is_allowed():
            logger.warning(f'小时限流触发: {client_ip}')
            return jsonify({
                'status': 'error',
                'message': '请求次数已达上限，请一小时后再试',
                'error_type': 'rate_limit_exceeded',
                'limit_type': 'per_hour'
            }), 429

        return f(*args, **kwargs)

    return decorated_function


def circuit_breaker(breaker_type='database'):
    """
    熔断器装饰器

    Args:
        breaker_type: 熔断器类型 ('database', 'external_api')

    Returns:
        函数执行结果或错误响应
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not Config.CIRCUIT_BREAKER_ENABLED:
                return f(*args, **kwargs)

            # 选择熔断器
            if breaker_type == 'database':
                breaker = database_circuit_breaker
            elif breaker_type == 'external_api':
                breaker = external_api_circuit_breaker
            else:
                return f(*args, **kwargs)

            # 检查是否允许调用
            if not breaker.is_allowed():
                logger.warning(f'熔断器已打开: {breaker_type}')
                return jsonify({
                    'status': 'error',
                    'message': '服务暂时不可用，请稍后再试',
                    'error_type': 'service_unavailable',
                    'breaker_state': breaker.get_state()
                }), 503

            try:
                result = f(*args, **kwargs)
                breaker.on_success()
                return result
            except Exception as e:
                breaker.on_failure()
                raise

        return decorated_function
    return decorator
