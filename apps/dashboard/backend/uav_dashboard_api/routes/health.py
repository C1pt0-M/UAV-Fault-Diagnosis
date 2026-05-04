# 系统健康检查路由
from flask import Blueprint, jsonify
from uav_dashboard_api import db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.user import User
from uav_dashboard_api.utils.logger import get_logger
from datetime import datetime
import time

bp = Blueprint('health', __name__)
logger = get_logger(__name__)

@bp.route('/health', methods=['GET'])
def health_check():
    """
    基础健康检查
    """
    return jsonify({
        'status': 'healthy',
        'service': 'drone-fault-management-backend',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'uptime': time.time()
    })

@bp.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    """
    详细健康检查，包括数据库状态
    """
    try:
        start_time = time.time()
        
        # 检查数据库连接
        db_status = check_database_connection()
        
        # 检查数据库数据
        data_status = check_database_data()
        
        # 计算响应时间
        response_time = round((time.time() - start_time) * 1000, 2)
        
        # 判断整体状态
        overall_status = 'healthy' if db_status['connected'] and data_status['initialized'] else 'unhealthy'
        
        return jsonify({
            'status': overall_status,
            'service': 'drone-fault-management-backend',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'response_time_ms': response_time,
            'database': {
                'connection': db_status,
                'data': data_status
            },
            'recommendations': get_recommendations(db_status, data_status)
        })
    except Exception as e:
        logger.error(f'健康检查失败: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@bp.route('/health/database', methods=['GET'])
def database_health_check():
    """
    数据库健康检查
    """
    try:
        start_time = time.time()
        
        # 检查数据库连接
        db_status = check_database_connection()
        
        # 检查数据库数据
        data_status = check_database_data()
        
        # 计算响应时间
        response_time = round((time.time() - start_time) * 1000, 2)
        
        return jsonify({
            'status': 'healthy' if db_status['connected'] and data_status['initialized'] else 'unhealthy',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'response_time_ms': response_time,
            'connection': db_status,
            'data': data_status,
            'recommendations': get_recommendations(db_status, data_status)
        })
    except Exception as e:
        logger.error(f'数据库健康检查失败: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@bp.route('/health/database/status', methods=['GET'])
def database_status():
    """
    数据库状态检查（简化版）
    """
    try:
        # 检查数据库连接
        db_status = check_database_connection()
        
        # 检查数据库数据
        data_status = check_database_data()
        
        return jsonify({
            'connected': db_status['connected'],
            'initialized': data_status['initialized'],
            'has_users': data_status['has_users'],
            'message': get_status_message(db_status, data_status)
        })
    except Exception as e:
        logger.error(f'数据库状态检查失败: {str(e)}')
        return jsonify({
            'connected': False,
            'initialized': False,
            'has_users': False,
            'message': f'检查失败: {str(e)}'
        }), 500

def check_database_connection():
    """
    检查数据库连接
    """
    try:
        # 执行简单查询测试连接
        db.session.execute(db.text('SELECT 1'))
        return {
            'connected': True,
            'message': '数据库连接正常'
        }
    except Exception as e:
        logger.error(f'数据库连接失败: {str(e)}')
        return {
            'connected': False,
            'message': f'数据库连接失败: {str(e)}'
        }

def check_database_data():
    """
    检查数据库数据
    """
    try:
        user_count = User.query.count()
        asset_count = Asset.query.count()
        run_count = RunRecord.query.count()
        is_initialized = user_count > 0

        return {
            'initialized': is_initialized,
            'has_users': user_count > 0,
            'user_count': user_count,
            'asset_count': asset_count,
            'run_count': run_count,
            'message': '数据库已初始化' if is_initialized else '数据库未初始化'
        }
    except Exception as e:
        logger.error(f'数据库数据检查失败: {str(e)}')
        return {
            'initialized': False,
            'has_users': False,
            'user_count': 0,
            'asset_count': 0,
            'run_count': 0,
            'message': f'数据检查失败: {str(e)}'
        }

def get_status_message(db_status, data_status):
    """
    获取状态消息
    """
    if not db_status['connected']:
        return '数据库连接失败，请检查数据库配置'
    
    if not data_status['initialized']:
        return '数据库未初始化，请先创建管理员账号'
    
    if not data_status['has_users']:
        return '数据库中没有用户数据，请先创建管理员账号'
    
    return '数据库状态正常'

def get_recommendations(db_status, data_status):
    """
    获取建议
    """
    recommendations = []
    
    if not db_status['connected']:
        recommendations.append({
            'type': 'error',
            'message': '数据库连接失败',
            'action': '请检查数据库配置和连接状态'
        })
    
    if db_status['connected'] and not data_status['initialized']:
        recommendations.append({
            'type': 'warning',
            'message': '数据库未初始化',
            'action': '首次启动后会自动初始化默认管理员账号'
        })
    
    if db_status['connected'] and data_status['initialized'] and not data_status['has_users']:
        recommendations.append({
            'type': 'warning',
            'message': '数据库中没有用户数据',
            'action': '请创建管理员账号后重试'
        })
    
    if db_status['connected'] and data_status['initialized'] and data_status['has_users']:
        recommendations.append({
            'type': 'success',
            'message': '系统状态正常',
            'action': '可以正常使用系统'
        })
    
    return recommendations
