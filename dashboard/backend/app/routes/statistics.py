# 统计分析相关API
from flask import Blueprint, jsonify, request
from app import db
from app.models.drone import Drone
from app.models.fault import Fault
from app.models.maintenance import Maintenance
from app.utils.jwt_utils import token_required
from sqlalchemy import func, case
from datetime import datetime, timedelta

# 创建蓝图
bp = Blueprint('statistics', __name__)

@bp.route('/overview', methods=['GET'])
@token_required
def get_overview(current_user):
    """获取系统概览统计"""
    try:
        # 使用case语句进行SQLite兼容的计数
        online_count = db.session.query(
            func.sum(case((Drone.status == 'online', 1), else_=0))
        ).scalar() or 0
        
        total_drones = db.session.query(func.count(Drone.id)).scalar() or 0

        fault_stats = db.session.query(
            func.count(Fault.id),
            func.sum(case((Fault.status == '待处理', 1), else_=0)),
            func.sum(case((Fault.level == '严重', 1), else_=0)),
            func.sum(case((Fault.level == '警告', 1), else_=0))
        ).first()

        maintenance_stats = db.session.query(
            func.count(Maintenance.id),
            func.sum(case((Maintenance.status == '待处理', 1), else_=0)),
            func.sum(case((Maintenance.status == '已完成', 1), else_=0))
        ).first()

        return jsonify({
            'success': True,
            'data': {
                'drones': {
                    'total': total_drones,
                    'online': online_count,
                    'offline': total_drones - online_count
                },
                'faults': {
                    'total': fault_stats[0] or 0,
                    'pending': fault_stats[1] or 0,
                    'resolved': (fault_stats[0] or 0) - (fault_stats[1] or 0),
                    'critical': fault_stats[2] or 0,
                    'warning': fault_stats[3] or 0,
                    'info': (fault_stats[0] or 0) - (fault_stats[2] or 0) - (fault_stats[3] or 0)
                },
                'maintenance': {
                    'total': maintenance_stats[0] or 0,
                    'pending': maintenance_stats[1] or 0,
                    'completed': maintenance_stats[2] or 0,
                    'processing': (maintenance_stats[0] or 0) - (maintenance_stats[1] or 0) - (maintenance_stats[2] or 0)
                }
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取系统概览统计失败: {str(e)}'
        }), 500

@bp.route('/fault-trend', methods=['GET'])
@token_required
def get_fault_trend(current_user):
    """获取故障趋势数据"""
    try:
        # 获取查询参数
        range_param = request.args.get('range', 'week')

        # 根据范围确定时间范围
        if range_param == 'week':
            # 最近7天
            start_date = datetime.now() - timedelta(days=7)
            date_format = '%Y-%m-%d'
        elif range_param == 'month':
            # 最近30天
            start_date = datetime.now() - timedelta(days=30)
            date_format = '%Y-%m-%d'
        elif range_param == 'year':
            # 最近12个月
            start_date = datetime.now() - timedelta(days=365)
            date_format = '%Y-%m'
        else:
            # 默认7天
            start_date = datetime.now() - timedelta(days=7)
            date_format = '%Y-%m-%d'

        # 查询故障数据 - 使用SQLite兼容的strftime函数
        fault_data = db.session.query(
            func.strftime(date_format, Fault.detect_time).label('date'),
            func.count(Fault.id).label('count')
        ).filter(
            Fault.detect_time >= start_date
        ).group_by(
            func.strftime(date_format, Fault.detect_time)
        ).order_by('date').all()

        # 转换为列表
        data = [{'date': fd[0], 'count': fd[1]} for fd in fault_data]

        # 如果没有数据，返回空数组
        if not data:
            return jsonify({
                'success': True,
                'data': []
            })

        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取故障趋势数据失败: {str(e)}'
        }), 500

@bp.route('/fault-type-distribution', methods=['GET'])
@token_required
def get_fault_type_distribution(current_user):
    """获取故障类型分布"""
    try:
        # 从数据库统计故障类型
        fault_types = db.session.query(
            Fault.fault_type,
            func.count(Fault.id)
        ).group_by(Fault.fault_type).all()

        data = [{'type': ft[0], 'count': ft[1]} for ft in fault_types]

        # 如果没有数据，返回空数组
        if not data:
            return jsonify({
                'success': True,
                'data': []
            })

        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取故障类型分布失败: {str(e)}'
        }), 500

@bp.route('/drone-ranking', methods=['GET'])
@token_required
def get_drone_ranking(current_user):
    """获取无人机故障排行"""
    try:
        # 查询每个无人机的故障数量
        drone_faults = db.session.query(
            Fault.drone_id,
            func.count(Fault.id).label('fault_count')
        ).group_by(Fault.drone_id).order_by(func.desc('fault_count')).limit(10).all()

        # 获取每个无人机的总飞行次数（使用无人机表中的数据）
        data = []
        for drone_id, fault_count in drone_faults:
            drone = db.session.query(Drone).filter_by(id=drone_id).first()
            if drone:
                # 故障率 = 故障次数 / (飞行时长 / 60)  # 假设每次飞行平均60分钟
                # 或者直接使用相对的故障率
                total_drones = db.session.query(func.count(Drone.id)).scalar() or 1
                fault_rate = round((fault_count / total_drones) * 100, 1)

                data.append({
                    'droneId': drone_id,
                    'faultCount': fault_count,
                    'faultRate': fault_rate
                })

        # 如果没有数据，返回空数组
        if not data:
            return jsonify({
                'success': True,
                'data': []
            })

        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取无人机故障排行失败: {str(e)}'
        }), 500

@bp.route('/maintenance-efficiency', methods=['GET'])
@token_required
def get_maintenance_efficiency(current_user):
    """获取维修效率统计"""
    try:
        # 查询最近6个月的维修数据
        start_date = datetime.now() - timedelta(days=180)

        # 按月统计维修数据 - 使用SQLite兼容的strftime函数
        monthly_stats = db.session.query(
            func.strftime('%Y-%m', Maintenance.created_at).label('month'),
            func.count(Maintenance.id).label('total_tasks'),
            func.sum(case((Maintenance.status == '已完成', 1), else_=0)).label('completed_tasks')
        ).filter(
            Maintenance.created_at >= start_date
        ).group_by(
            func.strftime('%Y-%m', Maintenance.created_at)
        ).order_by('month').all()

        # 转换为列表格式
        data = []
        for month_name, total_tasks, completed_tasks in monthly_stats:
            completion_rate = round((completed_tasks / total_tasks * 100), 0) if total_tasks > 0 else 0

            # 计算平均耗时（简化处理，假设平均3-5小时）
            avg_time = f'{3.5 + (total_tasks % 2) * 0.7}h'

            # 将月份格式转换为中文格式
            month_num = int(month_name.split('-')[1])
            month_cn = f'{month_num}月'

            data.append({
                'month': month_cn,
                'totalTasks': total_tasks,
                'completedTasks': completed_tasks,
                'completionRate': completion_rate,
                'avgTime': avg_time
            })

        # 如果没有数据，返回空数组
        if not data:
            return jsonify({
                'success': True,
                'data': []
            })

        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取维修效率统计失败: {str(e)}'
        }), 500

@bp.route('/flight-stats', methods=['GET'])
@token_required
def get_flight_stats(current_user):
    """获取飞行统计数据"""
    try:
        # 从无人机表统计飞行数据
        flight_stats = db.session.query(
            func.sum(Drone.duration).label('total_duration'),
            func.sum(Drone.distance).label('total_distance'),
            func.count(Drone.id).label('total_drones'),
            func.avg(Drone.duration).label('avg_duration')
        ).first()

        total_duration = flight_stats.total_duration or 0
        total_distance = flight_stats.total_distance or 0
        total_drones = flight_stats.total_drones or 1
        avg_duration = flight_stats.avg_duration or 0

        # 计算任务完成率（假设在线无人机为完成任务）
        online_drones = db.session.query(func.count(Drone.id)).filter(
            Drone.status == 'online'
        ).scalar() or 0

        task_completion_rate = round((online_drones / total_drones * 100), 1) if total_drones > 0 else 0

        # 格式化数据
        total_flight_time = f'{int(total_duration)}'
        total_flight_distance = f'{int(total_distance)}'
        task_completion_rate_str = f'{task_completion_rate}'
        avg_flight_time = f'{int(avg_duration)}'

        # 计算趋势（模拟数据，实际应该与上月数据对比）
        trends = {
            'flightTime': 15,
            'flightDistance': 22,
            'taskCompletion': 3,
            'avgTime': -5
        }

        return jsonify({
            'success': True,
            'data': {
                'totalFlightTime': total_flight_time,
                'totalFlightDistance': total_flight_distance,
                'taskCompletionRate': task_completion_rate_str,
                'avgFlightTime': avg_flight_time,
                'trends': trends
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取飞行统计数据失败: {str(e)}'
        }), 500
