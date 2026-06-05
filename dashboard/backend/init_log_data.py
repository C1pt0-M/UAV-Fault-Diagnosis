"""
日志数据初始化脚本
用于生成测试日志数据以便测试日志功能
"""
from app import create_app, db
from app.models.log import Log
from app.models.drone import Drone
from datetime import datetime, timedelta
import random

def init_log_data():
    """初始化日志数据"""
    app = create_app()

    with app.app_context():
        print("开始初始化日志数据...")

        # 清空现有日志数据
        print("清空现有日志数据...")
        Log.query.delete()
        db.session.commit()

        # 获取所有无人机
        drones = Drone.query.all()
        if not drones:
            print("警告：没有找到无人机数据，请先初始化无人机数据")
            return

        # 日志类型
        log_types = ['flight', 'fault', 'maintenance', 'system']
        flight_contents = [
            '无人机起飞成功，飞行高度120米',
            '无人机降落成功，飞行时长35分钟',
            '无人机起飞成功，执行巡检任务',
            '无人机完成航线飞行，返回基地',
            '无人机悬停高度稳定，准备降落',
            '无人机电池电量充足，继续执行任务',
            '无人机GPS信号良好，位置准确',
            '无人机风速适中，飞行平稳'
        ]
        fault_contents = [
            '检测到电机转速异常，触发故障警报',
            '电池电压过低，触发低电量警报',
            '传感器数据异常，需要检查',
            '通信信号不稳定，数据传输延迟',
            '陀螺仪校准失败，影响飞行稳定性',
            '电机温度过高，建议暂停飞行',
            'GPS信号丢失，无法定位',
            '气压传感器读数异常'
        ]
        maintenance_contents = [
            '完成电机保养维护，更换轴承',
            '完成传感器校准',
            '更换电池，完成充电测试',
            '检查并调整螺旋桨平衡',
            '清洁机身，检查外观完整性',
            '更新固件版本，优化性能',
            '检查通信模块，更换天线',
            '测试飞行控制器功能正常'
        ]
        system_contents = [
            '系统自动备份完成',
            '数据库优化完成',
            '系统版本更新成功',
            '定时任务执行完成',
            '系统日志清理完成',
            '缓存刷新成功',
            '系统健康检查通过',
            '用户权限更新完成'
        ]

        # 创建100条测试日志
        print("创建测试日志数据...")
        for i in range(1, 101):
            log_type = random.choice(log_types)
            
            # 根据类型选择内容
            if log_type == 'flight':
                content = random.choice(flight_contents)
            elif log_type == 'fault':
                content = random.choice(fault_contents)
            elif log_type == 'maintenance':
                content = random.choice(maintenance_contents)
            else:
                content = random.choice(system_contents)

            # 随机选择无人机（系统日志可能没有无人机）
            if log_type == 'system' and random.random() > 0.5:
                drone_id = None
            else:
                drone_id = random.choice(drones).id

            # 操作人
            if log_type == 'system':
                operator = '系统'
            elif log_type == 'maintenance':
                operator = f'维修员-{random.randint(1, 5)}'
            else:
                operator = '系统'

            # 创建时间（最近30天内）
            created_at = datetime.now() - timedelta(days=random.randint(0, 30))

            log = Log(
                id=f'LOG-{i:04d}',
                drone_id=drone_id,
                type=log_type,
                content=content,
                operator=operator,
                created_at=created_at
            )
            db.session.add(log)

        db.session.commit()
        print(f"已创建 100 条测试日志")

        # 统计日志数据
        print("\n日志数据统计：")
        print(f"- 总日志数: {Log.query.count()}")
        print(f"- 飞行日志: {Log.query.filter_by(type='flight').count()}")
        print(f"- 故障日志: {Log.query.filter_by(type='fault').count()}")
        print(f"- 维护日志: {Log.query.filter_by(type='maintenance').count()}")
        print(f"- 系统日志: {Log.query.filter_by(type='system').count()}")

        print("\n日志数据初始化完成！")

if __name__ == '__main__':
    init_log_data()
