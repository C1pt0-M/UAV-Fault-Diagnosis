"""
测试数据初始化脚本
用于生成测试数据以便测试统计和图表功能
"""
from app import create_app, db
from app.models.drone import Drone
from app.models.fault import Fault
from app.models.maintenance import Maintenance
from datetime import datetime, timedelta
import random

def init_test_data():
    """初始化测试数据"""
    app = create_app()

    with app.app_context():
        print("开始初始化测试数据...")

        # 清空现有数据
        print("清空现有数据...")
        Fault.query.delete()
        Maintenance.query.delete()
        Drone.query.delete()
        db.session.commit()

        # 创建无人机数据
        print("创建无人机数据...")
        drone_statuses = ['online', 'offline']
        fault_types = ['电机故障', '传感器异常', '电池故障', '通信故障', '导航系统故障']
        fault_levels = ['严重', '警告', '信息']
        fault_statuses = ['待处理', '处理中', '已解决']
        maintenance_statuses = ['待处理', '处理中', '已完成']
        maintenance_priorities = ['高', '中', '低']

        for i in range(1, 11):
            drone = Drone(
                id=f'DR-{i:03d}',
                name=f'无人机-{i}',
                status=random.choice(drone_statuses),
                battery=random.randint(20, 100),
                signal=random.choice(['强', '中', '弱']),
                height=random.uniform(0, 100),
                speed=random.uniform(0, 15),
                distance=random.uniform(0, 5000),
                duration=random.randint(0, 300),
                gps=random.choice(['良好', '一般', '差']),
                longitude=f'{random.uniform(110, 120):.6f}',
                latitude=f'{random.uniform(20, 30):.6f}',
                task_status=random.choice(['待命', '飞行中', '返航中', '降落中']),
                flight_mode=random.choice(['手动', '自动', '定点悬停'])
            )
            db.session.add(drone)

        db.session.commit()
        print(f"已创建 10 个无人机")

        # 创建故障数据
        print("创建故障数据...")
        for i in range(1, 51):
            fault = Fault(
                id=f'FT-{i:04d}',
                drone_id=f'DR-{random.randint(1, 10):03d}',
                fault_type=random.choice(fault_types),
                description=f'{random.choice(fault_types)}的详细描述',
                level=random.choice(fault_levels),
                status=random.choice(fault_statuses),
                detect_time=datetime.now() - timedelta(days=random.randint(0, 30)),
                resolved_time=datetime.now() - timedelta(days=random.randint(-1, 29)) if random.random() > 0.5 else None
            )
            db.session.add(fault)

        db.session.commit()
        print(f"已创建 50 个故障记录")

        # 创建维修任务数据
        print("创建维修任务数据...")
        for i in range(1, 31):
            maintenance = Maintenance(
                id=f'MT-{i:04d}',
                drone_id=f'DR-{random.randint(1, 10):03d}',
                title=f'维修任务-{i}',
                fault_type=random.choice(fault_types),
                description=f'{random.choice(fault_types)}的维修方案',
                priority=random.choice(maintenance_priorities),
                status=random.choice(maintenance_statuses),
                assignee=f'维修员-{random.randint(1, 5)}',
                steps='[]',
                parts='[]',
                records='[]',
                created_at=datetime.now() - timedelta(days=random.randint(0, 180))
            )
            db.session.add(maintenance)

        db.session.commit()
        print(f"已创建 30 个维修任务")

        print("\n测试数据初始化完成！")
        print("\n数据统计：")
        print(f"- 无人机数量: {Drone.query.count()}")
        print(f"- 在线无人机: {Drone.query.filter_by(status='online').count()}")
        print(f"- 故障记录: {Fault.query.count()}")
        print(f"- 待处理故障: {Fault.query.filter_by(status='待处理').count()}")
        print(f"- 维修任务: {Maintenance.query.count()}")
        print(f"- 已完成维修: {Maintenance.query.filter_by(status='已完成').count()}")

if __name__ == '__main__':
    init_test_data()
