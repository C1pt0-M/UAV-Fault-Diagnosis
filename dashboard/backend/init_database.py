"""
统一数据库初始化脚本
用于初始化所有必要的测试数据
"""
import sys
import io

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app import create_app, db
from app.models.user import User
from app.models.drone import Drone
from app.models.fault import Fault
from app.models.maintenance import Maintenance
from app.models.log import Log
from app.routes.auth import hash_password
from datetime import datetime, timedelta
import random

def init_database():
    """初始化数据库"""
    app = create_app()

    with app.app_context():
        print("=" * 60)
        print("开始初始化数据库...")
        print("=" * 60)

        # 创建所有表
        print("\n1. 创建数据库表...")
        db.create_all()
        print("✓ 数据库表创建成功")

        # 清空现有数据
        print("\n2. 清空现有数据...")
        Log.query.delete()
        Maintenance.query.delete()
        Fault.query.delete()
        Drone.query.delete()
        User.query.delete()
        db.session.commit()
        print("✓ 现有数据清空成功")

        # 创建用户数据
        print("\n3. 创建用户数据...")
        users = [
            User(
                username='admin',
                password=hash_password('admin123'),
                real_name='系统管理员',
                email='admin@plane.com',
                phone='13800138000',
                department='技术部',
                role='admin',
                status='active'
            ),
            User(
                username='user1',
                password=hash_password('user123'),
                real_name='张三',
                email='zhangsan@plane.com',
                phone='13800138001',
                department='维修部',
                role='user',
                status='active'
            ),
            User(
                username='user2',
                password=hash_password('user123'),
                real_name='李四',
                email='lisi@plane.com',
                phone='13800138002',
                department='飞行部',
                role='user',
                status='active'
            )
        ]
        for user in users:
            db.session.add(user)
        db.session.commit()
        print(f"✓ 已创建 {len(users)} 个用户")
        print("  - 管理员账号: admin / admin123")
        print("  - 普通用户: user1 / user123, user2 / user123")

        # 创建无人机数据
        print("\n4. 创建无人机数据...")
        drones = []
        for i in range(1, 11):
            drone = Drone(
                id=f'DR-{i:03d}',
                name=f'无人机-{i}',
                status=random.choice(['online', 'offline']),
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
            drones.append(drone)
            db.session.add(drone)
        db.session.commit()
        print(f"✓ 已创建 {len(drones)} 个无人机")

        # 创建故障数据
        print("\n5. 创建故障数据...")
        fault_types = ['电机故障', '传感器异常', '电池故障', '通信故障', '导航系统故障']
        fault_levels = ['严重', '警告', '信息']
        fault_statuses = ['待处理', '处理中', '已解决']
        
        faults = []
        for i in range(1, 51):
            fault = Fault(
                id=f'FLT-{i:04d}',
                drone_id=f'DR-{random.randint(1, 10):03d}',
                fault_type=random.choice(fault_types),
                description=f'{random.choice(fault_types)}的详细描述',
                level=random.choice(fault_levels),
                status=random.choice(fault_statuses),
                detect_time=datetime.now() - timedelta(days=random.randint(0, 30)),
                resolved_time=datetime.now() - timedelta(days=random.randint(-1, 29)) if random.random() > 0.5 else None
            )
            faults.append(fault)
            db.session.add(fault)
        db.session.commit()
        print(f"✓ 已创建 {len(faults)} 个故障记录")

        # 创建维修任务数据
        print("\n6. 创建维修任务数据...")
        maintenance_statuses = ['待处理', '处理中', '已完成']
        maintenance_priorities = ['高', '中', '低']
        
        maintenances = []
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
            maintenances.append(maintenance)
            db.session.add(maintenance)
        db.session.commit()
        print(f"✓ 已创建 {len(maintenances)} 个维修任务")

        # 创建日志数据
        print("\n7. 创建日志数据...")
        log_types = ['flight', 'fault', 'maintenance', 'system']
        flight_contents = [
            '无人机起飞成功，飞行高度120米',
            '无人机降落成功，飞行时长35分钟',
            '无人机起飞成功，执行巡检任务',
            '无人机完成航线飞行，返回基地'
        ]
        fault_contents = [
            '检测到电机转速异常，触发故障警报',
            '电池电压过低，触发低电量警报',
            '传感器数据异常，需要检查',
            '通信信号不稳定，数据传输延迟'
        ]
        maintenance_contents = [
            '完成电机保养维护，更换轴承',
            '完成传感器校准',
            '更换电池，完成充电测试',
            '检查并调整螺旋桨平衡'
        ]
        system_contents = [
            '系统自动备份完成',
            '数据库优化完成',
            '系统版本更新成功',
            '定时任务执行完成'
        ]
        
        logs = []
        for i in range(1, 101):
            log_type = random.choice(log_types)
            
            if log_type == 'flight':
                content = random.choice(flight_contents)
            elif log_type == 'fault':
                content = random.choice(fault_contents)
            elif log_type == 'maintenance':
                content = random.choice(maintenance_contents)
            else:
                content = random.choice(system_contents)

            if log_type == 'system' and random.random() > 0.5:
                drone_id = None
            else:
                drone_id = f'DR-{random.randint(1, 10):03d}'

            if log_type == 'system':
                operator = '系统'
            elif log_type == 'maintenance':
                operator = f'维修员-{random.randint(1, 5)}'
            else:
                operator = '系统'

            log = Log(
                id=f'LOG-{i:04d}',
                drone_id=drone_id,
                type=log_type,
                content=content,
                operator=operator,
                created_at=datetime.now() - timedelta(days=random.randint(0, 30))
            )
            logs.append(log)
            db.session.add(log)
        db.session.commit()
        print(f"✓ 已创建 {len(logs)} 条日志记录")

        # 统计数据
        print("\n" + "=" * 60)
        print("数据库初始化完成！")
        print("=" * 60)
        print("\n数据统计：")
        print(f"  用户数量: {User.query.count()}")
        print(f"  - 管理员: {User.query.filter_by(role='admin').count()}")
        print(f"  - 普通用户: {User.query.filter_by(role='user').count()}")
        print(f"\n  无人机数量: {Drone.query.count()}")
        print(f"  - 在线: {Drone.query.filter_by(status='online').count()}")
        print(f"  - 离线: {Drone.query.filter_by(status='offline').count()}")
        print(f"\n  故障记录: {Fault.query.count()}")
        print(f"  - 待处理: {Fault.query.filter_by(status='待处理').count()}")
        print(f"  - 处理中: {Fault.query.filter_by(status='处理中').count()}")
        print(f"  - 已解决: {Fault.query.filter_by(status='已解决').count()}")
        print(f"\n  维修任务: {Maintenance.query.count()}")
        print(f"  - 待处理: {Maintenance.query.filter_by(status='待处理').count()}")
        print(f"  - 处理中: {Maintenance.query.filter_by(status='处理中').count()}")
        print(f"  - 已完成: {Maintenance.query.filter_by(status='已完成').count()}")
        print(f"\n  日志记录: {Log.query.count()}")
        print(f"  - 飞行日志: {Log.query.filter_by(type='flight').count()}")
        print(f"  - 故障日志: {Log.query.filter_by(type='fault').count()}")
        print(f"  - 维护日志: {Log.query.filter_by(type='maintenance').count()}")
        print(f"  - 系统日志: {Log.query.filter_by(type='system').count()}")
        print("\n" + "=" * 60)
        print("登录信息：")
        print("  管理员: admin / admin123")
        print("  普通用户: user1 / user123, user2 / user123")
        print("=" * 60)

if __name__ == '__main__':
    init_database()
