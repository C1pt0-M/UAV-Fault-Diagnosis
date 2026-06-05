# 数据库初始化脚本
# 用于创建数据库和初始化示例数据
from app import create_app, db
from app.models.drone import Drone
from app.models.fault import Fault
from app.models.log import Log
from app.models.maintenance import Maintenance
from app.models.report import Report
from app.models.user import User
import bcrypt

def hash_password(password):
    """使用bcrypt加密密码"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def init_database():
    """初始化数据库"""
    app = create_app()

    with app.app_context():
        # 创建所有表
        db.create_all()
        print("数据库表创建成功！")

        # 检查是否已有用户数据
        if User.query.count() > 0:
            print("数据库已有用户数据，跳过用户初始化")
        else:
            # 创建默认管理员用户
            admin_user = User(
                username='admin',
                password=hash_password('admin123'),
                real_name='系统管理员',
                email='admin@example.com',
                phone='13800138000',
                role='admin',
                status='active',
                department='技术部'
            )
            db.session.add(admin_user)

            # 创建普通用户
            normal_users = [
                User(
                    username='zhangsan',
                    password=hash_password('123456'),
                    real_name='张三',
                    email='zhangsan@example.com',
                    phone='13800138001',
                    role='user',
                    status='active',
                    department='维修部'
                ),
                User(
                    username='lisi',
                    password=hash_password('123456'),
                    real_name='李四',
                    email='lisi@example.com',
                    phone='13800138002',
                    role='user',
                    status='active',
                    department='维修部'
                ),
                User(
                    username='wangwu',
                    password=hash_password('123456'),
                    real_name='王五',
                    email='wangwu@example.com',
                    phone='13800138003',
                    role='user',
                    status='active',
                    department='监控部'
                )
            ]
            db.session.add_all(normal_users)

            print("默认用户创建成功！")
            print("管理员账号: admin / admin123")
            print("普通用户: zhangsan / 123456")

        # 检查是否已有业务数据
        if Drone.query.count() > 0:
            print("数据库已有业务数据，跳过业务数据初始化")
            return

        # 创建示例无人机
        drones = [
            Drone(id='DR-001', name='无人机-001', status='online', battery=85, signal='强',
                   height=120, speed=15, distance=3200, duration=25, gps='强',
                   longitude='116.4074', latitude='39.9042', task_status='执行中', flight_mode='GPS模式'),
            Drone(id='DR-002', name='无人机-002', status='online', battery=62, signal='中',
                   height=80, speed=10, distance=1800, duration=18, gps='中',
                   longitude='116.3974', latitude='39.9142', task_status='返航中', flight_mode='姿态模式'),
            Drone(id='DR-003', name='无人机-003', status='offline', battery=100, signal='-',
                   task_status='待命'),
            Drone(id='DR-004', name='无人机-004', status='online', battery=45, signal='强',
                   height=150, speed=18, distance=4500, duration=32, gps='强',
                   longitude='116.4174', latitude='39.8942', task_status='执行中', flight_mode='GPS模式')
        ]
        db.session.add_all(drones)

        # 创建示例故障
        faults = [
            Fault(id='FLT001', drone_id='DR-001', fault_type='电机故障',
                   description='1号电机转速异常，可能存在轴承损坏', level='严重', status='待处理'),
            Fault(id='FLT002', drone_id='DR-002', fault_type='电池异常',
                   description='电池电压下降过快，可能存在电池老化', level='警告', status='处理中'),
            Fault(id='FLT003', drone_id='DR-003', fault_type='传感器故障',
                   description='IMU传感器数据不稳定', level='严重', status='待处理'),
            Fault(id='FLT004', drone_id='DR-004', fault_type='GPS信号弱',
                   description='GPS信号强度低于阈值，影响定位精度', level='警告', status='处理中')
        ]
        db.session.add_all(faults)

        # 创建示例日志
        logs = [
            Log(id='LOG001', drone_id='DR-001', type='flight',
                content='无人机起飞成功，飞行高度120米', operator='系统'),
            Log(id='LOG002', drone_id='DR-002', type='fault',
                content='检测到电机转速异常，触发故障警报', operator='系统'),
            Log(id='LOG003', drone_id='DR-003', type='maintenance',
                content='完成电机保养维护，更换轴承', operator='张三'),
            Log(id='LOG004', drone_id='DR-001', type='system',
                content='系统自动备份完成', operator='系统')
        ]
        db.session.add_all(logs)

        # 创建示例维修任务
        maintenance_tasks = [
            Maintenance(id='MT-2024-001', drone_id='DR-001', title='DR-001电机轴承更换',
                       fault_type='电机故障', description='1号电机转速异常，检测发现轴承损坏，需要更换',
                       priority='高', status='待处理', assignee='张三'),
            Maintenance(id='MT-2024-002', drone_id='DR-002', title='DR-002电池更换',
                       fault_type='电池异常', description='电池电压下降过快，需要更换新电池',
                       priority='中', status='待处理', assignee='李四')
        ]
        db.session.add_all(maintenance_tasks)

        # 创建示例报告
        reports = [
            Report(id='RPT202401001', title='DR-001电机故障检测报告', type='fault',
                   drone_id='DR-001', creator='张三', status='已审核',
                   summary='对DR-001无人机进行全面检测，发现1号电机存在异常，建议更换轴承'),
            Report(id='RPT202401002', title='DR-002季度维护报告', type='maintenance',
                   drone_id='DR-002', creator='李四', status='已提交',
                   summary='完成DR-002季度定期维护，更换电池、校准传感器')
        ]
        db.session.add_all(reports)

        # 提交所有更改
        db.session.commit()
        print("示例数据初始化成功！")
        print(f"已创建 {len(drones)} 个无人机")
        print(f"已创建 {len(faults)} 个故障记录")
        print(f"已创建 {len(logs)} 条日志")
        print(f"已创建 {len(maintenance_tasks)} 个维修任务")
        print(f"已创建 {len(reports)} 个报告")

if __name__ == '__main__':
    init_database()
