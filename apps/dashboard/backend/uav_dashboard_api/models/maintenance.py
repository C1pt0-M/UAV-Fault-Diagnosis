# 维修方案模型
from datetime import datetime
from uav_dashboard_api import db

class Maintenance(db.Model):
    """维修方案数据模型"""
    __tablename__ = 'maintenance_tasks'

    id = db.Column(db.String(20), primary_key=True, comment='任务编号')
    drone_id = db.Column(db.String(20), db.ForeignKey('drones.id'), comment='无人机ID')
    title = db.Column(db.String(200), nullable=False, comment='任务标题')
    fault_type = db.Column(db.String(100), comment='故障类型')
    description = db.Column(db.Text, comment='故障描述')
    priority = db.Column(db.String(20), default='中', comment='优先级: 高/中/低')
    status = db.Column(db.String(20), default='待处理', comment='状态: 待处理/处理中/已完成')
    assignee = db.Column(db.String(50), comment='负责人')
    steps = db.Column(db.Text, comment='维修步骤(JSON格式)')
    parts = db.Column(db.Text, comment='所需配件(JSON格式)')
    records = db.Column(db.Text, comment='维修记录(JSON格式)')
    created_at = db.Column(db.DateTime, default=datetime.now, comment='创建时间')
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, comment='更新时间')

    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'drone_id': self.drone_id,
            'title': self.title,
            'fault_type': self.fault_type,
            'description': self.description,
            'priority': self.priority,
            'status': self.status,
            'assignee': self.assignee,
            'steps': self.steps,
            'parts': self.parts,
            'records': self.records,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None
        }
