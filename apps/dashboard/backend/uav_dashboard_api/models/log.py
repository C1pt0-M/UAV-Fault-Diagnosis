# 日志模型
from datetime import datetime
from uav_dashboard_api import db

class Log(db.Model):
    """日志数据模型"""
    __tablename__ = 'logs'

    id = db.Column(db.String(20), primary_key=True, comment='日志ID')
    drone_id = db.Column(db.String(20), db.ForeignKey('drones.id'), comment='无人机ID')
    type = db.Column(db.String(20), nullable=False, comment='日志类型: flight/fault/maintenance/system')
    content = db.Column(db.Text, nullable=False, comment='日志内容')
    operator = db.Column(db.String(50), comment='操作人')
    created_at = db.Column(db.DateTime, default=datetime.now, comment='创建时间')

    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'drone_id': self.drone_id,
            'type': self.type,
            'content': self.content,
            'operator': self.operator,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }
