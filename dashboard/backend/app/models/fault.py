# 故障模型
from datetime import datetime
from app import db

class Fault(db.Model):
    """故障数据模型"""
    __tablename__ = 'faults'

    id = db.Column(db.String(20), primary_key=True, comment='故障编号')
    drone_id = db.Column(db.String(20), db.ForeignKey('drones.id'), index=True, comment='无人机ID')
    fault_type = db.Column(db.String(100), nullable=False, comment='故障类型')
    description = db.Column(db.Text, comment='故障描述')
    level = db.Column(db.String(20), default='信息', index=True, comment='级别: 严重/警告/信息')
    status = db.Column(db.String(20), default='待处理', index=True, comment='状态: 待处理/处理中/已解决')
    detect_time = db.Column(db.DateTime, default=datetime.now, index=True, comment='检测时间')
    resolved_time = db.Column(db.DateTime, comment='解决时间')
    created_at = db.Column(db.DateTime, default=datetime.now, comment='创建时间')
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, comment='更新时间')

    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'drone_id': self.drone_id,
            'fault_type': self.fault_type,
            'description': self.description,
            'level': self.level,
            'status': self.status,
            'detect_time': self.detect_time.strftime('%Y-%m-%d %H:%M:%S') if self.detect_time else None,
            'resolved_time': self.resolved_time.strftime('%Y-%m-%d %H:%M:%S') if self.resolved_time else None,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None
        }
