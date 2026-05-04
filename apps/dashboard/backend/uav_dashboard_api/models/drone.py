# 无人机模型
from datetime import datetime
from uav_dashboard_api import db

class Drone(db.Model):
    """无人机数据模型"""
    __tablename__ = 'drones'

    id = db.Column(db.String(20), primary_key=True, comment='无人机ID')
    name = db.Column(db.String(100), nullable=False, comment='无人机名称')
    status = db.Column(db.String(20), default='offline', index=True, comment='状态: online/offline')
    battery = db.Column(db.Integer, default=100, comment='电池电量(%)')
    signal = db.Column(db.String(20), comment='信号强度')
    height = db.Column(db.Float, default=0, comment='飞行高度(米)')
    speed = db.Column(db.Float, default=0, comment='飞行速度(m/s)')
    distance = db.Column(db.Float, default=0, comment='飞行距离(米)')
    duration = db.Column(db.Integer, default=0, comment='飞行时长(分钟)')
    gps = db.Column(db.String(20), comment='GPS信号')
    longitude = db.Column(db.String(50), comment='经度')
    latitude = db.Column(db.String(50), comment='纬度')
    task_status = db.Column(db.String(50), default='待命', comment='任务状态')
    flight_mode = db.Column(db.String(50), comment='飞行模式')
    created_at = db.Column(db.DateTime, default=datetime.now, comment='创建时间')
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, comment='更新时间')

    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status,
            'battery': self.battery,
            'signal': self.signal,
            'height': self.height,
            'speed': self.speed,
            'distance': self.distance,
            'duration': self.duration,
            'gps': self.gps,
            'longitude': self.longitude,
            'latitude': self.latitude,
            'task_status': self.task_status,
            'flight_mode': self.flight_mode,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None
        }
