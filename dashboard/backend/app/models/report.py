# 报告模型
from datetime import datetime
from app import db

class Report(db.Model):
    """报告数据模型"""
    __tablename__ = 'reports'

    id = db.Column(db.String(20), primary_key=True, comment='报告编号')
    title = db.Column(db.String(200), nullable=False, comment='报告标题')
    type = db.Column(db.String(20), nullable=False, comment='报告类型: fault/maintenance/flight/inspection')
    drone_id = db.Column(db.String(20), db.ForeignKey('drones.id'), comment='无人机ID')
    creator = db.Column(db.String(50), comment='创建人')
    status = db.Column(db.String(20), default='草稿', comment='状态: 草稿/已提交/已审核')
    summary = db.Column(db.Text, comment='报告摘要')
    content = db.Column(db.Text, comment='报告内容')
    created_at = db.Column(db.DateTime, default=datetime.now, comment='创建时间')
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, comment='更新时间')

    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'title': self.title,
            'type': self.type,
            'drone_id': self.drone_id,
            'creator': self.creator,
            'status': self.status,
            'summary': self.summary,
            'content': self.content,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None
        }
