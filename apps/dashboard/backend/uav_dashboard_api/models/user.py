# 用户数据模型
from datetime import datetime
from uav_dashboard_api import db

class User(db.Model):
    """用户模型"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False, index=True, comment='用户名')
    password = db.Column(db.String(255), nullable=False, comment='密码（加密）')
    real_name = db.Column(db.String(50), comment='真实姓名')
    email = db.Column(db.String(100), unique=True, index=True, comment='邮箱')
    phone = db.Column(db.String(20), comment='手机号')
    role = db.Column(db.String(20), default='user', comment='角色：admin, user')
    status = db.Column(db.String(10), default='active', index=True, comment='状态：active, inactive')
    avatar = db.Column(db.String(255), comment='头像URL')
    department = db.Column(db.String(50), comment='部门')
    last_login = db.Column(db.DateTime, comment='最后登录时间')
    login_count = db.Column(db.Integer, default=0, comment='登录次数')
    created_at = db.Column(db.DateTime, default=datetime.now, comment='创建时间')
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, comment='更新时间')

    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'username': self.username,
            'real_name': self.real_name,
            'email': self.email,
            'phone': self.phone,
            'role': self.role,
            'status': self.status,
            'avatar': self.avatar,
            'department': self.department,
            'last_login': self.last_login.strftime('%Y-%m-%d %H:%M:%S') if self.last_login else None,
            'login_count': self.login_count,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None
        }

    def __repr__(self):
        return f'<User {self.username}>'
