from datetime import datetime
import json

from flask import current_app, has_app_context

from uav_dashboard_api import db
from uav_dashboard_api.utils.path_store import sanitize_path_payload


class Asset(db.Model):
    """上传文件资产（ULG/CSV）。"""
    __tablename__ = 'assets'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(255), nullable=False, comment='原始文件名')
    kind = db.Column(db.String(16), nullable=False, index=True, comment='文件类型: ulg/csv')
    stored_path = db.Column(db.String(1024), nullable=False, unique=True, comment='本地存储绝对路径')
    size_bytes = db.Column(db.BigInteger, nullable=False, default=0, comment='文件大小(bytes)')
    sha256 = db.Column(db.String(64), nullable=False, index=True, comment='文件哈希')
    meta_json = db.Column(db.Text, nullable=True, comment='扩展元数据JSON')
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.now, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    def get_meta(self):
        if not self.meta_json:
            return {}
        try:
            return json.loads(self.meta_json)
        except Exception:
            return {}

    def set_meta(self, data: dict):
        self.meta_json = json.dumps(
            sanitize_path_payload(data or {}),
            ensure_ascii=False,
        )

    @property
    def prepared_csv_path(self):
        return self.get_meta().get('prepared_csv_path')

    @property
    def is_prepared(self):
        prepared_path = self.prepared_csv_path
        return bool(prepared_path)

    def to_dict(self):
        meta = self.get_meta()
        payload = {
            'id': self.id,
            'filename': self.filename,
            'kind': self.kind,
            'size_bytes': self.size_bytes,
            'sha256': self.sha256,
            'meta': meta,
            'created_by': self.created_by,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None,
        }
        if has_app_context() and current_app.config.get('EXPOSE_SERVER_PATHS', False):
            payload['stored_path'] = self.stored_path
        return payload
