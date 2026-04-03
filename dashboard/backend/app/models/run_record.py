from datetime import datetime
import json

from app import db
from app.utils.path_store import sanitize_path_payload


class RunRecord(db.Model):
    """异步任务执行记录。"""
    __tablename__ = 'runs'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    task = db.Column(db.String(64), nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, index=True, default='queued')
    progress = db.Column(db.Float, nullable=False, default=0.0)
    params_json = db.Column(db.Text, nullable=True)
    result_json = db.Column(db.Text, nullable=True)
    error = db.Column(db.Text, nullable=True)
    error_detail = db.Column(db.Text, nullable=True)
    asset_id = db.Column(db.Integer, db.ForeignKey('assets.id'), nullable=True, index=True)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.now, index=True)
    started_at = db.Column(db.DateTime, nullable=True)
    finished_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    def get_params(self):
        if not self.params_json:
            return {}
        try:
            return json.loads(self.params_json)
        except Exception:
            return {}

    def set_params(self, data: dict):
        self.params_json = json.dumps(data or {}, ensure_ascii=False)

    def get_result(self):
        if not self.result_json:
            return None
        try:
            return json.loads(self.result_json)
        except Exception:
            return None

    def set_result(self, data):
        self.result_json = (
            json.dumps(sanitize_path_payload(data), ensure_ascii=False)
            if data is not None
            else None
        )

    def to_dict(self):
        return {
            'id': self.id,
            'task': self.task,
            'status': self.status,
            'progress': self.progress,
            'params': self.get_params(),
            'result': self.get_result(),
            'error': self.error,
            'error_detail': self.error_detail,
            'asset_id': self.asset_id,
            'created_by': self.created_by,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'started_at': self.started_at.strftime('%Y-%m-%d %H:%M:%S') if self.started_at else None,
            'finished_at': self.finished_at.strftime('%Y-%m-%d %H:%M:%S') if self.finished_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None,
        }
