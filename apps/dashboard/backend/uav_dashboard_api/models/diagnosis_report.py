from datetime import datetime
import json

from uav_dashboard_api import db
from uav_dashboard_api.services.diagnosis_contract import build_diagnosis_payload


class DiagnosisReport(db.Model):
    """离线分类诊断报告。"""
    __tablename__ = 'diagnosis_reports'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    run_id = db.Column(db.Integer, db.ForeignKey('runs.id'), nullable=False, unique=True, index=True)
    asset_id = db.Column(db.Integer, db.ForeignKey('assets.id'), nullable=True, index=True)
    model_name = db.Column(db.String(64), nullable=False, default='')
    prediction_label = db.Column(db.String(64), nullable=False, index=True)
    confidence = db.Column(db.Float, nullable=False, default=0.0)
    probs_json = db.Column(db.Text, nullable=True)
    summary_json = db.Column(db.Text, nullable=True)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.now, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    def get_probs(self):
        if not self.probs_json:
            return {}
        try:
            return json.loads(self.probs_json)
        except Exception:
            return {}

    def set_probs(self, data: dict):
        self.probs_json = json.dumps(data or {}, ensure_ascii=False)

    def get_summary(self):
        if not self.summary_json:
            return {}
        try:
            return json.loads(self.summary_json)
        except Exception:
            return {}

    def set_summary(self, data: dict):
        self.summary_json = json.dumps(data or {}, ensure_ascii=False)

    def to_dict(self):
        summary = self.get_summary()
        diagnosis = build_diagnosis_payload(
            source='offline_report',
            label=self.prediction_label,
            confidence=self.confidence,
            severity=((summary.get('severity') or {}).get('level') if isinstance(summary, dict) else None),
            recommendations=(summary.get('recommendations') if isinstance(summary, dict) else []),
            model={
                'name': self.model_name,
                'task': 'offline_multiclass',
            },
            time_range=None,
        )
        payload = {
            'id': self.id,
            'run_id': self.run_id,
            'asset_id': self.asset_id,
            'model_name': self.model_name,
            'prediction_label': self.prediction_label,
            'confidence': self.confidence,
            'probs': self.get_probs(),
            'summary': summary,
            'diagnosis': diagnosis,
            'created_by': self.created_by,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None,
        }
        return payload
