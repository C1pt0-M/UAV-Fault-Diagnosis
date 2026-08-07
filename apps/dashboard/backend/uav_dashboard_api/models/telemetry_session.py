from datetime import datetime
import json

from uav_dashboard_api import db


def _normalize_recommendation_text(item):
    if isinstance(item, dict):
        return str(item.get('title') or item.get('description') or '').strip()
    return str(item or '').strip()


def _build_evidence_timeline(diagnosis, start_time, end_time):
    if not isinstance(diagnosis, dict):
        return []
    timeline = []
    lead_time_s = diagnosis.get('lead_time_s')
    evidence = diagnosis.get('evidence_summary', {}).get('top_evidence') or []
    recommendations = [text for text in (_normalize_recommendation_text(item) for item in (diagnosis.get('recommendations') or [])) if text]
    label = diagnosis.get('label') or 'fault'
    severity = diagnosis.get('severity') or 'unknown'
    stage_order = {'preview': 0, 'confirmed': 1, 'window_end': 2, 'action': 3}

    if lead_time_s is not None and start_time is not None:
        try:
            timeline.append({
                'ts': round(float(start_time) - float(lead_time_s), 3),
                'stage': 'preview',
                'title': '先兆预警',
                'description': f'{label} 在正式确认前出现可量化先兆',
                'status': 'derived',
                'evidence': evidence,
            })
        except (TypeError, ValueError):
            pass

    if start_time is not None:
        timeline.append({
            'ts': start_time,
            'stage': 'confirmed',
            'title': '故障确认',
            'description': f'{label} 已确认，严重度 {severity}',
            'status': 'observed',
            'evidence': evidence,
        })

    if end_time is not None:
        timeline.append({
            'ts': end_time,
            'stage': 'window_end',
            'title': '检测窗口结束',
            'description': '当前故障检测窗口结束',
            'status': 'derived',
            'evidence': evidence,
        })

    if recommendations and end_time is not None:
        timeline.append({
            'ts': end_time,
            'stage': 'action',
            'title': '处置建议',
            'description': '；'.join(recommendations),
            'status': 'recommended',
            'evidence': evidence,
        })

    return sorted(timeline, key=lambda item: (float(item.get('ts', 0)), stage_order.get(item.get('stage'), 99)))


class TelemetrySession(db.Model):
    """遥测采集会话记录。"""
    __tablename__ = 'telemetry_sessions'

    id = db.Column(db.String(20), primary_key=True, comment='会话ID: TS-XXXXXX')
    drone_id = db.Column(db.String(20), comment='关联无人机ID')
    session_type = db.Column(db.String(20), nullable=False, default='simulation', comment='类型: simulation/real')
    scenario_backend = db.Column(db.String(32), nullable=True, comment='场景后端: px4/replay')
    schema_version = db.Column(db.String(32), nullable=True, comment='遥测 schema 版本')
    source_path = db.Column(db.Text, nullable=True, comment='replay/数据源路径')
    meta_json = db.Column(db.Text, nullable=True, comment='扩展元信息 JSON')
    start_time = db.Column(db.DateTime, nullable=True, comment='开始时间')
    end_time = db.Column(db.DateTime, nullable=True, comment='结束时间')
    duration_seconds = db.Column(db.Float, nullable=True, comment='持续时长')
    frame_count = db.Column(db.Integer, default=0, comment='总帧数')
    csv_path = db.Column(db.Text, nullable=True, comment='CSV 文件路径')
    has_detection = db.Column(db.Boolean, default=False, comment='是否启用了实时检测')
    fault_events_count = db.Column(db.Integer, default=0, comment='检出故障事件数')
    status = db.Column(db.String(20), nullable=False, default='running', comment='状态: running/completed/failed')
    created_by = db.Column(db.String(50), nullable=True, comment='创建人')
    created_at = db.Column(db.DateTime, default=datetime.now, comment='创建时间')

    fault_events = db.relationship('TelemetryFaultEvent', backref='session', lazy='dynamic', cascade='all, delete-orphan')

    def get_meta(self):
        if not self.meta_json:
            return {}
        try:
            return json.loads(self.meta_json)
        except Exception:
            return {}

    def set_meta(self, data: dict):
        self.meta_json = json.dumps(data or {}, ensure_ascii=False)

    def to_dict(self):
        events = [item.to_dict() for item in self.fault_events.order_by(TelemetryFaultEvent.created_at.asc()).all()]
        latest_diagnosis = events[-1]['diagnosis'] if events else None
        return {
            'id': self.id,
            'drone_id': self.drone_id,
            'session_type': self.session_type,
            'scenario_backend': self.scenario_backend,
            'schema_version': self.schema_version,
            'source_path': self.source_path,
            'meta': self.get_meta(),
            'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else None,
            'end_time': self.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'frame_count': self.frame_count,
            'csv_path': self.csv_path,
            'has_detection': self.has_detection,
            'fault_events_count': self.fault_events_count,
            'fault_events': events,
            'latest_diagnosis': latest_diagnosis,
            'status': self.status,
            'created_by': self.created_by,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
        }


class TelemetryFaultEvent(db.Model):
    """实时检测中产生的故障事件。"""
    __tablename__ = 'telemetry_fault_events'

    id = db.Column(db.String(20), primary_key=True, comment='事件ID: FE-XXXXXX')
    session_id = db.Column(db.String(20), db.ForeignKey('telemetry_sessions.id'), nullable=False)
    start_time = db.Column(db.Float, nullable=False, comment='事件开始时间（飞行相对秒）')
    end_time = db.Column(db.Float, nullable=False, comment='事件结束时间')
    duration = db.Column(db.Float, nullable=True, comment='持续时长')
    max_fault_prob = db.Column(db.Float, nullable=True, comment='最大故障概率')
    mean_fault_prob = db.Column(db.Float, nullable=True, comment='平均故障概率')
    data_points = db.Column(db.Integer, nullable=True, comment='数据点数')
    severity = db.Column(db.String(20), nullable=True, comment='兼容严重度: low/medium/high/critical')
    diagnosis_json = db.Column(db.Text, nullable=True, comment='统一诊断对象JSON')
    created_at = db.Column(db.DateTime, default=datetime.now, comment='创建时间')

    def get_diagnosis(self):
        if not self.diagnosis_json:
            return {}
        try:
            return json.loads(self.diagnosis_json)
        except Exception:
            return {}

    def set_diagnosis(self, data: dict):
        self.diagnosis_json = json.dumps(data or {}, ensure_ascii=False)

    def to_dict(self):
        diagnosis = self.get_diagnosis()
        time_range = diagnosis.get('time_range') if isinstance(diagnosis, dict) else {}
        start_time = time_range.get('start_time', self.start_time)
        end_time = time_range.get('end_time', self.end_time)
        return {
            'id': self.id,
            'session_id': self.session_id,
            'start_time': start_time,
            'end_time': end_time,
            'duration': time_range.get('duration', self.duration),
            'max_fault_prob': self.max_fault_prob,
            'mean_fault_prob': self.mean_fault_prob,
            'data_points': self.data_points,
            'severity': diagnosis.get('severity') or self.severity,
            'severity_legacy': self.severity,
            'diagnosis': diagnosis,
            'evidence_timeline': _build_evidence_timeline(diagnosis, start_time, end_time),
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
        }
