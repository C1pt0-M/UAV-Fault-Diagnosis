from datetime import datetime
import json

from app import db


class AIConversation(db.Model):
    """AI 助手会话。"""
    __tablename__ = 'ai_conversations'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(255), nullable=False, default='新对话')
    scene = db.Column(db.String(32), nullable=False, default='general', index=True)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    report_id = db.Column(db.Integer, db.ForeignKey('diagnosis_reports.id'), nullable=True, index=True)
    run_id = db.Column(db.Integer, db.ForeignKey('runs.id'), nullable=True, index=True)
    meta_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, index=True)

    messages = db.relationship(
        'AIMessage',
        backref='conversation',
        cascade='all, delete-orphan',
        lazy='dynamic',
        order_by='AIMessage.created_at.asc()',
    )

    def get_meta(self):
        if not self.meta_json:
            return {}
        try:
            return json.loads(self.meta_json)
        except Exception:
            return {}

    def set_meta(self, data: dict):
        self.meta_json = json.dumps(data or {}, ensure_ascii=False)

    def to_dict(self, include_meta: bool = True):
        payload = {
            'id': self.id,
            'title': self.title,
            'scene': self.scene,
            'created_by': self.created_by,
            'report_id': self.report_id,
            'run_id': self.run_id,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None,
        }
        if include_meta:
            payload['meta'] = self.get_meta()
        return payload


class AIMessage(db.Model):
    """AI 助手会话消息。"""
    __tablename__ = 'ai_messages'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('ai_conversations.id'), nullable=False, index=True)
    role = db.Column(db.String(16), nullable=False, index=True)
    content = db.Column(db.Text, nullable=False)
    citations_json = db.Column(db.Text, nullable=True)
    used_context_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now, index=True)

    def get_citations(self):
        if not self.citations_json:
            return []
        try:
            data = json.loads(self.citations_json)
        except Exception:
            return []
        return data if isinstance(data, list) else []

    def set_citations(self, data):
        self.citations_json = json.dumps(data or [], ensure_ascii=False)

    def get_used_context(self):
        if not self.used_context_json:
            return {}
        try:
            data = json.loads(self.used_context_json)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def set_used_context(self, data: dict):
        self.used_context_json = json.dumps(data or {}, ensure_ascii=False)

    def to_dict(self):
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'role': self.role,
            'content': self.content,
            'citations': self.get_citations(),
            'used_context': self.get_used_context(),
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
        }
