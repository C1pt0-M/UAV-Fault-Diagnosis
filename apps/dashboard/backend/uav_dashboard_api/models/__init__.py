# 导入所有模型
from .ai_conversation import AIConversation, AIMessage
from .asset import Asset
from .diagnosis_report import DiagnosisReport
from .log import Log
from .run_record import RunRecord
from .telemetry_session import TelemetryFaultEvent, TelemetrySession
from .user import User

__all__ = [
    'Asset',
    'AIConversation',
    'AIMessage',
    'DiagnosisReport',
    'Log',
    'RunRecord',
    'TelemetryFaultEvent',
    'TelemetrySession',
    'User',
]
