# 导入所有模型
from .ai_conversation import AIConversation, AIMessage
from .asset import Asset
from .diagnosis_report import DiagnosisReport
from .drone import Drone
from .fault import Fault
from .log import Log
from .maintenance import Maintenance
from .report import Report
from .run_record import RunRecord
from .telemetry_session import TelemetryFaultEvent, TelemetrySession
from .user import User

__all__ = [
    'Asset',
    'AIConversation',
    'AIMessage',
    'DiagnosisReport',
    'Drone',
    'Fault',
    'Log',
    'Maintenance',
    'Report',
    'RunRecord',
    'TelemetryFaultEvent',
    'TelemetrySession',
    'User',
]
