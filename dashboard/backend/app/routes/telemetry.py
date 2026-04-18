"""实时遥测采集与故障检测 API。"""

from flask import Blueprint, current_app, jsonify, request

from app import db
from app.utils.jwt_utils import token_required

bp = Blueprint('telemetry', __name__)


def _get_manager():
    from app.services.telemetry_manager import telemetry_manager
    return telemetry_manager


def _ok(data=None, message='ok', status=200):
    return jsonify({'success': True, 'message': message, 'data': data}), status


def _error(message, status=400, data=None):
    return jsonify({'success': False, 'message': message, 'data': data}), status


def _telemetry_capabilities() -> dict:
    allowed_backends = list(current_app.config.get('TELEMETRY_ALLOWED_BACKENDS') or ())
    return {
        'enabled': bool(current_app.config.get('ENABLE_TELEMETRY', True)),
        'allowed_backends': allowed_backends,
        'px4_enabled': 'px4' in allowed_backends,
        'rfly_enabled': 'rfly' in allowed_backends,
        'blind_test_enabled': bool(current_app.config.get('ENABLE_BLIND_TEST', False)),
    }


# ── 启动数据采集 ──────────────────────────────────────────────

@bp.route('/start-collection', methods=['POST'])
@token_required
def start_collection(current_user):
    """启动遥测采集。"""
    data = request.get_json(silent=True) or {}
    drone_id = data.get('drone_id')
    session_type = data.get('session_type', 'simulation')
    sim_time_raw = data.get('sim_time', 120)
    allowed_backend_list = list(current_app.config.get('TELEMETRY_ALLOWED_BACKENDS') or ())
    allowed_backends = set(allowed_backend_list)
    default_backend = allowed_backend_list[0] if allowed_backend_list else 'replay'
    backend = str(data.get('backend', default_backend) or default_backend).strip().lower()
    replay_path = data.get('replay_path')
    rfly_options = {
        'px4mavctrl_path': data.get('rfly_px4mavctrl_path'),
    }

    try:
        sim_time = float(sim_time_raw)
    except (TypeError, ValueError):
        sim_time = 0.0 if backend == 'replay' else None
    if backend != 'replay':
        if sim_time is None:
            return _error('sim_time 必须为数值')
        if sim_time <= 0:
            return _error('sim_time 必须大于 0')
    else:
        # replay 模式按整段 CSV 回放，sim_time 仅用于兼容字段。
        if sim_time is None:
            sim_time = 0.0
    if backend not in {'px4', 'rfly', 'replay'}:
        return _error('backend 仅支持 px4、rfly 或 replay')
    if backend not in allowed_backends:
        return _error('backend_not_allowed')
    if backend == 'replay' and not str(replay_path or '').strip():
        return _error('replay backend 需要 replay_path')

    mgr = _get_manager()
    try:
        result = mgr.start_collection(
            drone_id=drone_id,
            session_type=session_type,
            sim_time=sim_time,
            backend=backend,
            replay_path=replay_path,
            rfly_options=rfly_options,
        )
        return _ok(result, '数据采集已启动')
    except RuntimeError as e:
        return _error(str(e))


# ── 停止数据采集 ──────────────────────────────────────────────

@bp.route('/stop-collection', methods=['POST'])
@token_required
def stop_collection(current_user):
    """停止采集，数据入库。"""
    mgr = _get_manager()
    try:
        result = mgr.stop_collection()
        return _ok(result, '数据采集已停止')
    except RuntimeError as e:
        return _error(str(e))


# ── 启动实时检测 ──────────────────────────────────────────────

@bp.route('/start-detection', methods=['POST'])
@token_required
def start_detection(current_user):
    """在运行中的采集会话上启动实时故障检测。"""
    data = request.get_json(silent=True) or {}
    requested_model = data.get('model')
    if requested_model is not None and not isinstance(requested_model, str):
        return _error('model 必须为字符串')
    mgr = _get_manager()
    try:
        result = mgr.start_detection(requested_model=requested_model)
        return _ok(result, '实时检测已启动')
    except RuntimeError as e:
        return _error(str(e))


# ── 停止实时检测 ──────────────────────────────────────────────

@bp.route('/stop-detection', methods=['POST'])
@token_required
def stop_detection(current_user):
    """停止实时检测（采集继续运行）。"""
    mgr = _get_manager()
    try:
        result = mgr.stop_detection()
        return _ok(result, '实时检测已停止')
    except RuntimeError as e:
        return _error(str(e))


# ── 获取运行状态 ──────────────────────────────────────────────

@bp.route('/status', methods=['GET'])
@token_required
def get_status(current_user):
    """获取当前采集/检测运行状态。"""
    mgr = _get_manager()
    payload = mgr.get_status()
    payload['telemetry_capabilities'] = _telemetry_capabilities()
    return _ok(payload)


# ── 获取实时数据 ──────────────────────────────────────────────

@bp.route('/realtime-data', methods=['GET'])
@token_required
def get_realtime_data(current_user):
    """获取实时遥测数据缓冲（用于前端图表）。"""
    mgr = _get_manager()
    return _ok(mgr.get_realtime_data())


# ── 获取会话列表 ──────────────────────────────────────────────

@bp.route('/sessions', methods=['GET'])
@token_required
def get_sessions(current_user):
    """获取历史遥测会话列表。"""
    from app.models.telemetry_session import TelemetrySession

    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    status = request.args.get('status')

    query = TelemetrySession.query
    if status:
        query = query.filter_by(status=status)

    pagination = query.order_by(TelemetrySession.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return _ok({
        'items': [s.to_dict() for s in pagination.items],
        'total': pagination.total,
        'page': page,
        'per_page': per_page,
    })


# ── 获取会话详情（含故障事件）────────────────────────────────

@bp.route('/sessions/<session_id>', methods=['GET'])
@token_required
def get_session_detail(current_user, session_id):
    """获取指定会话详情及其故障事件。"""
    from app.models.telemetry_session import TelemetrySession

    session = db.session.get(TelemetrySession, session_id)
    if session is None:
        return _error('会话不存在', 404)
    return _ok(session.to_dict())


# ── 盲测模式 ─────────────────────────────────────────────────

@bp.route('/blind-test/start', methods=['POST'])
@token_required
def start_blind_test(current_user):
    """启动盲测模式（随机故障，不泄露 ground truth）。"""
    if not current_app.config.get('ENABLE_BLIND_TEST', False):
        return _error('blind_test_disabled', status=403)
    data = request.get_json(silent=True) or {}
    drone_id = data.get('drone_id')
    sim_time = data.get('sim_time', 120)

    if drone_id is not None and not isinstance(drone_id, str):
        return _error('drone_id 必须为字符串')
    try:
        sim_time = float(sim_time)
    except (TypeError, ValueError):
        return _error('sim_time 必须为数值')
    if sim_time <= 0:
        return _error('sim_time 必须大于 0')

    mgr = _get_manager()
    try:
        result = mgr.start_blind_test(drone_id=drone_id, sim_time=sim_time)
        return _ok(result, '盲测已启动')
    except RuntimeError as e:
        return _error(str(e))


@bp.route('/blind-test/reveal', methods=['GET'])
@token_required
def reveal_blind_test(current_user):
    """揭晓盲测结果（仅停止后可调用）。"""
    mgr = _get_manager()
    try:
        result = mgr.get_blind_test_reveal()
        return _ok(result, '盲测结果揭晓')
    except RuntimeError as e:
        return _error(str(e))
