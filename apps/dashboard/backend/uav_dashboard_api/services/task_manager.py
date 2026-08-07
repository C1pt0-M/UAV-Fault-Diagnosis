from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import math
import threading
import traceback

from flask import current_app

from uav_dashboard_api import db
from uav_dashboard_api.models.run_record import RunRecord

_EXECUTOR_EXTENSION_KEY = 'dashboard_task_executor'
_executor_lock = threading.Lock()


def get_executor(app=None):
    """Return the executor owned by one Flask application instance."""
    app = app or current_app._get_current_object()
    with _executor_lock:
        executor = app.extensions.get(_EXECUTOR_EXTENSION_KEY)
        if executor is None:
            max_workers = int(app.config.get('TASK_MAX_WORKERS', 2))
            if max_workers <= 0:
                raise ValueError('TASK_MAX_WORKERS 必须大于 0')
            executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix='dashboard-runner',
            )
            app.extensions[_EXECUTOR_EXTENSION_KEY] = executor
        return executor


def shutdown_executor(app, *, wait=True):
    """Stop the executor owned by an application, mainly for tests and app shutdown."""
    with _executor_lock:
        executor = app.extensions.pop(_EXECUTOR_EXTENSION_KEY, None)
    if executor is not None:
        executor.shutdown(wait=wait, cancel_futures=not wait)


def mark_run_running(run_id: int):
    run = db.session.get(RunRecord, run_id)
    if run is None:
        return
    run.status = 'running'
    run.progress = max(run.progress or 0.0, 0.01)
    run.started_at = datetime.now()
    run.error = None
    run.error_detail = None
    db.session.commit()


def mark_run_progress(run_id: int, progress: float):
    run = db.session.get(RunRecord, run_id)
    if run is None:
        return
    value = float(progress)
    if not math.isfinite(value):
        raise ValueError('任务进度必须是有限数字')
    run.progress = min(1.0, max(0.0, value))
    db.session.commit()


def mark_run_success(run_id: int, result: dict | list | None):
    run = db.session.get(RunRecord, run_id)
    if run is None:
        return
    run.status = 'success'
    run.progress = 1.0
    run.set_result(result)
    run.error = None
    run.error_detail = None
    run.finished_at = datetime.now()
    db.session.commit()


def mark_run_error(run_id: int, error: str, detail: str | None = None):
    run = db.session.get(RunRecord, run_id)
    if run is None:
        return
    run.status = 'error'
    run.error = error
    run.error_detail = detail
    run.finished_at = datetime.now()
    db.session.commit()


def submit_background_run(app, run_id: int, worker_fn, *args, **kwargs):
    """提交后台任务，worker_fn 必须返回可JSON序列化对象。"""
    try:
        executor = get_executor(app)
        executor.submit(_run_worker_with_app, app, run_id, worker_fn, args, kwargs)
    except Exception as exc:
        with app.app_context():
            mark_run_error(run_id, f'任务提交失败: {exc}', traceback.format_exc())
        raise


def _run_worker_with_app(app, run_id: int, worker_fn, args, kwargs):
    with app.app_context():
        try:
            mark_run_running(run_id)
            result = worker_fn(run_id, *args, **kwargs)
            mark_run_success(run_id, result if result is not None else {})
        except Exception as exc:
            detail = traceback.format_exc()
            mark_run_error(run_id, str(exc), detail)
