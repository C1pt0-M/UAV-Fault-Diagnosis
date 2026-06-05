from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import traceback

from flask import current_app

from app import db
from app.models.run_record import RunRecord

_executor = None


def get_executor():
    global _executor
    if _executor is None:
        max_workers = int(current_app.config.get('TASK_MAX_WORKERS', 2))
        _executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='dashboard-runner')
    return _executor


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
    run.progress = float(progress)
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
    executor = get_executor()
    executor.submit(_run_worker_with_app, app, run_id, worker_fn, args, kwargs)


def _run_worker_with_app(app, run_id: int, worker_fn, args, kwargs):
    with app.app_context():
        try:
            mark_run_running(run_id)
            result = worker_fn(run_id, *args, **kwargs)
            mark_run_success(run_id, result if result is not None else {})
        except Exception as exc:
            detail = traceback.format_exc()
            mark_run_error(run_id, str(exc), detail)
