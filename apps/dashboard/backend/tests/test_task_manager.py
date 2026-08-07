from __future__ import annotations

import pytest
from flask import Flask

from uav_dashboard_api import db
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.services import task_manager


def _app(tmp_path, workers: int) -> Flask:
    app = Flask(f"task-test-{workers}")
    app.config.update(
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{tmp_path / f'run-{workers}.db'}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        TASK_MAX_WORKERS=workers,
    )
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return app


def test_executor_is_scoped_to_app_and_config(tmp_path):
    app_one = _app(tmp_path, 1)
    app_two = _app(tmp_path, 3)
    try:
        executor_one = task_manager.get_executor(app_one)
        executor_two = task_manager.get_executor(app_two)

        assert executor_one is not executor_two
        assert executor_one._max_workers == 1
        assert executor_two._max_workers == 3
    finally:
        task_manager.shutdown_executor(app_one)
        task_manager.shutdown_executor(app_two)
        with app_one.app_context():
            db.session.remove()


def test_mark_run_progress_rejects_non_finite_and_clamps_range(tmp_path):
    app = _app(tmp_path, 1)
    with app.app_context():
        run = RunRecord(task="offline_multiclass", status="running")
        db.session.add(run)
        db.session.commit()

        task_manager.mark_run_progress(run.id, 2.0)
        assert run.progress == 1.0

        with pytest.raises(ValueError, match="有限数字"):
            task_manager.mark_run_progress(run.id, float("nan"))

        db.session.remove()


def test_run_record_hides_traceback_unless_server_paths_are_exposed(tmp_path):
    app = _app(tmp_path, 1)
    with app.app_context():
        run = RunRecord(
            task="offline_multiclass",
            status="error",
            error="推理失败",
            error_detail=r"Traceback: D:\private\model.py",
        )
        db.session.add(run)
        db.session.commit()

        app.config["EXPOSE_SERVER_PATHS"] = False
        assert "error_detail" not in run.to_dict()

        app.config["EXPOSE_SERVER_PATHS"] = True
        assert run.to_dict()["error_detail"] == r"Traceback: D:\private\model.py"

        db.session.remove()


def test_submit_failure_marks_run_error(tmp_path, monkeypatch):
    app = _app(tmp_path, 1)
    with app.app_context():
        run = RunRecord(task="offline_multiclass", status="queued")
        db.session.add(run)
        db.session.commit()
        run_id = run.id

    def fail_submit(*args, **kwargs):
        raise RuntimeError("executor unavailable")

    class FailingExecutor:
        submit = fail_submit

    monkeypatch.setattr(task_manager, "get_executor", lambda app: FailingExecutor())

    with pytest.raises(RuntimeError, match="executor unavailable"):
        task_manager.submit_background_run(app, run_id, lambda _run_id: {})

    with app.app_context():
        run = db.session.get(RunRecord, run_id)
        assert run.status == "error"
        assert run.error == "任务提交失败: executor unavailable"
        assert "RuntimeError: executor unavailable" in (run.error_detail or "")
        db.session.remove()
