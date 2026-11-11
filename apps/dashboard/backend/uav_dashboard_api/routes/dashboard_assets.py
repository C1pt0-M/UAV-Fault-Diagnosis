from __future__ import annotations

from datetime import datetime
import uuid
from pathlib import Path

from flask import current_app, jsonify, request, send_file
from sqlalchemy import or_
from werkzeug.utils import secure_filename

from uav_dashboard_api import db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.services.flight_log_asset_input import (
    asset_payload_with_quality,
    build_csv_quality_meta,
    sha256_file,
)
from uav_dashboard_api.services.bridge import run_ulg_to_csv
from uav_dashboard_api.services.task_manager import mark_run_progress, submit_background_run
from uav_dashboard_api.utils.jwt_utils import token_required, is_admin
from uav_dashboard_api.utils.path_store import resolve_local_path, serialize_local_path


ALLOWED_EXTS = {".ulg": "ulg", ".csv": "csv"}

def _get_asset_or_404(asset_id: int, user=None):
    asset = db.session.get(Asset, asset_id)
    if asset is None:
        return None, (jsonify({"success": False, "message": "资产不存在"}), 404)
    if user is not None and not is_admin(user) and asset.created_by != getattr(user, 'id', None):
        return None, (jsonify({"success": False, "message": "资产不存在"}), 404)
    return asset, None

def _prepare_ulg_worker(run_id: int, asset_id: int):
    asset = db.session.get(Asset, asset_id)
    if asset is None:
        raise RuntimeError("资产不存在")

    mark_run_progress(run_id, 0.1)
    output_dir = Path(current_app.config["DERIVED_DIR"]) / f"asset_{asset.id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{Path(asset.filename).stem}_prepared.csv"

    bridge_info = run_ulg_to_csv(
        resolve_local_path(asset.stored_path, current_app.config),
        str(output_csv),
    )
    if not output_csv.exists():
        raise RuntimeError("转换完成但未生成CSV")
    mark_run_progress(run_id, 0.85)

    meta = asset.get_meta()
    meta["prepared_csv_path"] = str(output_csv)
    meta["prepared_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta.update(build_csv_quality_meta(output_csv))
    asset.set_meta(meta)
    db.session.commit()

    try:
        import pandas as pd

        sample_df = pd.read_csv(output_csv, nrows=1000)
        columns = list(sample_df.columns)
        n_rows = sum(1 for _ in output_csv.open("r", encoding="utf-8")) - 1
    except Exception:
        columns = []
        n_rows = None

    return {
        "asset_id": asset.id,
        "output_csv": serialize_local_path(output_csv, current_app.config),
        "n_rows": n_rows,
        "columns": columns,
        "bridge": bridge_info,
    }

def upload_asset(current_user):
    if "file" not in request.files:
        return jsonify({"success": False, "message": "缺少file字段"}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"success": False, "message": "未选择文件"}), 400

    original_filename = Path(file.filename).name
    ext = Path(original_filename).suffix.lower()
    kind = ALLOWED_EXTS.get(ext)
    if kind is None:
        return jsonify({"success": False, "message": "仅支持 .ulg 或 .csv 文件"}), 400

    save_dir = Path(current_app.config["UPLOAD_DIR"]) / kind
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = secure_filename(Path(original_filename).stem) or "asset"
    stored_name = (
        f"{datetime.now().strftime('%Y%m%d%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}_{safe_stem}{ext}"
    )
    stored_path = save_dir / stored_name
    try:
        file.save(stored_path)
        sha256 = sha256_file(stored_path)

        existing_asset = Asset.query.filter_by(sha256=sha256).first()
        if existing_asset:
            stored_path.unlink(missing_ok=True)
            return jsonify(
                {
                    "success": False,
                    "message": "文件已存在(文件指纹雷同)，跳过上传",
                    "data": asset_payload_with_quality(existing_asset),
                }
            ), 409

        asset = Asset(
            filename=original_filename,
            kind=kind,
            stored_path=serialize_local_path(stored_path, current_app.config),
            size_bytes=stored_path.stat().st_size,
            sha256=sha256,
            created_by=getattr(current_user, "id", None),
        )
        meta = {}
        if kind == "csv":
            meta["prepared_csv_path"] = str(stored_path)
            meta.update(build_csv_quality_meta(stored_path))
        asset.set_meta(meta)
        db.session.add(asset)
        db.session.commit()
    except Exception:
        db.session.rollback()
        stored_path.unlink(missing_ok=True)
        raise

    return jsonify(
        {
            "success": True,
            "message": "上传成功",
            "data": asset.to_dict(),
        }
    ), 201

def list_assets(current_user):
    kind = request.args.get("kind", "all")
    query = Asset.query
    if kind == "ts":
        query = query.filter(
            Asset.meta_json.like('%"source_type": "telemetry_session"%')
        )
    elif kind == "csv":
        query = query.filter(
            Asset.kind == "csv",
            or_(
                Asset.meta_json.is_(None),
                ~Asset.meta_json.like('%"source_type": "telemetry_session"%'),
            ),
        )
    elif kind == "ulg":
        query = query.filter_by(kind="ulg")
    if not is_admin(current_user):
        query = query.filter(Asset.created_by == current_user.id)
    assets = query.order_by(Asset.created_at.desc()).all()
    return jsonify(
        {
            "success": True,
            "data": [
                asset_payload_with_quality(item, fill_missing_quality=False)
                for item in assets
            ],
        }
    )

def get_asset(current_user, asset_id: int):
    asset, error = _get_asset_or_404(asset_id, current_user)
    if error:
        return error
    return jsonify({"success": True, "data": asset_payload_with_quality(asset)})

def download_asset(current_user, asset_id: int):
    asset, error = _get_asset_or_404(asset_id, current_user)
    if error:
        return error
    file_path = Path(resolve_local_path(asset.stored_path, current_app.config))
    if not file_path.exists():
        return jsonify({"success": False, "message": "文件不存在"}), 404
    return send_file(file_path, as_attachment=True, download_name=asset.filename)

def delete_asset(current_user, asset_id: int):
    asset, error = _get_asset_or_404(asset_id, current_user)
    if error:
        return error

    active_run = (
        RunRecord.query.filter(
            RunRecord.asset_id == asset.id, RunRecord.status.in_(["queued", "running"])
        )
        .order_by(RunRecord.created_at.desc())
        .first()
    )
    if active_run is not None:
        return jsonify(
            {
                "success": False,
                "message": "asset_has_active_runs",
                "run_id": active_run.id,
            }
        ), 409

    file_paths: list[Path] = []
    if asset.stored_path:
        file_paths.append(Path(resolve_local_path(asset.stored_path, current_app.config)))
    prepared_csv_path = asset.get_meta().get("prepared_csv_path")
    if prepared_csv_path:
        prepared_path = Path(resolve_local_path(prepared_csv_path, current_app.config))
        if prepared_path not in file_paths:
            file_paths.append(prepared_path)

    # 保留历史任务记录，但解除与资产的外键关联。
    RunRecord.query.filter_by(asset_id=asset.id).update(
        {"asset_id": None},
        synchronize_session=False,
    )
    db.session.delete(asset)
    db.session.commit()

    deleted_files = []
    missing_files = []
    failed_files = []
    for path in file_paths:
        try:
            if path.exists():
                path.unlink()
                deleted_files.append(str(path))
            else:
                missing_files.append(str(path))
        except Exception as exc:
            failed_files.append({"path": str(path), "error": str(exc)})

    return jsonify(
        {
            "success": True,
            "message": "资产已删除",
            "data": {
                "asset_id": asset_id,
                "deleted_files": deleted_files,
                "missing_files": missing_files,
                "failed_files": failed_files,
            },
        }
    )

def prepare_asset(current_user, asset_id: int):
    asset, error = _get_asset_or_404(asset_id)
    if error:
        return error
    if asset.kind != "ulg":
        return jsonify({"success": False, "message": "仅ULG文件需要prepare"}), 400

    csv_path = resolve_local_path(
        asset.get_meta().get("prepared_csv_path"),
        current_app.config,
    )
    if csv_path and Path(csv_path).exists():
        return jsonify(
            {
                "success": True,
                "message": "资产已prepare",
                "data": {"asset": asset.to_dict()},
            }
        )

    run = RunRecord(
        task="convert_ulg",
        status="queued",
        progress=0.0,
        asset_id=asset.id,
        created_by=getattr(current_user, "id", None),
    )
    run.set_params({"asset_id": asset.id, "filename": asset.filename})
    db.session.add(run)
    db.session.commit()

    submit_background_run(
        current_app._get_current_object(),
        run.id,
        _prepare_ulg_worker,
        asset.id,
    )
    return jsonify(
        {
            "success": True,
            "message": "prepare任务已提交",
            "data": run.to_dict(),
        }
    ), 202


def register_routes(bp):
    bp.add_url_rule(
        "/assets/upload", endpoint="upload_asset", view_func=token_required(upload_asset), methods=["POST"]
    )
    bp.add_url_rule(
        "/assets", endpoint="list_assets", view_func=token_required(list_assets), methods=["GET"]
    )
    bp.add_url_rule(
        "/assets/<int:asset_id>", endpoint="get_asset", view_func=token_required(get_asset), methods=["GET"]
    )
    bp.add_url_rule(
        "/assets/<int:asset_id>/download", endpoint="download_asset", view_func=token_required(download_asset), methods=["GET"]
    )
    bp.add_url_rule(
        "/assets/<int:asset_id>", endpoint="delete_asset", view_func=token_required(delete_asset), methods=["DELETE"]
    )
    bp.add_url_rule(
        "/assets/<int:asset_id>/prepare", endpoint="prepare_asset", view_func=token_required(prepare_asset), methods=["POST"]
    )
