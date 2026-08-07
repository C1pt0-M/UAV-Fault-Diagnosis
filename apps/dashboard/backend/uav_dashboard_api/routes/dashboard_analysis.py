from __future__ import annotations

from flask import jsonify, request

from uav_dashboard_api import db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.routes.dashboard_asset_input import (
    AssetNotPreparedError,
    _read_fault_prob_curve_csv,
    _read_fault_prob_curve_json,
    _resolve_asset_telemetry_session,
    _resolved_csv_path,
)
from uav_dashboard_api.services.analysis import build_fields_response, build_timeseries_response
from uav_dashboard_api.utils.jwt_utils import token_required, is_admin

def _get_asset_or_404(asset_id):
    asset = db.session.get(Asset, asset_id)
    if asset is None:
        return None, (jsonify({"success": False, "message": "资产不存在"}), 404)
    return asset, None


def _asset_visible(asset, user) -> bool:
    return is_admin(user) or asset.created_by == getattr(user, 'id', None)


def _assert_asset_visible(asset, user):
    if not _asset_visible(asset, user):
        return jsonify({"success": False, "message": "资产不存在"}), 404
    return None


def get_analysis_fields(current_user):
    asset_id = request.args.get("asset_id", type=int)
    if not asset_id:
        return jsonify({"success": False, "message": "asset_id 为必填"}), 400
    asset = db.session.get(Asset, asset_id)
    if asset is None:
        return jsonify({"success": False, "message": "资产不存在"}), 404
    denied = _assert_asset_visible(asset, current_user)
    if denied:
        return denied

    try:
        csv_path = _resolved_csv_path(asset, strict_prepare=True)
    except AssetNotPreparedError:
        return jsonify(
            {
                "success": False,
                "message": "asset_not_prepared",
                "prepare_endpoint": f"/api/assets/{asset.id}/prepare",
            }
        ), 409

    data = build_fields_response(csv_path)
    return jsonify({"success": True, "data": data})

def get_analysis_timeseries(current_user):
    asset_id = request.args.get("asset_id", type=int)
    fields_raw = request.args.get("fields", "").strip()
    start_t = request.args.get("start_t", type=float)
    end_t = request.args.get("end_t", type=float)
    max_points = request.args.get("max_points", default=2000, type=int)

    if not asset_id:
        return jsonify({"success": False, "message": "asset_id 为必填"}), 400
    fields = [item.strip() for item in fields_raw.split(",") if item.strip()]
    if not fields:
        return jsonify({"success": False, "message": "fields 不能为空"}), 400

    asset = db.session.get(Asset, asset_id)
    if asset is None:
        return jsonify({"success": False, "message": "资产不存在"}), 404
    denied = _assert_asset_visible(asset, current_user)
    if denied:
        return denied

    try:
        csv_path = _resolved_csv_path(asset, strict_prepare=True)
    except AssetNotPreparedError:
        return jsonify(
            {
                "success": False,
                "message": "asset_not_prepared",
                "prepare_endpoint": f"/api/assets/{asset.id}/prepare",
            }
        ), 409

    data = build_timeseries_response(
        csv_path=csv_path,
        fields=fields,
        start_t=start_t,
        end_t=end_t,
        max_points=max_points,
    )
    return jsonify({"success": True, "data": data})

def get_fault_prob_curve(current_user):
    asset_id = request.args.get("asset_id", type=int)
    if not asset_id:
        return jsonify({"success": False, "message": "asset_id 为必填"}), 400
    asset = db.session.get(Asset, asset_id)
    if asset is None:
        return jsonify({"success": False, "message": "资产不存在"}), 404
    denied = _assert_asset_visible(asset, current_user)
    if denied:
        return denied

    session = _resolve_asset_telemetry_session(asset)
    if session is None:
        return jsonify(
            {
                "success": True,
                "data": [],
                "meta": {
                    "source": "telemetry_session",
                    "available": False,
                    "reason": "asset_has_no_monitoring_session",
                },
            }
        )

    meta = session.get_meta()
    curve = _read_fault_prob_curve_json(meta.get("fault_prob_curve_path"))
    source = "fault_prob_curve_file"
    if not curve:
        curve = _read_fault_prob_curve_csv(session.csv_path)
        source = "session_csv"

    return jsonify(
        {
            "success": True,
            "data": curve,
            "meta": {
                "source": source,
                "available": bool(curve),
                "reason": None if curve else "session_has_no_fault_prob_series",
                "session_id": session.id,
            },
        }
    )


def register_routes(bp):
    bp.add_url_rule(
        "/analysis/fields", endpoint="get_analysis_fields", view_func=token_required(get_analysis_fields), methods=["GET"]
    )
    bp.add_url_rule(
        "/analysis/timeseries", endpoint="get_analysis_timeseries", view_func=token_required(get_analysis_timeseries), methods=["GET"]
    )
    bp.add_url_rule(
        "/analysis/fault_prob_curve", endpoint="get_fault_prob_curve", view_func=token_required(get_fault_prob_curve), methods=["GET"]
    )
