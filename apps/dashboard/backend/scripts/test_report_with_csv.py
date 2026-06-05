#!/usr/bin/env python3
"""用指定 CSV 跑离线诊断并生成增强诊断报告。"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import bcrypt

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parents[2]
UAV_FAULT_ROOT = REPO_ROOT / "packages" / "uav_fault"
for candidate in (BACKEND_ROOT, UAV_FAULT_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from uav_dashboard_api import create_app, db
from uav_dashboard_api.config import TestingConfig
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.user import User
from uav_dashboard_api.services.bridge import run_offline_5class
from uav_dashboard_api.services.diagnosis_contract import build_diagnosis_payload
from uav_dashboard_api.services.diagnosis_report import (
    _build_pdf_lines,
    create_or_get_diagnosis_report,
)
from uav_dashboard_api.services.model_registry import resolve_registered_model
from uav_dashboard_api.services.severity_recommendation import (
    evaluate_severity_and_recommendations,
)
from uav_dashboard_api.utils.path_store import resolve_local_path
from uav_fault.online.decision_engine import TemporalDecisionEngine


def parse_args():
    parser = argparse.ArgumentParser(description="用 CSV 测试诊断报告模块")
    parser.add_argument(
        "--csv-path",
        required=True,
        help="输入 CSV 路径",
    )
    parser.add_argument(
        "--model",
        default="transfer3_cnnlstm_normalproto",
        help="离线诊断模型名",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "runtime" / "report_test"),
        help="测试运行时目录",
    )
    return parser.parse_args()


def _count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8") as fp:
        return max(0, sum(1 for _ in fp) - 1)


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path).resolve()
    if not csv_path.exists():
        print(f"[ERROR] CSV 不存在: {csv_path}", file=sys.stderr)
        return 1

    runtime = Path(args.output_dir).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    db_path = runtime / "report_test.db"

    class LocalConfig(TestingConfig):
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_path}"
        RUNTIME_DIR = runtime
        UPLOAD_DIR = runtime / "uploads"
        DERIVED_DIR = runtime / "derived"
        RUNS_DIR = runtime / "runs"
        REPORTS_DIR = runtime / "reports"

    app = create_app(LocalConfig)
    with app.app_context():
        db.create_all()
        user = User.query.filter_by(username="report_tester").first()
        if user is None:
            hashed = bcrypt.hashpw(b"test123", bcrypt.gensalt()).decode("utf-8")
            user = User(
                username="report_tester",
                password=hashed,
                real_name="Report Tester",
                email="rt@example.com",
                role="admin",
                status="active",
            )
            db.session.add(user)
            db.session.commit()

        upload_dir = Path(app.config["UPLOAD_DIR"])
        upload_dir.mkdir(parents=True, exist_ok=True)
        stored = upload_dir / csv_path.name
        shutil.copy2(csv_path, stored)

        asset = Asset(
            filename=csv_path.name,
            kind="csv",
            stored_path=str(stored),
            size_bytes=stored.stat().st_size,
            sha256=hashlib.sha256(stored.read_bytes()).hexdigest(),
            created_by=user.id,
        )
        asset.set_meta({"prepared_csv_path": str(stored)})
        db.session.add(asset)
        db.session.flush()

        model_meta = resolve_registered_model(
            app.config,
            requested_name=args.model,
            capability="offline_detection",
        )
        if model_meta is None:
            print(f"[ERROR] 模型不可用: {args.model}", file=sys.stderr)
            return 1

        print("=== 模型信息 ===")
        print(
            json.dumps(
                {
                    "name": model_meta["name"],
                    "task": model_meta["task"],
                    "architecture": model_meta["architecture"],
                    "label_names": model_meta["label_names"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        run_dir = Path(app.config["RUNS_DIR"]) / f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        output_json = run_dir / "result.json"

        print("\n=== 执行离线推理 ===")
        prediction = run_offline_5class(
            csv_path=str(stored),
            model_dir=model_meta["path"],
            output_json=str(output_json),
        )
        pred = prediction.get("prediction") or {}
        print("预测结果:", json.dumps(pred, ensure_ascii=False, indent=2))
        print("概率分布:", json.dumps(prediction.get("probs", {}), ensure_ascii=False))

        severity_bundle = evaluate_severity_and_recommendations(
            csv_path=str(stored),
            fault_label=str(pred.get("label", "")).strip(),
            confidence=pred.get("confidence"),
        )
        print(
            "\n严重度与建议:",
            json.dumps(severity_bundle, ensure_ascii=False, indent=2),
        )

        diagnosis = build_diagnosis_payload(
            source="offline",
            label=str(pred.get("label", "")).strip() or "unknown",
            confidence=pred.get("confidence"),
            severity=(severity_bundle.get("severity") or {}).get("level"),
            recommendations=severity_bundle.get("recommendations", []),
            model={
                "name": model_meta["name"],
                "task": model_meta.get("task"),
                "architecture": model_meta.get("architecture"),
                "label_names": model_meta.get("label_names") or [],
            },
            time_range=None,
        )
        decision_engine = TemporalDecisionEngine()
        offline_decision = decision_engine.attach_offline_diagnosis(
            {
                "fault_label": diagnosis.get("label"),
                "severity": severity_bundle.get("severity"),
                "recommendations": severity_bundle.get("recommendations", []),
                "evidence": (severity_bundle.get("severity") or {}).get("evidence", []),
            }
        )
        diagnosis["decision"] = {
            "episode_id": offline_decision.episode_id,
            "state": offline_decision.state,
            "warning_stage": offline_decision.warning_stage,
            "risk_score": offline_decision.risk_score,
            "risk_trend": offline_decision.risk_trend,
            "warning_reason": offline_decision.warning_reason,
            "lead_time_s": offline_decision.lead_time_s,
            "fault_label": offline_decision.fault_label,
            "severity": offline_decision.severity,
            "evidence_summary": offline_decision.evidence_summary,
            "recommendations": offline_decision.recommendations,
        }

        debug = prediction.get("debug") if isinstance(prediction.get("debug"), dict) else {}
        debug["sample_count"] = _count_csv_rows(stored)
        debug["feature_dim"] = model_meta.get("feature_count") or debug.get("in_channels")

        run = RunRecord(
            task="offline_multiclass",
            status="success",
            progress=1.0,
            asset_id=asset.id,
            created_by=user.id,
        )
        run.set_params({"asset_id": asset.id, "model": args.model})
        run.set_result(
            {
                "asset_id": asset.id,
                "asset_kind": asset.kind,
                "input_csv": str(stored),
                "model": model_meta["name"],
                "prediction": prediction.get("prediction"),
                "probs": prediction.get("probs"),
                "labels": prediction.get("labels"),
                "topk": prediction.get("topk"),
                "severity": severity_bundle.get("severity"),
                "recommendations": severity_bundle.get("recommendations", []),
                "recommendation_meta": severity_bundle.get("recommendation_meta", {}),
                "diagnosis": diagnosis,
                "unified_decision": diagnosis["decision"],
                "debug": debug,
            }
        )
        db.session.add(run)
        db.session.commit()

        report, created = create_or_get_diagnosis_report(run, created_by=user.id)
        summary = report.get_summary()
        pdf_path = Path(resolve_local_path(report.pdf_path, app.config))

        print("\n=== 诊断报告 ===")
        print(f"report_id: {report.id}")
        print(f"新建报告: {created}")
        print(f"结论标签: {report.prediction_label}")
        print(f"置信度: {report.confidence:.4f}")
        print(f"PDF 路径: {pdf_path}")
        print(f"PDF 已生成: {pdf_path.exists()}")

        print("\n=== 增强字段 ===")
        for key in (
            "data_quality",
            "traceability",
            "model_card",
            "report_confidence",
            "review",
        ):
            print(f"--- {key} ---")
            print(json.dumps(summary.get(key, {}), ensure_ascii=False, indent=2))

        print("\n=== PDF 内容预览（前 45 行）===")
        for line in _build_pdf_lines(summary, report.get_probs())[:45]:
            print(line)

        result_path = runtime / "latest_report_result.json"
        result_path.write_text(
            json.dumps(
                {
                    "csv_path": str(csv_path),
                    "asset_id": asset.id,
                    "run_id": run.id,
                    "report_id": report.id,
                    "prediction_label": report.prediction_label,
                    "confidence": report.confidence,
                    "probs": report.get_probs(),
                    "summary": summary,
                    "pdf_path": str(pdf_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n完整结果已保存: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
