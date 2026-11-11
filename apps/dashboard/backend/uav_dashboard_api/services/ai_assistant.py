from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime
import json
import time

from flask import current_app
from sqlalchemy.exc import OperationalError

from uav_dashboard_api import db
from uav_dashboard_api.models.ai_conversation import AIConversation, AIMessage
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.services.ai_provider import AIProviderError, OpenAICompatibleProvider
from uav_dashboard_api.services.ai_settings import ai_settings_service
from uav_dashboard_api.services.knowledge_base import KnowledgeContextResult, knowledge_base


_AI_CHAT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ai-chat")


SCENE_LABELS = {
    "general": "通用问答",
    "report_analysis": "报告分析",
}

SCENE_DEFAULT_QUESTIONS = {
    "report_analysis": (
        "请围绕当前报告解释：结论、证据、风险、建议动作、不确定性与限制、是否需要人工确认。"
        "只能依据报告字段作答，不要重新生成独立诊断。"
    ),
}

FALLBACK_LABEL_TEXT_MAP = {
    "normal": "状态正常",
    "battery_failure": "电池故障",
    "motor_failure": "电机故障",
    "motor": "电机故障",
    "prop_imbalance": "螺旋桨故障",
    "propeller": "螺旋桨故障",
    "sensor": "传感器故障",
    "thrust_loss": "推力异常",
}

FALLBACK_SEVERITY_TEXT_MAP = {
    "none": "风险可控",
    "light": "轻度风险",
    "medium": "中度风险",
    "heavy": "高风险",
    "unknown": "待人工确认",
}

FALLBACK_PRIORITY_TEXT_MAP = {
    "P0": "立即处理",
    "P1": "优先处理",
    "P2": "安排检查",
    "P3": "持续观察",
}

FALLBACK_METRIC_TEXT_MAP = {
    "prediction_confidence": "模型判断置信度",
    "motor_imbalance_ratio": "电机输出均衡性异常",
    "esc_rpm_imbalance_ratio": "ESC 转速一致性异常",
    "battery_warning": "电池告警",
    "battery_warning_max": "电池告警触发",
    "battery_faults_max": "电池故障计数偏高",
    "battery_remaining_min": "剩余电量偏低",
    "battery_voltage_drop_ratio": "电压跌落偏大",
    "gyro_vibration_p95": "陀螺振动偏高",
    "accel_vibration_p95": "加速度振动偏高",
    "fault_intensity_mean": "故障强度均值偏高",
    "fault_severity_p95": "高分位故障严重度偏高",
    "thrust_saturation_ratio": "推力饱和比例偏高",
    "unalloc_thrust_z_p95": "未分配推力异常",
}

SYSTEM_PROMPT = """你是基于数据驱动的无人机故障智能诊断系统中的 AI 专家助手，在报告分析场景中充当“报告解释器”。
回答要求：
1. 优先依据给定的系统上下文、诊断报告、任务记录和补充背景资料回答，但要把这些信息自然融入判断，不要逐条复述。
2. 不要编造项目内不存在的事实；无法确认时直接说明“无法从当前资料确认”。
3. 输出使用中文，技术术语保留英文。
4. 报告分析场景默认使用六段式自然回答：
第一段以“结论：”开头，概括报告诊断结论；
第二段以“证据：”开头，说明模型判断、规则证据和关键指标；
第三段以“风险：”开头，说明严重度、报告可信度与数据质量对风险的影响；
第四段以“建议动作：”开头，说明报告中的处置建议；
第五段以“不确定性与限制：”开头，说明模型限制、数据缺口或不足以支持强结论的情况；
第六段以“需要人工确认：”开头，说明复核状态与建议的人工确认事项。
5. 非报告分析场景可继续使用“直接结论 / 原因解释 / 风险与建议”三段式。
6. 除非用户明确要求，否则不要使用 Markdown 标题或粗体，也不要复述内部字段名或项目内部标签。
7. 不要在正文中直接写“参考资料”“知识片段 1/2”等内部标签；如需说明依据，用自然语言概括即可。
8. 如果上下文里有概率、严重度、数据质量、追溯信息、推荐动作，可以自然引用并解释，但不要把原始 JSON 或整段检索文本原样搬进回答。
9. 当数据质量较差或关键字段缺失时，要明确提示“不足以支持强结论”，不要编造传感器、工况、模型指标或维修结果。"""


def _fallback_label_text(label: str | None) -> str:
    raw = str(label or "").strip()
    return FALLBACK_LABEL_TEXT_MAP.get(raw, raw or "待确认异常")


def _fallback_severity_text(level: str | None) -> str:
    raw = str(level or "unknown").strip()
    return FALLBACK_SEVERITY_TEXT_MAP.get(raw, FALLBACK_SEVERITY_TEXT_MAP["unknown"])


def _fallback_metric_text(metric: str | None) -> str:
    raw = str(metric or "").strip()
    return FALLBACK_METRIC_TEXT_MAP.get(raw, raw.replace("_", " ") or "关键指标")


def _fallback_percent(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "暂无"


def _fallback_recommendation_text(recommendations) -> str:
    if not isinstance(recommendations, list):
        return ""

    rows = []
    for item in recommendations[:3]:
        if isinstance(item, str):
            title = item.strip()
            priority = "P2"
        elif isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            priority = str(item.get("priority") or "P2").strip().upper()
        else:
            continue

        if title:
            rows.append(
                f"{FALLBACK_PRIORITY_TEXT_MAP.get(priority, FALLBACK_PRIORITY_TEXT_MAP['P2'])}：{title}"
            )

    return "；".join(rows)


def _fallback_evidence_text(summary: dict) -> str:
    metrics = []
    severity = summary.get("severity") if isinstance(summary, dict) else None
    if isinstance(severity, dict):
        evidence = severity.get("evidence")
        if isinstance(evidence, list):
            for item in evidence[:3]:
                if not isinstance(item, dict):
                    continue
                metric = str(item.get("metric") or "").strip()
                if metric:
                    metrics.append(_fallback_metric_text(metric))

    decision = summary.get("decision") if isinstance(summary, dict) else None
    if isinstance(decision, dict):
        evidence_summary = decision.get("evidence_summary")
        top_evidence = (
            evidence_summary.get("top_evidence")
            if isinstance(evidence_summary, dict)
            else None
        )
        if isinstance(top_evidence, list):
            for item in top_evidence[:3]:
                if not isinstance(item, dict):
                    continue
                metric = str(item.get("metric") or "").strip()
                label = _fallback_metric_text(metric)
                if metric and label not in metrics:
                    metrics.append(label)

    return "、".join(metrics[:3])


@dataclass
class SceneContext:
    scene: str
    report: DiagnosisReport | None
    run: RunRecord | None
    citations: list[dict]
    blocks: list[str]


class AIAssistantService:
    def __init__(self, app=None):
        self.app = app

    def init_app(self, app):
        self.app = app
        knowledge_base.init_app(app)

    def _config(self) -> dict:
        return ai_settings_service.effective_config((self.app or current_app).config)

    def _provider_client(self) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(self._config())

    def _scene_label(self, scene: str) -> str:
        return SCENE_LABELS.get(scene, SCENE_LABELS["general"])

    def _ensure_scene(self, scene: str | None) -> str:
        value = str(scene or "general").strip() or "general"
        return value if value in SCENE_LABELS else "general"

    def _get_report(self, report_id: int | None) -> DiagnosisReport | None:
        if not report_id:
            return None
        return db.session.get(DiagnosisReport, int(report_id))

    def _get_run(self, run_id: int | None) -> RunRecord | None:
        if not run_id:
            return None
        return db.session.get(RunRecord, int(run_id))

    def _build_report_context(
        self, report: DiagnosisReport
    ) -> tuple[list[str], list[dict]]:
        summary = report.get_summary()
        decision = (
            summary.get("decision")
            if isinstance(summary, dict) and isinstance(summary.get("decision"), dict)
            else None
        )
        severity = summary.get("severity") if isinstance(summary, dict) else None
        recommendations = summary.get("recommendations") if isinstance(summary, dict) else None
        data_quality = summary.get("data_quality") if isinstance(summary, dict) else None
        traceability = summary.get("traceability") if isinstance(summary, dict) else None
        model_card = summary.get("model_card") if isinstance(summary, dict) else None
        report_confidence = (
            summary.get("report_confidence") if isinstance(summary, dict) else None
        )
        review = summary.get("review") if isinstance(summary, dict) else None

        blocks = [
            "【诊断报告】",
            f"报告ID: {report.id}",
            f"资产ID: {report.asset_id}",
            f"模型: {report.model_name}",
            f"主要结论: {report.prediction_label}",
            f"置信度: {report.confidence:.4f}",
        ]

        if isinstance(severity, dict):
            blocks.append(
                "严重度: "
                f"{severity.get('level') or '-'}"
                f" / score={severity.get('score') if severity.get('score') is not None else '-'}"
                f" / method={severity.get('method') or '-'}"
            )

            evidence = severity.get("evidence")
            if isinstance(evidence, list) and evidence:
                metrics = [
                    str(item.get("metric") or "").strip()
                    for item in evidence[:2]
                    if isinstance(item, dict) and str(item.get("metric") or "").strip()
                ]
                if metrics:
                    blocks.append(f"关键证据: {', '.join(metrics)}")

        if isinstance(data_quality, dict):
            blocks.append(
                "数据质量: "
                f"level={data_quality.get('level') or 'unknown'}"
                f" / sample_count={data_quality.get('sample_count') if data_quality.get('sample_count') is not None else '-'}"
                f" / feature_count={data_quality.get('feature_count') if data_quality.get('feature_count') is not None else '-'}"
            )
            warnings = data_quality.get("warnings")
            if isinstance(warnings, list) and warnings:
                blocks.append(f"数据质量提示: {'；'.join(str(item) for item in warnings[:2])}")

        if isinstance(report_confidence, dict):
            blocks.append(
                "报告可信度: "
                f"level={report_confidence.get('level') or 'unknown'}"
                f" / score={report_confidence.get('score') if report_confidence.get('score') is not None else '-'}"
            )
            reasons = report_confidence.get("reasons")
            if isinstance(reasons, list) and reasons:
                blocks.append(f"可信度说明: {'；'.join(str(item) for item in reasons[:2])}")

        if isinstance(traceability, dict):
            blocks.append(
                "追溯信息: "
                f"run_id={traceability.get('run_id') or '-'}"
                f" / input_csv={traceability.get('input_csv') or '-'}"
                f" / rules_version={traceability.get('rules_version') or '-'}"
            )

        if isinstance(model_card, dict):
            class_set = model_card.get("class_set")
            if isinstance(class_set, list) and class_set:
                blocks.append(f"模型类别集: {', '.join(str(item) for item in class_set)}")
            limitations = model_card.get("limitations")
            if isinstance(limitations, list) and limitations:
                blocks.append(f"模型限制: {'；'.join(str(item) for item in limitations[:2])}")

        review_info = review if isinstance(review, dict) else {}
        blocks.append(
            "人工复核: "
            f"status={review_info.get('status') or 'pending'}"
            f" / conclusion={review_info.get('conclusion') or '待补充'}"
        )

        if isinstance(recommendations, list) and recommendations:
            recommendation_titles = [
                str(item.get("title") or "").strip()
                for item in recommendations[:2]
                if isinstance(item, dict) and str(item.get("title") or "").strip()
            ]
            if recommendation_titles:
                blocks.append(f"处置建议: {'；'.join(recommendation_titles)}")

        if isinstance(decision, dict):
            if decision.get("warning_reason"):
                blocks.append(f"统一决策说明: {decision.get('warning_reason')}")

            evidence_summary = decision.get("evidence_summary")
            if isinstance(evidence_summary, dict):
                top_evidence = evidence_summary.get("top_evidence") or []
                metrics = []
                for item in top_evidence[:2]:
                    if not isinstance(item, dict):
                        continue
                    metric = str(item.get("metric") or "").strip()
                    if metric:
                        metrics.append(metric)
                if metrics:
                    blocks.append(f"统一决策证据: {', '.join(metrics)}")

        citations = [
            {
                "source_type": "report",
                "source_id": str(report.id),
                "title": f"诊断报告 #{report.id}",
                "snippet": f"预测类别={report.prediction_label}, 置信度={report.confidence:.2%}",
            }
        ]
        return blocks, citations

    def _build_run_context(self, run: RunRecord) -> tuple[list[str], list[dict]]:
        params = run.get_params()
        result = run.get_result()
        prediction = result.get("prediction") if isinstance(result, dict) else None
        severity = result.get("severity") if isinstance(result, dict) else None
        recommendations = result.get("recommendations") if isinstance(result, dict) else None

        blocks = [
            "【任务记录】",
            f"任务ID: {run.id}",
            f"任务类型: {run.task}",
            f"状态: {run.status}",
            f"进度: {run.progress}",
            f"资产ID: {run.asset_id}",
        ]

        if isinstance(prediction, dict):
            blocks.append(
                f"任务结论: {prediction.get('label') or '-'} / "
                f"confidence={prediction.get('confidence') if prediction.get('confidence') is not None else '-'}"
            )
        if isinstance(severity, dict):
            blocks.append(
                f"风险级别: {severity.get('level') or '-'} / "
                f"score={severity.get('score') if severity.get('score') is not None else '-'}"
            )
        if isinstance(recommendations, list) and recommendations:
            recommendation_titles = []
            for item in recommendations[:3]:
                if isinstance(item, dict):
                    title = str(item.get("title") or "").strip()
                    if title:
                        recommendation_titles.append(title)
            if recommendation_titles:
                blocks.append(f"建议动作: {'；'.join(recommendation_titles)}")

        if isinstance(params, dict):
            model_name = str(params.get("model") or "").strip()
            if model_name:
                blocks.append(f"执行模型: {model_name}")

        citations = [
            {
                "source_type": "run",
                "source_id": str(run.id),
                "title": f"任务记录 #{run.id}",
                "snippet": f"task={run.task}, status={run.status}, asset_id={run.asset_id}",
            }
        ]
        if run.asset_id:
            asset = db.session.get(Asset, run.asset_id)
            if asset is not None:
                blocks.append(
                    f"关联资产: id={asset.id}, filename={asset.filename}, kind={asset.kind}"
                )
                citations.append(
                    {
                        "source_type": "asset",
                        "source_id": str(asset.id),
                        "title": f"资产 #{asset.id}",
                        "snippet": f"{asset.filename} ({asset.kind})",
                    }
                )
        return blocks, citations

    def _build_scene_context(
        self, *, scene: str, report_id: int | None, run_id: int | None
    ) -> SceneContext:
        report = self._get_report(report_id)
        run = self._get_run(run_id)
        blocks = [f"【当前场景】{self._scene_label(scene)}"]
        citations: list[dict] = []

        if report is not None:
            report_blocks, report_citations = self._build_report_context(report)
            blocks.extend(report_blocks)
            citations.extend(report_citations)
            if run is None and report.run_id:
                run = self._get_run(report.run_id)

        include_run_context = run is not None and not (
            scene == "report_analysis" and report is not None
        )
        if include_run_context:
            run_blocks, run_citations = self._build_run_context(run)
            blocks.extend(run_blocks)
            citations.extend(run_citations)

        return SceneContext(
            scene=scene,
            report=report,
            run=run,
            citations=citations,
            blocks=blocks,
        )

    def _build_context_summary(
        self, scene: str, report_id: int | None, run_id: int | None
    ) -> dict:
        return {
            "scene_label": self._scene_label(scene),
            "report_id": report_id,
            "run_id": run_id,
        }

    def _collect_report_terms(self, report: DiagnosisReport | None) -> list[str]:
        if report is None:
            return []
        summary = report.get_summary()
        terms: list[str] = [
            str(report.prediction_label or '').strip(),
            str(report.model_name or '').strip(),
        ]

        severity = summary.get('severity') if isinstance(summary, dict) else None
        if isinstance(severity, dict):
            terms.append(str(severity.get('level') or '').strip())
            terms.append(str(severity.get('method') or '').strip())

            evidence = severity.get('evidence')
            if isinstance(evidence, list):
                for item in evidence[:4]:
                    if not isinstance(item, dict):
                        continue
                    terms.append(str(item.get('metric') or '').strip())
                    terms.append(str(item.get('reason') or '').strip())

        recommendations = summary.get('recommendations') if isinstance(summary, dict) else None
        data_quality = summary.get('data_quality') if isinstance(summary, dict) else None
        report_confidence = summary.get('report_confidence') if isinstance(summary, dict) else None
        review = summary.get('review') if isinstance(summary, dict) else None
        if isinstance(recommendations, list):
            for item in recommendations[:4]:
                if not isinstance(item, dict):
                    continue
                terms.append(str(item.get('title') or '').strip())
                for reason_code in item.get('reason_codes') or []:
                    terms.append(str(reason_code or '').strip())

        decision = summary.get('decision') if isinstance(summary, dict) else None
        if isinstance(decision, dict):
            terms.append(str(decision.get('fault_label') or '').strip())
            terms.append(str(decision.get('severity') or '').strip())
            terms.append(str(decision.get('warning_reason') or '').strip())

            evidence_summary = decision.get('evidence_summary')
            if isinstance(evidence_summary, dict):
                terms.append(str(evidence_summary.get('fault_label') or '').strip())
                severity_info = evidence_summary.get('severity')
                if isinstance(severity_info, dict):
                    terms.append(str(severity_info.get('level') or '').strip())
                for item in evidence_summary.get('top_evidence') or []:
                    if not isinstance(item, dict):
                        continue
                    terms.append(str(item.get('metric') or '').strip())

        if isinstance(data_quality, dict):
            terms.append(str(data_quality.get('level') or '').strip())
            for warning in data_quality.get('warnings') or []:
                terms.append(str(warning or '').strip())
        if isinstance(report_confidence, dict):
            terms.append(str(report_confidence.get('level') or '').strip())
            for reason in report_confidence.get('reasons') or []:
                terms.append(str(reason or '').strip())
        if isinstance(review, dict):
            terms.append(str(review.get('status') or '').strip())
            terms.append(str(review.get('conclusion') or '').strip())

        seen = set()
        deduped = []
        for term in terms:
            normalized = str(term or '').strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _conversation_meta(self, conversation: AIConversation) -> dict:
        meta = conversation.get_meta()
        meta.setdefault("message_count", conversation.messages.count())
        meta.setdefault("last_message_preview", "")
        meta.setdefault("provider", "")
        meta.setdefault(
            "context_summary",
            self._build_context_summary(
                conversation.scene, conversation.report_id, conversation.run_id
            ),
        )
        return meta

    def list_conversations(self, *, user_id: int, limit: int = 20) -> list[dict]:
        rows = (
            AIConversation.query.filter_by(created_by=user_id)
            .order_by(AIConversation.updated_at.desc(), AIConversation.id.desc())
            .limit(max(1, min(int(limit), 100)))
            .all()
        )
        return [{**row.to_dict(), "meta": self._conversation_meta(row)} for row in rows]

    def get_conversation_detail(self, *, conversation_id: int, user_id: int) -> dict:
        conversation = db.session.get(AIConversation, conversation_id)
        if conversation is None or conversation.created_by != user_id:
            raise ValueError("会话不存在")
        return {
            "conversation": {
                **conversation.to_dict(),
                "meta": self._conversation_meta(conversation),
            },
            "messages": [
                item.to_dict()
                for item in conversation.messages.order_by(
                    AIMessage.created_at.asc()
                ).all()
            ],
        }

    def _create_conversation(
        self,
        *,
        scene: str,
        user_id: int,
        report_id: int | None,
        run_id: int | None,
        title: str | None,
    ) -> AIConversation:
        conversation = AIConversation(
            title=title
            or f"{self._scene_label(scene)} {datetime.now().strftime('%m-%d %H:%M')}",
            scene=scene,
            created_by=user_id,
            report_id=report_id,
            run_id=run_id,
        )
        conversation.set_meta(
            {
                "message_count": 0,
                "last_message_preview": "",
                "provider": "",
                "context_summary": self._build_context_summary(
                    scene, report_id, run_id
                ),
            }
        )
        db.session.add(conversation)
        db.session.flush()
        return conversation

    def _upsert_conversation(
        self,
        *,
        conversation_id: int | None,
        scene: str,
        user_id: int,
        report_id: int | None,
        run_id: int | None,
        title_hint: str | None = None,
    ) -> AIConversation:
        if conversation_id:
            conversation = db.session.get(AIConversation, int(conversation_id))
            if conversation is None or conversation.created_by != user_id:
                raise ValueError("会话不存在")
            conversation.scene = scene or conversation.scene
            if report_id is not None:
                conversation.report_id = report_id
            if run_id is not None:
                conversation.run_id = run_id
            if title_hint and conversation.messages.count() == 0:
                conversation.title = title_hint[:255]
            conversation.updated_at = datetime.now()
            return conversation
        return self._create_conversation(
            scene=scene,
            user_id=user_id,
            report_id=report_id,
            run_id=run_id,
            title=title_hint,
        )

    def _build_title(
        self,
        *,
        scene: str,
        message: str,
        report: DiagnosisReport | None,
        run: RunRecord | None,
    ) -> str:
        if report is not None:
            return f"报告 #{report.id} AI解读"
        if run is not None:
            return f"任务 #{run.id} AI复盘"
        compact = " ".join((message or "").strip().split())
        if compact:
            return compact[:40]
        return f"{self._scene_label(scene)} {datetime.now().strftime('%m-%d %H:%M')}"

    def _add_message(
        self,
        conversation: AIConversation,
        *,
        role: str,
        content: str,
        citations: list[dict] | None = None,
        used_context: dict | None = None,
    ) -> AIMessage:
        message = AIMessage(
            conversation_id=conversation.id,
            role=role,
            content=content,
        )
        message.set_citations(citations or [])
        message.set_used_context(used_context or {})
        db.session.add(message)
        db.session.flush()

        meta = conversation.get_meta()
        meta["message_count"] = int(meta.get("message_count", 0)) + 1
        meta["last_message_preview"] = content[:120]
        conversation.set_meta(meta)
        conversation.updated_at = datetime.now()
        return message

    def _history_messages(self, conversation: AIConversation) -> list[dict[str, str]]:
        rows = (
            conversation.messages.order_by(AIMessage.created_at.desc()).limit(8).all()
        )
        messages = []
        for item in reversed(rows):
            messages.append({"role": item.role, "content": item.content})
        return messages

    def _with_sqlite_retry(self, callback):
        retries = max(0, int(self._config().get("SQLITE_LOCK_RETRIES", 2)))
        delay = max(0.0, float(self._config().get("SQLITE_LOCK_RETRY_DELAY", 0.25)))
        last_error = None
        for attempt in range(retries + 1):
            try:
                return callback()
            except OperationalError as exc:
                db.session.rollback()
                if "database is locked" not in str(exc).lower() or attempt >= retries:
                    raise
                last_error = exc
                time.sleep(delay * (attempt + 1))
        if last_error is not None:
            raise last_error

    def _knowledge_query(self, scene_context: SceneContext, message: str) -> str:
        parts = [message]
        if scene_context.report is not None:
            parts.extend(self._collect_report_terms(scene_context.report))
        if scene_context.run is not None:
            parts.append(scene_context.run.task)
        return " ".join(part for part in parts if part)

    def _looks_like_general_knowledge_question(self, message: str) -> bool:
        text = str(message or "").strip()
        if not text:
            return False
        keywords = (
            "表现",
            "症状",
            "原因",
            "怎么修",
            "如何修",
            "如何处理",
            "如何排查",
            "是什么",
            "有哪些",
            "原理",
        )
        return any(keyword in text for keyword in keywords)

    def _scene_prompt_guidance(
        self, *, scene_context: SceneContext, user_message: str
    ) -> str:
        if scene_context.scene == "report_analysis":
            return (
                "当前场景为报告分析，你是报告解释器，不要重新生成独立诊断。\n"
                "请围绕报告字段解释，并按“结论 / 证据 / 风险 / 建议动作 / 不确定性与限制 / 需要人工确认”组织回答。\n"
                "如果用户继续追问“如何修复、如何排查、下一步怎么做”这类问题，"
                "请优先给出可执行的检查与修复建议，并尽量按步骤组织答案。\n"
                "当数据质量较差或关键字段缺失时，要明确提示不足以支持强结论。\n"
                "当现有报告无法唯一确定故障根因时，也应先给出基于当前证据的分步骤排查路径，"
                "再说明哪些步骤需要人工复核，而不是直接拒绝回答。"
            )

        if scene_context.scene == "general" and self._looks_like_general_knowledge_question(user_message):
            return (
                "当前场景为通用问答，且没有绑定具体报告或任务。\n"
                "请直接按无人机故障诊断领域的一般知识回答用户问题，"
                "不要默认把回答收束成“请绑定 report_id 或 run_id”。\n"
                "如果用户询问表现、原因、排查、修复等内容，可以给出通用且可执行的检查建议，"
                "同时明确这些建议属于一般性说明。"
            )

        return (
            "当前场景为通用问答。\n"
            "若缺少具体报告或任务上下文，请优先给出一般性领域知识回答，"
            "只有在用户明确要求结合当前系统记录做判断时，才提示绑定 report_id 或 run_id。"
        )

    def _format_knowledge_context(self, items: list[dict]) -> list[str]:
        blocks = []
        for idx, item in enumerate(items, start=1):
            blocks.append(
                f"【补充背景 {idx}】\n来源类型: {item.get('source_type')}\n标题: {item.get('title')}\n内容: {item.get('snippet')}"
            )
        return blocks

    def _used_context_payload(
        self,
        *,
        scene_context: SceneContext,
        knowledge_context: KnowledgeContextResult,
        provider: str,
        model: str,
        fallback_used: bool,
    ) -> dict:
        return {
            "scene": scene_context.scene,
            "report_id": scene_context.report.id if scene_context.report else None,
            "run_id": scene_context.run.id if scene_context.run else None,
            "knowledge_hits": len(knowledge_context.hits),
            "knowledge_retrieval_mode": knowledge_context.actual_mode,
            "knowledge_embedding_enabled": knowledge_context.index_embedding_enabled,
            "provider": provider,
            "model": model,
            "fallback_used": fallback_used,
        }

    def _fallback_answer(
        self, *, scene_context: SceneContext, message: str, knowledge_hits: list[dict]
    ) -> str:
        knowledge_enabled = bool(self._config().get("AI_ENABLE_KNOWLEDGE_BASE", False))
        conclusion = "当前我将基于通用无人机故障诊断知识给出一般性判断。"
        reason = "当前主要依据你的问题和系统里已有的上下文信息。"
        advice = "建议先根据当前给出的通用检查项进行排查，若需结合具体日志或报告做精确判断，再绑定 report_id 或 run_id。"

        if scene_context.report is not None:
            summary = scene_context.report.get_summary()
            severity = summary.get("severity") if isinstance(summary, dict) else {}
            if not isinstance(severity, dict):
                severity = {}
            recommendations = (
                summary.get("recommendations") if isinstance(summary, dict) else []
            )
            decision = summary.get("decision") if isinstance(summary, dict) else None
            data_quality = summary.get("data_quality") if isinstance(summary, dict) else {}
            if not isinstance(data_quality, dict):
                data_quality = {}
            report_confidence = (
                summary.get("report_confidence") if isinstance(summary, dict) else {}
            )
            if not isinstance(report_confidence, dict):
                report_confidence = {}
            review = summary.get("review") if isinstance(summary, dict) else {}
            if not isinstance(review, dict):
                review = {}
            model_card = summary.get("model_card") if isinstance(summary, dict) else {}
            if not isinstance(model_card, dict):
                model_card = {}
            label_text = _fallback_label_text(scene_context.report.prediction_label)
            severity_text = _fallback_severity_text(severity.get("level"))
            confidence_text = _fallback_percent(scene_context.report.confidence)
            evidence_text = _fallback_evidence_text(summary if isinstance(summary, dict) else {})
            recommendation_text = _fallback_recommendation_text(recommendations)
            warning_reason = (
                str(decision.get("warning_reason") or "").strip()
                if isinstance(decision, dict)
                else ""
            )
            quality_level = str(data_quality.get("level") or "unknown").strip() or "unknown"
            quality_warnings = data_quality.get("warnings")
            if not isinstance(quality_warnings, list):
                quality_warnings = []
            confidence_level = str(report_confidence.get("level") or "unknown").strip() or "unknown"
            confidence_reasons = report_confidence.get("reasons")
            if not isinstance(confidence_reasons, list):
                confidence_reasons = []
            review_status = str(review.get("status") or "pending").strip() or "pending"
            limitations = model_card.get("limitations")
            if not isinstance(limitations, list):
                limitations = []

            conclusion = (
                f"当前报告判断为 {label_text}，主分类置信度约为 {confidence_text}，"
                f"综合风险等级为 {severity_text}。"
            )
            reason_parts = [
                "这一判断主要来自报告中的诊断结论、严重度规则和统一决策信息。"
            ]
            if evidence_text:
                reason_parts.append(f"当前关键证据集中在 {evidence_text}。")
            if warning_reason:
                reason_parts.append(f"统一决策给出的风险说明是：{warning_reason}")
            reason = "".join(reason_parts)
            if recommendation_text:
                advice = (
                    f"建议按报告处置项执行：{recommendation_text}。"
                    "处理前后都应保留原始日志和复核记录，便于后续追踪同类异常。"
                )
            elif scene_context.report.prediction_label == "normal":
                advice = "建议保持例行巡检与飞前检查，并将本次结果作为正常样本留存。"
            else:
                advice = (
                    "建议优先复核与该故障类别相关的关键部件，并结合原始遥测、维护记录和飞行工况确认根因。"
                )

            if scene_context.scene == "report_analysis":
                risk_text = (
                    f"综合风险等级为 {severity_text}，报告可信度为 {confidence_level}。"
                )
                if quality_level in {"poor", "warning"}:
                    risk_text += f" 当前数据质量为 {quality_level}，结论支撑度有限。"
                uncertainty_parts = []
                if quality_level == "poor":
                    uncertainty_parts.append("输入数据质量较差，不足以支持强结论。")
                elif quality_level == "warning":
                    uncertainty_parts.append("输入数据质量信息不完整，建议结合原始日志复核。")
                if confidence_level == "low":
                    uncertainty_parts.append("报告可信度偏低，不宜单独作为最终处置依据。")
                if limitations:
                    uncertainty_parts.append("；".join(str(item) for item in limitations[:2]))
                if confidence_reasons:
                    uncertainty_parts.append("；".join(str(item) for item in confidence_reasons[:2]))
                if not uncertainty_parts:
                    uncertainty_parts.append("当前报告字段较完整，但仍需结合现场检查确认。")

                review_text = (
                    f"当前复核状态为 {review_status}。"
                    if review_status != "pending"
                    else "当前复核状态为待复核，建议维护人员结合原始日志、现场检查和维修记录完成确认。"
                )
                if quality_warnings:
                    review_text += f" 数据质量提示：{'；'.join(str(item) for item in quality_warnings[:2])}。"

                return "\n\n".join(
                    [
                        f"结论：{conclusion}",
                        f"证据：{reason}",
                        f"风险：{risk_text}",
                        f"建议动作：{advice}",
                        f"不确定性与限制：{' '.join(uncertainty_parts)}",
                        f"需要人工确认：{review_text}",
                    ]
                )
        elif scene_context.run is not None:
            conclusion = (
                f"当前任务 `{scene_context.run.task}` 的状态为 `{scene_context.run.status}`。"
            )
            reason = "这一判断主要依据任务状态、进度和关联结果摘要。"
            advice = "建议结合任务结论与关联资产继续分析，必要时生成报告后再追问。"
        elif scene_context.scene == "general" and self._looks_like_general_knowledge_question(message):
            text = str(message or "")
            if "电机" in text:
                conclusion = "电机故障通常会表现为振动增大、异响、转速不稳、推力下降或过热。"
                reason = "这些表现通常与轴承磨损、螺旋桨失衡、电机绕组异常或 ESC 输出异常有关，因此会同时反映在动力、振动和电气参数上。"
                advice = "建议优先检查电机与螺旋桨的物理状态，同时观察转速、电流、振动和温度等参数是否异常；若有日志或报告，可继续结合具体数据定位故障原因。"
            elif "修复" in text or "排查" in text:
                conclusion = "故障修复一般需要先判定问题出在哪个部件，再针对性处理。"
                reason = "无人机故障往往包括电机、传感器、供电或控制链路等多种原因，如果不先做基本排查，直接更换部件容易导致误判。"
                advice = "建议先检查连接状态、供电、传感器读数、执行器输出和显著的机械损伤，再根据症状决定是校准、维修还是更换部件。"

        if knowledge_hits:
            reason += " 同时命中了补充知识库内容，可作为原因解释和处理建议的辅助依据。"
        elif knowledge_enabled:
            reason += " 当前没有检索到额外知识库命中，因此主要依据系统记录作答。"
        else:
            reason += " 当前未启用知识库检索，因此主要依据系统记录作答。"

        return "\n\n".join(
            [
                f"直接结论：{conclusion}",
                f"原因解释：{reason}",
                f"风险与建议：{advice}",
            ]
        )

    def _persist_user_message(
        self,
        *,
        conversation_id: int | None,
        scene: str,
        user_id: int,
        report_id: int | None,
        run_id: int | None,
        title_hint: str,
        clean_message: str,
    ) -> int:
        def _operation():
            conversation = self._upsert_conversation(
                conversation_id=conversation_id,
                scene=scene,
                user_id=user_id,
                report_id=report_id,
                run_id=run_id,
                title_hint=title_hint,
            )
            self._add_message(conversation, role="user", content=clean_message)
            db.session.commit()
            return int(conversation.id)

        return self._with_sqlite_retry(_operation)

    def _persist_assistant_message(
        self,
        *,
        conversation_id: int,
        title_hint: str,
        answer: str,
        provider_name: str,
        combined_citations: list[dict],
        used_context: dict,
    ) -> tuple[AIConversation, AIMessage]:
        def _operation():
            conversation = db.session.get(AIConversation, conversation_id)
            if conversation is None:
                raise ValueError("会话不存在")
            assistant_message = self._add_message(
                conversation,
                role="assistant",
                content=answer,
                citations=combined_citations,
                used_context=used_context,
            )
            meta = conversation.get_meta()
            meta["provider"] = provider_name
            meta["context_summary"] = self._build_context_summary(
                conversation.scene,
                conversation.report_id,
                conversation.run_id,
            )
            conversation.set_meta(meta)
            if conversation.messages.count() <= 2 and title_hint:
                conversation.title = title_hint[:255]
            db.session.commit()
            return conversation, assistant_message

        return self._with_sqlite_retry(_operation)

    def _call_model(
        self,
        messages: list[dict],
        *,
        scene_context: SceneContext,
        message: str,
        knowledge_hits: list[dict],
    ) -> tuple[str, str, str, bool]:
        provider = self._provider_client()
        hard_timeout = float(self._config().get("AI_CHAT_HARD_TIMEOUT", 60))
        try:
            future = _AI_CHAT_EXECUTOR.submit(provider.chat, messages)
            result = future.result(timeout=hard_timeout)
            return (
                result["content"],
                result.get("provider", provider.provider_name),
                result.get("model", provider.model),
                False,
            )
        except FutureTimeoutError:
            current_app.logger.warning(
                "AI provider 调用超过总时限 %.2fs，已回退为上下文回答",
                hard_timeout,
            )
            return (
                self._fallback_answer(
                    scene_context=scene_context,
                    message=message,
                    knowledge_hits=knowledge_hits,
                ),
                "offline_fallback",
                "context_only",
                True,
            )
        except AIProviderError:
            return (
                self._fallback_answer(
                    scene_context=scene_context,
                    message=message,
                    knowledge_hits=knowledge_hits,
                ),
                "offline_fallback",
                "context_only",
                True,
            )

    def _build_prompt_messages(
        self,
        *,
        scene_context: SceneContext,
        knowledge_hits: list[dict],
        user_message: str,
        conversation: AIConversation,
    ) -> list[dict]:
        context_blocks = list(scene_context.blocks)
        context_blocks.extend(self._format_knowledge_context(knowledge_hits))
        context_text = "\n\n".join(context_blocks)
        history = self._history_messages(conversation)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": self._scene_prompt_guidance(
                    scene_context=scene_context,
                    user_message=user_message,
                ),
            },
            {
                "role": "system",
                "content": (
                    "以下是当前系统上下文与补充背景资料。这些内容仅作内部参考，请吸收后自然作答，"
                    "不要在正文中直接写“参考资料”“知识片段 1/2”之类的标签，也不要整段复述检索片段。\n\n"
                    f"{context_text}"
                ),
            },
            *history,
            {"role": "user", "content": user_message},
        ]

    def chat(
        self,
        *,
        user_id: int,
        message: str,
        scene: str | None = None,
        conversation_id: int | None = None,
        report_id: int | None = None,
        run_id: int | None = None,
    ) -> dict:
        clean_message = str(message or "").strip()
        if not clean_message:
            raise ValueError("message 不能为空")

        normalized_scene = self._ensure_scene(scene)
        scene_context = self._build_scene_context(
            scene=normalized_scene, report_id=report_id, run_id=run_id
        )
        title_hint = self._build_title(
            scene=normalized_scene,
            message=clean_message,
            report=scene_context.report,
            run=scene_context.run,
        )
        conversation_id = self._persist_user_message(
            conversation_id=conversation_id,
            scene=normalized_scene,
            user_id=user_id,
            report_id=scene_context.report.id if scene_context.report else report_id,
            run_id=scene_context.run.id if scene_context.run else run_id,
            title_hint=title_hint,
            clean_message=clean_message,
        )
        conversation = db.session.get(AIConversation, conversation_id)
        if conversation is None:
            raise ValueError("会话不存在")

        top_k = min(
            int(self._config().get("AI_TOP_K", 5)),
            int(self._config().get("AI_MAX_CONTEXT_CHUNKS", 6)),
        )
        knowledge_context = knowledge_base.retrieve_for_interpretation(
            self._knowledge_query(scene_context, clean_message),
            candidate_limit=top_k,
            max_items=min(2, top_k),
        )
        knowledge_hits = knowledge_context.hits
        prompt_messages = self._build_prompt_messages(
            scene_context=scene_context,
            knowledge_hits=knowledge_hits,
            user_message=clean_message,
            conversation=conversation,
        )
        answer, provider_name, model_name, fallback_used = self._call_model(
            prompt_messages,
            scene_context=scene_context,
            message=clean_message,
            knowledge_hits=knowledge_hits,
        )
        combined_citations = list(scene_context.citations) + knowledge_hits
        used_context = self._used_context_payload(
            scene_context=scene_context,
            knowledge_context=knowledge_context,
            provider=provider_name,
            model=model_name,
            fallback_used=fallback_used,
        )
        conversation, assistant_message = self._persist_assistant_message(
            conversation_id=conversation_id,
            title_hint=title_hint,
            answer=answer,
            provider_name=provider_name,
            combined_citations=combined_citations,
            used_context=used_context,
        )
        return {
            "conversation": {
                **conversation.to_dict(),
                "meta": self._conversation_meta(conversation),
            },
            "message": assistant_message.to_dict(),
            "citations": combined_citations,
            "used_context": used_context,
        }

    def analyze_report(
        self,
        *,
        user_id: int,
        report_id: int,
        question: str | None = None,
        conversation_id: int | None = None,
    ) -> dict:
        report = self._get_report(report_id)
        if report is None:
            raise ValueError("报告不存在")
        prompt = str(question or SCENE_DEFAULT_QUESTIONS["report_analysis"]).strip()
        return self.chat(
            user_id=user_id,
            message=prompt,
            scene="report_analysis",
            conversation_id=conversation_id,
            report_id=report.id,
            run_id=report.run_id,
        )


ai_assistant_service = AIAssistantService()
