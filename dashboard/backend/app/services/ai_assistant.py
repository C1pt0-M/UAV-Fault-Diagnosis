from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime
import json
import time

from flask import current_app
from sqlalchemy.exc import OperationalError

from app import db
from app.models.ai_conversation import AIConversation, AIMessage
from app.models.asset import Asset
from app.models.diagnosis_report import DiagnosisReport
from app.models.run_record import RunRecord
from app.services.ai_provider import AIProviderError, OpenAICompatibleProvider
from app.services.knowledge_base import knowledge_base


_AI_CHAT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ai-chat")


SCENE_LABELS = {
    "general": "通用问答",
    "report_analysis": "报告分析",
}

SCENE_DEFAULT_QUESTIONS = {
    "report_analysis": "请先给出直接结论，再解释主要原因，最后说明风险与处理建议。",
}

SYSTEM_PROMPT = """你是无人机故障诊断系统中的 AI 专家助手。
回答要求：
1. 优先依据给定的系统上下文、诊断报告、任务记录和补充背景资料回答，但要把这些信息自然融入判断，不要逐条复述。
2. 不要编造项目内不存在的事实；无法确认时直接说明“无法从当前资料确认”。
3. 输出使用中文，技术术语保留英文。
4. 默认使用三段式自然回答：
第一段以“直接结论：”开头，先给最终判断；
第二段以“原因解释：”开头，解释主要依据和原因；
第三段以“风险与建议：”开头，说明风险等级、下一步动作或处理建议。
5. 除非用户明确要求，否则不要使用 Markdown 标题或粗体，也不要复述内部字段名或项目内部标签。
6. 不要在正文中直接写“参考资料”“知识片段 1/2”等内部标签；如需说明依据，用自然语言概括即可。
7. 如果上下文里有概率、严重度、推荐动作，可以自然引用并解释，但不要把原始 JSON 或整段检索文本原样搬进回答。"""


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
        return dict((self.app or current_app).config)

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

        blocks = [
            "【诊断报告】",
            f"报告ID: {report.id}",
            f"任务ID: {report.run_id}",
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
                    for item in evidence[:4]
                    if isinstance(item, dict) and str(item.get("metric") or "").strip()
                ]
                if metrics:
                    blocks.append(f"关键证据: {', '.join(metrics)}")

        if isinstance(recommendations, list) and recommendations:
            recommendation_titles = [
                str(item.get("title") or "").strip()
                for item in recommendations[:3]
                if isinstance(item, dict) and str(item.get("title") or "").strip()
            ]
            if recommendation_titles:
                blocks.append(f"处置建议: {'；'.join(recommendation_titles)}")

        if isinstance(decision, dict):
            blocks.append(
                "统一决策结论: "
                f"state={decision.get('state') or '-'}"
                f", severity={decision.get('severity') or '-'}"
                f", warning_stage={decision.get('warning_stage') or '-'}"
            )
            if decision.get("warning_reason"):
                blocks.append(f"统一决策说明: {decision.get('warning_reason')}")

            evidence_summary = decision.get("evidence_summary")
            if isinstance(evidence_summary, dict):
                top_evidence = evidence_summary.get("top_evidence") or []
                metrics = []
                for item in top_evidence[:4]:
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

        if run is not None:
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

    def _format_knowledge_context(self, items: list[dict]) -> list[str]:
        blocks = []
        for idx, item in enumerate(items, start=1):
            blocks.append(
                f"【补充背景 {idx}】\n来源类型: {item.get('source_type')}\n标题: {item.get('title')}\n内容: {item.get('snippet')}"
            )
        return blocks

    def _select_knowledge_hits(
        self, *, query: str, hits: list[dict], max_items: int = 2
    ) -> list[dict]:
        if not hits:
            return []

        top_score = max(float(item.get("score") or 0.0) for item in hits)
        threshold = max(0.35, top_score * 0.6)
        selected = []

        for item in hits:
            score = float(item.get("score") or 0.0)
            snippet = str(item.get("snippet") or "")
            title = str(item.get("title") or "")
            if score < threshold:
                continue
            if not knowledge_base._is_text_quality_acceptable(f"{title}\n{snippet}"):
                continue
            selected.append(item)
            if len(selected) >= max_items:
                break

        return selected

    def _used_context_payload(
        self,
        *,
        scene_context: SceneContext,
        knowledge_hits: list[dict],
        provider: str,
        model: str,
        fallback_used: bool,
    ) -> dict:
        knowledge_info = knowledge_base.retrieval_info()
        knowledge_retrieval_mode = None
        if knowledge_hits:
            knowledge_retrieval_mode = str(
                knowledge_hits[0].get("retrieval_mode") or ""
            ).strip() or None
        if knowledge_retrieval_mode is None and knowledge_info.get("enabled"):
            knowledge_retrieval_mode = str(
                knowledge_info.get("retrieval_mode") or ""
            ).strip() or None

        return {
            "scene": scene_context.scene,
            "report_id": scene_context.report.id if scene_context.report else None,
            "run_id": scene_context.run.id if scene_context.run else None,
            "knowledge_hits": len(knowledge_hits),
            "knowledge_retrieval_mode": knowledge_retrieval_mode,
            "knowledge_embedding_enabled": bool(
                knowledge_info.get("embedding_enabled", False)
            ),
            "provider": provider,
            "model": model,
            "fallback_used": fallback_used,
        }

    def _fallback_answer(
        self, *, scene_context: SceneContext, message: str, knowledge_hits: list[dict]
    ) -> str:
        knowledge_enabled = bool(self._config().get("AI_ENABLE_KNOWLEDGE_BASE", False))
        knowledge_enabled = bool(self._config().get("AI_ENABLE_KNOWLEDGE_BASE", False))
        conclusion = "\u5f53\u524d\u6ca1\u6709\u7ed1\u5b9a\u5177\u4f53\u62a5\u544a\u6216\u4efb\u52a1\uff0c\u6211\u53ea\u80fd\u7ed9\u51fa\u4e00\u822c\u6027\u5224\u65ad\u3002"
        reason = "\u5f53\u524d\u4e3b\u8981\u4f9d\u636e\u4f60\u7684\u95ee\u9898\u548c\u7cfb\u7edf\u91cc\u5df2\u6709\u7684\u4e0a\u4e0b\u6587\u4fe1\u606f\u3002"
        advice = "\u5efa\u8bae\u7ed1\u5b9a\u5177\u4f53 report_id \u6216 run_id \u540e\u518d\u7ee7\u7eed\u5206\u6790\u3002"

        if scene_context.report is not None:
            conclusion = (
                f"\u5f53\u524d\u62a5\u544a\u66f4\u63a5\u8fd1 `{scene_context.report.prediction_label}`\uff0c"
                f"\u7f6e\u4fe1\u5ea6\u7ea6\u4e3a {scene_context.report.confidence:.2%}\u3002"
            )
            reason = "\u8fd9\u4e00\u5224\u65ad\u4e3b\u8981\u6765\u81ea\u62a5\u544a\u4e2d\u7684\u8bca\u65ad\u7ed3\u8bba\u3001\u4e25\u91cd\u5ea6\u548c\u7edf\u4e00\u51b3\u7b56\u4fe1\u606f\u3002"
            advice = "\u5efa\u8bae\u4f18\u5148\u7ed3\u5408\u62a5\u544a\u4e2d\u7684\u98ce\u9669\u7b49\u7ea7\u548c\u5904\u7f6e\u5efa\u8bae\u7ee7\u7eed\u590d\u6838\u5173\u952e\u90e8\u4ef6\u3002"
        elif scene_context.run is not None:
            conclusion = (
                f"\u5f53\u524d\u4efb\u52a1 `{scene_context.run.task}` \u7684\u72b6\u6001\u4e3a `{scene_context.run.status}`\u3002"
            )
            reason = "\u8fd9\u4e00\u5224\u65ad\u4e3b\u8981\u4f9d\u636e\u4efb\u52a1\u72b6\u6001\u3001\u8fdb\u5ea6\u548c\u5173\u8054\u7ed3\u679c\u6458\u8981\u3002"
            advice = "\u5efa\u8bae\u7ed3\u5408\u4efb\u52a1\u7ed3\u8bba\u4e0e\u5173\u8054\u8d44\u4ea7\u7ee7\u7eed\u5206\u6790\uff0c\u5fc5\u8981\u65f6\u751f\u6210\u62a5\u544a\u540e\u518d\u8ffd\u95ee\u3002"

        if knowledge_hits:
            reason += " \u540c\u65f6\u547d\u4e2d\u4e86\u8865\u5145\u77e5\u8bc6\u5e93\u5185\u5bb9\uff0c\u53ef\u4f5c\u4e3a\u539f\u56e0\u89e3\u91ca\u548c\u5904\u7406\u5efa\u8bae\u7684\u8f85\u52a9\u4f9d\u636e\u3002"
        elif knowledge_enabled:
            reason += " \u5f53\u524d\u6ca1\u6709\u68c0\u7d22\u5230\u989d\u5916\u77e5\u8bc6\u5e93\u547d\u4e2d\uff0c\u56e0\u6b64\u4e3b\u8981\u4f9d\u636e\u7cfb\u7edf\u8bb0\u5f55\u4f5c\u7b54\u3002"
        else:
            reason += " \u5f53\u524d\u672a\u542f\u7528\u77e5\u8bc6\u5e93\u68c0\u7d22\uff0c\u56e0\u6b64\u4e3b\u8981\u4f9d\u636e\u7cfb\u7edf\u8bb0\u5f55\u4f5c\u7b54\u3002"

        return "\n\n".join(
            [
                f"\u76f4\u63a5\u7ed3\u8bba\uff1a{conclusion}",
                f"\u539f\u56e0\u89e3\u91ca\uff1a{reason}",
                f"\u98ce\u9669\u4e0e\u5efa\u8bae\uff1a{advice}",
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
        knowledge_hits = knowledge_base.search(
            self._knowledge_query(scene_context, clean_message),
            top_k=top_k,
        )
        knowledge_hits = self._select_knowledge_hits(
            query=clean_message,
            hits=knowledge_hits,
            max_items=min(2, top_k),
        )
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
            knowledge_hits=knowledge_hits,
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
