from flask import Blueprint, current_app, jsonify, request

from uav_dashboard_api.services.ai_provider import AIProviderError, OpenAICompatibleProvider
from uav_dashboard_api.services.ai_settings import ai_settings_service
from uav_dashboard_api.utils.jwt_utils import admin_required, token_required

bp = Blueprint("settings", __name__)


@bp.route("/ai", methods=["GET"])
@token_required
@admin_required
def get_ai_settings(current_user):
    return jsonify(
        {
            "success": True,
            "data": ai_settings_service.public_payload(current_app.config),
        }
    )


@bp.route("/ai", methods=["PUT"])
@token_required
@admin_required
def update_ai_settings(current_user):
    payload = request.get_json(silent=True) or {}
    data = ai_settings_service.save_public_settings(current_app.config, payload)
    return jsonify(
        {
            "success": True,
            "message": "AI 配置已保存",
            "data": data,
        }
    )


@bp.route("/ai/test", methods=["POST"])
@token_required
@admin_required
def test_ai_settings(current_user):
    payload = request.get_json(silent=True) or {}
    effective_config = ai_settings_service.effective_config(current_app.config, payload)
    provider = OpenAICompatibleProvider(effective_config)
    messages = [
        {"role": "system", "content": "You are a connection test assistant."},
        {"role": "user", "content": "Reply with ok."},
    ]
    try:
        result = provider.chat(messages, temperature=0)
    except AIProviderError as exc:
        return jsonify(
            {
                "success": True,
                "data": {
                    "ok": False,
                    "message": ai_settings_service.redact_secrets(
                        str(exc), current_app.config, payload
                    ),
                },
            }
        )

    return jsonify(
        {
            "success": True,
            "data": {
                "ok": True,
                "provider": result.get("provider", provider.provider_name),
                "model": result.get("model", provider.model),
            },
        }
    )
