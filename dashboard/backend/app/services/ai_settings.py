from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PUBLIC_TO_CONFIG_KEYS = {
    "provider": "AI_PROVIDER",
    "base_url": "AI_BASE_URL",
    "api_key": "AI_API_KEY",
    "model": "AI_MODEL",
    "embedding_base_url": "AI_EMBEDDING_BASE_URL",
    "embedding_api_key": "AI_EMBEDDING_API_KEY",
    "embedding_model": "AI_EMBEDDING_MODEL",
    "http_timeout": "AI_HTTP_TIMEOUT",
    "rag_mode": "AI_RAG_MODE",
}

SECRET_FIELDS = {"api_key", "embedding_api_key"}


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _clean_base_url(value: Any) -> str:
    return _clean_string(value).rstrip("/")


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _mask_secret(value: str) -> str:
    clean = _clean_string(value)
    if not clean:
        return ""
    if len(clean) <= 8:
        return f"{clean[:2]}...{clean[-2:]}"
    return f"{clean[:4]}...{clean[-4:]}"


class AISettingsService:
    def _settings_path(self, config: dict) -> Path:
        raw_path = config.get("AI_SETTINGS_PATH")
        if raw_path:
            path = Path(raw_path)
        else:
            path = Path(config["RUNTIME_DIR"]) / "ai_settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _read_runtime_settings(self, config: dict) -> dict:
        path = self._settings_path(config)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_runtime_settings(self, config: dict, settings: dict) -> None:
        path = self._settings_path(config)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def _default_public_settings(self, config: dict) -> dict:
        base_url = _clean_base_url(config.get("AI_BASE_URL", ""))
        api_key = _clean_string(config.get("AI_API_KEY", ""))
        return {
            "provider": _clean_string(config.get("AI_PROVIDER", "openai_compatible"))
            or "openai_compatible",
            "base_url": base_url,
            "api_key": api_key,
            "model": _clean_string(config.get("AI_MODEL", "")),
            "embedding_base_url": _clean_base_url(
                config.get("AI_EMBEDDING_BASE_URL") or base_url
            ),
            "embedding_api_key": _clean_string(
                config.get("AI_EMBEDDING_API_KEY") or api_key
            ),
            "embedding_model": _clean_string(config.get("AI_EMBEDDING_MODEL", "")),
            "http_timeout": _safe_int(config.get("AI_HTTP_TIMEOUT"), 45),
            "rag_mode": _clean_string(config.get("AI_RAG_MODE", "bm25")).lower()
            or "bm25",
        }

    def _normalize_runtime_settings(self, raw: dict) -> dict:
        normalized = {}
        for field in PUBLIC_TO_CONFIG_KEYS:
            if field not in raw:
                continue
            value = raw.get(field)
            if field in {"base_url", "embedding_base_url"}:
                normalized[field] = _clean_base_url(value)
            elif field == "http_timeout":
                normalized[field] = _safe_int(value, 45)
            elif field == "rag_mode":
                normalized[field] = _clean_string(value).lower() or "bm25"
            else:
                normalized[field] = _clean_string(value)
        return normalized

    def effective_public_settings(
        self,
        config: dict,
        candidate: dict | None = None,
    ) -> dict:
        settings = self._default_public_settings(config)
        runtime_settings = self._normalize_runtime_settings(
            self._read_runtime_settings(config)
        )
        settings.update(runtime_settings)

        if "base_url" in runtime_settings and "embedding_base_url" not in runtime_settings:
            settings["embedding_base_url"] = settings["base_url"]
        if "api_key" in runtime_settings and "embedding_api_key" not in runtime_settings:
            settings["embedding_api_key"] = settings["api_key"]

        if candidate:
            candidate_settings = self._normalize_runtime_settings(candidate)
            for field, value in candidate_settings.items():
                if field in SECRET_FIELDS and value == "":
                    continue
                settings[field] = value
            if "base_url" in candidate_settings and "embedding_base_url" not in candidate_settings:
                settings["embedding_base_url"] = settings["base_url"]
            if "api_key" in candidate_settings and "embedding_api_key" not in candidate_settings:
                settings["embedding_api_key"] = settings["api_key"]

        return settings

    def effective_config(self, config: dict, candidate: dict | None = None) -> dict:
        merged = dict(config)
        public_settings = self.effective_public_settings(merged, candidate)
        for public_key, config_key in PUBLIC_TO_CONFIG_KEYS.items():
            merged[config_key] = public_settings[public_key]
        return merged

    def public_payload(self, config: dict, candidate: dict | None = None) -> dict:
        settings = self.effective_public_settings(config, candidate)
        api_key = settings.pop("api_key", "")
        embedding_api_key = settings.pop("embedding_api_key", "")
        settings.update(
            {
                "api_key_set": bool(api_key),
                "api_key_preview": _mask_secret(api_key),
                "embedding_api_key_set": bool(embedding_api_key),
                "embedding_api_key_preview": _mask_secret(embedding_api_key),
            }
        )
        return settings

    def save_public_settings(self, config: dict, payload: dict) -> dict:
        existing = self._normalize_runtime_settings(self._read_runtime_settings(config))
        next_settings = dict(existing)

        for field in PUBLIC_TO_CONFIG_KEYS:
            if field in SECRET_FIELDS:
                continue
            if field not in payload:
                continue
            value = payload.get(field)
            if field in {"base_url", "embedding_base_url"}:
                next_settings[field] = _clean_base_url(value)
            elif field == "http_timeout":
                next_settings[field] = _safe_int(value, 45)
            elif field == "rag_mode":
                next_settings[field] = _clean_string(value).lower() or "bm25"
            else:
                next_settings[field] = _clean_string(value)

        for field in SECRET_FIELDS:
            clear_field = f"clear_{field}"
            if payload.get(clear_field) is True:
                next_settings[field] = ""
            elif field in payload and _clean_string(payload.get(field)):
                next_settings[field] = _clean_string(payload.get(field))

        self._write_runtime_settings(config, next_settings)
        return self.public_payload(config)

    def redact_secrets(self, text: str, config: dict, candidate: dict | None = None) -> str:
        redacted = str(text or "")
        settings = self.effective_public_settings(config, candidate)
        for field in SECRET_FIELDS:
            secret = _clean_string(settings.get(field))
            if secret:
                redacted = redacted.replace(secret, "***")
        return redacted


ai_settings_service = AISettingsService()
