from __future__ import annotations

import json
from urllib import error, request


class AIProviderError(RuntimeError):
    """LLM provider 调用失败。"""


class OpenAICompatibleProvider:
    """兼容 OpenAI Chat Completions 的轻量 Provider。"""

    def __init__(self, config: dict):
        self.base_url = str(config.get('AI_BASE_URL', '')).rstrip('/')
        self.api_key = str(config.get('AI_API_KEY', '')).strip()
        self.model = str(config.get('AI_MODEL', '')).strip()
        self.embedding_base_url = str(
            config.get('AI_EMBEDDING_BASE_URL', '')
        ).rstrip('/')
        self.embedding_api_key = str(
            config.get('AI_EMBEDDING_API_KEY', '')
        ).strip()
        self.embedding_model = str(config.get('AI_EMBEDDING_MODEL', '')).strip()
        self.timeout = int(config.get('AI_HTTP_TIMEOUT', 45))
        self.provider_name = str(config.get('AI_PROVIDER', 'openai_compatible')).strip() or 'openai_compatible'

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    @property
    def is_embedding_configured(self) -> bool:
        return bool(
            self.embedding_base_url and self.embedding_api_key and self.embedding_model
        )

    def chat(self, messages: list[dict], *, temperature: float = 0.2) -> dict:
        if not self.is_configured:
            raise AIProviderError('AI provider 未配置完整，请检查 AI_BASE_URL / AI_API_KEY / AI_MODEL')

        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'stream': False,
        }
        endpoint = f'{self.base_url}/chat/completions'
        req = request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            },
            method='POST',
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode('utf-8')
        except error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='ignore')
            raise AIProviderError(f'AI provider HTTP {exc.code}: {detail or exc.reason}') from exc
        except error.URLError as exc:
            raise AIProviderError(f'AI provider 网络错误: {exc.reason}') from exc
        except Exception as exc:
            raise AIProviderError(f'AI provider 调用失败: {exc}') from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AIProviderError('AI provider 返回非 JSON 响应') from exc

        choices = data.get('choices') if isinstance(data, dict) else None
        if not choices:
            raise AIProviderError('AI provider 返回为空')
        message = choices[0].get('message') if isinstance(choices[0], dict) else None
        content = message.get('content') if isinstance(message, dict) else ''
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    parts.append(str(item.get('text', '')))
            content = '\n'.join(part for part in parts if part)
        content = str(content or '').strip()
        if not content:
            raise AIProviderError('AI provider 未返回有效内容')

        usage = data.get('usage') if isinstance(data.get('usage'), dict) else {}
        return {
            'provider': self.provider_name,
            'model': data.get('model') or self.model,
            'content': content,
            'usage': usage,
        }

    def embed_texts(self, texts: list[str]) -> dict:
        if not self.is_embedding_configured:
            raise AIProviderError(
                'Embedding provider 未配置完整，请检查 AI_EMBEDDING_BASE_URL / AI_EMBEDDING_API_KEY / AI_EMBEDDING_MODEL'
            )

        normalized = [str(text or '').strip() for text in texts if str(text or '').strip()]
        if not normalized:
            return {
                'provider': self.provider_name,
                'model': self.embedding_model,
                'embeddings': [],
            }

        payload = {
            'model': self.embedding_model,
            'input': normalized,
            'encoding_format': 'float',
        }
        endpoint = f'{self.embedding_base_url}/embeddings'
        req = request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.embedding_api_key}',
            },
            method='POST',
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode('utf-8')
        except error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='ignore')
            raise AIProviderError(
                f'Embedding provider HTTP {exc.code}: {detail or exc.reason}'
            ) from exc
        except error.URLError as exc:
            raise AIProviderError(f'Embedding provider 网络错误: {exc.reason}') from exc
        except Exception as exc:
            raise AIProviderError(f'Embedding provider 调用失败: {exc}') from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AIProviderError('Embedding provider 返回非 JSON 响应') from exc

        rows = data.get('data') if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise AIProviderError('Embedding provider 返回为空')

        rows = sorted(
            [item for item in rows if isinstance(item, dict)],
            key=lambda item: int(item.get('index', 0)),
        )
        embeddings = []
        for item in rows:
            vector = item.get('embedding')
            if not isinstance(vector, list):
                raise AIProviderError('Embedding provider 未返回有效向量')
            embeddings.append([float(value) for value in vector])

        if len(embeddings) != len(normalized):
            raise AIProviderError('Embedding provider 返回向量数量与输入数量不一致')

        return {
            'provider': self.provider_name,
            'model': data.get('model') or self.embedding_model,
            'embeddings': embeddings,
        }
