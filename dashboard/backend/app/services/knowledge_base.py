from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Iterable

from flask import current_app

from app.models.diagnosis_report import DiagnosisReport
from app.models.run_record import RunRecord
from app.services.ai_provider import AIProviderError, OpenAICompatibleProvider


TOKEN_RE = re.compile(r'[A-Za-z0-9_]+|[一-鿿]')
SUPPORTED_EXTS = {'.md', '.txt', '.json', '.yaml', '.yml'}
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
MAX_FILE_SIZE = 256 * 1024
RETRIEVAL_MODES = {'bm25', 'hybrid', 'semantic'}


QUERY_SPLIT_RE = re.compile(r'怎么|如何|怎样|为何|为什么|是否|能否|可以|需要|应该|请问')
MAX_RESULTS_PER_SOURCE = 1
SNIPPET_PREVIEW_CHARS = 220

INTENT_SYNONYMS = {
    'repair': ('修复', '维修', '检修', '排查', '更换', '校准', '恢复'),
    'reason': ('原因', '起因', '导致', '为什么', '为何'),
    'prevention': ('预防', '避免', '防止', '降低'),
}


class KnowledgeBaseService:
    """本地轻量知识库，支持 BM25 + optional embeddings 的 hybrid RAG。"""

    def __init__(self, app=None):
        self.app = app

    def init_app(self, app):
        self.app = app

    def _config(self) -> dict:
        return dict((self.app or current_app).config)

    def _enabled(self) -> bool:
        return bool(self._config().get('AI_ENABLE_KNOWLEDGE_BASE', False))

    def _requested_retrieval_mode(self) -> str:
        value = str(self._config().get('AI_RAG_MODE', 'hybrid')).strip().lower()
        return value if value in RETRIEVAL_MODES else 'hybrid'

    def _embedding_client(self) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(self._config())

    def _empty_index_payload(self) -> dict:
        return {
            'built_at': None,
            'documents': [],
            'chunks': [],
            'df': {},
            'avg_token_len': 0.0,
            'retrieval_mode': 'bm25',
            'requested_retrieval_mode': self._requested_retrieval_mode(),
            'embedding_enabled': False,
            'embedding_provider': '',
            'embedding_model': '',
            'embedding_error': None,
            'embedding_dim': 0,
        }

    def _runtime_dir(self) -> Path:
        path = Path(self._config()['AI_RUNTIME_DIR'])
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _index_path(self) -> Path:
        return self._runtime_dir() / 'knowledge_index.json'

    def _knowledge_root(self) -> Path:
        path = Path(self._config()['AI_KNOWLEDGE_DIR'])
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _project_doc_paths(self) -> list[Path]:
        raw = self._config().get('AI_PROJECT_DOC_PATHS', ())
        paths = []
        for item in raw:
            path = Path(item)
            if path.exists():
                paths.append(path)
        return paths

    def _index_meta(self, index: dict) -> dict:
        retrieval_mode = str(index.get('retrieval_mode') or 'bm25').strip().lower()
        if retrieval_mode not in RETRIEVAL_MODES:
            retrieval_mode = 'bm25'
        return {
            'enabled': True,
            'built_at': index.get('built_at'),
            'knowledge_root': str(self._knowledge_root()),
            'retrieval_mode': retrieval_mode,
            'requested_retrieval_mode': str(index.get('requested_retrieval_mode') or self._requested_retrieval_mode()),
            'embedding_enabled': bool(index.get('embedding_enabled', False)),
            'embedding_provider': str(index.get('embedding_provider') or ''),
            'embedding_model': str(index.get('embedding_model') or ''),
            'embedding_error': index.get('embedding_error'),
            'embedding_dim': int(index.get('embedding_dim') or 0),
        }

    def _empty_sources_payload(self) -> dict:
        index = self._empty_index_payload()
        meta = self._index_meta(index)
        meta.update(
            {
                'enabled': False,
                'total_documents': 0,
                'total_chunks': 0,
                'by_type': {},
                'items': [],
            }
        )
        return meta

    def retrieval_info(self) -> dict:
        if not self._enabled():
            return self._empty_sources_payload()
        return self._index_meta(self._load_index())

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text or '')]

    def _extract_query_focus_terms(self, query: str) -> list[str]:
        text = str(query or '').strip().lower()
        if not text:
            return []

        parts = [
            item.strip()
            for item in QUERY_SPLIT_RE.split(text)
            if item and item.strip()
        ]
        if not parts:
            parts = [text]

        focus_terms = []
        for item in parts:
            normalized = re.sub(r'[^\w\u4e00-\u9fff]+', '', item)
            if len(normalized) >= 2:
                focus_terms.append(normalized)
        return focus_terms

    def _intent_groups_for_query(self, query: str) -> list[tuple[str, ...]]:
        normalized = str(query or '').strip().lower()
        if not normalized:
            return []

        groups = []
        for synonyms in INTENT_SYNONYMS.values():
            if any(term in normalized for term in synonyms):
                groups.append(tuple(synonyms))
        return groups

    def _matches_query_intent(self, *, query: str, title: str, text: str) -> bool:
        haystack = f'{title}\n{text}'.lower()
        for group in self._intent_groups_for_query(query):
            if not any(term in haystack for term in group):
                return False
        return True

    def _build_display_snippet(self, text: str, query: str) -> str:
        normalized = str(text or '').strip()
        if not normalized:
            return ''

        lowered = normalized.lower()
        focus_terms = self._extract_query_focus_terms(query)
        match_index = -1
        match_length = 0

        for term in sorted(focus_terms, key=len, reverse=True):
            current_index = lowered.find(term.lower())
            if current_index >= 0:
                match_index = current_index
                match_length = len(term)
                break

        if match_index < 0:
            for group in self._intent_groups_for_query(query):
                for term in group:
                    current_index = lowered.find(term.lower())
                    if current_index >= 0:
                        match_index = current_index
                        match_length = len(term)
                        break
                if match_index >= 0:
                    break

        if match_index < 0:
            return normalized[:SNIPPET_PREVIEW_CHARS]

        context_before = min(12, SNIPPET_PREVIEW_CHARS // 6)
        context_after = max(SNIPPET_PREVIEW_CHARS - context_before, match_length + 80)
        start = max(0, match_index - context_before)
        end = min(len(normalized), match_index + match_length + context_after)

        if start > 0:
            boundary = max(
                normalized.rfind('。', 0, match_index),
                normalized.rfind('！', 0, match_index),
                normalized.rfind('？', 0, match_index),
                normalized.rfind('\n', 0, match_index),
            )
            if boundary >= 0:
                start = boundary + 1
            if (match_index - start) <= 12:
                start = match_index

        snippet = normalized[start:end].strip()
        if start > 0:
            snippet = f'...{snippet}'
        if end < len(normalized):
            snippet = f'{snippet}...'
        return snippet

    def _index_is_stale(self, index: dict) -> bool:
        documents = index.get('documents', [])
        if not isinstance(documents, list):
            return True

        for document in documents:
            if not isinstance(document, dict):
                return True

            source_type = str(document.get('source_type') or '')
            source_id = str(document.get('source_id') or '').strip()

            if source_type == 'report' and source_id:
                try:
                    if db.session.get(DiagnosisReport, int(source_id)) is None:
                        return True
                except Exception:
                    return True

            if source_type == 'run' and source_id:
                try:
                    if db.session.get(RunRecord, int(source_id)) is None:
                        return True
                except Exception:
                    return True

        return False

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[str]:
        normalized = str(text or '').strip()
        if not normalized:
            return []
        lines = [line.strip() for line in normalized.splitlines()]
        paragraphs = [line for line in lines if line]
        joined = '\n'.join(paragraphs) if paragraphs else normalized
        if len(joined) <= chunk_size:
            return [joined]

        chunks = []
        start = 0
        while start < len(joined):
            end = min(len(joined), start + chunk_size)
            chunks.append(joined[start:end].strip())
            if end >= len(joined):
                break
            start = max(0, end - overlap)
        return [item for item in chunks if item]

    def _iter_doc_files(self) -> Iterable[dict]:
        seen = set()
        for root in [self._knowledge_root(), *self._project_doc_paths()]:
            if root.is_file():
                candidates = [root]
            else:
                candidates = sorted(path for path in root.rglob('*') if path.is_file())
            for path in candidates:
                if path.suffix.lower() not in SUPPORTED_EXTS:
                    continue
                if path.stat().st_size > MAX_FILE_SIZE:
                    continue
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                yield {
                    'kind': 'file',
                    'source_type': 'knowledge_base',
                    'source_id': key,
                    'title': path.name,
                    'path': key,
                    'text': path.read_text(encoding='utf-8', errors='ignore'),
                    'updated_at': datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                }

    def _iter_report_records(self) -> Iterable[dict]:
        rows = (
            DiagnosisReport.query.order_by(
                DiagnosisReport.updated_at.desc(), DiagnosisReport.created_at.desc()
            )
            .limit(100)
            .all()
        )
        for report in rows:
            summary = report.get_summary()
            parts = [
                f'报告ID: {report.id}',
                f'任务ID: {report.run_id}',
                f'模型: {report.model_name}',
                f'预测类别: {report.prediction_label}',
                f'置信度: {report.confidence}',
                f'摘要: {json.dumps(summary, ensure_ascii=False)}',
            ]
            yield {
                'kind': 'report',
                'source_type': 'report',
                'source_id': str(report.id),
                'title': f'诊断报告 #{report.id}',
                'path': '',
                'text': '\n'.join(parts),
                'updated_at': report.updated_at.strftime('%Y-%m-%d %H:%M:%S') if report.updated_at else None,
            }

    def _iter_run_records(self) -> Iterable[dict]:
        rows = (
            RunRecord.query.order_by(RunRecord.updated_at.desc(), RunRecord.created_at.desc())
            .limit(100)
            .all()
        )
        for run in rows:
            result = run.get_result()
            params = run.get_params()
            parts = [
                f'任务ID: {run.id}',
                f'任务类型: {run.task}',
                f'状态: {run.status}',
                f'资产ID: {run.asset_id}',
                f'参数: {json.dumps(params, ensure_ascii=False)}',
                f'结果: {json.dumps(result, ensure_ascii=False)}',
            ]
            yield {
                'kind': 'run',
                'source_type': 'run',
                'source_id': str(run.id),
                'title': f'任务记录 #{run.id}',
                'path': '',
                'text': '\n'.join(parts),
                'updated_at': run.updated_at.strftime('%Y-%m-%d %H:%M:%S') if run.updated_at else None,
            }

    def _normalize_embedding(self, vector: list[float]) -> list[float]:
        values = [float(item) for item in vector]
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 1e-12:
            return [0.0 for _ in values]
        return [round(value / norm, 6) for value in values]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right))))

    def _embed_chunk_texts(self, chunks: list[dict]) -> tuple[bool, str, str, str | None, int]:
        provider = self._embedding_client()
        if not provider.is_embedding_configured or not chunks:
            return False, '', '', None, 0

        batch_size = max(1, int(self._config().get('AI_EMBED_BATCH_SIZE', 16)))
        raw_embeddings: list[list[float]] = []
        provider_name = provider.provider_name
        model_name = provider.embedding_model
        try:
            for offset in range(0, len(chunks), batch_size):
                batch = chunks[offset : offset + batch_size]
                result = provider.embed_texts([item['text'] for item in batch])
                provider_name = str(result.get('provider') or provider.provider_name)
                model_name = str(result.get('model') or provider.embedding_model)
                raw_embeddings.extend(result.get('embeddings') or [])
        except AIProviderError as exc:
            current_app.logger.warning('知识库 embedding 构建失败，回退为 BM25: %s', exc)
            return False, '', '', str(exc), 0

        if len(raw_embeddings) != len(chunks):
            error_text = 'embedding 数量与 chunk 数量不一致'
            current_app.logger.warning('知识库 embedding 构建失败，回退为 BM25: %s', error_text)
            return False, '', '', error_text, 0

        normalized = [self._normalize_embedding(item) for item in raw_embeddings]
        for chunk, embedding in zip(chunks, normalized):
            chunk['embedding'] = embedding
        dim = len(normalized[0]) if normalized else 0
        return True, provider_name, model_name, None, dim

    def rebuild_index(self) -> dict:
        if not self._enabled():
            path = self._index_path()
            if path.exists():
                path.unlink()
            return {
                'enabled': False,
                'built_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'documents': 0,
                'chunks': 0,
                'knowledge_root': str(self._knowledge_root()),
                'retrieval_mode': 'bm25',
                'requested_retrieval_mode': self._requested_retrieval_mode(),
                'embedding_enabled': False,
                'embedding_provider': '',
                'embedding_model': '',
                'embedding_error': None,
                'embedding_dim': 0,
            }

        documents = []
        chunks = []
        doc_id = 0
        for item in [
            *list(self._iter_doc_files()),
            *list(self._iter_report_records()),
            *list(self._iter_run_records()),
        ]:
            doc_id += 1
            documents.append(
                {
                    'doc_id': doc_id,
                    'kind': item['kind'],
                    'source_type': item['source_type'],
                    'source_id': item['source_id'],
                    'title': item['title'],
                    'path': item['path'],
                    'updated_at': item['updated_at'],
                }
            )
            for idx, chunk_text in enumerate(self._chunk_text(item['text'])):
                tokens = self._tokenize(chunk_text)
                if not tokens:
                    continue
                token_counts = Counter(tokens)
                chunks.append(
                    {
                        'doc_id': doc_id,
                        'chunk_id': f'{doc_id}:{idx}',
                        'source_type': item['source_type'],
                        'source_id': item['source_id'],
                        'title': item['title'],
                        'path': item['path'],
                        'text': chunk_text,
                        'token_counts': dict(token_counts),
                        'token_len': sum(token_counts.values()),
                    }
                )

        df = Counter()
        for chunk in chunks:
            df.update(chunk['token_counts'].keys())

        avg_len = sum(chunk['token_len'] for chunk in chunks) / max(1, len(chunks))
        requested_mode = self._requested_retrieval_mode()
        retrieval_mode = 'bm25'
        embedding_enabled = False
        embedding_provider = ''
        embedding_model = ''
        embedding_error = None
        embedding_dim = 0
        if requested_mode in {'hybrid', 'semantic'}:
            (
                embedding_enabled,
                embedding_provider,
                embedding_model,
                embedding_error,
                embedding_dim,
            ) = self._embed_chunk_texts(chunks)
            if embedding_enabled:
                retrieval_mode = requested_mode

        payload = {
            'built_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'documents': documents,
            'chunks': chunks,
            'df': dict(df),
            'avg_token_len': avg_len,
            'retrieval_mode': retrieval_mode,
            'requested_retrieval_mode': requested_mode,
            'embedding_enabled': embedding_enabled,
            'embedding_provider': embedding_provider,
            'embedding_model': embedding_model,
            'embedding_error': embedding_error,
            'embedding_dim': embedding_dim,
        }
        self._index_path().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        return {
            'enabled': True,
            'built_at': payload['built_at'],
            'documents': len(documents),
            'chunks': len(chunks),
            'knowledge_root': str(self._knowledge_root()),
            'retrieval_mode': retrieval_mode,
            'requested_retrieval_mode': requested_mode,
            'embedding_enabled': embedding_enabled,
            'embedding_provider': embedding_provider,
            'embedding_model': embedding_model,
            'embedding_error': embedding_error,
            'embedding_dim': embedding_dim,
        }

    def _load_index(self) -> dict:
        if not self._enabled():
            return self._empty_index_payload()
        path = self._index_path()
        if not path.exists():
            self.rebuild_index()
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            self.rebuild_index()
            data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            return self._empty_index_payload()
        if self._index_is_stale(data):
            self.rebuild_index()
            data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else self._empty_index_payload()

    def list_sources(self) -> dict:
        if not self._enabled():
            return self._empty_sources_payload()
        index = self._load_index()
        documents = index.get('documents', [])
        grouped = Counter(doc.get('source_type', 'unknown') for doc in documents)
        payload = self._index_meta(index)
        payload.update(
            {
                'total_documents': len(documents),
                'total_chunks': len(index.get('chunks', [])),
                'by_type': dict(grouped),
                'items': documents,
            }
        )
        return payload

    def _bm25_scores(self, index: dict, query_tokens: list[str]) -> dict[str, float]:
        if not query_tokens:
            return {}
        df = index.get('df', {})
        chunks = index.get('chunks', [])
        avg_len = float(index.get('avg_token_len') or 1.0)
        total_docs = max(1, len(chunks))
        k1 = 1.2
        b = 0.75
        scores: dict[str, float] = {}
        for chunk in chunks:
            token_counts = chunk.get('token_counts', {})
            token_len = max(1, int(chunk.get('token_len') or 1))
            score = 0.0
            for token in query_tokens:
                tf = float(token_counts.get(token, 0))
                if tf <= 0:
                    continue
                doc_freq = int(df.get(token, 0))
                idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                denom = tf + k1 * (1 - b + b * token_len / avg_len)
                score += idf * ((tf * (k1 + 1)) / max(1e-9, denom))
            if score > 0:
                scores[str(chunk.get('chunk_id'))] = score
        return scores

    def _semantic_scores(self, index: dict, query: str) -> tuple[dict[str, float], str]:
        if not query or not index.get('embedding_enabled'):
            return {}, 'bm25'
        provider = self._embedding_client()
        if not provider.is_embedding_configured:
            return {}, 'bm25'
        try:
            result = provider.embed_texts([query])
        except AIProviderError as exc:
            current_app.logger.warning('知识库 query embedding 失败，回退为 BM25: %s', exc)
            return {}, 'bm25'
        embeddings = result.get('embeddings') or []
        if not embeddings:
            return {}, 'bm25'
        query_vector = self._normalize_embedding(embeddings[0])
        scores = {}
        for chunk in index.get('chunks', []):
            vector = chunk.get('embedding')
            if not isinstance(vector, list):
                continue
            score = self._cosine_similarity(query_vector, vector)
            if score > 0:
                scores[str(chunk.get('chunk_id'))] = score
        return scores, self._index_meta(index)['retrieval_mode']

    def search(self, query: str, *, top_k: int = 5) -> list[dict]:
        if not self._enabled():
            return []
        clean_query = str(query or '').strip()
        if not clean_query:
            return []

        index = self._load_index()
        chunks = index.get('chunks', [])
        if not chunks:
            return []

        lexical_scores = self._bm25_scores(index, self._tokenize(clean_query))
        retrieval_mode = self._index_meta(index)['retrieval_mode']
        semantic_scores = {}
        if retrieval_mode in {'hybrid', 'semantic'}:
            semantic_scores, actual_mode = self._semantic_scores(index, clean_query)
            if not semantic_scores:
                retrieval_mode = 'bm25'
            elif actual_mode in RETRIEVAL_MODES:
                retrieval_mode = actual_mode

        max_lexical = max(lexical_scores.values(), default=0.0)
        results = []
        seen = set()
        source_counts = Counter()
        for chunk in chunks:
            chunk_id = str(chunk.get('chunk_id'))
            lexical_raw = float(lexical_scores.get(chunk_id, 0.0))
            lexical = lexical_raw / max_lexical if max_lexical > 0 else 0.0
            semantic = float(semantic_scores.get(chunk_id, 0.0))

            if retrieval_mode == 'semantic':
                score = semantic
            elif retrieval_mode == 'hybrid':
                score = 0.45 * lexical + 0.55 * semantic
            else:
                score = lexical_raw

            if score <= 0:
                continue

            title = str(chunk.get('title') or '')
            text = str(chunk.get('text') or '')
            if not self._matches_query_intent(
                query=clean_query,
                title=title,
                text=text,
            ):
                continue

            key = (chunk.get('source_type'), chunk.get('source_id'), chunk.get('text'))
            if key in seen:
                continue
            source_key = (chunk.get('source_type'), chunk.get('source_id'))
            if source_counts[source_key] >= MAX_RESULTS_PER_SOURCE:
                continue
            seen.add(key)
            source_counts[source_key] += 1
            results.append(
                {
                    'source_type': chunk.get('source_type'),
                    'source_id': chunk.get('source_id'),
                    'title': title,
                    'path': chunk.get('path'),
                    'snippet': self._build_display_snippet(text, clean_query),
                    'score': round(score, 4),
                    'retrieval_mode': retrieval_mode,
                    'score_breakdown': {
                        'lexical': round(lexical, 4),
                        'lexical_raw': round(lexical_raw, 4),
                        'semantic': round(semantic, 4),
                    },
                }
            )
        results.sort(key=lambda item: item['score'], reverse=True)
        return results[: max(1, int(top_k))]


knowledge_base = KnowledgeBaseService()
