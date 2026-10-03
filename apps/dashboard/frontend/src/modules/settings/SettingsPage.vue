<template>
  <div class="page-stack settings-page">
    <header class="page-header">
      <div class="page-header__left">
        <h1 class="page-title">系统设置</h1>
        <p class="page-description">管理用户信息、界面外观与模型接入配置。</p>
      </div>
    </header>

    <div class="settings-layout">
      <aside class="settings-nav">
        <div class="settings-nav__item" :class="{ 'is-active': activeTab === 'account' }" @click="activeTab = 'account'">
          用户信息
        </div>
        <div class="settings-nav__item" :class="{ 'is-active': activeTab === 'appearance' }" @click="activeTab = 'appearance'">
          外观与主题
        </div>
        <div class="settings-nav__item" :class="{ 'is-active': activeTab === 'ai' }" @click="activeTab = 'ai'">
          AI 接入
        </div>
      </aside>

      <main class="settings-content">
        <section v-show="activeTab === 'account'" class="settings-section">
          <h2 class="settings-section__title">用户信息</h2>
          <div class="page-panel settings-panel">
            <div class="form-row">
              <label>登录账号</label>
              <div class="form-value">{{ storedUser?.username || '未知用户' }}</div>
            </div>
            <div class="form-row">
              <label>角色权限</label>
              <div class="form-value">
                <span class="status-badge status-badge--ok">{{ roleLabel }}</span>
              </div>
            </div>
            <div class="form-row">
              <label>账号状态</label>
              <div class="form-value">正常</div>
            </div>
            <div class="form-actions form-actions--spaced">
              <el-button type="danger" plain @click="logout">退出登录</el-button>
            </div>
          </div>
        </section>

        <section v-show="activeTab === 'appearance'" class="settings-section">
          <h2 class="settings-section__title">外观与主题</h2>
          <div class="page-panel settings-panel">
            <div class="theme-picker">
              <div class="theme-card" :class="{ 'is-active': !isDarkMode }" @click="setTheme(false)">
                <div class="theme-card__preview theme-card__preview--light">
                  <div class="preview-header"></div>
                  <div class="preview-body"></div>
                </div>
                <div class="theme-card__label">浅色模式 (Light)</div>
              </div>

              <div class="theme-card" :class="{ 'is-active': isDarkMode }" @click="setTheme(true)">
                <div class="theme-card__preview theme-card__preview--dark">
                  <div class="preview-header"></div>
                  <div class="preview-body"></div>
                </div>
                <div class="theme-card__label">深色工业风 (Dark)</div>
              </div>
            </div>
          </div>
        </section>

        <section v-show="activeTab === 'ai'" class="settings-section">
          <h2 class="settings-section__title">AI 接入配置</h2>

          <h3 class="settings-config-title">AI 助手模型</h3>
          <div class="page-panel settings-panel">
            <p class="form-hint settings-intro">
              仅用于 AI 助手的对话与报告解读，不会作为知识库 Embedding 配置使用。
            </p>

            <div class="form-grid">
              <div class="form-group">
                <label>供应商名称 (Provider)</label>
                <div class="provider-field" :class="{ 'is-custom': aiConfig.llmProvider === 'custom' }">
                  <select v-model="aiConfig.llmProvider" class="styled-select provider-field__select">
                    <option value="openai">OpenAI</option>
                    <option value="deepseek">DeepSeek</option>
                    <option value="aliyun">阿里云百炼 (Qwen)</option>
                    <option value="custom">自定义</option>
                  </select>
                  <input
                    v-if="aiConfig.llmProvider === 'custom'"
                    v-model="aiConfig.customProviderName"
                    type="text"
                    class="styled-input provider-field__custom"
                    placeholder="输入自定义 provider..."
                  />
                </div>
              </div>

              <div class="form-group">
                <label>API 调用格式 (Format)</label>
                <select v-model="aiConfig.llmFormat" class="styled-select">
                  <option value="openai">OpenAI 兼容格式 (默认)</option>
                  <option value="anthropic" disabled>Anthropic 格式 (暂未接入)</option>
                  <option value="ollama" disabled>Ollama 格式 (暂未接入)</option>
                </select>
              </div>
            </div>

            <div class="form-group form-group--spaced">
              <label>请求地址 (API Endpoint)</label>
              <input v-model="aiConfig.llmEndpoint" type="text" class="styled-input" placeholder="https://api.openai.com/v1" />
              <p class="form-hint">填写 OpenAI-compatible Base URL，例如 /v1/chat/completions 前面的部分。</p>
            </div>

            <div class="form-group form-group--spaced">
              <label>模型名称 (Model)</label>
              <input v-model="aiConfig.llmModel" type="text" class="styled-input" placeholder="deepseek-ai/DeepSeek-V3" />
              <p class="form-hint">作为 chat/completions 请求中的 model 参数。</p>
            </div>

            <div class="form-group form-group--spaced">
              <label>API 密钥 (API Key)</label>
              <input
                v-model="aiConfig.llmApiKey"
                type="password"
                class="styled-input"
                :placeholder="aiKeyStatus.set ? '已配置，留空不修改' : 'sk-...'"
              />
              <p v-if="!aiKeyStatus.set" class="form-hint">保存后写入后端 runtime 配置，不再保存到浏览器 localStorage。</p>
              <p v-else class="form-hint">已配置：{{ aiKeyStatus.preview || '******' }}，留空保存时不会覆盖。</p>
            </div>

            <div class="form-actions form-actions--footer">
              <el-button @click="testAiConfig" :loading="aiConfigTesting">测试 AI 助手</el-button>
              <el-button type="primary" @click="saveAiConfig" :loading="aiConfigSaving">保存 AI 助手配置</el-button>
            </div>
          </div>

          <h3 class="settings-config-title settings-config-title--spaced">知识库检索与 Embedding</h3>
          <div class="page-panel settings-panel">
            <p class="form-hint settings-intro">
              独立配置知识库向量接口。Hybrid 和 Semantic 模式要求地址、密钥与模型均配置完整。
            </p>

            <div class="form-grid">
              <div class="form-group">
                <label>检索模式 (RAG Mode)</label>
                <select v-model="embeddingConfig.ragMode" class="styled-select">
                  <option value="bm25">BM25 关键词检索</option>
                  <option value="hybrid">Hybrid 混合检索</option>
                  <option value="semantic">Semantic 向量检索</option>
                </select>
              </div>

              <div class="form-group">
                <label>Embedding 模型</label>
                <input
                  v-model="embeddingConfig.model"
                  type="text"
                  class="styled-input"
                  placeholder="Qwen/Qwen3-Embedding-8B"
                />
              </div>
            </div>

            <div class="form-group form-group--spaced">
              <label>Embedding 请求地址</label>
              <input
                v-model="embeddingConfig.endpoint"
                type="text"
                class="styled-input"
                placeholder="https://api.siliconflow.cn/v1"
              />
              <p class="form-hint">填写 OpenAI-compatible Base URL，后端会调用其 /embeddings 接口。</p>
            </div>

            <div class="form-group form-group--spaced">
              <label>Embedding API 密钥</label>
              <input
                v-model="embeddingConfig.apiKey"
                type="password"
                class="styled-input"
                :placeholder="embeddingKeyStatus.set ? '已配置，留空不修改' : 'sk-...'"
              />
              <p v-if="!embeddingKeyStatus.set" class="form-hint">该密钥独立保存，不会复用 AI 助手密钥。</p>
              <p v-else class="form-hint">已配置：{{ embeddingKeyStatus.preview || '******' }}，留空保存时不会覆盖。</p>
            </div>

            <div class="form-actions form-actions--footer">
              <el-button @click="testEmbeddingConfig" :loading="embeddingConfigTesting">测试 Embedding</el-button>
              <el-button type="primary" @click="saveEmbeddingConfig" :loading="embeddingConfigSaving">保存 Embedding 配置</el-button>
            </div>
          </div>
        </section>
      </main>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElButton } from 'element-plus'
import { authApi, settingsApi } from '@/shared/services'
import { parseStoredUser, getRoleLabel, clearStoredSession } from '@/shared/utils/user'

const router = useRouter()
const activeTab = ref('account')

const storedUser = computed(() => parseStoredUser(localStorage.getItem('user')))
const roleLabel = computed(() => getRoleLabel(storedUser.value))

async function logout() {
  try {
    await authApi.logout()
  } catch {
  } finally {
    clearStoredSession()
    router.push('/login')
  }
}

const isDarkMode = ref(localStorage.getItem('theme-mode') === 'dark')

function setTheme(dark) {
  isDarkMode.value = dark
  localStorage.setItem('theme-mode', dark ? 'dark' : 'light')
  if (dark) {
    document.documentElement.setAttribute('data-theme', 'dark')
  } else {
    document.documentElement.removeAttribute('data-theme')
  }
}

const aiConfig = ref({
  llmProvider: 'openai',
  customProviderName: '',
  llmFormat: 'openai',
  llmEndpoint: 'https://api.openai.com/v1',
  llmModel: 'deepseek-ai/DeepSeek-V3',
  llmApiKey: ''
})

const embeddingConfig = ref({
  endpoint: '',
  model: '',
  apiKey: '',
  ragMode: 'bm25'
})

const aiKeyStatus = ref({
  set: false,
  preview: ''
})
const embeddingKeyStatus = ref({
  set: false,
  preview: ''
})
const aiConfigSaving = ref(false)
const aiConfigTesting = ref(false)
const embeddingConfigSaving = ref(false)
const embeddingConfigTesting = ref(false)

function normalizeProvider(provider) {
  const value = String(provider || '').trim()
  if (['openai', 'deepseek', 'aliyun'].includes(value)) {
    return value
  }
  if (!value || value === 'openai_compatible') {
    return 'openai'
  }
  aiConfig.value.customProviderName = value
  return 'custom'
}

function applyBackendLlmSettings(data = {}) {
  aiConfig.value.llmProvider = normalizeProvider(data.provider)
  aiConfig.value.llmEndpoint = data.base_url || ''
  aiConfig.value.llmModel = data.model || ''
  aiConfig.value.llmApiKey = ''
  aiKeyStatus.value = {
    set: Boolean(data.api_key_set),
    preview: data.api_key_preview || ''
  }
}

function applyBackendEmbeddingSettings(data = {}) {
  embeddingConfig.value.endpoint = data.embedding_base_url || ''
  embeddingConfig.value.model = data.embedding_model || ''
  embeddingConfig.value.apiKey = ''
  embeddingConfig.value.ragMode = data.rag_mode || 'bm25'
  embeddingKeyStatus.value = {
    set: Boolean(data.embedding_api_key_set),
    preview: data.embedding_api_key_preview || ''
  }
}

function applyBackendAiSettings(data = {}) {
  applyBackendLlmSettings(data)
  applyBackendEmbeddingSettings(data)
}

function buildAiSettingsPayload() {
  const provider =
    aiConfig.value.llmProvider === 'custom'
      ? aiConfig.value.customProviderName.trim() || 'openai_compatible'
      : aiConfig.value.llmProvider
  const payload = {
    provider,
    base_url: aiConfig.value.llmEndpoint.trim(),
    model: aiConfig.value.llmModel.trim()
  }
  const apiKey = aiConfig.value.llmApiKey.trim()
  if (apiKey) {
    payload.api_key = apiKey
  }
  return payload
}

function buildEmbeddingSettingsPayload() {
  const payload = {
    embedding_base_url: embeddingConfig.value.endpoint.trim(),
    embedding_model: embeddingConfig.value.model.trim(),
    rag_mode: embeddingConfig.value.ragMode
  }
  const apiKey = embeddingConfig.value.apiKey.trim()
  if (apiKey) {
    payload.embedding_api_key = apiKey
  }
  return payload
}

onMounted(async () => {
  try {
    const response = await settingsApi.getAiSettings()
    applyBackendAiSettings(response.data || {})
  } catch (error) {
    ElMessage.error(error.message || 'AI 配置加载失败')
  }
})

async function saveAiConfig() {
  aiConfigSaving.value = true
  try {
    const response = await settingsApi.updateAiSettings(buildAiSettingsPayload())
    applyBackendLlmSettings(response.data || {})
    ElMessage.success('AI 助手配置已保存，新会话将立即生效')
  } catch (error) {
    ElMessage.error(error.message || 'AI 助手配置保存失败')
  } finally {
    aiConfigSaving.value = false
  }
}

async function saveEmbeddingConfig() {
  embeddingConfigSaving.value = true
  try {
    const response = await settingsApi.updateAiSettings(buildEmbeddingSettingsPayload())
    applyBackendEmbeddingSettings(response.data || {})
    ElMessage.success('Embedding 配置已保存，请重建知识库索引')
  } catch (error) {
    ElMessage.error(error.message || 'Embedding 配置保存失败')
  } finally {
    embeddingConfigSaving.value = false
  }
}

async function testAiConfig() {
  aiConfigTesting.value = true
  try {
    const response = await settingsApi.testAiSettings(buildAiSettingsPayload())
    const result = response.data || {}
    if (result.ok) {
      ElMessage.success(`AI 助手连接测试成功：${result.provider || '-'} / ${result.model || '-'}`)
    } else {
      ElMessage.error(result.message || 'AI 助手连接测试失败')
    }
  } catch (error) {
    ElMessage.error(error.message || 'AI 助手连接测试失败')
  } finally {
    aiConfigTesting.value = false
  }
}

async function testEmbeddingConfig() {
  embeddingConfigTesting.value = true
  try {
    const response = await settingsApi.testEmbeddingSettings(buildEmbeddingSettingsPayload())
    const result = response.data || {}
    if (result.ok) {
      ElMessage.success(`Embedding 连接测试成功：${result.provider || '-'} / ${result.model || '-'}`)
    } else {
      ElMessage.error(result.message || 'Embedding 连接测试失败')
    }
  } catch (error) {
    ElMessage.error(error.message || 'Embedding 连接测试失败')
  } finally {
    embeddingConfigTesting.value = false
  }
}
</script>

<style scoped>
.settings-layout {
  display: flex;
  gap: var(--space-xl);
  align-items: flex-start;
}

.settings-nav {
  width: 200px;
  display: flex;
  flex-direction: column;
  gap: var(--space-xs);
  flex-shrink: 0;
}

.settings-nav__item {
  padding: 10px 14px;
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.settings-nav__item:hover {
  background: var(--bg-panel-raised);
  color: var(--text-primary);
}

.settings-nav__item.is-active {
  background: var(--brand-soft);
  color: var(--brand-strong);
  font-weight: 600;
}

.settings-content {
  flex: 1;
  min-width: 0;
  max-width: 800px;
}

.settings-section__title {
  margin: 0 0 var(--space-md);
  font-size: 16px;
  font-weight: 600;
}

.settings-config-title {
  margin: 0 0 var(--space-sm);
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.settings-config-title--spaced {
  margin-top: var(--space-xl);
}

.settings-panel {
  padding: var(--space-xl);
}

.form-row {
  display: flex;
  padding: 12px 0;
  border-bottom: 1px solid var(--line-faint);
}
.form-row:last-of-type {
  border-bottom: none;
}
.form-row label {
  width: 120px;
  color: var(--text-tertiary);
  font-size: 13px;
}
.form-value {
  color: var(--text-primary);
  font-weight: 500;
  font-size: 14px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.form-group--spaced {
  margin-top: var(--space-lg);
}
.form-group label {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary);
}
.form-hint {
  margin: 0;
  font-size: 12px;
  color: var(--text-tertiary);
}

.styled-select, .styled-input {
  width: 100%;
  max-width: 400px;
  height: 36px;
  padding: 0 12px;
  border-radius: var(--radius-md);
  border: 1px solid var(--line-soft);
  background: var(--bg-canvas);
  color: var(--text-primary);
  outline: none;
  transition: border-color 0.2s;
}
.styled-select:focus, .styled-input:focus {
  border-color: var(--brand);
}

.provider-field {
  display: flex;
  gap: var(--space-sm);
}

.provider-field__select {
  flex: 1;
}

.provider-field.is-custom .provider-field__select {
  flex: 0 0 140px;
}

.provider-field__custom {
  flex: 1;
}

.theme-picker {
  display: flex;
  gap: var(--space-lg);
}

.theme-card {
  width: 160px;
  cursor: pointer;
  border-radius: var(--radius-md);
  transition: transform 0.2s;
}

.theme-card:hover {
  transform: translateY(-2px);
}

.theme-card__preview {
  height: 100px;
  border-radius: var(--radius-md);
  border: 2px solid var(--line-soft);
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  transition: border-color 0.2s;
}

.theme-card.is-active .theme-card__preview {
  border-color: var(--brand);
}

.theme-card__preview--light {
  background: #f4f5f7;
}
.theme-card__preview--light .preview-header { background: #fff; height: 16px; border-radius: 4px; }
.theme-card__preview--light .preview-body { background: #fff; flex: 1; border-radius: 4px; }

.theme-card__preview--dark {
  --theme-preview-canvas: #1b2230;
  --theme-preview-panel: #242c3a;

  background: var(--theme-preview-canvas);
}
.theme-card__preview--dark .preview-header { background: var(--theme-preview-panel); height: 16px; border-radius: 4px; }
.theme-card__preview--dark .preview-body { background: var(--theme-preview-panel); flex: 1; border-radius: 4px; }

.theme-card__label {
  margin-top: 12px;
  text-align: center;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary);
}

.theme-card.is-active .theme-card__label {
  color: var(--brand);
  font-weight: 600;
}

.form-actions--spaced {
  margin-top: var(--space-xl);
}

.form-actions--footer {
  margin-top: var(--space-xl);
  padding-top: var(--space-lg);
  border-top: 1px solid var(--line-faint);
}

.settings-intro {
  margin-bottom: var(--space-lg);
}
.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  max-width: 560px;
}
.form-grid .styled-select {
  max-width: 100%;
}
.styled-input {
  max-width: 560px;
}
</style>
