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
          <div class="page-panel settings-panel">
            <p class="form-hint settings-intro">
              诊断模型由模型资产清单统一管理，并在诊断中心或实时监测页面选择；此处只配置 AI 助手使用的大语言模型。
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
              <el-button @click="testAiConfig" :loading="aiConfigTesting">测试连接</el-button>
              <el-button type="primary" @click="saveAiConfig" :loading="aiConfigSaving">保存配置</el-button>
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

const aiKeyStatus = ref({
  set: false,
  preview: ''
})
const aiConfigSaving = ref(false)
const aiConfigTesting = ref(false)

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

function applyBackendAiSettings(data = {}) {
  aiConfig.value.llmProvider = normalizeProvider(data.provider)
  aiConfig.value.llmEndpoint = data.base_url || aiConfig.value.llmEndpoint
  aiConfig.value.llmModel = data.model || aiConfig.value.llmModel
  aiConfig.value.llmApiKey = ''
  aiKeyStatus.value = {
    set: Boolean(data.api_key_set),
    preview: data.api_key_preview || ''
  }
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
    applyBackendAiSettings(response.data || {})
    ElMessage.success('AI 配置已保存，新会话将立即生效')
  } catch (error) {
    ElMessage.error(error.message || 'AI 配置保存失败')
  } finally {
    aiConfigSaving.value = false
  }
}

async function testAiConfig() {
  aiConfigTesting.value = true
  try {
    const response = await settingsApi.testAiSettings(buildAiSettingsPayload())
    const result = response.data || {}
    if (result.ok) {
      ElMessage.success(`AI 连接测试成功：${result.provider || '-'} / ${result.model || '-'}`)
    } else {
      ElMessage.error(result.message || 'AI 连接测试失败')
    }
  } catch (error) {
    ElMessage.error(error.message || 'AI 连接测试失败')
  } finally {
    aiConfigTesting.value = false
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
