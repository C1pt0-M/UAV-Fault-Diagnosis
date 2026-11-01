import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const routerPush = vi.hoisted(() => vi.fn())
const authApiMock = vi.hoisted(() => ({
  logout: vi.fn()
}))
const settingsApiMock = vi.hoisted(() => ({
  getAiSettings: vi.fn(),
  updateAiSettings: vi.fn(),
  testAiSettings: vi.fn(),
  testEmbeddingSettings: vi.fn()
}))
const messageMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn()
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPush })
}))

vi.mock('element-plus', () => ({
  ElMessage: messageMock,
  ElButton: {
    name: 'ElButton',
    emits: ['click'],
    template: '<button type="button" @click="$emit(\'click\')"><slot /></button>'
  }
}))

vi.mock('@/shared/services', () => ({
  authApi: authApiMock,
  settingsApi: settingsApiMock
}))

import SettingsPage from '@/modules/settings/SettingsPage.vue'
import { applyRuntimeCapabilities } from '@/shared/state/runtimeCapabilities'

function mountSettingsPage() {
  return mount(SettingsPage)
}

describe('SettingsPage AI settings', () => {
  beforeEach(() => {
    routerPush.mockReset()
    authApiMock.logout.mockReset()
    settingsApiMock.getAiSettings.mockReset()
    settingsApiMock.updateAiSettings.mockReset()
    settingsApiMock.testAiSettings.mockReset()
    settingsApiMock.testEmbeddingSettings.mockReset()
    messageMock.success.mockReset()
    messageMock.error.mockReset()
    messageMock.warning.mockReset()
    localStorage.clear()
    applyRuntimeCapabilities({ ai: true, telemetry: true })
    localStorage.setItem('user', JSON.stringify({ username: 'admin', role: 'admin' }))
    settingsApiMock.getAiSettings.mockResolvedValue({
      data: {
        provider: 'deepseek',
        base_url: 'https://runtime.example/v1',
        model: 'deepseek-chat',
        api_key_set: true,
        api_key_preview: 'runt...-key',
        embedding_base_url: 'https://embedding.example/v1',
        embedding_model: 'embedding-model',
        embedding_api_key_set: true,
        embedding_api_key_preview: 'embe...-key',
        rag_mode: 'hybrid'
      }
    })
  })

  it('loads AI settings from backend instead of localStorage', async () => {
    localStorage.setItem(
      'ai-settings',
      JSON.stringify({
        llmEndpoint: 'https://local-only.example/v1',
        llmApiKey: 'local-secret'
      })
    )

    const wrapper = mountSettingsPage()
    await flushPromises()

    expect(settingsApiMock.getAiSettings).toHaveBeenCalledTimes(1)
    expect(wrapper.vm.aiConfig.llmProvider).toBe('deepseek')
    expect(wrapper.vm.aiConfig.llmEndpoint).toBe('https://runtime.example/v1')
    expect(wrapper.vm.aiConfig.llmModel).toBe('deepseek-chat')
    expect(wrapper.vm.aiConfig.llmApiKey).toBe('')
    expect(wrapper.vm.embeddingConfig.endpoint).toBe('https://embedding.example/v1')
    expect(wrapper.vm.embeddingConfig.model).toBe('embedding-model')
    expect(wrapper.vm.embeddingConfig.ragMode).toBe('hybrid')
    expect(wrapper.vm.embeddingConfig.apiKey).toBe('')
    expect(wrapper.text()).toContain('runt...-key')
    expect(wrapper.text()).toContain('embe...-key')
  })

  it('saves AI settings to backend and does not write ai-settings localStorage', async () => {
    settingsApiMock.updateAiSettings.mockResolvedValue({
      data: {
        provider: 'openai',
        base_url: 'https://saved.example/v1',
        model: 'gpt-4o-mini',
        api_key_set: true,
        api_key_preview: 'save...-key'
      }
    })
    const wrapper = mountSettingsPage()
    await flushPromises()

    wrapper.vm.aiConfig.llmProvider = 'openai'
    wrapper.vm.aiConfig.llmEndpoint = 'https://saved.example/v1'
    wrapper.vm.aiConfig.llmModel = 'gpt-4o-mini'
    wrapper.vm.aiConfig.llmApiKey = 'saved-secret-key'
    await wrapper.vm.saveAiConfig()
    await flushPromises()

    expect(settingsApiMock.updateAiSettings).toHaveBeenCalledWith({
      provider: 'openai',
      base_url: 'https://saved.example/v1',
      model: 'gpt-4o-mini',
      api_key: 'saved-secret-key'
    })
    expect(localStorage.getItem('ai-settings')).toBeNull()
    expect(messageMock.success).toHaveBeenCalled()
  })

  it('saves explicit Embedding settings independently from AI assistant settings', async () => {
    settingsApiMock.updateAiSettings.mockResolvedValue({
      data: {
        embedding_base_url: 'https://saved-embedding.example/v1',
        embedding_model: 'saved-embedding-model',
        embedding_api_key_set: true,
        embedding_api_key_preview: 'save...-key',
        rag_mode: 'semantic'
      }
    })
    const wrapper = mountSettingsPage()
    await flushPromises()

    wrapper.vm.embeddingConfig.endpoint = 'https://saved-embedding.example/v1'
    wrapper.vm.embeddingConfig.model = 'saved-embedding-model'
    wrapper.vm.embeddingConfig.apiKey = 'saved-embedding-secret'
    wrapper.vm.embeddingConfig.ragMode = 'semantic'
    await wrapper.vm.saveEmbeddingConfig()
    await flushPromises()

    expect(settingsApiMock.updateAiSettings).toHaveBeenCalledWith({
      embedding_base_url: 'https://saved-embedding.example/v1',
      embedding_model: 'saved-embedding-model',
      embedding_api_key: 'saved-embedding-secret',
      rag_mode: 'semantic'
    })
    expect(wrapper.vm.aiConfig.llmEndpoint).toBe('https://runtime.example/v1')
    expect(wrapper.vm.embeddingConfig.apiKey).toBe('')
    expect(wrapper.text()).toContain('save...-key')
  })

  it('tests backend AI connection using current form values', async () => {
    settingsApiMock.testAiSettings.mockResolvedValue({
      data: { ok: true, provider: 'deepseek', model: 'deepseek-chat' }
    })
    const wrapper = mountSettingsPage()
    await flushPromises()

    wrapper.vm.aiConfig.llmApiKey = 'candidate-secret'
    await wrapper.vm.testAiConfig()
    await flushPromises()

    expect(settingsApiMock.testAiSettings).toHaveBeenCalledWith({
      provider: 'deepseek',
      base_url: 'https://runtime.example/v1',
      model: 'deepseek-chat',
      api_key: 'candidate-secret'
    })
    expect(messageMock.success).toHaveBeenCalled()
  })

  it.each(['disabled', 'non-admin'])('does not load or expose AI settings when %s', async (mode) => {
    if (mode === 'disabled') applyRuntimeCapabilities({ ai: false })
    else localStorage.setItem('user', JSON.stringify({ username: 'operator', role: 'user' }))
    const wrapper = mountSettingsPage()
    await flushPromises()
    expect(settingsApiMock.getAiSettings).not.toHaveBeenCalled()
    expect(wrapper.findAll('.settings-nav__item').map(item => item.text())).not.toContain('AI 接入')
    expect(wrapper.findAll('input[type="password"]')).toHaveLength(0)
    expect(wrapper.get('[role="status"]').text()).toContain('账号与外观设置仍可使用')
    await wrapper.vm.saveAiConfig()
    await wrapper.vm.saveEmbeddingConfig()
    await wrapper.vm.testAiConfig()
    await wrapper.vm.testEmbeddingConfig()
    expect(settingsApiMock.updateAiSettings).not.toHaveBeenCalled()
    expect(settingsApiMock.testAiSettings).not.toHaveBeenCalled()
    expect(settingsApiMock.testEmbeddingSettings).not.toHaveBeenCalled()
  })

  it('returns to account settings if AI capability is revoked', async () => {
    const wrapper = mountSettingsPage()
    await flushPromises()
    wrapper.vm.activeTab = 'ai'
    applyRuntimeCapabilities({ ai: false })
    await flushPromises()
    expect(wrapper.vm.activeTab).toBe('account')
    expect(wrapper.findAll('input[type="password"]')).toHaveLength(0)
  })

  it('tests Embedding connection using only Embedding form values', async () => {
    settingsApiMock.testEmbeddingSettings.mockResolvedValue({
      data: { ok: true, provider: 'openai_compatible', model: 'embedding-model' }
    })
    const wrapper = mountSettingsPage()
    await flushPromises()

    wrapper.vm.embeddingConfig.apiKey = 'embedding-candidate-secret'
    await wrapper.vm.testEmbeddingConfig()
    await flushPromises()

    expect(settingsApiMock.testEmbeddingSettings).toHaveBeenCalledWith({
      embedding_base_url: 'https://embedding.example/v1',
      embedding_model: 'embedding-model',
      embedding_api_key: 'embedding-candidate-secret',
      rag_mode: 'hybrid'
    })
    expect(messageMock.success).toHaveBeenCalled()
  })
})
