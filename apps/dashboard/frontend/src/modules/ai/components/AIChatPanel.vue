<template>
  <article class="section-card ai-chat">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ aiCopy.chatTitle }}</div>
        <div class="section-card__desc">{{ aiCopy.chatDescription }}</div>
      </div>
    </div>

    <div class="section-card__body">
      <div ref="messageStreamRef" class="message-stream" v-loading="loadingMessages">
        <div v-if="messages.length === 0" class="empty-chat">
          {{ aiCopy.emptyChat }}
        </div>

        <article
          v-for="item in messages"
          :key="item.id"
          class="message-card"
          :class="item.role"
        >
          <div class="message-card__role">{{ item.role === 'assistant' ? aiCopy.heroTag : aiCopy.userRole }}</div>
          <div v-if="item.role === 'assistant' && parseAssistantSections(item.content).summary" class="message-card__summary">
            {{ parseAssistantSections(item.content).summary }}
          </div>
          <div class="message-card__content">
            <template v-if="item.role === 'assistant'">
              <p
                v-for="paragraph in parseAssistantSections(item.content).paragraphs"
                :key="paragraph"
                class="message-card__paragraph"
              >
                {{ paragraph }}
              </p>
            </template>
            <template v-else>
              {{ item.content }}
            </template>
          </div>

          <details
            v-if="item.role === 'assistant' && item.citations?.length"
            class="message-card__references"
          >
            <summary class="message-card__references-title">{{ aiCopy.referencesLabel }}（{{ item.citations.length }}）</summary>
            <div class="reference-list">
              <article
                v-for="citation in item.citations"
                :key="`${citation.source_type}-${citation.source_id}-${citation.title}`"
                class="reference-item"
              >
                <div class="reference-item__title">
                  {{ formatSourceType(citation.source_type) }}: {{ citation.title }}
                </div>
                <div v-if="citation.snippet" class="reference-item__snippet">
                  {{ aiCopy.referenceSnippet }}: {{ citation.snippet }}
                </div>
              </article>
            </div>
          </details>
        </article>
      </div>

      <div class="composer-box">
        <el-input
          v-model="draftComposer"
          type="textarea"
          :rows="4"
          resize="none"
          :placeholder="aiCopy.composerPlaceholder"
        />
        <div class="composer-actions">
          <el-button :disabled="!canSeedAnalysis || sending" @click="$emit('seed-analysis')">
            {{ aiCopy.seedAnalysisButton }}
          </el-button>
          <el-button type="primary" :loading="sending" @click="$emit('send')">
            {{ aiCopy.sendButton }}
          </el-button>
        </div>
      </div>
    </div>
  </article>
</template>

<script setup>
import { nextTick, ref, watch } from 'vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElInput from 'element-plus/es/components/input/index.mjs'
import { aiCopy, formatSourceType } from '@/modules/ai/aiView'

const props = defineProps({
  loadingMessages: {
    type: Boolean,
    default: false
  },
  messages: {
    type: Array,
    default: () => []
  },
  sending: {
    type: Boolean,
    default: false
  },
  canSeedAnalysis: {
    type: Boolean,
    default: false
  }
})

defineEmits(['send', 'seed-analysis'])

const draftComposer = defineModel('composer', {
  type: String,
  required: true
})

const messageStreamRef = ref(null)

function scrollToLatest() {
  const container = messageStreamRef.value
  if (!container) return

  const top = container.scrollHeight || 0
  if (typeof container.scrollTo === 'function') {
    container.scrollTo({
      top,
      behavior: 'smooth'
    })
    return
  }

  container.scrollTop = top
}

watch(
  () => props.messages.map((item) => item.id).join('|'),
  async () => {
    await nextTick()
    scrollToLatest()
  },
  { flush: 'post' }
)

function parseAssistantSections(content) {
  const normalized = String(content || '').trim()
  if (!normalized) {
    return { summary: '', paragraphs: [] }
  }

  const sections = normalized
    .split(/\n\s*\n/)
    .map((item) => item.trim())
    .filter(Boolean)

  const summaryIndex = sections.findIndex((item) => item.startsWith('直接结论：') || item.startsWith('结论：'))
  const summary = summaryIndex >= 0 ? sections[summaryIndex] : ''
  const paragraphs = sections.filter((_, index) => index !== summaryIndex)

  return {
    summary,
    paragraphs: paragraphs.length ? paragraphs : (summary ? [] : sections)
  }
}
</script>

<style scoped>
.ai-chat {
  overflow: hidden;
}

.message-stream {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 360px;
  max-height: 560px;
  overflow: auto;
  padding-right: 6px;
}

.empty-chat {
  display: grid;
  place-items: center;
  min-height: 240px;
  border: 1px dashed var(--line-strong);
  border-radius: 18px;
  color: var(--text-tertiary);
}

.message-card {
  padding: 14px 16px;
  border-radius: var(--radius-lg);
  border: 1px solid var(--line-soft);
  background: var(--bg-panel-raised);
}

.message-card.assistant {
  background: var(--bg-panel);
  border-left: 2px solid var(--brand);
}

.message-card__role {
  margin-bottom: 6px;
  font-size: 12px;
  color: var(--text-tertiary);
}

.message-card__content {
  line-height: 1.8;
}

.message-card__summary {
  margin-bottom: 10px;
  padding: 10px 12px;
  border-radius: var(--radius-md);
  background: var(--brand-glow);
  border: 1px solid var(--brand-soft);
  color: var(--text-primary);
  font-weight: 700;
  line-height: 1.7;
}

.message-card__paragraph {
  margin: 0 0 10px;
  white-space: pre-wrap;
}

.message-card__paragraph:last-child {
  margin-bottom: 0;
}

.message-card__references {
  margin-top: 12px;
  border-top: 1px dashed var(--line-strong);
  padding-top: 10px;
  color: var(--text-secondary);
}

.message-card__references-title {
  font-size: 12px;
  font-weight: 700;
  color: var(--brand-strong);
  cursor: pointer;
  user-select: none;
}

.reference-list {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.reference-item {
  padding: 10px 12px;
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  border: 1px solid var(--line-soft);
}

.reference-item__title {
  font-weight: 600;
  color: var(--text-primary);
}

.reference-item__snippet {
  margin-top: 4px;
  line-height: 1.6;
}

.composer-box {
  margin-top: 18px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.composer-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}
</style>
