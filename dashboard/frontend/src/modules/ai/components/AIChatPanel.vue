<template>
  <article class="section-card">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ aiCopy.chatTitle }}</div>
        <div class="section-card__desc">{{ aiCopy.chatDescription }}</div>
      </div>
    </div>

    <div class="section-card__body">
      <div class="message-stream" v-loading="loadingMessages">
        <div v-if="messages.length === 0" class="empty-chat">
          {{ aiCopy.emptyChat }}
        </div>

        <article
          v-for="item in messages"
          :key="item.id"
          class="message-card"
          :class="item.role"
        >
          <div class="message-card__role">{{ item.role === 'assistant' ? aiCopy.heroTag : '用户' }}</div>
          <div class="message-card__content">{{ item.content }}</div>

          <details
            v-if="item.role === 'assistant' && item.citations?.length"
            class="message-card__references"
          >
            <summary>{{ aiCopy.referencesLabel }}（{{ item.citations.length }}）</summary>
            <ul class="reference-list">
              <li
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
              </li>
            </ul>
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
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElInput from 'element-plus/es/components/input/index.mjs'
import { aiCopy, formatSourceType } from '@/modules/ai/aiView'

defineProps({
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
</script>

<style scoped>
.message-stream {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 360px;
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
  border-radius: 16px;
  border: 1px solid var(--line-soft);
  background: #ffffff;
}

.message-card.assistant {
  background: #f8fbfb;
}

.message-card__role {
  margin-bottom: 6px;
  font-size: 12px;
  color: var(--text-tertiary);
}

.message-card__content {
  white-space: pre-wrap;
  line-height: 1.8;
}

.message-card__references {
  margin-top: 12px;
  border-top: 1px dashed var(--line-strong);
  padding-top: 10px;
  color: var(--text-secondary);
}

.message-card__references summary {
  cursor: pointer;
  user-select: none;
  color: var(--brand-strong);
}

.reference-list {
  margin: 10px 0 0;
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 8px;
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
