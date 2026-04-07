<template>
  <article class="section-card ai-list">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ aiCopy.conversationListTitle }}</div>
        <div class="section-card__desc">{{ aiCopy.conversationListDescription }}</div>
      </div>
      <el-button type="primary" plain size="small" @click="$emit('start-fresh')">
        {{ aiCopy.newConversation }}
      </el-button>
    </div>

    <div class="section-card__body">
      <div class="conversation-list" v-loading="loading">
        <button
          v-for="item in normalizedConversations"
          :key="item.id"
          class="conversation-item"
          :class="{ 'is-active': currentConversationId === item.id }"
          @click="$emit('open-conversation', item.id)"
        >
          <div class="conversation-item__top">
            <strong>{{ item.title || aiCopy.conversationFallbackTitle(item.id) }}</strong>
            <small>{{ item.updated_at || '-' }}</small>
          </div>

          <div class="conversation-item__meta">
            <span>{{ aiCopy.conversationScenePrefix }}：{{ item.sceneLabel }}</span>
            <span v-if="item.reportId">{{ aiCopy.conversationReportPrefix }}：#{{ item.reportId }}</span>
            <span v-if="item.runId">{{ aiCopy.conversationRunPrefix }}：#{{ item.runId }}</span>
          </div>

          <p v-if="item.lastPreview" class="conversation-item__preview">{{ item.lastPreview }}</p>
        </button>

        <el-empty
          v-if="!loading && normalizedConversations.length === 0"
          :description="aiCopy.emptyConversationList"
          :image-size="54"
        />
      </div>
    </div>
  </article>
</template>

<script setup>
import { computed } from 'vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElEmpty from 'element-plus/es/components/empty/index.mjs'
import { aiCopy, normalizeConversationSummary } from '@/modules/ai/aiView'

const props = defineProps({
  conversations: {
    type: Array,
    default: () => []
  },
  currentConversationId: {
    type: Number,
    default: null
  },
  loading: {
    type: Boolean,
    default: false
  }
})

defineEmits(['open-conversation', 'start-fresh'])

const normalizedConversations = computed(() => {
  return (props.conversations || []).map((item) => ({
    ...item,
    ...normalizeConversationSummary(item)
  }))
})
</script>

<style scoped>
.ai-list {
  overflow: hidden;
}

.conversation-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.conversation-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 14px;
  border-radius: 16px;
  border: 1px solid var(--line-soft);
  background: #f8fbfb;
  cursor: pointer;
  text-align: left;
}

.conversation-item.is-active {
  border-color: var(--brand);
  background: #edf8f6;
}

.conversation-item__top {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.conversation-item__top small,
.conversation-item__meta {
  color: var(--text-tertiary);
}

.conversation-item__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 12px;
  font-size: 12px;
}

.conversation-item__preview {
  margin: 0;
  color: var(--text-secondary);
  line-height: 1.7;
  font-size: 13px;
}
</style>
