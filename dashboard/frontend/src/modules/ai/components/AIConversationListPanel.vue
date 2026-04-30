<template>
  <article class="section-card ai-list" :class="{ 'ai-list--rail-expanded': railExpanded }">
    <div class="section-card__header">
      <div class="ai-list__heading">
        <div class="section-card__title">{{ aiCopy.conversationListTitle }}</div>
        <div class="section-card__desc">{{ aiCopy.conversationListDescription }}</div>
      </div>
      <div class="ai-list__actions">
        <el-button class="ai-list__rail-toggle" plain size="small" @click="$emit('toggle-rail')">
          {{ railExpanded ? '收起' : '会话' }}
        </el-button>
        <el-button class="ai-list__new" type="primary" plain size="small" @click="$emit('start-fresh')">
          {{ aiCopy.newConversation }}
        </el-button>
      </div>
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
  },
  railExpanded: {
    type: Boolean,
    default: false
  }
})

defineEmits(['open-conversation', 'start-fresh', 'toggle-rail'])

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

.ai-list__actions {
  display: flex;
  gap: var(--space-xs);
  align-items: center;
  flex-shrink: 0;
}

.ai-list__rail-toggle {
  display: none;
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
  padding: 10px 12px;
  border-radius: var(--radius-md);
  border: 1px solid var(--line-soft);
  background: var(--bg-panel-raised);
  cursor: pointer;
  text-align: left;
  transition: border-color 0.15s, background 0.15s;
}

.conversation-item:hover {
  background: var(--bg-panel-strong);
}

.conversation-item.is-active {
  border-color: var(--brand);
  border-left: 2px solid var(--brand);
  background: var(--brand-glow);
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

@media (max-width: 1560px) {
  .ai-list {
    min-height: 128px;
  }

  .ai-list__rail-toggle {
    display: inline-flex;
  }

  .ai-list:not(.ai-list--rail-expanded) {
    width: 56px;
  }

  .ai-list:not(.ai-list--rail-expanded) .section-card__header {
    display: grid;
    place-items: center;
    padding: var(--space-sm);
  }

  .ai-list:not(.ai-list--rail-expanded) .ai-list__heading,
  .ai-list:not(.ai-list--rail-expanded) .section-card__body,
  .ai-list:not(.ai-list--rail-expanded) .ai-list__new {
    display: none;
  }

  .ai-list:not(.ai-list--rail-expanded) .ai-list__actions {
    display: grid;
    gap: var(--space-xs);
  }

  .ai-list:not(.ai-list--rail-expanded) .ai-list__rail-toggle {
    width: 36px;
    height: 96px;
    padding: 0;
    writing-mode: vertical-rl;
    letter-spacing: 0.12em;
  }

  .ai-list.ai-list--rail-expanded {
    width: 260px;
  }
}

@media (max-width: 1080px) {
  .ai-list,
  .ai-list.ai-list--rail-expanded {
    width: auto;
  }

  .ai-list:not(.ai-list--rail-expanded) .section-card__header,
  .ai-list:not(.ai-list--rail-expanded) .ai-list__heading,
  .ai-list:not(.ai-list--rail-expanded) .section-card__body,
  .ai-list:not(.ai-list--rail-expanded) .ai-list__new {
    display: revert;
  }

  .ai-list:not(.ai-list--rail-expanded) .section-card__header {
    display: flex;
  }

  .ai-list__rail-toggle {
    display: none;
  }
}
</style>
