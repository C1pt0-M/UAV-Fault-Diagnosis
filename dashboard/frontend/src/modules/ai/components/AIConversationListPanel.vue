<template>
  <article class="section-card">
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
          v-for="item in conversations"
          :key="item.id"
          class="conversation-item"
          :class="{ 'is-active': currentConversationId === item.id }"
          @click="$emit('open-conversation', item.id)"
        >
          <strong>{{ item.title || `会话 #${item.id}` }}</strong>
          <small>{{ item.updated_at || '-' }}</small>
        </button>
        <el-empty
          v-if="!loading && conversations.length === 0"
          :description="aiCopy.emptyConversationList"
          :image-size="54"
        />
      </div>
    </div>
  </article>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElEmpty from 'element-plus/es/components/empty/index.mjs'
import { aiCopy } from '@/modules/ai/aiView'

defineProps({
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
</script>

<style scoped>
.conversation-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.conversation-item {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
  padding: 14px;
  border-radius: 16px;
  border: 1px solid var(--line-soft);
  background: #f8fbfb;
  cursor: pointer;
}

.conversation-item.is-active {
  border-color: var(--brand);
  background: #edf8f6;
}

.conversation-item small {
  color: var(--text-tertiary);
}
</style>
