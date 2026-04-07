<template>
  <article class="section-card ai-context">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ aiCopy.contextTitle }}</div>
        <div class="section-card__desc">{{ aiCopy.contextDescription }}</div>
      </div>
      <el-button plain size="small" :loading="reindexing" @click="$emit('reindex-knowledge')">
        {{ aiCopy.reindexKnowledgeButton }}
      </el-button>
    </div>

    <div class="section-card__body">
      <el-form label-position="top">
        <el-form-item :label="aiCopy.sceneLabel">
          <el-select v-model="selectedScene">
            <el-option
              v-for="(label, value) in aiCopy.sceneLabels"
              :key="value"
              :label="label"
              :value="value"
            />
          </el-select>
        </el-form-item>

        <el-form-item v-if="selectedScene === 'report_analysis'" :label="aiCopy.reportIdLabel">
          <el-input v-model="selectedReportIdInput" :placeholder="aiCopy.reportIdPlaceholder" />
        </el-form-item>
      </el-form>

      <div class="context-box">
        <div class="context-box__title">当前分析上下文</div>
        <div class="context-box__grid">
          <div><strong>{{ aiCopy.currentConversation }}：</strong>{{ contextSummary.conversationId || aiCopy.newConversationLabel }}</div>
          <div><strong>{{ aiCopy.currentScene }}：</strong>{{ contextSummary.sceneLabel }}</div>
          <div><strong>{{ aiCopy.boundReport }}：</strong>{{ contextSummary.reportId || '-' }}</div>
          <div><strong>{{ aiCopy.boundRun }}：</strong>{{ contextSummary.runId || '-' }}</div>
          <div><strong>{{ aiCopy.model }}：</strong>{{ contextSummary.model || '-' }}</div>
          <div><strong>{{ aiCopy.retrievalMode }}：</strong>{{ contextSummary.retrievalMode || knowledge.retrieval_mode || '-' }}</div>
          <div><strong>{{ aiCopy.fallback }}：</strong>{{ contextSummary.fallbackUsed ? aiCopy.fallbackEnabled : aiCopy.fallbackDisabled }}</div>
        </div>
      </div>

      <div class="knowledge-status">
        <div class="knowledge-status__title">{{ aiCopy.knowledgeStatusTitle }}</div>
        <div class="knowledge-status__grid">
          <div><strong>{{ aiCopy.knowledgeDocuments }}：</strong>{{ knowledge.total_documents ?? '-' }}</div>
          <div><strong>{{ aiCopy.knowledgeChunks }}：</strong>{{ knowledge.total_chunks || 0 }}</div>
          <div><strong>{{ aiCopy.requestedRetrievalMode }}：</strong>{{ knowledge.requested_retrieval_mode || '-' }}</div>
          <div><strong>{{ aiCopy.currentRetrievalMode }}：</strong>{{ knowledge.retrieval_mode || '-' }}</div>
          <div><strong>{{ aiCopy.embeddingStatus }}：</strong>{{ embeddingStatusText }}</div>
          <div><strong>{{ aiCopy.embeddingModel }}：</strong>{{ knowledge.embedding_model || '-' }}</div>
          <div><strong>{{ aiCopy.embeddingError }}：</strong>{{ knowledge.embedding_error || '-' }}</div>
        </div>
      </div>
    </div>
  </article>
</template>

<script setup>
import { computed } from 'vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import { ElForm, ElFormItem } from 'element-plus/es/components/form/index.mjs'
import ElInput from 'element-plus/es/components/input/index.mjs'
import { ElOption, ElSelect } from 'element-plus/es/components/select/index.mjs'
import { aiCopy } from '@/modules/ai/aiView'

const props = defineProps({
  contextSummary: {
    type: Object,
    required: true
  },
  knowledge: {
    type: Object,
    default: () => ({ total_chunks: 0, retrieval_mode: 'bm25' })
  },
  reindexing: {
    type: Boolean,
    default: false
  }
})

defineEmits(['reindex-knowledge'])

const selectedScene = defineModel('scene', {
  type: String,
  required: true
})

const selectedReportIdInput = defineModel('reportIdInput', {
  type: String,
  required: true
})

const embeddingStatusText = computed(() => {
  if (props.knowledge.embedding_enabled) return aiCopy.embeddingEnabled
  if (props.knowledge.embedding_error) return aiCopy.embeddingErrorState
  return aiCopy.embeddingDisabled
})
</script>

<style scoped>
.ai-context {
  overflow: hidden;
}

.context-box {
  margin-top: 8px;
  padding: 16px 18px;
  border-radius: 18px;
  background: linear-gradient(180deg, #f7fbfc 0%, #eef6f7 100%);
  border: 1px solid var(--line-soft);
}

.context-box__title,
.knowledge-status__title {
  font-size: 13px;
  font-weight: 700;
  color: var(--text-primary);
}

.context-box__grid,
.knowledge-status__grid {
  display: grid;
  gap: 8px;
  margin-top: 10px;
  color: var(--text-secondary);
  line-height: 1.8;
}

.knowledge-status {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px dashed var(--line-strong);
}
</style>
