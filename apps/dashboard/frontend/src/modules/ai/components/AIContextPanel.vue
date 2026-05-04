<template>
  <article class="section-card ai-context" :class="`ai-context--${panelVariant}`">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ panelTitle }}</div>
        <div class="section-card__desc">{{ panelDescription }}</div>
      </div>
      <el-button
        v-if="showStatusBlocks"
        plain
        size="small"
        :loading="reindexing"
        @click="$emit('reindex-knowledge')"
      >
        {{ aiCopy.reindexKnowledgeButton }}
      </el-button>
    </div>

    <div class="section-card__body">
      <el-form v-if="showSceneControls" class="scene-form" label-position="top">
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

      <div v-if="showStatusBlocks" class="context-box">
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

      <div v-if="showStatusBlocks" class="knowledge-status">
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
  },
  panelVariant: {
    type: String,
    default: 'full',
    validator: (value) => ['full', 'scene', 'status'].includes(value)
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

const showSceneControls = computed(() => props.panelVariant !== 'status')
const showStatusBlocks = computed(() => props.panelVariant !== 'scene')
const panelTitle = computed(() => (
  props.panelVariant === 'scene' ? '分析场景' : aiCopy.contextTitle
))
const panelDescription = computed(() => (
  props.panelVariant === 'scene'
    ? '先确定任务类型和报告编号，聊天主区会按当前分析上下文回答。'
    : aiCopy.contextDescription
))

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

.ai-context--scene {
  border-left: 3px solid var(--brand);
}

.scene-form {
  display: grid;
  grid-template-columns: minmax(220px, 320px) minmax(180px, 260px);
  gap: var(--space-md);
  align-items: end;
}

.scene-form :deep(.el-form-item) {
  margin-bottom: 0;
}

.scene-form :deep(.el-select) {
  width: 100%;
}

.context-box {
  margin-top: var(--space-md);
  padding: var(--space-lg);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  border: 1px solid var(--line-soft);
}

.ai-context--status .context-box {
  margin-top: 0;
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
  gap: var(--space-sm);
  margin-top: var(--space-md);
  color: var(--text-secondary);
  line-height: 1.7;
}

.knowledge-status {
  margin-top: var(--space-lg);
  padding-top: var(--space-lg);
  border-top: 1px dashed var(--line-strong);
}

@media (max-width: 860px) {
  .scene-form {
    grid-template-columns: 1fr;
  }
}
</style>
