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
        <div class="context-box__title">{{ aiCopy.contextBoxTitle }}</div>
        <div class="kv-list">
          <div class="kv-row">
            <span>{{ aiCopy.currentConversation }}</span>
            <strong>{{ contextSummary.conversationId || aiCopy.newConversationLabel }}</strong>
          </div>
          <div class="kv-row">
            <span>{{ aiCopy.currentScene }}</span>
            <strong>{{ contextSummary.sceneLabel }}</strong>
          </div>
          <div class="kv-row">
            <span>{{ aiCopy.boundReport }}</span>
            <strong>{{ contextSummary.reportId || '-' }}</strong>
          </div>
          <div class="kv-row">
            <span>{{ aiCopy.boundRun }}</span>
            <strong>{{ contextSummary.runId || '-' }}</strong>
          </div>
          <div class="kv-row">
            <span>{{ aiCopy.model }}</span>
            <strong>{{ contextSummary.model || '-' }}</strong>
          </div>
          <div class="kv-row">
            <span>{{ aiCopy.retrievalMode }}</span>
            <strong>{{ contextSummary.retrievalMode || knowledge.retrieval_mode || '-' }}</strong>
          </div>
          <div class="kv-row">
            <span>{{ aiCopy.fallback }}</span>
            <strong>{{ contextSummary.fallbackUsed ? aiCopy.fallbackEnabled : aiCopy.fallbackDisabled }}</strong>
          </div>
        </div>
      </div>

      <div v-if="showStatusBlocks" class="knowledge-status">
        <div class="knowledge-status__title">{{ aiCopy.knowledgeStatusTitle }}</div>
        <div class="kv-list">
          <div class="kv-row">
            <span>{{ aiCopy.knowledgeDocuments }}</span>
            <strong>{{ knowledge.total_documents ?? '-' }}</strong>
          </div>
          <div class="kv-row">
            <span>{{ aiCopy.knowledgeChunks }}</span>
            <strong>{{ knowledge.total_chunks || 0 }}</strong>
          </div>
          <div class="kv-row">
            <span>{{ aiCopy.requestedRetrievalMode }}</span>
            <strong>{{ knowledge.requested_retrieval_mode || '-' }}</strong>
          </div>
          <div class="kv-row">
            <span>{{ aiCopy.currentRetrievalMode }}</span>
            <strong>{{ knowledge.retrieval_mode || '-' }}</strong>
          </div>
          <div class="kv-row">
            <span>{{ aiCopy.embeddingStatus }}</span>
            <strong>{{ embeddingStatusText }}</strong>
          </div>
          <div class="kv-row">
            <span>{{ aiCopy.embeddingModel }}</span>
            <strong>{{ knowledge.embedding_model || '-' }}</strong>
          </div>
          <div class="kv-row">
            <span>{{ aiCopy.embeddingError }}</span>
            <strong>{{ knowledge.embedding_error || '-' }}</strong>
          </div>
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
  props.panelVariant === 'scene' ? aiCopy.scenePanelTitle : aiCopy.contextTitle
))
const panelDescription = computed(() => (
  props.panelVariant === 'scene'
    ? aiCopy.scenePanelDescription
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
  min-height: 690px;
  overflow: hidden;
}

.ai-context--scene {
  min-height: 0;
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
  padding: 14px;
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
  font-weight: 800;
  color: var(--text-primary);
}

.kv-list {
  display: grid;
  gap: 8px;
  margin-top: 12px;
}

.kv-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding-bottom: 7px;
  border-bottom: 1px solid var(--line-soft);
  color: var(--text-secondary);
  line-height: 1.55;
}

.kv-row:last-child {
  padding-bottom: 0;
  border-bottom: 0;
}

.kv-row span {
  color: var(--text-secondary);
}

.kv-row strong {
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 12px;
  text-align: right;
  overflow-wrap: anywhere;
}

.knowledge-status {
  margin-top: var(--space-lg);
  padding-top: var(--space-lg);
  border-top: 1px dashed var(--line-strong);
}

@media (max-width: 1080px) {
  .ai-context {
    min-height: 0;
  }
}

@media (max-width: 860px) {
  .scene-form {
    grid-template-columns: 1fr;
  }
}
</style>
