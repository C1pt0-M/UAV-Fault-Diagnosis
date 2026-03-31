<template>
  <article class="section-card">
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
        <div><strong>{{ aiCopy.currentConversation }}：</strong>{{ contextSummary.conversationId || aiCopy.newConversationLabel }}</div>
        <div><strong>{{ aiCopy.currentScene }}：</strong>{{ contextSummary.sceneLabel }}</div>
        <div><strong>{{ aiCopy.boundReport }}：</strong>{{ contextSummary.reportId || '-' }}</div>
        <div><strong>{{ aiCopy.boundRun }}：</strong>{{ contextSummary.runId || '-' }}</div>
        <div><strong>{{ aiCopy.knowledgeChunks }}：</strong>{{ knowledge.total_chunks || 0 }}</div>
        <div><strong>{{ aiCopy.retrievalMode }}：</strong>{{ contextSummary.retrievalMode || knowledge.retrieval_mode || '-' }}</div>
        <div><strong>{{ aiCopy.provider }}：</strong>{{ contextSummary.provider || '-' }}</div>
        <div><strong>{{ aiCopy.model }}：</strong>{{ contextSummary.model || '-' }}</div>
        <div><strong>{{ aiCopy.fallback }}：</strong>{{ contextSummary.fallbackUsed ? aiCopy.fallbackEnabled : aiCopy.fallbackDisabled }}</div>
      </div>
    </div>
  </article>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
import { ElForm, ElFormItem } from 'element-plus/es/components/form/index.mjs'
import ElInput from 'element-plus/es/components/input/index.mjs'
import { ElOption, ElSelect } from 'element-plus/es/components/select/index.mjs'
import { aiCopy } from '@/modules/ai/aiView'

defineProps({
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
</script>

<style scoped>
.context-box {
  line-height: 1.9;
  color: var(--text-secondary);
}
</style>
