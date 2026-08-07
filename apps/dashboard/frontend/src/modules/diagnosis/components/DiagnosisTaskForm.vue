<template>
  <article class="section-card diagnosis-launch">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ diagnosisCopy.formTitle }}</div>
        <div class="section-card__desc">{{ diagnosisCopy.formDescription }}</div>
      </div>
      <el-tooltip :content="diagnosisCopy.refreshButton" placement="top">
        <el-button
          class="diagnosis-launch__refresh"
          circle
          :icon="Refresh"
          :aria-label="diagnosisCopy.refreshButton"
          @click="$emit('refresh')"
        />
      </el-tooltip>
    </div>
    <div class="section-card__body diagnosis-launch__body">
      <el-form label-position="top" class="diagnosis-form-grid">
        <el-form-item :label="diagnosisCopy.assetLabel">
          <el-select
            v-model="selectedAssetId"
            clearable
            filterable
            :placeholder="diagnosisCopy.assetPlaceholder"
          >
            <el-option
              v-for="asset in assets"
              :key="asset.id"
              :label="`#${asset.id} ${asset.filename}`"
              :value="asset.id"
            />
          </el-select>
          <div v-if="assetSummaryItems.length" class="diagnosis-selection-meta">
            <span v-for="item in assetSummaryItems" :key="item.label">
              {{ item.label }} <strong>{{ item.value }}</strong>
            </span>
          </div>
        </el-form-item>

        <el-form-item :label="diagnosisCopy.modelLabel">
          <el-select
            v-model="selectedModel"
            :placeholder="diagnosisCopy.modelPlaceholder"
          >
            <el-option
              v-for="model in models"
              :key="model.name"
              :label="formatDiagnosisModelName(model.display_name || model.name)"
              :value="model.name"
            />
          </el-select>
          <div v-if="modelContextText" class="diagnosis-model-context">
            {{ modelContextText }}
          </div>
        </el-form-item>
      </el-form>

      <div v-if="stage.key === 'running'" class="diagnosis-running-state" role="status">
        <span class="status-dot status-dot--active"></span>
        <div>
          <strong>正在分析日志</strong>
          <small>系统正在完成特征处理、模型推理与结果整理。</small>
        </div>
      </div>

      <div class="diagnosis-actions">
        <el-button type="primary" size="large" :loading="running" @click="$emit('submit')">
          {{ stage.key === 'completed' ? diagnosisCopy.rerunButton : diagnosisCopy.submitButton }}
        </el-button>
      </div>
    </div>
  </article>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
import { ElForm, ElFormItem } from 'element-plus/es/components/form/index.mjs'
import { ElOption, ElSelect } from 'element-plus/es/components/select/index.mjs'
import ElTooltip from 'element-plus/es/components/tooltip/index.mjs'
import { Refresh } from '@element-plus/icons-vue'
import { diagnosisCopy, formatDiagnosisModelName } from '@/modules/diagnosis/diagnosisView'

defineProps({
  assetSummaryItems: {
    type: Array,
    default: () => []
  },
  assets: {
    type: Array,
    default: () => []
  },
  modelContextText: {
    type: String,
    default: ''
  },
  models: {
    type: Array,
    default: () => []
  },
  running: {
    type: Boolean,
    default: false
  },
  stage: {
    type: Object,
    required: true
  }
})

defineEmits(['submit', 'refresh'])

const selectedAssetId = defineModel('selectedAssetId', {
  default: null
})

const selectedModel = defineModel('selectedModel', {
  type: String,
  required: true
})
</script>

<style scoped>
.diagnosis-launch {
  overflow: hidden;
}

.diagnosis-launch__body {
  padding: var(--space-lg);
}

.diagnosis-launch__refresh {
  flex: 0 0 auto;
}

.diagnosis-form-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 4px;
}

.diagnosis-selection-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  margin-top: 8px;
  color: var(--text-tertiary);
  font-size: 12px;
}

.diagnosis-selection-meta strong {
  margin-left: 3px;
  color: var(--text-secondary);
}

.diagnosis-model-context {
  margin-top: 8px;
  color: var(--text-tertiary);
  font-size: 12px;
  line-height: 1.5;
}

.diagnosis-running-state {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin-top: 4px;
  padding: 12px 0;
  border-top: 1px solid var(--line-soft);
}

.diagnosis-running-state strong,
.diagnosis-running-state small {
  display: block;
}

.diagnosis-running-state strong {
  color: var(--text-primary);
  font-size: 13px;
}

.diagnosis-running-state small {
  margin-top: 3px;
  color: var(--text-tertiary);
  font-size: 11px;
}

.diagnosis-actions {
  margin-top: 8px;
}

.diagnosis-actions :deep(.el-button) {
  width: 100%;
}

:deep(.diagnosis-form-grid .el-form-item) {
  margin-bottom: 8px;
}

@media (max-width: 760px) {
  .diagnosis-launch__body {
    padding: var(--space-md);
  }
}
</style>
