<template>
  <article class="section-card diagnosis-launch">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ diagnosisCopy.formTitle }}</div>
        <div class="section-card__desc">{{ diagnosisCopy.formDescription }}</div>
      </div>
      <span class="diagnosis-launch__badge">{{ assets.length }} 个可用资产</span>
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
        </el-form-item>

        <el-form-item :label="diagnosisCopy.modelLabel">
          <el-select
            v-model="selectedModel"
            :placeholder="diagnosisCopy.modelPlaceholder"
          >
            <el-option
              v-for="model in models"
              :key="model.name"
              :label="model.display_name || model.name"
              :value="model.name"
            />
          </el-select>
        </el-form-item>

        <el-form-item :label="diagnosisCopy.taskTypeLabel">
          <el-select model-value="offline_multiclass" disabled>
            <el-option :label="diagnosisCopy.taskTypeValue" value="offline_multiclass" />
          </el-select>
        </el-form-item>

        <el-form-item :label="diagnosisCopy.windowLabel">
          <el-input :model-value="diagnosisCopy.windowPlaceholder" disabled />
        </el-form-item>
      </el-form>

      <section class="diagnosis-summary-block">
        <h4>日志概览</h4>
        <div class="diagnosis-mini-grid">
          <article
            v-for="item in assetSummaryItems"
            :key="item.label"
            class="diagnosis-param-card"
          >
            <span class="diagnosis-param-card__label">{{ item.label }}</span>
            <strong class="diagnosis-param-card__value" :title="String(item.value)">{{ item.value }}</strong>
          </article>
        </div>
      </section>

      <section class="diagnosis-summary-block">
        <h4>模型概览</h4>
        <div class="diagnosis-mini-grid">
          <article
            v-for="item in modelSummaryItems"
            :key="item.label"
            class="diagnosis-param-card"
          >
            <span class="diagnosis-param-card__label">{{ item.label }}</span>
            <strong class="diagnosis-param-card__value" :title="String(item.value)">{{ item.value }}</strong>
          </article>
        </div>
      </section>

      <div class="diagnosis-flow">
        <div
          v-for="step in flowSteps"
          :key="step.label"
          class="diagnosis-flow__step"
          :class="`diagnosis-flow__step--${step.status}`"
        >
          <span class="status-dot" :class="`status-dot--${step.status === 'ok' ? 'ok' : step.status}`"></span>
          <strong>{{ step.label }}</strong>
          <small>{{ step.time }}</small>
        </div>
      </div>

      <div class="diagnosis-actions">
        <el-button type="primary" size="large" :loading="running" @click="$emit('submit')">
          {{ stage.key === 'completed' ? diagnosisCopy.rerunButton : diagnosisCopy.submitButton }}
        </el-button>
        <el-button plain @click="$emit('refresh')">
          {{ diagnosisCopy.refreshButton }}
        </el-button>
      </div>
    </div>
  </article>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
import { ElForm, ElFormItem } from 'element-plus/es/components/form/index.mjs'
import ElInput from 'element-plus/es/components/input/index.mjs'
import { ElOption, ElSelect } from 'element-plus/es/components/select/index.mjs'
import { diagnosisCopy } from '@/modules/diagnosis/diagnosisView'

defineProps({
  assetSummaryItems: {
    type: Array,
    default: () => []
  },
  assets: {
    type: Array,
    default: () => []
  },
  flowSteps: {
    type: Array,
    default: () => []
  },
  modelSummaryItems: {
    type: Array,
    default: () => []
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
  padding: 16px var(--space-lg);
}

.diagnosis-launch__badge {
  display: inline-flex;
  padding: 6px 10px;
  border-radius: 999px;
  background: var(--brand-soft);
  color: var(--brand-strong);
  font-size: 12px;
  font-weight: 700;
}

.diagnosis-form-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  column-gap: 12px;
  row-gap: 6px;
}

.diagnosis-summary-block {
  margin-top: 10px;
}

.diagnosis-summary-block h4 {
  margin: 0 0 8px;
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 800;
}

.diagnosis-mini-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}

.diagnosis-param-card {
  min-width: 0;
  min-height: 52px;
  padding: 9px 10px;
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  border: 1px solid var(--line-soft);
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 5px;
}

.diagnosis-param-card__label {
  font-size: 11px;
  font-weight: 700;
  color: var(--text-tertiary);
}

.diagnosis-param-card__value {
  min-width: 0;
  font-size: 13px;
  line-height: 1.25;
  font-weight: 800;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.diagnosis-flow {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 8px;
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px dashed var(--line-soft);
}

.diagnosis-flow__step {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 5px;
  color: var(--text-tertiary);
}

.diagnosis-flow__step strong {
  color: var(--text-primary);
  font-size: 12px;
}

.diagnosis-flow__step small {
  font-size: 11px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.diagnosis-actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 14px;
  justify-content: flex-end;
}

:deep(.diagnosis-form-grid .el-form-item) {
  margin-bottom: 8px;
}

@media (max-width: 1480px) {
  .diagnosis-form-grid,
  .diagnosis-mini-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 760px) {
  .diagnosis-form-grid,
  .diagnosis-mini-grid,
  .diagnosis-flow {
    grid-template-columns: 1fr;
  }
}
</style>
