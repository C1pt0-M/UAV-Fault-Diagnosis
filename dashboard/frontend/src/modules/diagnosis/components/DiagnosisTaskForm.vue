<template>
  <article class="section-card diagnosis-launch">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ diagnosisCopy.formTitle }}</div>
        <div class="section-card__desc">{{ diagnosisCopy.formDescription }}</div>
      </div>
      <span class="diagnosis-launch__badge">{{ assets.length }} 个可用资产</span>
    </div>
    <div class="section-card__body">
      <div class="diagnosis-launch__summary">
        <article class="diagnosis-param-card">
          <span class="diagnosis-param-card__label">可用资产</span>
          <strong class="diagnosis-param-card__value">{{ assets.length }}</strong>
          <small class="diagnosis-param-card__meta">CSV / ULog 诊断输入</small>
        </article>
        <article class="diagnosis-param-card">
          <span class="diagnosis-param-card__label">当前资产</span>
          <strong class="diagnosis-param-card__value">{{ currentAssetName }}</strong>
          <small class="diagnosis-param-card__meta">{{ currentAssetMeta }}</small>
        </article>
        <article class="diagnosis-param-card">
          <span class="diagnosis-param-card__label">诊断模型</span>
          <strong class="diagnosis-param-card__value">{{ currentModelDisplayName }}</strong>
          <small v-if="currentModelRawName" class="diagnosis-param-card__meta diagnosis-param-card__meta--mono">
            {{ currentModelRawName }}
          </small>
          <small v-else class="diagnosis-param-card__meta">请选择离线诊断模型</small>
        </article>
      </div>

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
      </el-form>

      <div class="diagnosis-actions">
        <el-button type="primary" size="large" :loading="running" @click="$emit('submit')">
          {{ diagnosisCopy.submitButton }}
        </el-button>
        <el-button plain @click="$emit('refresh')">
          {{ diagnosisCopy.refreshButton }}
        </el-button>
      </div>
    </div>
  </article>
</template>

<script setup>
import { computed } from 'vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import { ElForm, ElFormItem } from 'element-plus/es/components/form/index.mjs'
import { ElOption, ElSelect } from 'element-plus/es/components/select/index.mjs'
import { diagnosisCopy } from '@/modules/diagnosis/diagnosisView'

const props = defineProps({
  assets: {
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

const currentAsset = computed(() => {
  if (selectedAssetId.value == null) return null
  return props.assets.find((asset) => asset.id === selectedAssetId.value) || null
})

const currentAssetName = computed(() => currentAsset.value?.filename || '尚未选择资产')
const currentAssetMeta = computed(() => {
  if (!currentAsset.value) return '等待选择诊断输入'
  return `资产 #${currentAsset.value.id}`
})

const currentModel = computed(() => {
  if (!selectedModel.value) return null
  return props.models.find((model) => model.name === selectedModel.value) || {
    name: selectedModel.value,
    display_name: selectedModel.value
  }
})

const currentModelDisplayName = computed(() => {
  if (!selectedModel.value) return '尚未选择模型'
  return currentModel.value.display_name || currentModel.value.name || selectedModel.value
})

const currentModelRawName = computed(() => {
  if (!selectedModel.value) return ''
  return currentModel.value.name || selectedModel.value
})
</script>

<style scoped>
.diagnosis-launch {
  overflow: hidden;
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

.diagnosis-launch__summary {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--space-md);
  margin-bottom: var(--space-lg);
}

.diagnosis-param-card {
  min-height: 76px;
  padding: var(--space-md) var(--space-lg);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  border: 1px solid var(--line-soft);
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 6px;
}

.diagnosis-param-card__label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-tertiary);
  letter-spacing: 0.04em;
}

.diagnosis-param-card__value {
  min-width: 0;
  font-size: 15px;
  line-height: 1.3;
  font-weight: 700;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.diagnosis-param-card__meta {
  min-width: 0;
  font-size: 11px;
  line-height: 1.3;
  color: var(--text-tertiary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.diagnosis-param-card__meta--mono {
  font-family: var(--font-mono);
}

.diagnosis-form-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.diagnosis-actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 8px;
  justify-content: flex-end;
  padding-top: var(--space-md);
  border-top: 1px dashed var(--line-soft);
}

@media (max-width: 1480px) {
  .diagnosis-launch__summary,
  .diagnosis-form-grid {
    grid-template-columns: 1fr;
  }
}
</style>
