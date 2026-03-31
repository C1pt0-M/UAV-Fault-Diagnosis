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
import ElButton from 'element-plus/es/components/button/index.mjs'
import { ElForm, ElFormItem } from 'element-plus/es/components/form/index.mjs'
import { ElOption, ElSelect } from 'element-plus/es/components/select/index.mjs'
import { diagnosisCopy } from '@/modules/diagnosis/diagnosisView'

defineProps({
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
}
</style>
