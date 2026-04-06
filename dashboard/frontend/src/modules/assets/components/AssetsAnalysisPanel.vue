<template>
  <section class="section-card">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ assetsCopy.analysisTitle }}</div>
        <div class="section-card__desc">{{ assetsCopy.analysisDescription }}</div>
      </div>
    </div>

    <div class="section-card__body">
      <div v-if="annotationItems.length" class="analysis-annotations">
        <div class="analysis-annotations__label">故障发生阶段</div>
        <div class="analysis-annotations__list">
          <span v-for="item in annotationItems" :key="item.field" class="analysis-annotations__item">
            <strong>{{ item.displayName }}</strong>
            <small v-if="item.sample_values?.length">{{ item.sample_values.join(' / ') }}</small>
          </span>
        </div>
      </div>

      <div class="analysis-controls">
        <el-select
          v-model="assetIdModel"
          style="width: 320px"
          filterable
          :placeholder="assetsCopy.analysisAssetPlaceholder"
        >
          <el-option
            v-for="asset in assets"
            :key="asset.id"
            :label="`#${asset.id} ${asset.filename}`"
            :value="asset.id"
          />
        </el-select>

        <el-select
          v-model="fieldsModel"
          multiple
          collapse-tags
          collapse-tags-tooltip
          filterable
          style="width: 420px"
          :placeholder="assetsCopy.analysisFieldsPlaceholder"
        >
          <el-option
            v-for="field in fieldOptions"
            :key="field.value"
            :label="field.label"
            :value="field.value"
          />
        </el-select>

        <el-input-number v-model="maxPointsModel" :min="200" :max="20000" :step="200" />
        <el-input-number v-model="startTModel" :step="1" placeholder="start_t" />
        <el-input-number v-model="endTModel" :step="1" placeholder="end_t" />

        <el-button :loading="loadingFields" @click="$emit('reload-fields')">
          {{ assetsCopy.analysisRefreshFields }}
        </el-button>
        <el-button type="primary" :loading="loadingSeries" @click="$emit('load-series')">
          {{ assetsCopy.analysisLoadSeries }}
        </el-button>
      </div>

      <p v-if="errorMessage" class="analysis-error">{{ errorMessage }}</p>

      <TimeSeriesChart
        :x-data="chartTimes"
        :series="chartSeries"
        :height="360"
        :empty-text="assetsCopy.analysisNoSeries"
      />
    </div>
  </section>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElInputNumber from 'element-plus/es/components/input-number/index.mjs'
import { ElOption, ElSelect } from 'element-plus/es/components/select/index.mjs'
import TimeSeriesChart from '@/shared/components/TimeSeriesChart.vue'
import { assetsCopy } from '@/modules/assets/assetsView'

defineProps({
  assets: {
    type: Array,
    default: () => []
  },
  fieldOptions: {
    type: Array,
    default: () => []
  },
  chartTimes: {
    type: Array,
    default: () => []
  },
  chartSeries: {
    type: Array,
    default: () => []
  },
  loadingFields: {
    type: Boolean,
    default: false
  },
  loadingSeries: {
    type: Boolean,
    default: false
  },
  errorMessage: {
    type: String,
    default: ''
  },
  annotationItems: {
    type: Array,
    default: () => []
  }
})

defineEmits(['reload-fields', 'load-series'])

const assetIdModel = defineModel('analysisAssetId', {
  type: Number,
  default: null
})
const fieldsModel = defineModel('analysisFields', {
  type: Array,
  default: () => []
})
const startTModel = defineModel('analysisStartT', {
  type: Number,
  default: undefined
})
const endTModel = defineModel('analysisEndT', {
  type: Number,
  default: undefined
})
const maxPointsModel = defineModel('analysisMaxPoints', {
  type: Number,
  default: 2000
})
</script>

<style scoped>
.analysis-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.analysis-error {
  margin: 0 0 12px;
  color: var(--danger);
}

.analysis-annotations {
  margin-bottom: 14px;
  padding: 14px 16px;
  border-radius: 16px;
  border: 1px solid rgba(16, 50, 64, 0.08);
  background: #f7fbfc;
}

.analysis-annotations__label {
  margin-bottom: 10px;
  font-size: 12px;
  font-weight: 700;
  color: var(--text-tertiary);
  letter-spacing: 0.08em;
}

.analysis-annotations__list {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.analysis-annotations__item {
  display: inline-flex;
  flex-direction: column;
  gap: 4px;
  padding: 10px 12px;
  border-radius: 14px;
  background: white;
  border: 1px solid rgba(16, 50, 64, 0.08);
}

.analysis-annotations__item strong {
  font-size: 13px;
  color: var(--text-primary);
}

.analysis-annotations__item small {
  font-size: 12px;
  color: var(--text-secondary);
}
</style>
