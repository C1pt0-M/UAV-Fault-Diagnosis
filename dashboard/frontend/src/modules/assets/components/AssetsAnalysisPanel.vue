<template>
  <section class="section-card">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ assetsCopy.analysisTitle }}</div>
        <div class="section-card__desc">{{ assetsCopy.analysisDescription }}</div>
      </div>
    </div>

    <div class="section-card__body">
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
</style>
