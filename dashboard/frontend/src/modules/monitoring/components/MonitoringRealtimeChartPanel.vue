<template>
  <section class="section-card">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ monitoringCopy.realtimeCurveTitle }}</div>
        <div class="section-card__desc">{{ monitoringCopy.realtimeCurveDescription }}</div>
      </div>
      <el-select
        v-model="selectedFieldsModel"
        multiple
        collapse-tags
        collapse-tags-tooltip
        style="width: 420px"
        :placeholder="monitoringCopy.realtimeCurveFieldPlaceholder"
      >
        <el-option
          v-for="item in fieldOptions"
          :key="item.value"
          :label="item.label"
          :value="item.value"
        />
      </el-select>
    </div>
    <div class="section-card__body">
      <TimeSeriesChart
        :x-data="chartTimes"
        :series="chartSeries"
        :height="340"
        :empty-text="monitoringCopy.realtimeCurveEmpty"
      />
    </div>
  </section>
</template>

<script setup>
import { ElOption, ElSelect } from 'element-plus/es/components/select/index.mjs'
import TimeSeriesChart from '@/shared/components/TimeSeriesChart.vue'
import { monitoringCopy } from '@/modules/monitoring/monitoringView'

defineProps({
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
  }
})

const selectedFieldsModel = defineModel('selectedFields', {
  type: Array,
  default: () => []
})
</script>
