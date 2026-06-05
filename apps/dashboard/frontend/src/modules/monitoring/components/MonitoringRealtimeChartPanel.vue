<template>
  <section class="section-card monitoring-realtime">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ monitoringCopy.realtimeCurveTitle }}</div>
        <div class="section-card__desc">{{ monitoringCopy.realtimeCurveDescription }}</div>
      </div>
    </div>
    <div class="section-card__body monitoring-realtime__body">
      <div class="realtime-chart-group realtime-chart-group--main">
        <div class="realtime-chart-group__header">
          <h4 class="realtime-chart-group__title">故障概率趋势</h4>
          <span class="realtime-chart-group__hint">阈值与告警由后端检测结果驱动</span>
        </div>
        <TimeSeriesChart
          :x-data="chartTimes"
          :series="faultProbabilitySeries"
          :fault-prob-data="faultProbData"
          :height="310"
          :empty-text="monitoringCopy.realtimeCurveEmpty"
          :show-slider="true"
          :show-legend="true"
          :show-threshold-line="true"
          :threshold-value="0.7"
          threshold-label="风险阈值 0.70"
          x-axis-name="time_s"
        />
      </div>

      <div class="realtime-chart-grid">
        <div class="realtime-chart-group">
          <h4 class="realtime-chart-group__title">{{ monitoringCopy.motorChartTitle }}</h4>
          <TimeSeriesChart
            :x-data="chartTimes"
            :series="motorSeries"
            :fault-prob-data="faultProbData"
            :height="190"
            :empty-text="monitoringCopy.realtimeCurveEmpty"
            :show-slider="false"
            :show-legend="false"
            :show-threshold-line="false"
          />
        </div>

        <div class="realtime-chart-group">
          <h4 class="realtime-chart-group__title">{{ monitoringCopy.attitudeChartTitle }}</h4>
          <TimeSeriesChart
            :x-data="chartTimes"
            :series="attitudeSeries"
            :fault-prob-data="faultProbData"
            :height="190"
            :empty-text="monitoringCopy.realtimeCurveEmpty"
            :show-slider="false"
            :show-legend="false"
            :show-threshold-line="false"
          />
        </div>

        <div class="realtime-chart-group">
          <h4 class="realtime-chart-group__title">{{ monitoringCopy.positionChartTitle }}</h4>
          <TimeSeriesChart
            :x-data="chartTimes"
            :series="positionSeries"
            :fault-prob-data="faultProbData"
            :height="190"
            :empty-text="monitoringCopy.realtimeCurveEmpty"
            :show-slider="false"
            :show-legend="false"
            :show-threshold-line="false"
          />
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import TimeSeriesChart from '@/shared/components/TimeSeriesChart.vue'
import { monitoringCopy } from '@/modules/monitoring/monitoringView'

const props = defineProps({
  chartTimes: { type: Array, default: () => [] },
  faultProbData: { type: Array, default: () => [] },
  motorSeries: { type: Array, default: () => [] },
  attitudeSeries: { type: Array, default: () => [] },
  positionSeries: { type: Array, default: () => [] }
})

const faultProbabilitySeries = computed(() => {
  if (!Array.isArray(props.faultProbData) || props.faultProbData.length === 0) return []
  return [
    {
      name: '故障概率',
      data: props.faultProbData
    }
  ]
})
</script>

<style scoped>
.monitoring-realtime {
  overflow: hidden;
}

.monitoring-realtime__body {
  padding: var(--space-lg);
}

.realtime-chart-group {
  min-width: 0;
  padding: 12px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
}

.realtime-chart-group--main {
  margin-bottom: var(--space-md);
}

.realtime-chart-group__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-md);
}

.realtime-chart-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--space-md);
}

.realtime-chart-group__title {
  margin: 0 0 8px;
  font-size: 13px;
  color: var(--text-secondary);
  font-weight: 800;
}

.realtime-chart-group__hint {
  margin-bottom: 8px;
  font-size: 11px;
  color: var(--text-tertiary);
  white-space: nowrap;
}

@media (max-width: 1320px) {
  .realtime-chart-grid {
    grid-template-columns: 1fr;
  }
}
</style>
