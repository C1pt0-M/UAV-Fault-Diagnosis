<template>
  <section class="section-card monitoring-realtime">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ monitoringCopy.realtimeCurveTitle }}</div>
        <div class="section-card__desc">{{ monitoringCopy.realtimeCurveDescription }}</div>
      </div>
    </div>
    <div class="section-card__body">
      <div class="realtime-chart-group">
        <h4 class="realtime-chart-group__title">{{ monitoringCopy.motorChartTitle }}</h4>
        <TimeSeriesChart
          :x-data="chartTimes"
          :series="motorSeries"
          :fault-prob-data="faultProbData"
          :height="300"
          :empty-text="monitoringCopy.realtimeCurveEmpty"
        />
      </div>

      <div class="realtime-chart-grid">
        <div class="realtime-chart-group">
          <h4 class="realtime-chart-group__title">{{ monitoringCopy.attitudeChartTitle }}</h4>
          <TimeSeriesChart
            :x-data="chartTimes"
            :series="attitudeSeries"
            :fault-prob-data="faultProbData"
            :height="260"
            :empty-text="monitoringCopy.realtimeCurveEmpty"
          />
        </div>

        <div class="realtime-chart-group">
          <h4 class="realtime-chart-group__title">{{ monitoringCopy.positionChartTitle }}</h4>
          <TimeSeriesChart
            :x-data="chartTimes"
            :series="positionSeries"
            :fault-prob-data="faultProbData"
            :height="260"
            :empty-text="monitoringCopy.realtimeCurveEmpty"
          />
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import TimeSeriesChart from '@/shared/components/TimeSeriesChart.vue'
import { monitoringCopy } from '@/modules/monitoring/monitoringView'

defineProps({
  chartTimes: { type: Array, default: () => [] },
  faultProbData: { type: Array, default: () => [] },
  motorSeries: { type: Array, default: () => [] },
  attitudeSeries: { type: Array, default: () => [] },
  positionSeries: { type: Array, default: () => [] }
})
</script>

<style scoped>
.monitoring-realtime {
  overflow: hidden;
}

.realtime-chart-group {
  margin-bottom: 14px;
}

.realtime-chart-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.realtime-chart-group__title {
  margin: 0 0 8px;
  font-size: 13px;
  color: var(--text-secondary);
  font-weight: 700;
}

@media (max-width: 1480px) {
  .realtime-chart-grid {
    grid-template-columns: 1fr;
  }
}
</style>
