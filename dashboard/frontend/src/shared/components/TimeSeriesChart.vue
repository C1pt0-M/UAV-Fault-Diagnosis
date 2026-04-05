<template>
  <div class="timeseries-chart">
    <div ref="chartRef" class="timeseries-chart__canvas" :style="{ height: `${height}px` }" />
    <div v-if="!hasSeriesData" class="timeseries-chart__empty">{{ emptyText }}</div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { init } from 'echarts/core'

use([LineChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, CanvasRenderer])

const props = defineProps({
  xData: {
    type: Array,
    default: () => []
  },
  series: {
    type: Array,
    default: () => []
  },
  height: {
    type: Number,
    default: 340
  },
  emptyText: {
    type: String,
    default: '暂无曲线数据'
  },
  xAxisName: {
    type: String,
    default: 'time_s'
  }
})

const chartRef = ref(null)
let chartInstance = null

const hasSeriesData = computed(() => {
  return props.series.some((item) => Array.isArray(item.data) && item.data.length > 0)
})

function buildSeries() {
  return props.series.map((item) => ({
    name: item.name,
    type: 'line',
    smooth: false,
    showSymbol: false,
    sampling: 'lttb',
    data: item.data || [],
    connectNulls: true
  }))
}

function renderChart() {
  if (!chartInstance) return
  chartInstance.setOption(
    {
      color: ['#0f766e', '#1d4ed8', '#d97706', '#be123c', '#7c3aed', '#0891b2', '#15803d', '#b45309'],
      animation: false,
      grid: { left: 50, right: 24, top: 36, bottom: 42 },
      tooltip: { trigger: 'axis' },
      legend: {
        top: 6,
        type: 'scroll',
        textStyle: { color: '#49616d' }
      },
      xAxis: {
        type: 'category',
        name: props.xAxisName,
        boundaryGap: false,
        data: props.xData || [],
        axisLine: { lineStyle: { color: 'rgba(16, 50, 64, 0.25)' } },
        axisLabel: { color: '#738791' }
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLine: { lineStyle: { color: 'rgba(16, 50, 64, 0.25)' } },
        splitLine: { lineStyle: { color: 'rgba(16, 50, 64, 0.08)' } },
        axisLabel: { color: '#738791' }
      },
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        { type: 'slider', height: 16, bottom: 8, start: 0, end: 100 }
      ],
      series: buildSeries()
    },
    true
  )
}

function initChart() {
  if (!chartRef.value || chartInstance) return
  chartInstance = init(chartRef.value)
  renderChart()
}

function disposeChart() {
  if (chartInstance) {
    chartInstance.dispose()
    chartInstance = null
  }
}

function onResize() {
  chartInstance?.resize()
}

watch(
  () => [props.xData, props.series],
  () => {
    renderChart()
  },
  { deep: true }
)

onMounted(() => {
  initChart()
  window.addEventListener('resize', onResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', onResize)
  disposeChart()
})
</script>

<style scoped>
.timeseries-chart {
  position: relative;
}

.timeseries-chart__canvas {
  width: 100%;
}

.timeseries-chart__empty {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: var(--text-tertiary);
  pointer-events: none;
}
</style>
