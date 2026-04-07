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
  },
  faultProbData: {
    type: Array,
    default: () => []
  }
})

const chartRef = ref(null)
let chartInstance = null

const hasSeriesData = computed(() => {
  return props.series.some((item) => Array.isArray(item.data) && item.data.length > 0)
})

function buildLineSeries() {
  return props.series.map((item) => ({
    name: item.name,
    type: 'line',
    yAxisIndex: 0,
    smooth: false,
    showSymbol: false,
    sampling: 'lttb',
    data: item.data || [],
    connectNulls: true,
    z: 3
  }))
}

function clamp01(value) {
  if (!Number.isFinite(value)) return 0
  if (value <= 0) return 0
  if (value >= 1) return 1
  return value
}

function smoothFaultProbSeries(values, alpha = 0.14) {
  const out = []
  let prev = 0
  let hasPrev = false

  for (const value of values) {
    const parsed = Number(value)
    const current = Number.isFinite(parsed) ? clamp01(parsed) : (hasPrev ? prev : 0)
    if (!hasPrev) {
      prev = current
      hasPrev = true
    } else {
      prev = prev + alpha * (current - prev)
    }
    out.push(prev)
  }

  // Backward smoothing pass reduces thin vertical strip artifacts.
  const back = [...out]
  let tail = back.length ? back[back.length - 1] : 0
  for (let i = back.length - 2; i >= 0; i -= 1) {
    tail = tail + alpha * (back[i] - tail)
    back[i] = (back[i] + tail) / 2
  }

  return back
}

function buildFaultProbAreaSeries() {
  const xData = Array.isArray(props.xData) ? props.xData : []
  const probs = Array.isArray(props.faultProbData) ? props.faultProbData : []
  const n = Math.min(xData.length, probs.length)

  if (n <= 1) return []

  const smoothed = smoothFaultProbSeries(probs.slice(0, n), 0.14)
  const aligned = xData.map((_, index) => {
    if (index >= n) return null
    return clamp01(smoothed[index])
  })

  // Accentuate high-risk tail while preserving continuous low-to-high shape.
  const highAligned = aligned.map((value) => {
    if (!Number.isFinite(value)) return null
    return Math.max(0, (value - 0.55) / 0.45)
  })

  return [
    {
      id: '__fault_prob_area__',
      name: '__fault_prob_area__',
      type: 'line',
      xAxisIndex: 0,
      yAxisIndex: 1,
      data: aligned,
      smooth: true,
      symbol: 'none',
      silent: true,
      tooltip: { show: false },
      emphasis: { disabled: true },
      lineStyle: {
        width: 0,
        opacity: 0
      },
      areaStyle: {
        color: 'rgba(245, 158, 11, 0.22)'
      },
      z: 0
    },
    {
      id: '__fault_prob_area_high__',
      name: '__fault_prob_area_high__',
      type: 'line',
      xAxisIndex: 0,
      yAxisIndex: 1,
      data: highAligned,
      smooth: true,
      symbol: 'none',
      silent: true,
      tooltip: { show: false },
      emphasis: { disabled: true },
      lineStyle: {
        width: 0,
        opacity: 0
      },
      areaStyle: {
        color: 'rgba(220, 38, 38, 0.28)'
      },
      z: 1
    }
  ]
}

function renderChart() {
  if (!chartInstance) return

  const lineSeries = buildLineSeries()
  const faultProbAreaSeries = buildFaultProbAreaSeries()

  chartInstance.setOption(
    {
      color: ['#0f766e', '#1d4ed8', '#d97706', '#be123c', '#7c3aed', '#0891b2', '#15803d', '#b45309'],
      animation: false,
      grid: { left: 50, right: 24, top: 36, bottom: 42 },
      tooltip: { trigger: 'axis' },
      legend: {
        top: 6,
        type: 'scroll',
        textStyle: { color: '#49616d' },
        data: lineSeries.map((item) => item.name)
      },
      xAxis: {
        type: 'category',
        name: props.xAxisName,
        boundaryGap: false,
        data: props.xData || [],
        axisLine: { lineStyle: { color: 'rgba(16, 50, 64, 0.25)' } },
        axisLabel: { color: '#738791' }
      },
      yAxis: [
        {
          type: 'value',
          scale: true,
          axisLine: { lineStyle: { color: 'rgba(16, 50, 64, 0.25)' } },
          splitLine: { lineStyle: { color: 'rgba(16, 50, 64, 0.08)' } },
          axisLabel: { color: '#738791' }
        },
        {
          type: 'value',
          min: 0,
          max: 1,
          show: false,
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { show: false },
          splitLine: { show: false }
        }
      ],
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        { type: 'slider', height: 16, bottom: 8, start: 0, end: 100 }
      ],
      series: [...faultProbAreaSeries, ...lineSeries]
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
  () => [props.xData, props.series, props.faultProbData],
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
