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
  },
  showSlider: {
    type: Boolean,
    default: true
  },
  showLegend: {
    type: Boolean,
    default: true
  },
  showThresholdLine: {
    type: Boolean,
    default: false
  },
  thresholdValue: {
    type: Number,
    default: 0.7
  },
  thresholdLabel: {
    type: String,
    default: ''
  }
})

const chartRef = ref(null)
let chartInstance = null

const hasSeriesData = computed(() => {
  return props.series.some((item) => Array.isArray(item.data) && item.data.length > 0)
})

function buildLineSeries(chartTheme) {
  return props.series.map((item, index) => ({
    name: item.name,
    type: 'line',
    yAxisIndex: 0,
    smooth: false,
    showSymbol: false,
    sampling: 'lttb',
    data: item.data || [],
    connectNulls: true,
    z: 3,
    ...(index === 0 && props.showThresholdLine
      ? {
          markLine: {
            silent: true,
            symbol: 'none',
            data: [
              {
                yAxis: props.thresholdValue,
                name: props.thresholdLabel || `阈值 ${props.thresholdValue}`
              }
            ],
            label: {
              formatter: props.thresholdLabel || `阈值 ${props.thresholdValue}`,
              color: chartTheme.legendTextColor,
              fontSize: 11
            },
            lineStyle: {
              color: chartTheme.warnLineColor,
              type: 'dashed',
              width: 1.2
            }
          }
        }
      : {})
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

function getCssToken(name, fallback) {
  if (typeof window === 'undefined') return fallback
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return value || fallback
}

function toRgbParts(color) {
  const value = String(color || '').trim()
  if (value.startsWith('#') && value.length === 7) {
    return [
      Number.parseInt(value.slice(1, 3), 16),
      Number.parseInt(value.slice(3, 5), 16),
      Number.parseInt(value.slice(5, 7), 16)
    ]
  }

  const rgbMatch = value.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/)
  if (rgbMatch) {
    return rgbMatch.slice(1, 4).map((part) => Number.parseInt(part, 10))
  }

  return [0, 0, 0]
}

function colorWithAlpha(color, alpha) {
  const [red, green, blue] = toRgbParts(color)
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`
}

function resolveChartTheme() {
  const brand = getCssToken('--brand', '#2563eb')
  const active = getCssToken('--status-active', '#2563eb')
  const warn = getCssToken('--status-warn', '#d97706')
  const danger = getCssToken('--status-danger', '#dc2626')
  const pending = getCssToken('--status-pending', '#7c3aed')
  const ok = getCssToken('--status-ok', '#16a34a')
  const text = getCssToken('--text-tertiary', '#9ca3af')
  const legendText = getCssToken('--text-secondary', '#4b5563')
  const lineSoft = getCssToken('--line-soft', '#e5e7eb')
  const lineStrong = getCssToken('--line-strong', '#d1d5db')

  return {
    palette: [brand, active, warn, danger, pending, ok],
    textColor: text,
    legendTextColor: legendText,
    axisColor: lineStrong,
    splitLineColor: lineSoft,
    sliderFillColor: colorWithAlpha(brand, 0.12),
    sliderDataColor: colorWithAlpha(brand, 0.08),
    sliderHandleColor: colorWithAlpha(brand, 0.38),
    warnLineColor: colorWithAlpha(warn, 0.86),
    warnAreaColor: colorWithAlpha(warn, 0.22),
    dangerAreaColor: colorWithAlpha(danger, 0.28)
  }
}

const hasVisibleDataZoom = computed(() => {
  return hasSeriesData.value && (props.showSlider || props.xData.length > 1)
})

const chartGrid = computed(() => ({
  left: 50,
  right: 24,
  top: props.showLegend ? 36 : 16,
  bottom: hasSeriesData.value && props.showSlider ? 34 : 20
}))

function buildDataZoom(chartTheme) {
  if (!hasVisibleDataZoom.value) return []

  const zoom = [
    { type: 'inside', start: 0, end: 100 }
  ]

  if (props.showSlider) {
    zoom.push({
      type: 'slider',
      height: 10,
      bottom: 8,
      start: 0,
      end: 100,
      borderColor: chartTheme.splitLineColor,
      backgroundColor: chartTheme.sliderDataColor,
      fillerColor: chartTheme.sliderFillColor,
      handleStyle: { color: chartTheme.sliderHandleColor },
      dataBackground: {
        lineStyle: { color: chartTheme.sliderHandleColor },
        areaStyle: { color: chartTheme.sliderDataColor }
      },
      selectedDataBackground: {
        lineStyle: { color: chartTheme.sliderHandleColor },
        areaStyle: { color: chartTheme.sliderFillColor }
      },
      textStyle: { color: chartTheme.textColor }
    })
  }

  return zoom
}

function buildFaultProbAreaSeries(chartTheme) {
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
        color: chartTheme.warnAreaColor
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
        color: chartTheme.dangerAreaColor
      },
      z: 1
    }
  ]
}

function renderChart() {
  if (!chartInstance) return

  const chartTheme = resolveChartTheme()
  const lineSeries = buildLineSeries(chartTheme)
  const faultProbAreaSeries = buildFaultProbAreaSeries(chartTheme)

  chartInstance.setOption(
    {
      color: chartTheme.palette,
      animation: false,
      grid: chartGrid.value,
      tooltip: { trigger: 'axis' },
      legend: {
        show: props.showLegend,
        top: 6,
        type: 'scroll',
        textStyle: { color: chartTheme.legendTextColor },
        data: lineSeries.map((item) => item.name)
      },
      xAxis: {
        type: 'category',
        name: props.xAxisName,
        boundaryGap: false,
        data: props.xData || [],
        axisLine: { lineStyle: { color: chartTheme.axisColor } },
        axisLabel: { color: chartTheme.textColor }
      },
      yAxis: [
        {
          type: 'value',
          scale: true,
          axisLine: { lineStyle: { color: chartTheme.axisColor } },
          splitLine: { lineStyle: { color: chartTheme.splitLineColor } },
          axisLabel: { color: chartTheme.textColor }
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
      dataZoom: buildDataZoom(chartTheme),
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
