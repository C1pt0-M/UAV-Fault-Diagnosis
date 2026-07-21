<template>
  <section class="section-card link-monitor" data-test="5g-link-monitor">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">5G链路运行监测</div>
        <div class="section-card__desc">遥测上行时延、可靠性与服务等级</div>
      </div>
      <span class="link-state">
        <span class="status-dot" :class="`status-dot--${linkTone}`"></span>
        {{ linkStateLabel }}
      </span>
    </div>

    <div class="section-card__body link-monitor__body">
      <div class="link-kpis">
        <div v-for="item in kpis" :key="item.label" class="link-kpi">
          <span>{{ item.label }}</span>
          <strong :title="item.value">{{ item.value }}</strong>
          <small>{{ item.meta }}</small>
        </div>
      </div>

      <div class="mec-strip" data-test="mec-status-strip">
        <div><span>推理位置</span><strong>{{ mecLocationLabel }}</strong></div>
        <div><span>MEC进程</span><strong>{{ mecProcessLabel }}</strong></div>
        <div><span>平均推理</span><strong>{{ formatMs(inference.avg_inference_ms) }}</strong></div>
        <div><span>P95推理</span><strong>{{ formatMs(inference.p95_inference_ms) }}</strong></div>
        <div><span>平均决策</span><strong>{{ formatMs(inference.avg_decision_latency_ms) }}</strong></div>
        <div><span>P95决策</span><strong>{{ formatMs(inference.p95_decision_latency_ms) }}</strong></div>
        <div><span>推理次数</span><strong>{{ formatInteger(inference.inference_count) }}</strong></div>
      </div>

      <div class="link-monitor__workspace">
        <div class="link-chart">
          <div class="link-chart__header">
            <strong>端到端时延趋势</strong>
            <span>SLA P95 {{ formatMs(slaThreshold) }}</span>
          </div>
          <TimeSeriesChart
            :x-data="times"
            :series="latencyChartSeries"
            :height="220"
            empty-text="等待5G链路遥测"
            :show-slider="false"
            :show-legend="false"
            :show-threshold-line="true"
            :threshold-value="slaThreshold"
            threshold-label="SLA上限"
            x-axis-name="time_s"
          />
        </div>

        <div class="link-assurance">
          <div class="link-assurance__row"><span>链路进程</span><strong>{{ processLabel }}</strong></div>
          <div class="link-assurance__row"><span>SLA状态</span><strong :class="`text-${slaTone}`">{{ slaLabel }}</strong></div>
          <div class="link-assurance__row"><span>接收包</span><strong>{{ formatInteger(networkLink.rx_packets) }}</strong></div>
          <div class="link-assurance__row"><span>丢失包</span><strong>{{ formatInteger(networkLink.lost_packets) }}</strong></div>
          <div class="link-assurance__row"><span>链路证据</span><strong>{{ networkLink.metrics_csv ? '已落盘' : '待生成' }}</strong></div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import TimeSeriesChart from '@/shared/components/TimeSeriesChart.vue'

const props = defineProps({
  networkLink: { type: Object, default: () => ({}) },
  mec: { type: Object, default: () => ({}) },
  inference: { type: Object, default: () => ({}) },
  selectedProfile: { type: String, default: 'enhanced_5g' },
  processes: { type: Object, default: () => ({}) },
  times: { type: Array, default: () => [] },
  latencySeries: { type: Array, default: () => [] },
  lossSeries: { type: Array, default: () => [] },
  integritySeries: { type: Array, default: () => [] }
})

const profileSla = { baseline: 20, enhanced_5g: 40, degraded: 120, direct: 10 }

function formatMs(value) {
  if (value === null || value === undefined || value === '') return '-'
  const parsed = Number(value)
  return Number.isFinite(parsed) ? `${parsed.toFixed(parsed >= 100 ? 0 : 1)} ms` : '-'
}
function formatRate(value) {
  if (value === null || value === undefined || value === '') return '-'
  const parsed = Number(value)
  return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(2)}%` : '-'
}
function formatInteger(value) {
  if (value === null || value === undefined || value === '') return '-'
  const parsed = Number(value)
  return Number.isFinite(parsed) ? Math.round(parsed).toLocaleString('zh-CN') : '-'
}

const effectiveProfile = computed(() => (
  props.networkLink.enabled ? props.networkLink.profile : props.selectedProfile
))
const slaThreshold = computed(() => profileSla[effectiveProfile.value] || 40)
const p95 = computed(() => Number(props.networkLink.p95_latency_ms))
const loss = computed(() => Number(props.networkLink.loss_rate))
const hasMetrics = computed(() => (
  props.networkLink.p95_latency_ms !== null &&
  props.networkLink.p95_latency_ms !== undefined &&
  props.networkLink.loss_rate !== null &&
  props.networkLink.loss_rate !== undefined &&
  Number.isFinite(p95.value) &&
  Number.isFinite(loss.value)
))
const slaPassed = computed(() => hasMetrics.value && p95.value <= slaThreshold.value && loss.value <= 0.01)
const slaTone = computed(() => !hasMetrics.value ? 'pending' : slaPassed.value ? 'ok' : 'danger')
const slaLabel = computed(() => !hasMetrics.value ? '等待数据' : slaPassed.value ? '达标' : '超出阈值')
const processLabel = computed(() => {
  const status = props.processes.network_link
  if (status === 'running') return '运行中'
  if (status === 'failed') return '异常'
  if (status === 'stopped') return '已停止'
  return props.networkLink.enabled ? '等待启动' : '未启用'
})
const mecLocationLabel = computed(() => (
  props.mec.location === 'mec' ? 'MEC 边缘节点' : '中心平台'
))
const mecProcessLabel = computed(() => {
  if (!props.mec.enabled) return '未启用'
  const status = props.processes.mec || props.mec.status
  if (status === 'running' || status === 'active') return '运行中'
  if (status === 'failed') return '异常'
  if (status === 'stopped') return '已停止'
  return '正在建立'
})
const linkTone = computed(() => {
  if (props.processes.network_link === 'failed') return 'danger'
  if (props.networkLink.status === 'active') return slaPassed.value ? 'ok' : 'warn'
  if (props.processes.network_link === 'running') return 'active'
  return 'offline'
})
const linkStateLabel = computed(() => {
  if (linkTone.value === 'danger') return '链路异常'
  if (linkTone.value === 'warn') return '链路受限'
  if (linkTone.value === 'ok') return '链路稳定'
  if (linkTone.value === 'active') return '正在建立'
  return '未运行'
})
const latencyChartSeries = computed(() => {
  if (!props.latencySeries.some((item) => Number.isFinite(Number(item)))) return []
  return [{ name: 'E2E latency', data: props.latencySeries }]
})
const kpis = computed(() => [
  { label: '链路场景', value: effectiveProfile.value || '-', meta: props.networkLink.enabled ? '运行profile' : '待启动profile' },
  { label: '平均时延', value: hasMetrics.value ? formatMs(props.networkLink.avg_latency_ms) : '-', meta: '端到端' },
  { label: 'P95时延', value: hasMetrics.value ? formatMs(props.networkLink.p95_latency_ms) : '-', meta: `SLA ${formatMs(slaThreshold.value)}` },
  { label: '丢包率', value: hasMetrics.value ? formatRate(props.networkLink.loss_rate) : '-', meta: `趋势点 ${props.lossSeries.length}` },
  { label: '数据完整率', value: hasMetrics.value ? formatRate(props.networkLink.data_integrity) : '-', meta: `趋势点 ${props.integritySeries.length}` }
])
</script>

<style scoped>
.link-monitor { overflow: hidden; }
.link-monitor__body { padding: var(--space-lg); }
.link-state { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; font-weight: 800; }
.link-kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); border: 1px solid var(--line-soft); border-radius: var(--radius-md); overflow: hidden; }
.link-kpi { min-width: 0; padding: 12px 14px; background: var(--bg-panel-raised); border-right: 1px solid var(--line-soft); }
.link-kpi:last-child { border-right: 0; }
.link-kpi span, .link-kpi small { display: block; color: var(--text-tertiary); font-size: 11px; }
.link-kpi strong { display: block; margin: 6px 0 4px; overflow: hidden; color: var(--text-primary); font-size: 19px; text-overflow: ellipsis; white-space: nowrap; }
.link-monitor__workspace { display: grid; grid-template-columns: minmax(0, 1fr) 240px; gap: var(--space-md); margin-top: var(--space-md); }
.mec-strip { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); margin-top: var(--space-md); border: 1px solid var(--line-soft); border-radius: var(--radius-md); background: var(--bg-panel-raised); }
.mec-strip > div { min-width: 0; padding: 10px 14px; border-right: 1px solid var(--line-soft); }
.mec-strip > div:last-child { border-right: 0; }
.mec-strip span, .mec-strip strong { display: block; }
.mec-strip span { color: var(--text-tertiary); font-size: 11px; }
.mec-strip strong { margin-top: 5px; overflow: hidden; color: var(--text-primary); font-size: 13px; text-overflow: ellipsis; white-space: nowrap; }
.link-chart, .link-assurance { border: 1px solid var(--line-soft); border-radius: var(--radius-md); background: var(--bg-panel-raised); }
.link-chart { min-width: 0; padding: 12px; }
.link-chart__header { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 6px; font-size: 12px; color: var(--text-secondary); }
.link-chart__header span { color: var(--text-tertiary); }
.link-assurance { padding: 4px 14px; }
.link-assurance__row { display: flex; justify-content: space-between; gap: 12px; padding: 12px 0; border-bottom: 1px solid var(--line-soft); font-size: 12px; }
.link-assurance__row:last-child { border-bottom: 0; }
.link-assurance__row span { color: var(--text-tertiary); }
.link-assurance__row strong { color: var(--text-primary); text-align: right; }
.text-ok { color: var(--status-ok) !important; }
.text-danger { color: var(--status-danger) !important; }
.text-pending { color: var(--status-pending) !important; }
@media (max-width: 1120px) { .link-kpis, .mec-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); } .link-kpi, .mec-strip > div { border-bottom: 1px solid var(--line-soft); } .link-monitor__workspace { grid-template-columns: 1fr; } }
@media (max-width: 640px) { .link-kpis, .mec-strip { grid-template-columns: 1fr; } .link-kpi, .mec-strip > div { border-right: 0; } }
</style>
