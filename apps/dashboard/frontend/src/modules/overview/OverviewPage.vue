<template>
  <div class="overview-workbench">
    <section class="overview-statusbar">
      <div class="statusbar-item statusbar-item--online">
        <span class="status-dot status-dot--ok"></span>
        <span>系统在线</span>
      </div>
      <div class="statusbar-divider"></div>
      <div class="statusbar-item">
        <span>最近会话:</span>
        <strong>{{ sessionLabel }}</strong>
      </div>
      <div class="statusbar-divider"></div>
      <div class="statusbar-item">
        <span>后端状态:</span>
        <strong class="text-ok">{{ telemetrySummary.backend_status || '正常' }}</strong>
      </div>
      <div class="statusbar-divider"></div>
      <div class="statusbar-item statusbar-item--model">
        <span>模型:</span>
        <strong class="statusbar-value--truncate" :title="modelRawName">{{ modelDisplayName }}</strong>
      </div>
      <div class="statusbar-divider"></div>
      <div class="statusbar-item">
        <span>最近同步:</span>
        <strong>{{ telemetrySummary.last_sync || '-' }}</strong>
      </div>
      <button class="refresh-button" type="button" title="刷新总览" @click="loadOverview">
        <RefreshRight />
      </button>
    </section>

    <section class="overview-kpis">
      <a
        v-for="metric in kpiCards"
        :key="metric.key"
        :href="metric.route"
        class="kpi-card"
        :class="`kpi-card--${metric.tone}`"
        @click.prevent="goTo(metric.route)"
      >
        <div class="kpi-card__icon" aria-hidden="true">
          <component :is="metric.icon" />
        </div>
        <div class="kpi-card__content">
          <span class="kpi-card__label">{{ metric.label }}</span>
          <strong class="kpi-card__value">{{ metric.value }}</strong>
          <span class="kpi-card__meta">{{ metric.meta }}</span>
        </div>
      </a>
    </section>

    <section class="overview-main-grid">
      <article class="panel-card panel-card--trend">
        <header class="panel-card__header">
          <div>
            <h2>近 24h 诊断风险趋势</h2>
            <p>基于最近诊断任务、告警事件与故障概率记录聚合</p>
          </div>
        </header>
        <div v-if="riskPoints.length" ref="riskChartEl" class="risk-chart"></div>
        <div v-else class="empty-state empty-state--trend">
          <TrendCharts />
          <strong>暂无近 24h 风险趋势</strong>
          <span>近 24h 无诊断报告或告警事件</span>
          <span>生成新诊断报告或启动实时检测后自动形成趋势</span>
        </div>
      </article>

    </section>

    <section class="overview-secondary-grid">
      <article class="panel-card">
        <header class="panel-card__header panel-card__header--compact">
          <h2>当前检测模型</h2>
          <span>用于实时检测与故障概率曲线</span>
        </header>
        <div class="model-detail">
          <div class="model-detail__top">
            <div class="model-detail__name">
              <strong>{{ modelDisplayName }}</strong>
              <span :title="modelRawName">{{ modelRawName }}</span>
            </div>
            <span>更新 {{ modelSummary.updated_at_text || '-' }}</span>
          </div>
          <div class="model-detail__grid">
            <div>
              <span>窗口大小</span>
              <strong>{{ valueOrDash(modelSummary.window_size) }}</strong>
            </div>
            <div>
              <span>特征数量</span>
              <strong>{{ valueOrDash(modelSummary.feature_count) }}</strong>
            </div>
          </div>
          <div class="capability-tags">
            <span
              v-for="capability in capabilityLabels"
              :key="capability"
              class="capability-tag"
            >
              {{ capability }}
            </span>
          </div>
          <div class="model-score-grid">
            <div>
              <span>macro F1</span>
              <strong>{{ metricNumber(modelSummary.metrics?.macro_f1) }}</strong>
            </div>
            <div>
              <span>weighted F1</span>
              <strong>{{ metricNumber(modelSummary.metrics?.weighted_f1) }}</strong>
            </div>
            <div>
              <span>accuracy</span>
              <strong>{{ metricNumber(modelSummary.metrics?.accuracy) }}</strong>
            </div>
            <div>
              <span>avg confidence</span>
              <strong>{{ metricNumber(faultSummary.avg_confidence) }}</strong>
            </div>
          </div>
        </div>
      </article>

      <article class="panel-card">
        <header class="panel-card__header panel-card__header--compact">
          <div>
            <h2>最近告警</h2>
            <span>基于告警日志记录</span>
          </div>
          <a class="panel-action" href="/monitoring" @click.prevent="goTo('/monitoring')">查看监测</a>
        </header>
        <div v-if="recentWarnings.length" class="warning-list">
          <div v-for="item in recentWarnings" :key="item.id" class="warning-row">
            <span class="mono">{{ item.time || '-' }}</span>
            <span>{{ item.asset || '-' }}</span>
            <strong>{{ evidenceLabelText(item.message || '故障告警') }}</strong>
            <span class="status-badge" :class="severityBadgeClass(item.severity)">
              {{ severityText(item.severity) }}
            </span>
          </div>
        </div>
        <div v-else class="empty-state empty-state--small">暂无确认告警</div>
      </article>
    </section>

    <section class="overview-bottom-grid">
      <article class="panel-card">
        <header class="panel-card__header panel-card__header--compact">
          <div>
            <h2>最近任务队列</h2>
            <span>离线诊断任务</span>
          </div>
          <a class="panel-action" href="/diagnosis" @click.prevent="goTo('/diagnosis')">进入诊断</a>
        </header>
        <div v-if="recentTasks.length" class="task-table">
          <div class="task-table__head">
            <span>任务 ID</span>
            <span>资产</span>
            <span>诊断结论</span>
            <span>置信度</span>
            <span>状态</span>
            <span>更新时间</span>
          </div>
          <div v-for="task in recentTasks.slice(0, 5)" :key="task.id" class="task-table__row">
            <span class="mono">T-{{ task.id }}</span>
            <span>{{ assetDisplayName(task.asset_id ? `Asset #${task.asset_id}` : '-') }}</span>
            <span>{{ taskPredictionLabel(task) }}</span>
            <span class="mono">{{ taskConfidence(task) }}</span>
            <span class="status-badge" :class="statusBadgeClass(task.status)">
              {{ normalizeStatusText(task.status) }}
            </span>
            <span class="mono">{{ formatShortTime(task.updated_at || task.created_at) }}</span>
          </div>
        </div>
        <div v-else class="empty-state empty-state--small">暂无任务记录</div>
      </article>

      <article class="panel-card">
        <header class="panel-card__header panel-card__header--compact">
          <div>
            <h2>相似历史报告</h2>
            <span>参考</span>
          </div>
          <a class="panel-action" href="/reports" @click.prevent="goTo('/reports')">查看报告</a>
        </header>
        <div v-if="similarReports.length" class="similar-report-list">
          <div class="similar-report-list__head">
            <span>报告 ID</span>
            <span>资产</span>
            <span>故障类型</span>
            <span>置信度</span>
          </div>
          <div v-for="report in similarReports" :key="report.id" class="similar-report-list__row">
            <span class="mono">RPT-{{ report.id }}</span>
            <span>{{ assetDisplayName(report.asset || '-') }}</span>
            <span>{{ report.fault_label || '-' }}</span>
            <span class="mono">{{ percentDecimal(report.confidence) }}</span>
          </div>
        </div>
        <div v-else class="empty-state empty-state--small">暂无可参考历史报告</div>
      </article>
    </section>

    <p class="overview-provenance">
      统计基于日志资产、诊断任务、诊断报告与 telemetry fault events 聚合。
    </p>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import {
  Bell,
  Box,
  CircleCheck,
  Document,
  Finished,
  RefreshRight,
  TrendCharts,
  Warning
} from '@element-plus/icons-vue'
import { overviewApi } from '@/shared/services'

const overview = ref({})
const loading = ref(false)
const riskChartEl = ref(null)
const router = useRouter()
let riskChart = null

const assetSummary = computed(() => overview.value.asset_summary || {
  total: overview.value.assets_total || 0,
  csv_total: overview.value.assets_by_kind?.csv || 0,
  ulg_total: overview.value.assets_by_kind?.ulg || 0,
  diagnosable_total: overview.value.assets_total || 0
})
const taskSummary = computed(() => overview.value.task_summary || {
  completed_total: overview.value.runs_by_status?.success || 0,
  success_total: overview.value.runs_by_status?.success || 0,
  failed_total: overview.value.runs_by_status?.error || 0,
  recent: overview.value.recent_runs || []
})
const warningSummary = computed(() => overview.value.warning_summary || {
  confirmed_total: 0,
  severity_counts: {},
  recent: []
})
const faultSummary = computed(() => overview.value.fault_summary || {
  dominant_label_text: dominantFaultLabelFromLegacy.value,
  dominant_ratio: 0,
  avg_confidence: null,
  report_count: 0
})
const modelSummary = computed(() => overview.value.model_summary || { metrics: {}, capabilities: {} })
const telemetrySummary = computed(() => overview.value.telemetry_summary || {})
const riskPoints = computed(() => overview.value.risk_trend?.points || [])
const recentWarnings = computed(() => warningSummary.value.recent || [])
const recentTasks = computed(() => taskSummary.value.recent || overview.value.recent_runs || [])
const similarReports = computed(() => overview.value.similar_reports || [])

const modelRawName = computed(() => modelSummary.value.name || '暂无注册模型')
const modelDisplayName = computed(() => modelSummary.value.display_name || friendlyModelName(modelSummary.value))

const legacyDistribution = computed(() => overview.value.fault_distribution || { normal: 0, motor: 0, sensor: 0 })
const dominantFaultLabelFromLegacy = computed(() => {
  const entries = Object.entries(legacyDistribution.value)
  if (!entries.length) return '暂无数据'
  const [key, count] = entries.sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0))[0]
  if (!count) return '暂无数据'
  return faultLabelText(key)
})

const sessionLabel = computed(() => {
  const current = telemetrySummary.value.current_session_id
  const latest = telemetrySummary.value.latest_session_id
  if (current) return current
  if (latest) return latest
  return '暂无当前会话'
})

const kpiCards = computed(() => [
  {
    key: 'assets',
    label: '日志资产总数',
    value: valueOrDash(assetSummary.value.total),
    meta: `CSV ${assetSummary.value.csv_total || 0} / ULG ${assetSummary.value.ulg_total || 0}`,
    tone: 'brand',
    icon: Document,
    route: '/assets'
  },
  {
    key: 'diagnosable',
    label: '可诊断资产数',
    value: valueOrDash(assetSummary.value.diagnosable_total),
    meta: `CSV + 已预处理 ULG`,
    tone: 'success',
    icon: Box,
    route: '/assets'
  },
  {
    key: 'completed',
    label: '已完成任务数',
    value: valueOrDash(taskSummary.value.completed_total),
    meta: `成功 ${taskSummary.value.success_total || 0} / 失败 ${taskSummary.value.failed_total || 0}`,
    tone: 'warning',
    icon: Finished,
    route: '/diagnosis'
  },
  {
    key: 'warnings',
    label: '确认告警数',
    value: valueOrDash(warningSummary.value.confirmed_total),
    meta: `高危 ${warningSummary.value.severity_counts?.high || 0} / 中危 ${warningSummary.value.severity_counts?.medium || 0} / 低危 ${warningSummary.value.severity_counts?.low || 0}`,
    tone: 'danger',
    icon: Bell,
    route: '/monitoring'
  },
  {
    key: 'fault',
    label: '故障分类概览',
    value: faultSummary.value.dominant_label_text || '暂无数据',
    meta: `基于 ${faultSummary.value.report_count || 0} 份报告，占比 ${faultSummary.value.dominant_ratio || 0}%`,
    tone: 'pending',
    icon: Warning,
    route: '/reports'
  },
  {
    key: 'confidence',
    label: '平均诊断置信度',
    value: metricNumber(faultSummary.value.avg_confidence),
    meta: `基于 ${faultSummary.value.report_count || 0} 个报告均值`,
    tone: 'brand',
    icon: CircleCheck,
    route: '/reports'
  }
])

const capabilityLabels = computed(() => {
  const caps = modelSummary.value.capabilities || {}
  const labels = []
  if (caps.offline_detection) labels.push('离线诊断')
  if (caps.realtime_detection) labels.push('实时检测')
  if (caps.fault_prob_curve) labels.push('故障概率曲线')
  return labels.length ? labels : ['等待模型能力信息']
})

function valueOrDash(value) {
  return value === null || value === undefined || value === '' ? '-' : value
}

function metricNumber(value) {
  if (value === null || value === undefined || value === '') return '-'
  const number = Number(value)
  return Number.isFinite(number) ? number.toFixed(2) : value
}

function percentDecimal(value) {
  if (value === null || value === undefined || value === '') return '-'
  const number = Number(value)
  if (!Number.isFinite(number)) return value
  return `${Math.round(number * 100)}%`
}

function friendlyModelName(model) {
  const architecture = String(model?.architecture || model?.model || '').trim()
  const name = String(model?.name || '').trim()
  const lookup = `${architecture} ${name}`.toLowerCase()
  if (lookup.includes('kg') && lookup.includes('mstca')) return 'KG-MSTCA 在线检测模型'
  if (lookup.includes('mstca')) return 'MSTCA 在线检测模型'
  if (lookup.includes('cnn1d') || lookup.includes('cnn_1d')) return 'CNN1D 在线检测模型'
  if (lookup.includes('lstm')) return 'LSTM 离线诊断模型'
  return name || '暂无注册模型'
}

function evidenceLabelText(name) {
  const raw = String(name || '').trim()
  const map = {
    fault_probability: '故障概率异常',
    fault_prob: '故障概率异常',
    max_fault_prob: '最大故障概率异常',
    motor_current_rms: '电机电流 RMS 升高',
    phase_current_imbalance: '相电流不平衡',
    motor_current_imbalance: '电机电流不平衡',
    line_temperature: '线组温度升高',
    temperature: '温度异常',
    pwm_duty_cycle: 'PWM 占空比波动',
    voltage_sag: '电压纹波异常',
    battery_voltage: '电池电压异常',
    imu_drift: 'IMU 数据漂移',
    gyro_bias: '陀螺仪偏置异常',
    barometer: '气压计信号异常',
    warning_stage: '告警阶段',
    lead_time_s: '预测提前量',
    risk_score: '风险评分'
  }
  return map[raw.toLowerCase()] || raw || '故障告警'
}

function assetDisplayName(value) {
  const raw = String(value || '').trim()
  if (!raw || raw === '-') return '-'
  return raw.replace(/^Asset\s+#/i, '资产 #')
}

function goTo(route) {
  if (!route) return
  if (router?.push) {
    router.push(route)
  }
}

function normalizeStatusText(status) {
  if (status === 'success') return '已完成'
  if (status === 'error' || status === 'failed') return '失败'
  if (status === 'running') return '进行中'
  if (status === 'queued') return '排队中'
  return status || '未知'
}

function statusBadgeClass(status) {
  if (status === 'success') return 'status-badge--ok'
  if (status === 'error' || status === 'failed') return 'status-badge--danger'
  if (status === 'running') return 'status-badge--active'
  if (status === 'queued') return 'status-badge--pending'
  return 'status-badge--offline'
}

function severityBadgeClass(severity) {
  if (severity === 'high' || severity === 'critical') return 'status-badge--danger'
  if (severity === 'medium') return 'status-badge--warn'
  if (severity === 'low') return 'status-badge--ok'
  return 'status-badge--offline'
}

function severityText(severity) {
  if (severity === 'high' || severity === 'critical') return '高危'
  if (severity === 'medium') return '中危'
  if (severity === 'low') return '低危'
  return '未知'
}

function faultLabelText(label) {
  const map = {
    normal: '正常',
    motor: '电机故障',
    motor_failure: '电机故障',
    sensor: '传感器故障',
    battery_failure: '电池故障',
    prop_imbalance: '旋翼异常',
    thrust_loss: '推力损失'
  }
  return map[label] || label || '暂无数据'
}

function taskPredictionLabel(task) {
  const label = task?.result?.prediction?.label || task?.result?.diagnosis?.label
  return faultLabelText(label)
}

function taskConfidence(task) {
  const confidence = task?.result?.prediction?.confidence || task?.result?.diagnosis?.confidence
  return metricNumber(confidence)
}

function formatShortTime(ts) {
  if (!ts) return '-'
  try {
    return new Date(ts).toLocaleString('zh-CN', {
      hour: '2-digit',
      minute: '2-digit'
    })
  } catch {
    return ts
  }
}

function renderRiskChart() {
  if (!riskChartEl.value || !riskPoints.value.length) return
  if (!riskChart) {
    riskChart = echarts.init(riskChartEl.value)
  }
  const threshold = overview.value.risk_trend?.threshold ?? 0.7
  riskChart.setOption({
    grid: { left: 36, right: 24, top: 28, bottom: 32 },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: riskPoints.value.map((item) => item.time),
      axisLine: { lineStyle: { color: '#d1d5db' } },
      axisLabel: { color: '#64748b', fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      splitLine: { lineStyle: { color: '#edf2f7' } },
      axisLabel: { color: '#64748b', fontSize: 11 }
    },
    series: [
      {
        name: '风险趋势',
        type: 'line',
        smooth: true,
        symbolSize: 7,
        data: riskPoints.value.map((item) => item.risk),
        lineStyle: { color: '#2563eb', width: 2 },
        itemStyle: { color: '#2563eb' },
        areaStyle: { color: 'rgba(37, 99, 235, 0.08)' },
        markLine: {
          symbol: 'none',
          lineStyle: { color: '#dc2626', type: 'dashed' },
          label: {
            formatter: '风险',
            color: '#dc2626',
            position: 'insideEndTop',
            distance: 6
          },
          data: [{ yAxis: threshold }]
        },
        markPoint: {
          symbol: 'pin',
          symbolSize: 30,
          itemStyle: { color: '#f97316' },
          data: riskPoints.value
            .map((item, index) => item.marker ? { coord: [index, item.risk], name: '告警事件' } : null)
            .filter(Boolean)
        }
      }
    ]
  })
}

async function loadOverview() {
  loading.value = true
  try {
    const response = await overviewApi.getOverview()
    overview.value = response.data || {}
    await nextTick()
    renderRiskChart()
  } catch (error) {
    ElMessage.error(`加载总览失败：${error.message || '未知错误'}`)
  } finally {
    loading.value = false
  }
}

watch(riskPoints, async () => {
  await nextTick()
  renderRiskChart()
})

onMounted(loadOverview)

onBeforeUnmount(() => {
  if (riskChart) {
    riskChart.dispose()
    riskChart = null
  }
})
</script>

<style scoped>
.overview-workbench {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.overview-statusbar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px 10px;
  padding: 7px 10px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-lg);
  background: var(--bg-panel);
  box-shadow: var(--shadow-sm);
  font-size: 12px;
  color: var(--text-secondary);
  min-width: 0;
}

.statusbar-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  white-space: nowrap;
}

.statusbar-item strong {
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 12px;
}

.statusbar-item--model {
  flex: 1 1 220px;
}

.statusbar-value--truncate {
  display: inline-block;
  min-width: 0;
  max-width: 260px;
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: bottom;
}

.statusbar-item--online {
  color: var(--status-ok);
  font-weight: 600;
}

.statusbar-divider {
  width: 1px;
  height: 14px;
  background: var(--line-soft);
  flex-shrink: 0;
}

.refresh-button {
  margin-left: auto;
  width: 28px;
  height: 28px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  cursor: pointer;
  color: var(--brand);
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.refresh-button svg {
  width: 14px;
  height: 14px;
}

.text-ok {
  color: var(--status-ok) !important;
}

.overview-kpis {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.kpi-card {
  display: grid;
  grid-template-columns: 30px minmax(0, 1fr);
  gap: 10px;
  min-height: 96px;
  padding: 14px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-lg);
  background: var(--bg-panel);
  box-shadow: var(--shadow-sm);
  color: inherit;
  text-decoration: none;
  transition: border-color 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease;
}

.kpi-card:hover {
  border-color: rgba(37, 99, 235, 0.28);
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}

.kpi-card__icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: var(--radius-md);
}

.kpi-card__icon svg {
  width: 16px;
  height: 16px;
}

.kpi-card--brand .kpi-card__icon { color: var(--brand); background: var(--brand-soft); }
.kpi-card--success .kpi-card__icon { color: var(--status-ok); background: var(--status-ok-soft); }
.kpi-card--warning .kpi-card__icon { color: var(--status-warn); background: var(--status-warn-soft); }
.kpi-card--danger .kpi-card__icon { color: var(--status-danger); background: var(--status-danger-soft); }
.kpi-card--pending .kpi-card__icon { color: var(--status-pending); background: var(--status-pending-soft); }

.kpi-card__content {
  min-width: 0;
}

.kpi-card__label {
  display: block;
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 600;
}

.kpi-card__value {
  display: block;
  margin-top: 7px;
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 26px;
  line-height: 1;
}

.kpi-card__meta {
  display: block;
  margin-top: 8px;
  color: var(--text-tertiary);
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.overview-main-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
  align-items: stretch;
}

.overview-secondary-grid {
  display: grid;
  grid-template-columns: minmax(420px, 0.9fr) minmax(420px, 1.1fr);
  gap: 10px;
  align-items: stretch;
}

.panel-card {
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-lg);
  background: var(--bg-panel);
  box-shadow: var(--shadow-sm);
  overflow: hidden;
}

.panel-card__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 14px;
  border-bottom: 1px solid var(--line-soft);
  background: var(--bg-panel-raised);
}

.panel-card__header--compact {
  align-items: center;
}

.panel-card__header h2 {
  margin: 0;
  color: var(--text-primary);
  font-size: 14px;
  font-weight: 700;
}

.panel-card__header p,
.panel-card__header span {
  margin: 2px 0 0;
  color: var(--text-tertiary);
  font-size: 12px;
}

.panel-action {
  flex-shrink: 0;
  align-self: center;
  padding: 3px 8px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-sm);
  background: var(--bg-panel);
  color: var(--brand) !important;
  font-size: 12px;
  font-weight: 600;
  line-height: 1.4;
  text-decoration: none;
}

.panel-action:hover {
  border-color: var(--brand);
  background: var(--brand-soft);
}

.panel-card--trend {
  min-height: 334px;
}

.risk-chart {
  height: 270px;
  padding: 8px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-height: 180px;
  color: var(--text-tertiary);
  font-size: 13px;
  text-align: center;
  padding: 16px;
}

.empty-state svg {
  width: 22px;
  height: 22px;
  color: var(--brand);
}

.empty-state strong {
  color: var(--text-secondary);
  font-size: 13px;
}

.empty-state span {
  color: var(--text-tertiary);
  font-size: 12px;
}

.empty-state--small {
  min-height: 96px;
}

.empty-state--trend {
  min-height: 248px;
}

.model-detail {
  padding: 12px 14px;
}

.warning-list {
  padding: 8px 12px 12px;
}

.warning-row {
  display: grid;
  grid-template-columns: 60px 78px minmax(0, 1fr) 50px;
  align-items: center;
  gap: 8px;
  padding: 8px 0;
  border-bottom: 1px solid var(--line-faint);
  color: var(--text-secondary);
  font-size: 12px;
}

.warning-row:last-child {
  border-bottom: none;
}

.warning-row strong {
  color: var(--text-primary);
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.mono {
  font-family: var(--font-mono);
}

.model-detail__top,
.model-detail__grid,
.model-score-grid {
  display: grid;
  gap: 10px;
}

.model-detail__top {
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  color: var(--text-secondary);
  font-size: 12px;
}

.model-detail__name {
  min-width: 0;
}

.model-detail__name strong {
  display: block;
  color: var(--text-primary);
  font-size: 13px;
}

.model-detail__name span {
  display: block;
  margin-top: 3px;
  overflow: hidden;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-detail__grid,
.model-score-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 12px;
}

.model-detail__grid div,
.model-score-grid div {
  padding: 8px;
  border: 1px solid var(--line-faint);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
}

.model-detail__grid span,
.model-score-grid span {
  display: block;
  color: var(--text-tertiary);
  font-size: 11px;
}

.model-detail__grid strong,
.model-score-grid strong {
  display: block;
  margin-top: 4px;
  color: var(--text-primary);
  font-family: var(--font-mono);
}

.capability-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 12px;
}

.capability-tag {
  padding: 3px 7px;
  border-radius: var(--radius-sm);
  background: var(--brand-soft);
  color: var(--brand);
  font-size: 11px;
  font-weight: 600;
}

.overview-bottom-grid {
  display: grid;
  grid-template-columns: minmax(500px, 1.2fr) minmax(360px, 0.8fr);
  gap: 12px;
}

.task-table,
.similar-report-list {
  padding: 10px 12px 12px;
}

.task-table__head,
.task-table__row {
  display: grid;
  grid-template-columns: 94px 78px minmax(110px, 1fr) 74px 72px 76px;
  gap: 8px;
  align-items: center;
  font-size: 12px;
}

.similar-report-list__head,
.similar-report-list__row {
  display: grid;
  grid-template-columns: 82px 78px minmax(100px, 1fr) 62px;
  gap: 8px;
  align-items: center;
  font-size: 12px;
}

.task-table__head,
.similar-report-list__head {
  padding: 0 0 8px;
  color: var(--text-tertiary);
  border-bottom: 1px solid var(--line-soft);
}

.task-table__row,
.similar-report-list__row {
  padding: 9px 0;
  color: var(--text-secondary);
  border-bottom: 1px solid var(--line-faint);
}

.task-table__row:last-child,
.similar-report-list__row:last-child {
  border-bottom: none;
}

.overview-provenance {
  margin: -2px 2px 0;
  color: var(--text-tertiary);
  font-size: 12px;
  line-height: 1.6;
}

@media (max-width: 1500px) {
  .overview-bottom-grid {
    grid-template-columns: minmax(0, 1fr) minmax(320px, 0.8fr);
  }
}

@media (min-width: 1680px) {
  .overview-kpis {
    grid-template-columns: repeat(6, minmax(0, 1fr));
  }
}

@media (max-width: 1180px) {
  .overview-secondary-grid,
  .overview-bottom-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 920px) {
  .overview-kpis {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .overview-kpis {
    grid-template-columns: 1fr;
  }

  .task-table__head,
  .task-table__row {
    grid-template-columns: 88px minmax(0, 1fr) 64px;
  }

  .task-table__head span:nth-child(2),
  .task-table__head span:nth-child(4),
  .task-table__head span:nth-child(6),
  .task-table__row span:nth-child(2),
  .task-table__row span:nth-child(4),
  .task-table__row span:nth-child(6) {
    display: none;
  }
}
</style>
