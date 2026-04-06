<template>
  <div class="page-stack model-compare-page">
    <section class="page-panel compare-hero" :class="{ 'compare-hero--realtime': mode === 'realtime' }">
      <div class="compare-hero__main">
        <div class="compare-hero__title-row">
          <div>
            <h2 class="compare-hero__title">模型对比</h2>
            <p class="compare-hero__desc">{{ modeMeta.description }}</p>
          </div>

          <div class="compare-mode-switch" role="tablist" aria-label="模型对比模式切换">
            <button
              type="button"
              class="compare-mode-switch__btn"
              :class="{ 'is-active': mode === 'offline' }"
              @click="mode = 'offline'"
            >
              离线分类对比
            </button>
            <button
              type="button"
              class="compare-mode-switch__btn"
              :class="{ 'is-active': mode === 'realtime' }"
              @click="mode = 'realtime'"
            >
              实时检测对比
            </button>
          </div>
        </div>

        <div v-if="mode === 'offline'" class="compare-hero__summary-row">
          <div v-if="offlineChampion" class="compare-hero__winner">
            <div class="compare-hero__winner-label">当前最优</div>
            <strong>{{ offlineChampion.displayTitle }}</strong>
            <small>{{ offlineChampion.stageLabel }}</small>
            <div class="compare-hero__winner-metrics">
              <span>Real Macro F1 {{ formatScore(offlineChampion.realMacroF1) }}</span>
              <span>Balanced Acc {{ formatScore(offlineChampion.realBalancedAccuracy) }}</span>
            </div>
          </div>
        </div>

        <div v-else class="compare-hero__summary-row compare-hero__summary-row--realtime">
          <div class="compare-hero__winner compare-hero__winner--realtime">
            <div class="compare-hero__winner-label">当前资产</div>
            <strong>{{ realtimeSelectedAssetLabel }}</strong>
            <small>{{ realtimeSelectedAssetHint }}</small>
            <div class="compare-hero__winner-metrics">
              <span>模型数 {{ realtimeEntries.length }}</span>
              <span>采样点 {{ realtimePointCount }}</span>
            </div>
          </div>

          <div v-if="realtimeChampion" class="compare-hero__winner compare-hero__winner--realtime">
            <div class="compare-hero__winner-label">响应最快</div>
            <strong>{{ realtimeChampion.displayTitle }}</strong>
            <small>{{ realtimeChampion.stageLabel }}</small>
            <div class="compare-hero__winner-metrics">
              <span>首次告警 {{ formatTime(realtimeChampion.firstAlertTime) }}</span>
              <span>峰值概率 {{ formatScore(realtimeChampion.peakFaultProb) }}</span>
            </div>
          </div>
        </div>
      </div>

      <div v-if="mode === 'offline'" class="compare-hero__stats">
        <article class="metric-card">
          <div class="metric-card__label">当前最优模型</div>
          <div class="metric-card__value">{{ offlineChampion ? offlineChampion.shortName : '--' }}</div>
          <div class="metric-card__meta">{{ offlineChampion ? offlineChampion.stageLabel : '等待模型数据' }}</div>
        </article>
        <article class="metric-card">
          <div class="metric-card__label">最佳 Real Macro F1</div>
          <div class="metric-card__value">{{ offlineChampion ? formatScore(offlineChampion.realMacroF1) : '--' }}</div>
          <div class="metric-card__meta">真实域总体诊断能力</div>
        </article>
        <article class="metric-card">
          <div class="metric-card__label">相对基线提升</div>
          <div class="metric-card__value">{{ offlineBaselineDelta }}</div>
          <div class="metric-card__meta">相对 `BiLSTM + FocalLoss` 的 Real Macro F1 增益</div>
        </article>
        <article class="metric-card">
          <div class="metric-card__label">参与对比模型</div>
          <div class="metric-card__value">{{ offlineModels.length }}</div>
          <div class="metric-card__meta">基础模型 / 结构升级 / 最终方案</div>
        </article>
      </div>

      <div v-else class="compare-hero__stats">
        <article class="metric-card">
          <div class="metric-card__label">最早告警</div>
          <div class="metric-card__value">{{ realtimeChampion ? formatTime(realtimeChampion.firstAlertTime) : '--' }}</div>
          <div class="metric-card__meta">越早说明在线响应越敏感</div>
        </article>
        <article class="metric-card">
          <div class="metric-card__label">最高峰值概率</div>
          <div class="metric-card__value">{{ realtimeChampion ? formatScore(realtimeChampion.peakFaultProb) : '--' }}</div>
          <div class="metric-card__meta">同一遥测段上的故障响应峰值</div>
        </article>
        <article class="metric-card">
          <div class="metric-card__label">平均波动</div>
          <div class="metric-card__value">{{ realtimeChampion ? formatScore(realtimeChampion.stdFaultProb) : '--' }}</div>
          <div class="metric-card__meta">数值越小表示曲线越稳定</div>
        </article>
        <article class="metric-card">
          <div class="metric-card__label">参与对比模型</div>
          <div class="metric-card__value">{{ realtimeEntries.length }}</div>
          <div class="metric-card__meta">在线检测模型 / 同一段遥测回放</div>
        </article>
      </div>
    </section>

    <section v-if="mode === 'offline'" class="section-card">
      <div class="section-card__header">
        <div>
          <div class="section-card__title">模型对比卡片</div>
          <div class="section-card__desc">每张卡片展示一个模型的测试域与真实域表现，并给出相对基线的差值。</div>
        </div>
      </div>
      <div class="section-card__body compare-card-grid">
        <article
          v-for="model in offlineModels"
          :key="model.name"
          class="compare-card"
          :class="{ 'is-champion': offlineChampion?.name === model.name }"
          :style="{ '--accent': model.accent }"
        >
          <header class="compare-card__header">
            <div>
              <div class="compare-card__tag">{{ model.stageLabel }}</div>
              <h3 class="compare-card__title">{{ model.displayTitle }}</h3>
              <p class="compare-card__subtitle">{{ model.summary }}</p>
            </div>
            <div class="compare-card__rank">#{{ model.rank }}</div>
          </header>

          <div class="compare-card__stats">
            <div class="compare-stat">
              <span>Test Macro F1</span>
              <strong>{{ formatScore(model.testMacroF1) }}</strong>
            </div>
            <div class="compare-stat">
              <span>Real Macro F1</span>
              <strong>{{ formatScore(model.realMacroF1) }}</strong>
            </div>
            <div class="compare-stat">
              <span>Balanced Acc</span>
              <strong>{{ formatScore(model.realBalancedAccuracy) }}</strong>
            </div>
            <div class="compare-stat">
              <span>Δ vs BiLSTM</span>
              <strong>{{ formatDelta(model.deltaVsBaseline) }}</strong>
            </div>
          </div>

          <div class="compare-card__bars">
            <div class="compare-bar">
              <div class="compare-bar__label">Real</div>
              <div class="compare-bar__track">
                <div class="compare-bar__fill" :style="{ width: `${percent(model.realMacroF1)}%` }"></div>
              </div>
              <div class="compare-bar__value">{{ formatScore(model.realMacroF1) }}</div>
            </div>
            <div class="compare-bar">
              <div class="compare-bar__label">Test</div>
              <div class="compare-bar__track">
                <div class="compare-bar__fill compare-bar__fill--soft" :style="{ width: `${percent(model.testMacroF1)}%` }"></div>
              </div>
              <div class="compare-bar__value">{{ formatScore(model.testMacroF1) }}</div>
            </div>
          </div>

          <footer class="compare-card__footer">
            <span>{{ model.name }}</span>
            <span>Val F1 {{ formatScore(model.bestValF1) }}</span>
          </footer>
        </article>
      </div>
    </section>

    <section v-else class="section-card">
      <div class="section-card__header compare-realtime__header">
        <div>
          <div class="section-card__title">实时遥测对比</div>
          <div class="section-card__desc">同一段 CSV 遥测回放上，对比多个在线模型的 fault_prob 曲线与告警响应。</div>
        </div>

        <div class="compare-realtime__controls">
          <label class="compare-select">
            <span>遥测资产</span>
            <select v-model="selectedRealtimeAssetId" :disabled="realtimeLoading || !realtimeAssets.length">
              <option v-for="asset in realtimeAssets" :key="asset.id" :value="asset.id">
                {{ asset.filename }}
              </option>
            </select>
          </label>
          <button type="button" class="compare-refresh-btn" :disabled="realtimeLoading" @click="reloadRealtimeComparison">
            {{ realtimeLoading ? '加载中…' : '重新加载' }}
          </button>
        </div>
      </div>

      <div class="section-card__body">
        <div v-if="realtimeLoading" class="compare-empty">正在加载实时对比数据…</div>
        <div v-else-if="realtimeError" class="compare-empty compare-empty--error">{{ realtimeError }}</div>
        <div v-else-if="!realtimeEntries.length" class="compare-empty">
          {{ realtimeAssets.length ? '未找到可对比的在线模型或曲线数据。' : '暂无可用于实时对比的 CSV 资产。' }}
        </div>
        <div v-else class="compare-realtime-layout">
          <TimeSeriesChart
            :x-data="realtimeChart.xData"
            :series="realtimeChart.series"
            :height="360"
            empty-text="暂无实时曲线数据"
            x-axis-name="time_s"
          />

          <table class="compare-table compare-table--realtime">
            <thead>
              <tr>
                <th>模型</th>
                <th>首次告警</th>
                <th>峰值 fault_prob</th>
                <th>平均 fault_prob</th>
                <th>波动(Std)</th>
                <th>曲线点数</th>
                <th>原始 Real Macro F1</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="model in realtimeEntries" :key="`${model.name}-realtime`">
                <td>
                  <strong>{{ model.displayTitle }}</strong>
                  <div class="compare-table__sub">{{ model.name }}</div>
                </td>
                <td>{{ formatTime(model.firstAlertTime) }}</td>
                <td>{{ formatScore(model.peakFaultProb) }}</td>
                <td>{{ formatScore(model.meanFaultProb) }}</td>
                <td>{{ formatScore(model.stdFaultProb) }}</td>
                <td>{{ model.pointCount }}</td>
                <td>{{ formatScore(model.registryRealMacroF1) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <section v-if="mode === 'offline'" class="section-card">
      <div class="section-card__header">
        <div>
          <div class="section-card__title">指标总表</div>
          <div class="section-card__desc">原始模型指标与相对基线差值，适合直接截图展示。</div>
        </div>
      </div>
      <div class="section-card__body">
        <div v-if="offlineLoading" class="compare-empty">正在加载模型信息…</div>
        <div v-else-if="offlineError" class="compare-empty compare-empty--error">{{ offlineError }}</div>
        <div v-else-if="!offlineModels.length" class="compare-empty">未发现可对比的离线模型。</div>
        <div v-else class="compare-table-wrap">
          <table class="compare-table">
            <thead>
              <tr>
                <th>模型</th>
                <th>阶段</th>
                <th>Test Macro F1</th>
                <th>Real Macro F1</th>
                <th>Balanced Acc</th>
                <th>Real Accuracy</th>
                <th>Δ vs BiLSTM</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="model in offlineModels" :key="`${model.name}-row`">
                <td>
                  <strong>{{ model.displayTitle }}</strong>
                  <div class="compare-table__sub">{{ model.name }}</div>
                </td>
                <td>{{ model.stageLabel }}</td>
                <td>{{ formatScore(model.testMacroF1) }}</td>
                <td>{{ formatScore(model.realMacroF1) }}</td>
                <td>{{ formatScore(model.realBalancedAccuracy) }}</td>
                <td>{{ formatScore(model.realAccuracy) }}</td>
                <td>{{ formatDelta(model.deltaVsBaseline) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import TimeSeriesChart from '@/shared/components/TimeSeriesChart.vue'
import { analysisApi, assetApi, modelApi } from '@/shared/services'

const mode = ref('offline')
const offlineLoading = ref(false)
const offlineError = ref('')
const offlineRawModels = ref([])

const realtimeLoading = ref(false)
const realtimeError = ref('')
const realtimeRawModels = ref([])
const realtimeAssets = ref([])
const selectedRealtimeAssetId = ref(null)
const realtimeCurves = ref([])
const isBootstrappingRealtime = ref(false)

const alertThreshold = 0.6

const offlineSpotlight = {
  transfer3_bilstm_fewshot: {
    displayTitle: 'BiLSTM + FocalLoss',
    shortName: 'BiLSTM',
    stageLabel: '基础模型',
    summary: 'few-shot 主协议下的基础强基线，用于衡量结构升级和最终方案是否真正有效。',
    accent: '#0f766e'
  },
  transfer3_cnnlstm_fewshot: {
    displayTitle: 'CNN-LSTM + FocalLoss',
    shortName: 'CNN-LSTM',
    stageLabel: '结构升级',
    summary: '在基础混合结构上加入多尺度卷积、raw skip path 和更稳的归一化策略。',
    accent: '#22577a'
  },
  transfer3_cnnlstm_normalproto: {
    displayTitle: 'CNN-LSTM + Normal Prototype',
    shortName: 'NormalProto',
    stageLabel: '最终方案',
    summary: '在结构升级 backbone 上定向修正 normal 边界，最终拿到当前最佳真实域结果。',
    accent: '#9a3412'
  }
}

const realtimeSpotlight = {
  online_detector_cnn1d_rfly: {
    displayTitle: 'CNN1D + Rfly',
    stageLabel: '实时主模型',
    summary: '现阶段用于在线 fault_prob 曲线对比的主模型。',
    accent: '#0f766e'
  },
  online_detector_kg_mstca_rfly: {
    displayTitle: 'KG-MSTCA + Rfly',
    stageLabel: '时序增强',
    summary: '基于时序建模与特征融合的在线检测模型。',
    accent: '#22577a'
  },
  online_detector_CNN: {
    displayTitle: 'CNN Online',
    stageLabel: '兼容模型',
    summary: '保留的在线兼容模型，用于对比响应曲线。',
    accent: '#be123c'
  },
  online_detector_final: {
    displayTitle: 'Online Final',
    stageLabel: '最终方案',
    summary: '在线检测的最终候选模型。',
    accent: '#7c3aed'
  }
}

function formatScore(value) {
  const number = Number(value)
  if (Number.isNaN(number)) return '--'
  return number.toFixed(4)
}

function formatDelta(value) {
  const number = Number(value)
  if (Number.isNaN(number)) return '--'
  const sign = number > 0 ? '+' : ''
  return `${sign}${number.toFixed(4)}`
}

function formatTime(value) {
  const number = Number(value)
  if (Number.isNaN(number)) return '--'
  return `${number.toFixed(2)}s`
}

function percent(value) {
  const number = Number(value)
  if (Number.isNaN(number)) return 0
  return Math.max(0, Math.min(number * 100, 100))
}

function parseMetric(value) {
  const number = Number(value)
  return Number.isNaN(number) ? NaN : number
}

function mean(values) {
  if (!values.length) return NaN
  return values.reduce((sum, item) => sum + item, 0) / values.length
}

function standardDeviation(values) {
  if (!values.length) return NaN
  const avg = mean(values)
  const variance = mean(values.map((value) => (value - avg) ** 2))
  return Math.sqrt(variance)
}

function firstAlertTime(curve) {
  const hit = curve.find((point) => Number(point.fault_prob) >= alertThreshold)
  return hit ? Number(hit.time_s) : NaN
}

function countAlertSegments(curve) {
  let count = 0
  let inSegment = false
  for (const point of curve) {
    const value = Number(point.fault_prob)
    if (value >= alertThreshold && !inSegment) {
      count += 1
      inSegment = true
      continue
    }
    if (value < alertThreshold) {
      inSegment = false
    }
  }
  return count
}

function normalizeAssetLabel(asset) {
  if (!asset) return '--'
  return asset.filename || asset.name || `资产 #${asset.id}`
}

const modeMeta = computed(() => {
  if (mode.value === 'realtime') {
    return {
      description: '对比多个在线检测模型在同一段实时遥测回放上的 fault_prob 曲线、首次告警时间与稳定性。'
    }
  }
  return {
    description: '对比当前重点离线模型在测试集与真实域上的关键指标，重点观察 Real Macro F1、Balanced Accuracy 与结构升级带来的增益。'
  }
})

const offlineBaselineModel = computed(() => {
  return offlineRawModels.value.find((item) => item.name === 'transfer3_bilstm_fewshot') || null
})

const offlineModels = computed(() => {
  const baseline = offlineBaselineModel.value
  const baselineRealMacroF1 = parseMetric(baseline?.metrics_external?.macro_f1)

  const entries = offlineRawModels.value
    .filter((item) => item.task === 'offline_multiclass')
    .filter((item) => Object.prototype.hasOwnProperty.call(offlineSpotlight, item.name))
    .map((item) => {
      const meta = offlineSpotlight[item.name]
      const realMetrics = item.metrics_external || {}
      const realMacroF1 = parseMetric(realMetrics.macro_f1)
      return {
        ...item,
        ...meta,
        testMacroF1: parseMetric(item.metrics?.macro_f1),
        bestValF1: parseMetric(item.metrics?.best_val_f1),
        realMacroF1,
        realBalancedAccuracy: parseMetric(realMetrics.balanced_accuracy),
        realAccuracy: parseMetric(realMetrics.accuracy),
        deltaVsBaseline:
          Number.isNaN(realMacroF1) || Number.isNaN(baselineRealMacroF1)
            ? NaN
            : realMacroF1 - baselineRealMacroF1
      }
    })
    .sort((a, b) => {
      const aReal = Number.isNaN(a.realMacroF1) ? -1 : a.realMacroF1
      const bReal = Number.isNaN(b.realMacroF1) ? -1 : b.realMacroF1
      const aBalanced = Number.isNaN(a.realBalancedAccuracy) ? -1 : a.realBalancedAccuracy
      const bBalanced = Number.isNaN(b.realBalancedAccuracy) ? -1 : b.realBalancedAccuracy
      const aAccuracy = Number.isNaN(a.realAccuracy) ? -1 : a.realAccuracy
      const bAccuracy = Number.isNaN(b.realAccuracy) ? -1 : b.realAccuracy
      return bReal - aReal || bBalanced - aBalanced || bAccuracy - aAccuracy
    })

  return entries.map((item, index) => ({
    ...item,
    rank: index + 1
  }))
})

const offlineChampion = computed(() => offlineModels.value[0] || null)
const offlineBaselineDelta = computed(() => {
  if (!offlineChampion.value || !offlineBaselineModel.value) return '--'
  const baselineMetric = parseMetric(offlineBaselineModel.value.metrics_external?.macro_f1)
  if (Number.isNaN(offlineChampion.value.realMacroF1) || Number.isNaN(baselineMetric)) return '--'
  return formatDelta(offlineChampion.value.realMacroF1 - baselineMetric)
})

const realtimeModels = computed(() => {
  return realtimeRawModels.value
    .map((item) => ({
      ...item,
      ...realtimeSpotlight[item.name],
      displayTitle: realtimeSpotlight[item.name]?.displayTitle || item.label || item.name,
      stageLabel: realtimeSpotlight[item.name]?.stageLabel || '在线模型',
      summary: realtimeSpotlight[item.name]?.summary || '在线检测模型',
      accent: realtimeSpotlight[item.name]?.accent || '#0f766e'
    }))
    .slice(0, 4)
})

const realtimeEntries = computed(() => {
  return realtimeCurves.value
    .map((item) => {
      const curve = Array.isArray(item.curve) ? item.curve : []
      const probs = curve.map((point) => Number(point.fault_prob)).filter((value) => !Number.isNaN(value))
      const registryMetrics = item.model?.metrics_external || item.model?.metrics || {}
      return {
        name: item.model?.name || item.name,
        displayTitle: item.model?.displayTitle || item.model?.label || item.name,
        stageLabel: item.model?.stageLabel || '在线模型',
        accent: item.model?.accent || '#0f766e',
        curve,
        firstAlertTime: firstAlertTime(curve),
        peakFaultProb: probs.length ? Math.max(...probs) : NaN,
        meanFaultProb: mean(probs),
        stdFaultProb: standardDeviation(probs),
        pointCount: curve.length,
        alertSegments: countAlertSegments(curve),
        registryRealMacroF1: parseMetric(registryMetrics.macro_f1),
        registryBalancedAccuracy: parseMetric(registryMetrics.balanced_accuracy)
      }
    })
    .sort((a, b) => {
      const aAlert = Number.isNaN(a.firstAlertTime) ? Number.POSITIVE_INFINITY : a.firstAlertTime
      const bAlert = Number.isNaN(b.firstAlertTime) ? Number.POSITIVE_INFINITY : b.firstAlertTime
      if (aAlert !== bAlert) return aAlert - bAlert

      const aPeak = Number.isNaN(a.peakFaultProb) ? -1 : a.peakFaultProb
      const bPeak = Number.isNaN(b.peakFaultProb) ? -1 : b.peakFaultProb
      if (aPeak !== bPeak) return bPeak - aPeak

      const aStd = Number.isNaN(a.stdFaultProb) ? Number.POSITIVE_INFINITY : a.stdFaultProb
      const bStd = Number.isNaN(b.stdFaultProb) ? Number.POSITIVE_INFINITY : b.stdFaultProb
      return aStd - bStd
    })
})

const realtimeChampion = computed(() => realtimeEntries.value[0] || null)
const realtimeSelectedAsset = computed(() => {
  return realtimeAssets.value.find((item) => item.id === selectedRealtimeAssetId.value) || null
})
const realtimeSelectedAssetLabel = computed(() => normalizeAssetLabel(realtimeSelectedAsset.value))
const realtimeSelectedAssetHint = computed(() => {
  if (!realtimeSelectedAsset.value) return '请选择一个已上传的 CSV 资产作为同段遥测输入'
  return realtimeSelectedAsset.value.kind === 'csv'
    ? '该资产作为同一段实时遥测回放输入'
    : '当前资产可用于回放分析'
})

const realtimePointCount = computed(() => {
  return realtimeEntries.value.reduce((max, item) => Math.max(max, item.pointCount || 0), 0)
})

const realtimeChart = computed(() => {
  const times = Array.from(
    new Set(
      realtimeCurves.value.flatMap((item) => (Array.isArray(item.curve) ? item.curve : []).map((point) => Number(point.time_s)))
    )
  )
    .filter((value) => !Number.isNaN(value))
    .sort((a, b) => a - b)

  const timeLabels = times.map((value) => value.toFixed(2))
  const series = realtimeCurves.value.map((item) => {
    const curveMap = new Map(
      (Array.isArray(item.curve) ? item.curve : []).map((point) => [Number(point.time_s).toFixed(2), Number(point.fault_prob)])
    )
    return {
      name: item.model?.displayTitle || item.model?.label || item.name,
      data: timeLabels.map((label) => {
        const value = curveMap.get(label)
        return value == null || Number.isNaN(value) ? null : value
      })
    }
  })

  return {
    xData: timeLabels,
    series
  }
})

async function loadOfflineModels() {
  offlineLoading.value = true
  offlineError.value = ''
  try {
    const response = await modelApi.getModels({ task: 'offline_multiclass', capability: 'offline_detection' })
    offlineRawModels.value = response.data || []
  } catch (error) {
    offlineError.value = `加载模型对比数据失败：${error.message || '未知错误'}`
    ElMessage.error(offlineError.value)
  } finally {
    offlineLoading.value = false
  }
}

async function loadRealtimeMeta() {
  realtimeLoading.value = true
  realtimeError.value = ''
  isBootstrappingRealtime.value = true
  try {
    const [modelsResponse, assetsResponse] = await Promise.all([
      modelApi.getModels({ task: 'online_binary', capability: 'realtime_detection' }),
      assetApi.list('csv')
    ])
    realtimeRawModels.value = modelsResponse.data || []
    realtimeAssets.value = (assetsResponse.data || []).filter((item) => item.kind === 'csv')

    if (!selectedRealtimeAssetId.value && realtimeAssets.value.length) {
      selectedRealtimeAssetId.value = realtimeAssets.value[0].id
    }
  } catch (error) {
    realtimeError.value = `加载实时对比元数据失败：${error.message || '未知错误'}`
    ElMessage.error(realtimeError.value)
  } finally {
    realtimeLoading.value = false
    isBootstrappingRealtime.value = false
  }
}

async function loadRealtimeComparison() {
  if (mode.value !== 'realtime') return
  if (!selectedRealtimeAssetId.value) {
    realtimeCurves.value = []
    return
  }
  if (!realtimeModels.value.length) {
    realtimeCurves.value = []
    return
  }

  realtimeLoading.value = true
  realtimeError.value = ''
  try {
    const curves = await Promise.all(
      realtimeModels.value.map(async (model) => {
        const response = await analysisApi.faultProbCurve(selectedRealtimeAssetId.value, model.name)
        return {
          name: model.name,
          model,
          curve: response.data || []
        }
      })
    )
    realtimeCurves.value = curves.filter((item) => Array.isArray(item.curve) && item.curve.length > 0)
  } catch (error) {
    realtimeError.value = `加载实时对比曲线失败：${error.message || '未知错误'}`
    ElMessage.error(realtimeError.value)
  } finally {
    realtimeLoading.value = false
  }
}

async function reloadRealtimeComparison() {
  await loadRealtimeComparison()
}

watch(mode, async (value) => {
  if (value === 'realtime') {
    if (!realtimeRawModels.value.length || !realtimeAssets.value.length) {
      await loadRealtimeMeta()
    }
    await loadRealtimeComparison()
  }
})

watch(selectedRealtimeAssetId, async () => {
  if (mode.value === 'realtime' && !isBootstrappingRealtime.value) {
    await loadRealtimeComparison()
  }
})

watch(
  () => realtimeModels.value.map((item) => item.name).join('|'),
  async () => {
    if (mode.value === 'realtime' && !isBootstrappingRealtime.value) {
      await loadRealtimeComparison()
    }
  }
)

onMounted(async () => {
  await loadOfflineModels()
  await loadRealtimeMeta()
})
</script>

<style scoped>
.compare-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(420px, 0.9fr);
  gap: 22px;
  padding: 24px;
  background: linear-gradient(135deg, rgba(16, 50, 64, 0.96) 0%, rgba(15, 118, 110, 0.9) 100%);
  color: #f4fbfb;
}

.compare-hero--realtime {
  background: linear-gradient(135deg, rgba(24, 32, 56, 0.98) 0%, rgba(15, 118, 110, 0.9) 100%);
}

.compare-hero__main {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.compare-hero__title-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
}

.compare-hero__title {
  margin: 0;
  font-size: 38px;
  line-height: 1.05;
}

.compare-hero__desc {
  margin: 14px 0 0;
  max-width: 760px;
  color: rgba(244, 251, 251, 0.84);
  line-height: 1.8;
}

.compare-mode-switch {
  display: inline-flex;
  gap: 8px;
  padding: 6px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.14);
}

.compare-mode-switch__btn {
  border: 0;
  border-radius: 999px;
  padding: 11px 16px;
  background: transparent;
  color: rgba(244, 251, 251, 0.88);
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.compare-mode-switch__btn:hover {
  transform: translateY(-1px);
}

.compare-mode-switch__btn.is-active {
  background: rgba(255, 255, 255, 0.95);
  color: #103240;
}

.compare-hero__summary-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 12px;
}

.compare-hero__summary-row--realtime {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.compare-hero__winner {
  align-self: stretch;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 10px;
  padding: 20px;
  border-radius: 22px;
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.16);
}

.compare-hero__winner--realtime {
  min-height: 160px;
}

.compare-hero__winner-label {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: rgba(244, 251, 251, 0.72);
}

.compare-hero__winner strong {
  font-size: 24px;
}

.compare-hero__winner small {
  color: rgba(244, 251, 251, 0.8);
}

.compare-hero__winner-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 8px;
}

.compare-hero__winner-metrics span {
  padding: 10px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.1);
  font-size: 12px;
}

.compare-hero__stats {
  display: grid;
  gap: 12px;
}

.compare-card-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}

.compare-card {
  --accent: var(--brand);
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding: 18px;
  border-radius: 22px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 251, 252, 0.98) 100%);
  border: 1px solid rgba(16, 50, 64, 0.08);
  position: relative;
  overflow: hidden;
}

.compare-card::before {
  content: '';
  position: absolute;
  inset: 0 auto 0 0;
  width: 6px;
  background: var(--accent);
}

.compare-card.is-champion {
  box-shadow: 0 20px 40px rgba(16, 50, 64, 0.14);
  border-color: rgba(16, 50, 64, 0.16);
}

.compare-card__header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

.compare-card__tag {
  display: inline-flex;
  align-items: center;
  padding: 7px 10px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--accent) 14%, white);
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
}

.compare-card__title {
  margin: 12px 0 0;
  font-size: 22px;
  line-height: 1.15;
}

.compare-card__subtitle {
  margin: 8px 0 0;
  color: var(--text-secondary);
  line-height: 1.7;
}

.compare-card__rank {
  min-width: 46px;
  height: 46px;
  display: grid;
  place-items: center;
  border-radius: 16px;
  background: linear-gradient(135deg, var(--accent) 0%, color-mix(in srgb, var(--accent) 74%, white) 100%);
  color: white;
  font-size: 18px;
  font-weight: 800;
}

.compare-card__stats {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.compare-stat {
  padding: 12px 14px;
  border-radius: 16px;
  background: #f4f7f8;
  border: 1px solid rgba(16, 50, 64, 0.06);
}

.compare-stat span {
  display: block;
  font-size: 12px;
  color: var(--text-tertiary);
}

.compare-stat strong {
  display: block;
  margin-top: 8px;
  font-size: 20px;
}

.compare-card__bars {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.compare-bar {
  display: grid;
  grid-template-columns: 56px minmax(0, 1fr) 56px;
  gap: 10px;
  align-items: center;
}

.compare-bar__label,
.compare-bar__value {
  font-size: 12px;
  font-weight: 700;
  color: var(--text-secondary);
}

.compare-bar__track {
  height: 10px;
  border-radius: 999px;
  background: rgba(16, 50, 64, 0.08);
  overflow: hidden;
}

.compare-bar__fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--accent) 0%, color-mix(in srgb, var(--accent) 55%, white) 100%);
}

.compare-bar__fill--soft {
  opacity: 0.6;
}

.compare-card__footer {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding-top: 8px;
  border-top: 1px dashed rgba(16, 50, 64, 0.14);
  font-size: 12px;
  color: var(--text-tertiary);
}

.compare-empty {
  padding: 30px 20px;
  text-align: center;
  color: var(--text-secondary);
}

.compare-empty--error {
  color: var(--danger);
}

.compare-table-wrap {
  overflow-x: auto;
}

.compare-table {
  width: 100%;
  border-collapse: collapse;
  min-width: 920px;
}

.compare-table--realtime {
  min-width: 1100px;
  margin-top: 18px;
}

.compare-table th,
.compare-table td {
  padding: 14px 12px;
  border-bottom: 1px solid var(--line-soft);
  text-align: left;
}

.compare-table th {
  font-size: 12px;
  letter-spacing: 0.08em;
  color: var(--text-tertiary);
  text-transform: uppercase;
}

.compare-table__sub {
  margin-top: 4px;
  font-size: 12px;
  color: var(--text-tertiary);
}

.compare-realtime__header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-end;
}

.compare-realtime__controls {
  display: flex;
  align-items: flex-end;
  gap: 12px;
  flex-wrap: wrap;
}

.compare-select {
  display: inline-flex;
  flex-direction: column;
  gap: 6px;
  color: var(--text-secondary);
  font-size: 12px;
}

.compare-select select {
  min-width: 280px;
  height: 40px;
  padding: 0 12px;
  border-radius: 12px;
  border: 1px solid rgba(16, 50, 64, 0.14);
  background: #fff;
  color: var(--text-primary);
}

.compare-refresh-btn {
  height: 40px;
  padding: 0 16px;
  border: 0;
  border-radius: 12px;
  background: #103240;
  color: #fff;
  font-weight: 700;
  cursor: pointer;
}

.compare-refresh-btn:disabled {
  opacity: 0.65;
  cursor: not-allowed;
}

.compare-realtime-layout {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

@media (max-width: 1600px) {
  .compare-card-grid {
    grid-template-columns: 1fr;
  }

  .compare-hero {
    grid-template-columns: 1fr;
  }

  .compare-hero__summary-row--realtime {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 1180px) {
  .compare-hero__title-row,
  .compare-realtime__header {
    flex-direction: column;
    align-items: stretch;
  }

  .compare-mode-switch {
    align-self: flex-start;
  }

  .compare-select select {
    min-width: 240px;
  }
}
</style>
