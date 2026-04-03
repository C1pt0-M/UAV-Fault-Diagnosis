<template>
  <div class="page-stack model-compare-page">
    <section class="page-panel compare-hero">
      <div>
        <h2 class="compare-hero__title">模型对比</h2>
        <p class="compare-hero__desc">
          对比当前重点离线模型在测试集与真实域上的关键指标，重点观察 `Real Macro F1`、
          `Balanced Accuracy` 与结构升级带来的增益。
        </p>
      </div>
      <div v-if="champion" class="compare-hero__winner">
        <div class="compare-hero__winner-label">当前最优</div>
        <strong>{{ champion.displayTitle }}</strong>
        <small>{{ champion.stageLabel }}</small>
        <div class="compare-hero__winner-metrics">
          <span>Real Macro F1 {{ formatScore(champion.realMacroF1) }}</span>
          <span>Balanced Acc {{ formatScore(champion.realBalancedAccuracy) }}</span>
        </div>
      </div>
    </section>

    <section class="metric-grid">
      <article class="metric-card">
        <div class="metric-card__label">当前最优模型</div>
        <div class="metric-card__value">{{ champion ? champion.shortName : '--' }}</div>
        <div class="metric-card__meta">{{ champion ? champion.stageLabel : '等待模型数据' }}</div>
      </article>
      <article class="metric-card">
        <div class="metric-card__label">最佳 Real Macro F1</div>
        <div class="metric-card__value">{{ champion ? formatScore(champion.realMacroF1) : '--' }}</div>
        <div class="metric-card__meta">真实域总体诊断能力</div>
      </article>
      <article class="metric-card">
        <div class="metric-card__label">相对基线提升</div>
        <div class="metric-card__value">{{ baselineDelta }}</div>
        <div class="metric-card__meta">相对 `BiLSTM + FocalLoss` 的 Real Macro F1 增益</div>
      </article>
      <article class="metric-card">
        <div class="metric-card__label">参与对比模型</div>
        <div class="metric-card__value">{{ comparisonModels.length }}</div>
        <div class="metric-card__meta">基础模型 / 结构升级 / 最终方案</div>
      </article>
    </section>

    <section class="section-card">
      <div class="section-card__header">
        <div>
          <div class="section-card__title">模型对比卡片</div>
          <div class="section-card__desc">每张卡片展示一个模型的测试域与真实域表现，并给出相对基线的差值。</div>
        </div>
      </div>
      <div class="section-card__body compare-card-grid">
        <article
          v-for="model in comparisonModels"
          :key="model.name"
          class="compare-card"
          :class="{ 'is-champion': champion?.name === model.name }"
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

    <section class="section-card">
      <div class="section-card__header">
        <div>
          <div class="section-card__title">指标总表</div>
          <div class="section-card__desc">原始模型指标与相对基线差值，适合直接截图展示。</div>
        </div>
      </div>
      <div class="section-card__body">
        <div v-if="loading" class="compare-empty">正在加载模型信息…</div>
        <div v-else-if="errorMessage" class="compare-empty compare-empty--error">{{ errorMessage }}</div>
        <div v-else-if="!comparisonModels.length" class="compare-empty">未发现可对比的离线模型。</div>
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
              <tr v-for="model in comparisonModels" :key="`${model.name}-row`">
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
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { modelApi } from '@/shared/services'

const loading = ref(false)
const errorMessage = ref('')
const rawModels = ref([])

const spotlight = {
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

function percent(value) {
  const number = Number(value)
  if (Number.isNaN(number)) return 0
  return Math.max(0, Math.min(number * 100, 100))
}

function parseMetric(value) {
  const number = Number(value)
  return Number.isNaN(number) ? NaN : number
}

const baselineModel = computed(() => {
  return rawModels.value.find((item) => item.name === 'transfer3_bilstm_fewshot') || null
})

const comparisonModels = computed(() => {
  const baseline = baselineModel.value
  const baselineRealMacroF1 = parseMetric(baseline?.metrics_external?.macro_f1)
  const entries = rawModels.value
    .filter((item) => item.task === 'offline_multiclass')
    .filter((item) => Object.prototype.hasOwnProperty.call(spotlight, item.name))
    .map((item) => {
      const meta = spotlight[item.name]
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
      return (
        bReal - aReal ||
        bBalanced - aBalanced ||
        bAccuracy - aAccuracy
      )
    })

  return entries.map((item, index) => ({
    ...item,
    rank: index + 1
  }))
})

const champion = computed(() => comparisonModels.value[0] || null)
const baselineDelta = computed(() => {
  if (!champion.value || !baselineModel.value) return '--'
  const baselineMetric = parseMetric(baselineModel.value.metrics_external?.macro_f1)
  if (Number.isNaN(champion.value.realMacroF1) || Number.isNaN(baselineMetric)) return '--'
  return formatDelta(champion.value.realMacroF1 - baselineMetric)
})

async function loadModels() {
  loading.value = true
  errorMessage.value = ''
  try {
    const response = await modelApi.getModels({ task: 'offline_multiclass', capability: 'offline_detection' })
    rawModels.value = response.data || []
  } catch (error) {
    errorMessage.value = `加载模型对比数据失败：${error.message || '未知错误'}`
    ElMessage.error(errorMessage.value)
  } finally {
    loading.value = false
  }
}

onMounted(loadModels)
</script>

<style scoped>
.compare-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.8fr);
  gap: 22px;
  padding: 24px;
  background: linear-gradient(135deg, rgba(16, 50, 64, 0.96) 0%, rgba(15, 118, 110, 0.9) 100%);
  color: #f4fbfb;
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

@media (max-width: 1600px) {
  .compare-card-grid {
    grid-template-columns: 1fr;
  }

  .compare-hero {
    grid-template-columns: 1fr;
  }
}
</style>
