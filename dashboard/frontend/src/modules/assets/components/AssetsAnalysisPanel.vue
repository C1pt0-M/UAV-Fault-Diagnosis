<template>
  <section class='section-card'>
    <div class='section-card__header'>
      <div class='header-info'>
        <div class='header-title-row'>
          <el-icon class='header-icon'><Odometer /></el-icon>
          <div class='section-card__title'>{{ assetsCopy.analysisTitle }}</div>
        </div>
        <div class='section-card__desc'>{{ assetsCopy.analysisDescription }}</div>
      </div>
    </div>

    <div class='section-card__body'>
      <div v-if='annotationItems.length' class='analysis-annotations'>
        <div class='analysis-annotations__label'>
          <el-icon><Warning /></el-icon>
          <span>检测到时段内故障事件</span>
        </div>
        <div class='analysis-annotations__list'>
          <div v-for='item in annotationItems' :key='item.field' class='annotation-card'>
            <div class='annotation-card__main'>
              <strong>{{ item.displayName }}</strong>
              <small v-if='item.sample_values?.length'>{{ item.sample_values.join(" / ") }}</small>
            </div>
            <div class='annotation-card__state'>
              <el-tag size='small' type='danger' effect='dark' round>! 已注入</el-tag>
            </div>
          </div>
        </div>
      </div>

      <div class='analysis-grid-controls'>
        <div class='control-field'>
          <label class='field-label'>选择日志</label>
          <el-select
            v-model='assetIdModel'
            filterable
            class='control-select'
            :placeholder='assetsCopy.analysisAssetPlaceholder'
          >
            <el-option
              v-for='asset in assets'
              :key='asset.id'
              :label='"#" + asset.id + " " + asset.filename'
              :value='asset.id'
            />
          </el-select>
        </div>

        <div class='control-field'>
          <label class='field-label'>选择字段 (多选)</label>
          <el-select
            v-model='fieldsModel'
            multiple
            collapse-tags
            collapse-tags-tooltip
            filterable
            class='control-select'
            :placeholder='assetsCopy.analysisFieldsPlaceholder'
          >
            <el-option
              v-for='field in fieldOptions'
              :key='field.value'
              :label='field.label'
              :value='field.value'
            />
          </el-select>
        </div>

        <div class='control-field'>
          <label class='field-label'>最大点数</label>
          <el-input-number v-model='maxPointsModel' :min='200' :max='20000' :step='200' class='control-num' />
        </div>

        <div class='control-field'>
          <label class='field-label'>时间窗 (s)</label>
          <div class='range-inputs'>
            <el-input-number v-model='startTModel' :step='1' placeholder='start' class='small-num' />
            <span class='range-sep'>-</span>
            <el-input-number v-model='endTModel' :step='1' placeholder='end' class='small-num' />
          </div>
        </div>

        <div class='control-actions'>
          <el-button link class='refresh-btn' :loading='loadingFields' @click='$emit("reload-fields")'>
            <el-icon><Refresh /></el-icon>
            <span>刷新字段</span>
          </el-button>
          <el-button type='primary' :loading='loadingSeries' round class='load-btn' @click='$emit("load-series")'>
            <el-icon><Search /></el-icon>
            <span>{{ assetsCopy.analysisLoadSeries }}</span>
          </el-button>
        </div>
      </div>

      <p v-if='errorMessage' class='analysis-error'>{{ errorMessage }}</p>

      <div class='analysis-probability-bar' :class='{ "is-active": faultProbStats.available }'>
        <div class='prob-label'>故障预警概率</div>
        <div v-if='faultProbStats.available' class='prob-values'>
          <div class='prob-metric'>
            <span class='label'>当前</span>
            <span class='value'>{{ faultProbStats.current }}</span>
          </div>
          <div class='prob-metric'>
            <span class='label'>峰值</span>
            <span class='value' :class='{ "is-high": Number(faultProbStats.peak) > 0.6 }'>{{ faultProbStats.peak }}</span>
          </div>
          <div class='prob-metric'>
            <span class='label'>均值</span>
            <span class='value'>{{ faultProbStats.mean }}</span>
          </div>
        </div>
        <div v-else class='prob-empty'>
          <el-icon><InfoFilled /></el-icon>
          <span>该资产暂无可用概率曲线</span>
        </div>
      </div>

      <div class='chart-container'>
        <TimeSeriesChart
          :x-data='chartTimes'
          :series='chartSeries'
          :fault-prob-data='chartFaultProb'
          :height='420'
          :empty-text='assetsCopy.analysisNoSeries'
        />
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElIcon from 'element-plus/es/components/icon/index.mjs'
import ElInputNumber from 'element-plus/es/components/input-number/index.mjs'
import { ElOption, ElSelect } from 'element-plus/es/components/select/index.mjs'
import ElTag from 'element-plus/es/components/tag/index.mjs'
import { Odometer, Warning, Refresh, Search, InfoFilled } from '@element-plus/icons-vue'
import TimeSeriesChart from '@/shared/components/TimeSeriesChart.vue'
import { assetsCopy } from '@/modules/assets/assetsView'

const props = defineProps({
  assets: { type: Array, default: () => [] },
  fieldOptions: { type: Array, default: () => [] },
  chartTimes: { type: Array, default: () => [] },
  chartSeries: { type: Array, default: () => [] },
  chartFaultProb: { type: Array, default: () => [] },
  loadingFields: { type: Boolean, default: false },
  loadingSeries: { type: Boolean, default: false },
  errorMessage: { type: String, default: '' },
  annotationItems: { type: Array, default: () => [] }
})

const faultProbStats = computed(() => {
  const values = (Array.isArray(props.chartFaultProb) ? props.chartFaultProb : [])
    .map((item) => Number(item))
    .filter((item) => Number.isFinite(item))
  if (!values.length) {
    return { available: false, current: '--', peak: '--', mean: '--' }
  }
  const current = values[values.length - 1]
  const peak = values.reduce((a, b) => Math.max(a, b), -Infinity)
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  return {
    available: true,
    current: current.toFixed(3),
    peak: peak.toFixed(3),
    mean: mean.toFixed(3)
  }
})

defineEmits(['reload-fields', 'load-series'])

const assetIdModel = defineModel('analysisAssetId', { type: Number, default: null })
const fieldsModel = defineModel('analysisFields', { type: Array, default: () => [] })
const startTModel = defineModel('analysisStartT', { type: Number, default: undefined })
const endTModel = defineModel('analysisEndT', { type: Number, default: undefined })
const maxPointsModel = defineModel('analysisMaxPoints', { type: Number, default: 2000 })
</script>

<style scoped>
.header-info { display: flex; flex-direction: column; gap: 4px; }
.header-title-row { display: flex; align-items: center; gap: 12px; }
.header-icon { font-size: 20px; color: var(--primary); }

.analysis-annotations { margin-bottom: 24px; padding: 20px; border-radius: 20px; background: rgba(239, 68, 68, 0.03); border: 1px solid rgba(239, 68, 68, 0.1); }
.analysis-annotations__label { display: flex; align-items: center; gap: 8px; margin-bottom: 16px; font-size: 14px; font-weight: 700; color: var(--danger); }
.analysis-annotations__list { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
.annotation-card { position: relative; padding: 16px; background: white; border-radius: 16px; border: 1px solid rgba(0,0,0,0.05); display: flex; justify-content: space-between; align-items: flex-start; }
.annotation-card__main strong { display: block; font-size: 14px; margin-bottom: 4px; color: var(--text-primary); }
.annotation-card__main small { font-size: 12px; color: var(--text-tertiary); font-family: var(--font-mono); }

.analysis-grid-controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 24px; margin-bottom: 24px; padding: 24px; background: var(--bg-soft); border-radius: 20px; align-items: flex-end; }
.control-field { display: flex; flex-direction: column; gap: 8px; }
.field-label { font-size: 12px; font-weight: 700; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 0.1em; }
.range-inputs { display: flex; align-items: center; gap: 8px; }
.range-sep { color: var(--text-tertiary); }
.control-actions { display: flex; align-items: center; gap: 16px; }

.analysis-probability-bar { margin: 12px 0 24px; display: flex; align-items: center; justify-content: space-between; padding: 12px 24px; border-radius: 16px; background: #f8fafc; border: 1px solid rgba(0,0,0,0.04); transition: all 0.3s ease; }
.analysis-probability-bar.is-active { background: linear-gradient(90deg, #f0fdfa 0%, #faffff 100%); border-color: rgba(16, 185, 129, 0.2); }
.prob-label { font-size: 13px; font-weight: 700; color: var(--text-secondary); }
.prob-values { display: flex; gap: 32px; }
.prob-metric { display: flex; align-items: baseline; gap: 8px; }
.prob-metric .label { font-size: 12px; color: var(--text-tertiary); }
.prob-metric .value { font-size: 18px; font-weight: 700; font-family: var(--font-mono); color: var(--text-primary); }
.prob-metric .value.is-high { color: var(--danger); text-shadow: 0 0 12px rgba(239, 68, 68, 0.2); }
.prob-empty { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-tertiary); }

.chart-container { background: white; border-radius: 20px; border: 1px solid rgba(0,0,0,0.03); padding: 16px; }
</style>
