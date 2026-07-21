<template>
  <section class="section-card profile-comparison" data-test="5g-session-comparison">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">三档实时会话对比</div>
        <div class="section-card__desc">各profile最新完成的RflySim实时会话</div>
      </div>
      <el-button size="small" plain @click="$emit('refresh')">
        <el-icon><Refresh /></el-icon>
        刷新
      </el-button>
    </div>

    <div class="section-card__body profile-comparison__body">
      <div class="profile-grid">
        <article
          v-for="item in normalizedProfiles"
          :key="item.profile"
          class="profile-result"
          :class="`profile-result--${item.profile}`"
        >
          <div class="profile-result__header">
            <div>
              <span>{{ profileLabels[item.profile] }}</span>
              <strong>{{ item.profile }}</strong>
            </div>
            <span class="status-badge" :class="item.available ? 'status-badge--ok' : 'status-badge--offline'">
              {{ item.available ? '已有实测' : '待采集' }}
            </span>
          </div>

          <template v-if="item.available">
            <code>{{ item.session_id }}</code>
            <dl>
              <div><dt>平均时延</dt><dd>{{ formatMs(item.avg_latency_ms) }}</dd></div>
              <div><dt>P95时延</dt><dd>{{ formatMs(item.p95_latency_ms) }}</dd></div>
              <div><dt>丢包率</dt><dd>{{ formatRate(item.loss_rate) }}</dd></div>
              <div><dt>完整率</dt><dd>{{ formatRate(item.data_integrity) }}</dd></div>
              <div><dt>遥测帧</dt><dd>{{ formatInteger(item.frame_count) }}</dd></div>
              <div><dt>采集时间</dt><dd>{{ item.start_time || '-' }}</dd></div>
            </dl>
          </template>
          <div v-else class="profile-result__empty">暂无已完成实时会话</div>
        </article>
      </div>

      <div class="comparison-impact" :class="{ 'comparison-impact--pending': !hasImpact }">
        <strong>5G增强链路相对弱网</strong>
        <template v-if="hasImpact">
          <span>平均时延降低 {{ formatMs(impact.latency_reduction_ms) }}</span>
          <span>P95降低 {{ formatMs(impact.p95_reduction_ms) }}</span>
          <span>丢包率降低 {{ formatPoints(impact.loss_reduction) }}</span>
          <span>完整率提升 {{ formatPoints(impact.integrity_improvement) }}</span>
        </template>
        <span v-else>完成 enhanced_5g 与 degraded 实时会话后生成改善量</span>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { Refresh } from '@element-plus/icons-vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElIcon from 'element-plus/es/components/icon/index.mjs'

const props = defineProps({ comparison: { type: Object, default: () => ({}) } })
defineEmits(['refresh'])

const profileOrder = ['baseline', 'enhanced_5g', 'degraded']
const profileLabels = {
  baseline: '理想基线',
  enhanced_5g: '5G增强链路',
  degraded: '弱网场景'
}
const normalizedProfiles = computed(() => {
  const source = Array.isArray(props.comparison.profiles) ? props.comparison.profiles : []
  return profileOrder.map((profile) => (
    source.find((item) => item.profile === profile) || { profile, available: false }
  ))
})
const impact = computed(() => props.comparison.enhanced_vs_degraded || {})
const hasImpact = computed(() => (
  impact.value.latency_reduction_ms !== null &&
  impact.value.latency_reduction_ms !== undefined &&
  Number.isFinite(Number(impact.value.latency_reduction_ms))
))

function formatMs(value) {
  const parsed = Number(value)
  return value !== null && value !== undefined && Number.isFinite(parsed)
    ? `${parsed.toFixed(parsed >= 100 ? 0 : 1)} ms`
    : '-'
}
function formatRate(value) {
  const parsed = Number(value)
  return value !== null && value !== undefined && Number.isFinite(parsed)
    ? `${(parsed * 100).toFixed(2)}%`
    : '-'
}
function formatPoints(value) {
  const parsed = Number(value)
  return value !== null && value !== undefined && Number.isFinite(parsed)
    ? `${(parsed * 100).toFixed(2)} pct`
    : '-'
}
function formatInteger(value) {
  const parsed = Number(value)
  return value !== null && value !== undefined && Number.isFinite(parsed)
    ? Math.round(parsed).toLocaleString('zh-CN')
    : '-'
}
</script>

<style scoped>
.profile-comparison { overflow: hidden; }
.profile-comparison__body { padding: var(--space-lg); }
.profile-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--space-md); }
.profile-result { min-width: 0; padding: 14px; border: 1px solid var(--line-soft); border-top: 3px solid var(--status-offline); border-radius: var(--radius-md); background: var(--bg-panel-raised); }
.profile-result--baseline { border-top-color: var(--text-tertiary); }
.profile-result--enhanced_5g { border-top-color: var(--brand); }
.profile-result--degraded { border-top-color: var(--status-warn); }
.profile-result__header { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
.profile-result__header span:first-child { display: block; color: var(--text-tertiary); font-size: 11px; }
.profile-result__header strong { display: block; margin-top: 3px; font-size: 15px; color: var(--text-primary); }
.profile-result code { display: block; margin: 12px 0; color: var(--brand-strong); font-size: 12px; }
.profile-result dl { margin: 0; }
.profile-result dl div { display: grid; grid-template-columns: 88px minmax(0, 1fr); gap: 10px; padding: 7px 0; border-bottom: 1px solid var(--line-soft); font-size: 12px; }
.profile-result dl div:last-child { border-bottom: 0; }
.profile-result dt { color: var(--text-tertiary); }
.profile-result dd { min-width: 0; margin: 0; color: var(--text-primary); font-weight: 700; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.profile-result__empty { min-height: 178px; display: grid; place-items: center; color: var(--text-tertiary); font-size: 12px; }
.comparison-impact { display: flex; align-items: center; flex-wrap: wrap; gap: 10px 20px; margin-top: var(--space-md); padding: 12px 14px; border-left: 3px solid var(--brand); background: var(--brand-soft); color: var(--text-secondary); font-size: 12px; }
.comparison-impact strong { color: var(--brand-strong); }
.comparison-impact--pending { border-left-color: var(--status-pending); background: var(--bg-panel-raised); }
@media (max-width: 1120px) { .profile-grid { grid-template-columns: 1fr; } }
</style>
