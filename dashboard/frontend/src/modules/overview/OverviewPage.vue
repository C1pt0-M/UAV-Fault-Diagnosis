<template>
  <div class="page-stack">
    <section class="overview-hero page-panel">
      <div>
        <span class="overview-hero__tag">系统总览</span>
        <h2>面向运维与演示的统一故障监测闭环</h2>
        <p>
          从实时遥测、离线诊断、日志资产到报告分析与 AI 问答，
          当前工作台将关键流程收口为更稳定的五大模块。
        </p>
      </div>
      <div class="overview-hero__focus">
        <span>当前主能力</span>
        <strong>三分类离线诊断 + 在线监测</strong>
        <small>围绕任务、会话和报告建立统一分析入口</small>
      </div>
    </section>

    <section class="metric-grid">
      <article class="metric-card">
        <span class="metric-card__label">日志资产</span>
        <div class="metric-card__value">{{ overview.assets_total || 0 }}</div>
        <div class="metric-card__meta">当前累计可用的 ULG / CSV 资产数</div>
      </article>
      <article class="metric-card">
        <span class="metric-card__label">诊断任务</span>
        <div class="metric-card__value">{{ overview.runs_total || 0 }}</div>
        <div class="metric-card__meta">离线诊断与预处理任务总数</div>
      </article>
      <article class="metric-card">
        <span class="metric-card__label">异常占比</span>
        <div class="metric-card__value">{{ abnormalRate }}%</div>
        <div class="metric-card__meta">基于现有诊断结果计算的异常比例</div>
      </article>
      <article class="metric-card">
        <span class="metric-card__label">电机故障</span>
        <div class="metric-card__value">{{ faultDistribution.motor || 0 }}</div>
        <div class="metric-card__meta">当前累计识别出的电机故障次数</div>
      </article>
    </section>

    <section class="two-column">
      <article class="section-card">
        <div class="section-card__header">
          <div>
            <div class="section-card__title">最近任务</div>
            <div class="section-card__desc">快速查看最近提交的诊断与预处理任务。</div>
          </div>
        </div>
        <div class="section-card__body">
          <el-table :data="recentRuns" stripe empty-text="暂无任务记录">
            <el-table-column prop="id" label="任务 ID" width="110" />
            <el-table-column prop="task" label="任务类型" />
            <el-table-column prop="status" label="状态" width="120" />
            <el-table-column prop="created_at" label="创建时间" width="200" />
          </el-table>
        </div>
      </article>

      <article class="section-card">
        <div class="section-card__header">
          <div>
            <div class="section-card__title">故障分布</div>
            <div class="section-card__desc">当前离线诊断结果的标签分布。</div>
          </div>
        </div>
        <div class="section-card__body">
          <div class="distribution-list">
            <div class="distribution-item">
              <span>正常</span>
              <strong>{{ faultDistribution.normal || 0 }}</strong>
            </div>
            <div class="distribution-item">
              <span>电机故障</span>
              <strong>{{ faultDistribution.motor || 0 }}</strong>
            </div>
            <div class="distribution-item">
              <span>传感器故障</span>
              <strong>{{ faultDistribution.sensor || 0 }}</strong>
            </div>
          </div>
        </div>
      </article>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElTable, ElTableColumn } from 'element-plus/es/components/table/index.mjs'
import { ElMessage } from 'element-plus'
import { overviewApi } from '@/utils/api'

const overview = ref({})
const recentRuns = ref([])

const faultDistribution = computed(() => overview.value?.fault_distribution || { normal: 0, motor: 0, sensor: 0 })
const abnormalRate = computed(() => {
  const total = Object.values(faultDistribution.value).reduce((sum, value) => sum + Number(value || 0), 0)
  const abnormal = Number(faultDistribution.value.motor || 0) + Number(faultDistribution.value.sensor || 0)
  if (!total) return '0.0'
  return ((abnormal / total) * 100).toFixed(1)
})

async function loadOverview() {
  try {
    const response = await overviewApi.getOverview()
    overview.value = response.data || {}
    recentRuns.value = response.data?.recent_runs || []
  } catch (error) {
    ElMessage.error(`加载总览失败：${error.message || '未知错误'}`)
  }
}

onMounted(loadOverview)
</script>

<style scoped>
.overview-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.8fr);
  gap: 20px;
  padding: 28px;
  background: linear-gradient(135deg, #103240 0%, #0f766e 100%);
  color: #f4fbfb;
}

.overview-hero__tag {
  display: inline-flex;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.14);
  font-size: 12px;
  font-weight: 700;
}

.overview-hero h2 {
  margin: 14px 0 0;
  font-size: 34px;
}

.overview-hero p {
  margin: 12px 0 0;
  max-width: 640px;
  line-height: 1.8;
  color: rgba(244, 251, 251, 0.84);
}

.overview-hero__focus {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 20px;
  border-radius: 22px;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.12);
}

.overview-hero__focus span,
.overview-hero__focus small {
  color: rgba(244, 251, 251, 0.78);
}

.overview-hero__focus strong {
  font-size: 28px;
  line-height: 1.3;
}

.distribution-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.distribution-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  border-radius: 16px;
  background: #f5faf9;
  border: 1px solid var(--line-soft);
}

.distribution-item strong {
  font-size: 20px;
}
</style>
