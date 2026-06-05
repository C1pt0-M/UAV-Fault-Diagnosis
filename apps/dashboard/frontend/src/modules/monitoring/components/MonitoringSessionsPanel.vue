<template>
  <section class="section-card monitoring-sessions">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ monitoringCopy.sessionsTitle }}</div>
        <div class="section-card__desc">{{ monitoringCopy.sessionsDescription }}</div>
      </div>
      <el-button size="small" plain @click="$emit('refresh')">
        <el-icon><Refresh /></el-icon>
        {{ monitoringCopy.refreshAction }}
      </el-button>
    </div>
    <div class="section-card__body monitoring-sessions__body">
      <div v-if="sessions.length" class="session-table">
        <div class="session-table__head">
          <span>会话 ID</span>
          <span>资产 / 机型</span>
          <span>开始时间</span>
          <span>持续时间</span>
          <span>事件数</span>
          <span>状态</span>
          <span>操作</span>
        </div>
        <div
          v-for="row in pagedSessions"
          :key="row.id"
          class="session-table__row"
        >
          <strong class="session-table__mono">#{{ row.id }}</strong>
          <span>{{ row.drone_id || '-' }}</span>
          <span>{{ row.start_time || '-' }}</span>
          <span>{{ formatDuration(row.duration_seconds) }}</span>
          <span>{{ row.fault_events_count || 0 }}</span>
          <span class="status-badge" :class="sessionStatusClass(row.status)">
            {{ sessionStatusLabel(row.status) }}
          </span>
          <span class="session-table__actions">
            <el-button link type="primary" @click="$emit('open-detail', row.id)">
              {{ monitoringCopy.detailAction }}
            </el-button>
            <el-button link type="warning" @click="$emit('continue-diagnosis', row.id)">
              {{ monitoringCopy.continueAction }}
            </el-button>
          </span>
        </div>
        <div v-if="totalSessions > pageSize" class="table-pagination">
          <el-pagination
            background
            layout="prev, pager, next"
            :current-page="currentPage"
            :page-size="pageSize"
            :total="totalSessions"
            @current-change="currentPage = $event"
          />
        </div>
      </div>
      <el-empty v-else :description="monitoringCopy.sessionsEmpty" :image-size="60" />
    </div>
  </section>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { Refresh } from '@element-plus/icons-vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElIcon from 'element-plus/es/components/icon/index.mjs'
import ElEmpty from 'element-plus/es/components/empty/index.mjs'
import ElPagination from 'element-plus/es/components/pagination/index.mjs'
import { monitoringCopy } from '@/modules/monitoring/monitoringView'

const props = defineProps({
  sessions: { type: Array, default: () => [] }
})

defineEmits(['refresh', 'open-detail', 'continue-diagnosis'])

const pageSize = 5
const currentPage = ref(1)
const totalSessions = computed(() => props.sessions.length)
const pagedSessions = computed(() => {
  const start = (currentPage.value - 1) * pageSize
  return props.sessions.slice(start, start + pageSize)
})

watch(totalSessions, () => {
  const maxPage = Math.max(1, Math.ceil(totalSessions.value / pageSize))
  if (currentPage.value > maxPage) currentPage.value = maxPage
})

function formatDuration(seconds) {
  const parsed = Number(seconds || 0)
  if (!Number.isFinite(parsed) || parsed <= 0) return '-'
  if (parsed < 60) return `${parsed.toFixed(0)} s`
  const minutes = Math.floor(parsed / 60)
  const remain = Math.floor(parsed % 60)
  return `${minutes}m ${String(remain).padStart(2, '0')}s`
}

function sessionStatusLabel(status) {
  const value = String(status || '').toLowerCase()
  if (['running', 'active', 'collecting', 'detecting'].includes(value)) return '运行中'
  if (['completed', 'done', 'finished'].includes(value)) return '已完成'
  if (['failed', 'error'].includes(value)) return '异常'
  return status || '-'
}

function sessionStatusClass(status) {
  const value = String(status || '').toLowerCase()
  if (['running', 'active', 'collecting', 'detecting'].includes(value)) return 'status-badge--active'
  if (['completed', 'done', 'finished'].includes(value)) return 'status-badge--ok'
  if (['failed', 'error'].includes(value)) return 'status-badge--danger'
  return 'status-badge--offline'
}
</script>

<style scoped>
.monitoring-sessions {
  overflow: hidden;
}

.monitoring-sessions__body {
  padding: var(--space-lg);
}

.session-table {
  width: 100%;
  overflow-x: auto;
}

.session-table__head,
.session-table__row {
  display: grid;
  grid-template-columns: minmax(120px, 1.1fr) minmax(120px, 1fr) minmax(150px, 1.2fr) 92px 68px 78px minmax(150px, 1fr);
  gap: 12px;
  align-items: center;
  min-width: 900px;
}

.session-table__head {
  padding-bottom: 10px;
  color: var(--text-tertiary);
  font-size: 11px;
  font-weight: 800;
  border-bottom: 1px solid var(--line-soft);
}

.session-table__row {
  min-height: 42px;
  border-bottom: 1px solid var(--line-soft);
  color: var(--text-secondary);
  font-size: 12px;
}

.session-table__row > span,
.session-table__row > strong {
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.session-table__mono {
  font-family: var(--font-mono);
  color: var(--brand-strong);
}

.session-table__actions {
  display: flex;
  align-items: center;
  gap: 8px;
  overflow: visible !important;
}

.table-pagination {
  display: flex;
  justify-content: flex-end;
  padding-top: 12px;
}
</style>
