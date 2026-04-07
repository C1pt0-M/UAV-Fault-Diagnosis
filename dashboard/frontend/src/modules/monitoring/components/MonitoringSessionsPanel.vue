<template>
  <section class="section-card monitoring-sessions">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ monitoringCopy.sessionsTitle }}</div>
        <div class="section-card__desc">{{ monitoringCopy.sessionsDescription }}</div>
      </div>
      <el-button circle @click="$emit('refresh')">
        <el-icon><Refresh /></el-icon>
      </el-button>
    </div>
    <div class="section-card__body">
      <div v-if="sessions.length" class="session-list">
        <article v-for="row in sessions" :key="row.id" class="session-card">
          <div class="session-card__head">
            <div>
              <strong>#{{ row.id }}</strong>
              <span>{{ row.drone_id || '-' }}</span>
            </div>
            <div class="session-card__badge">{{ row.fault_events_count || 0 }} 个事件</div>
          </div>

          <p class="session-card__meta">{{ monitoringCopy.startTimeLabel }}：{{ row.start_time || '-' }}</p>
          <p class="session-card__meta">{{ monitoringCopy.durationLabel }}：{{ Number(row.duration_seconds || 0).toFixed(1) }}</p>

          <div class="session-card__actions">
            <el-button link type="primary" @click="$emit('open-detail', row.id)">{{ monitoringCopy.detailAction }}</el-button>
            <el-button link type="warning" @click="$emit('continue-diagnosis', row.id)">{{ monitoringCopy.continueAction }}</el-button>
          </div>
        </article>
      </div>
      <el-empty v-else :description="monitoringCopy.sessionsEmpty" :image-size="60" />
    </div>
  </section>
</template>

<script setup>
import { Refresh } from '@element-plus/icons-vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElIcon from 'element-plus/es/components/icon/index.mjs'
import ElEmpty from 'element-plus/es/components/empty/index.mjs'
import { monitoringCopy } from '@/modules/monitoring/monitoringView'

defineProps({
  sessions: { type: Array, default: () => [] }
})

defineEmits(['refresh', 'open-detail', 'continue-diagnosis'])
</script>

<style scoped>
.session-list {
  display: grid;
  gap: 12px;
}

.session-card {
  padding: 16px;
  border-radius: 18px;
  border: 1px solid var(--line-soft);
  background: linear-gradient(180deg, #fbfdfd 0%, #ffffff 100%);
}

.session-card__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.session-card__head strong {
  display: block;
  font-size: 18px;
}

.session-card__head span {
  color: var(--text-tertiary);
  font-size: 12px;
}

.session-card__badge {
  display: inline-flex;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(15, 118, 110, 0.1);
  color: var(--brand-strong);
  font-size: 12px;
  font-weight: 700;
}

.session-card__meta {
  margin: 8px 0 0;
  color: var(--text-secondary);
  line-height: 1.7;
}

.session-card__actions {
  display: flex;
  gap: 12px;
  margin-top: 10px;
}
</style>
