<template>
  <section class="two-column">
    <article class="section-card">
      <div class="section-card__header">
        <div>
          <div class="section-card__title">{{ monitoringCopy.alertsTitle }}</div>
          <div class="section-card__desc">{{ monitoringCopy.alertsDescription }}</div>
        </div>
      </div>
      <div class="section-card__body">
        <el-empty
          v-if="alerts.length === 0"
          :description="monitoringCopy.alertsEmpty"
          :image-size="60"
        />
        <div v-else class="alert-list">
          <el-alert
            v-for="item in alerts"
            :key="item.id || `${item.start_time}-${item.end_time}`"
            :title="item.diagnosis?.label || item.label || monitoringCopy.unknownFault"
            type="warning"
            :closable="false"
            show-icon
          >
            <template #default>
              <div class="alert-item__content">
                <span>时间窗口：{{ item.start_time || '-' }} ~ {{ item.end_time || '-' }}</span>
                <span>严重度：{{ item.diagnosis?.severity || item.severity || monitoringCopy.pendingSeverity }}</span>
              </div>
            </template>
          </el-alert>
        </div>
      </div>
    </article>

    <article class="section-card">
      <div class="section-card__header">
        <div>
          <div class="section-card__title">{{ monitoringCopy.sampleTitle }}</div>
          <div class="section-card__desc">{{ monitoringCopy.sampleDescription }}</div>
        </div>
      </div>
      <div class="section-card__body">
        <el-descriptions :column="1" border>
          <el-descriptions-item
            v-for="item in sampleSummaryItems"
            :key="item.label"
            :label="item.label"
          >
            {{ item.value }}
          </el-descriptions-item>
        </el-descriptions>
      </div>
    </article>
  </section>
</template>

<script setup>
import ElAlert from 'element-plus/es/components/alert/index.mjs'
import { ElDescriptions, ElDescriptionsItem } from 'element-plus/es/components/descriptions/index.mjs'
import ElEmpty from 'element-plus/es/components/empty/index.mjs'
import { monitoringCopy } from '@/modules/monitoring/monitoringView'

defineProps({
  alerts: {
    type: Array,
    default: () => []
  },
  sampleSummaryItems: {
    type: Array,
    default: () => []
  }
})
</script>

<style scoped>
.alert-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.alert-item__content {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 6px;
  color: var(--text-secondary);
}
</style>
