<template>
  <section class="section-card">
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
      <el-table :data="sessions" stripe :empty-text="monitoringCopy.sessionsEmpty">
        <el-table-column prop="id" label="会话 ID" width="120" />
        <el-table-column prop="drone_id" label="无人机" width="140" />
        <el-table-column prop="start_time" label="开始时间" width="180" />
        <el-table-column prop="duration_seconds" label="时长（秒）" width="120">
          <template #default="{ row }">
            {{ Number(row.duration_seconds || 0).toFixed(1) }}
          </template>
        </el-table-column>
        <el-table-column prop="fault_events_count" label="故障事件" width="120" />
        <el-table-column label="操作" width="220">
          <template #default="{ row }">
            <el-button link type="primary" @click="$emit('open-detail', row.id)">查看详情</el-button>
            <el-button link type="warning" @click="$emit('continue-diagnosis', row.id)">继续诊断</el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </section>
</template>

<script setup>
import { Refresh } from '@element-plus/icons-vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElIcon from 'element-plus/es/components/icon/index.mjs'
import { ElTable, ElTableColumn } from 'element-plus/es/components/table/index.mjs'
import { monitoringCopy } from '@/modules/monitoring/monitoringView'

defineProps({
  sessions: {
    type: Array,
    default: () => []
  }
})

defineEmits(['refresh', 'open-detail', 'continue-diagnosis'])
</script>
