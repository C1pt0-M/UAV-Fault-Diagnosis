<template>
  <el-dialog v-model="dialogVisible" :title="monitoringCopy.sessionDetailTitle" width="720px">
    <template v-if="sessionDetail">
      <el-descriptions :column="1" border>
        <el-descriptions-item :label="monitoringCopy.sessionIdLabel">{{ sessionDetail.id || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="monitoringCopy.droneIdLabel">{{ sessionDetail.drone_id || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="monitoringCopy.statusLabel">{{ sessionDetail.status || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="monitoringCopy.faultEventsLabel">{{ sessionDetail.fault_events_count || 0 }}</el-descriptions-item>
      </el-descriptions>
    </template>
    <el-empty
      v-else
      :description="monitoringCopy.sessionDetailEmpty"
      :image-size="60"
    />
  </el-dialog>
</template>

<script setup>
import { computed } from 'vue'
import { ElDescriptions, ElDescriptionsItem } from 'element-plus/es/components/descriptions/index.mjs'
import ElDialog from 'element-plus/es/components/dialog/index.mjs'
import ElEmpty from 'element-plus/es/components/empty/index.mjs'
import { monitoringCopy } from '@/modules/monitoring/monitoringView'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  sessionDetail: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['update:visible'])

const dialogVisible = computed({
  get: () => props.visible,
  set: (value) => emit('update:visible', value)
})
</script>
