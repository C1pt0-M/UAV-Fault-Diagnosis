<template>
  <section class="section-card">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ monitoringCopy.controlTitle }}</div>
        <div class="section-card__desc">{{ monitoringCopy.controlDescription }}</div>
      </div>
    </div>
    <div class="section-card__body">
      <div class="monitoring-controls">
        <div class="monitoring-controls__inputs">
          <el-select v-model="selectedBackend" style="width: 150px">
            <el-option
              v-for="option in monitoringBackendOptions"
              :key="option.value"
              :label="option.label"
              :value="option.value"
            />
          </el-select>

          <el-select v-model="selectedDuration" style="width: 140px">
            <el-option
              v-for="option in monitoringDurationOptions"
              :key="option.value"
              :label="option.label"
              :value="option.value"
            />
          </el-select>

          <el-select
            v-model="selectedModel"
            clearable
            style="width: 240px"
            :placeholder="monitoringCopy.modelPlaceholder"
          >
            <el-option
              v-for="item in onlineModels"
              :key="item.name"
              :label="item.display_name || item.name"
              :value="item.name"
            />
          </el-select>

          <el-select
            v-if="selectedBackend === 'replay'"
            v-model="selectedReplayAssetId"
            clearable
            filterable
            style="width: 260px"
            :placeholder="monitoringCopy.replayAssetPlaceholder"
          >
            <el-option
              v-for="asset in replayAssets"
              :key="asset.id"
              :label="`#${asset.id} ${asset.filename}`"
              :value="asset.id"
            />
          </el-select>
        </div>

        <div class="monitoring-controls__actions">
          <el-button
            type="primary"
            :loading="actionLoading.startCollection"
            @click="$emit('start-collection')"
          >
            启动采集
          </el-button>
          <el-button
            type="success"
            plain
            :loading="actionLoading.startDetection"
            :disabled="currentState !== 'COLLECTING'"
            @click="$emit('start-detection')"
          >
            启动检测
          </el-button>
          <el-button
            type="warning"
            plain
            :loading="actionLoading.startBlindTest"
            :disabled="currentState !== 'IDLE' || selectedBackend === 'replay'"
            @click="$emit('start-blind-test')"
          >
            启动盲测
          </el-button>
          <el-button
            type="info"
            plain
            :loading="actionLoading.stopDetection"
            :disabled="currentState !== 'DETECTING'"
            @click="$emit('stop-detection')"
          >
            停止检测
          </el-button>
          <el-button
            type="danger"
            :loading="actionLoading.stopCollection"
            :disabled="currentState === 'IDLE'"
            @click="$emit('stop-collection')"
          >
            停止采集
          </el-button>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
import { ElOption, ElSelect } from 'element-plus/es/components/select/index.mjs'
import {
  monitoringBackendOptions,
  monitoringCopy,
  monitoringDurationOptions
} from '@/modules/monitoring/monitoringView'

defineProps({
  actionLoading: {
    type: Object,
    required: true
  },
  currentState: {
    type: String,
    required: true
  },
  onlineModels: {
    type: Array,
    default: () => []
  },
  replayAssets: {
    type: Array,
    default: () => []
  }
})

defineEmits([
  'start-collection',
  'start-detection',
  'start-blind-test',
  'stop-detection',
  'stop-collection'
])

const selectedBackend = defineModel('selectedBackend', {
  type: String,
  required: true
})

const selectedDuration = defineModel('selectedDuration', {
  type: Number,
  required: true
})

const selectedModel = defineModel('selectedModel', {
  type: String,
  required: true
})

const selectedReplayAssetId = defineModel('selectedReplayAssetId', {
  default: null
})
</script>

<style scoped>
.monitoring-controls {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.monitoring-controls__inputs,
.monitoring-controls__actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
</style>
