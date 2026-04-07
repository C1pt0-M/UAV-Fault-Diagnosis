<template>
  <section class="section-card monitoring-console">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ monitoringCopy.controlTitle }}</div>
        <div class="section-card__desc">{{ monitoringCopy.controlDescription }}</div>
      </div>
    </div>
    <div class="section-card__body">
      <div class="monitoring-console__summary">
        <article class="monitoring-console__pill">
          <span>当前状态</span>
          <strong>{{ currentStateText }}</strong>
        </article>
        <article class="monitoring-console__pill">
          <span>数据源</span>
          <strong>{{ currentBackendLabel }}</strong>
        </article>
        <article class="monitoring-console__pill">
          <span>在线模型</span>
          <strong>{{ currentModelLabel }}</strong>
        </article>
      </div>

      <div class="monitoring-controls">
        <div class="monitoring-controls__inputs">
          <el-select v-model="selectedBackend" style="width: 150px">
            <el-option
              v-for="option in backendOptions"
              :key="option.value"
              :label="option.label"
              :value="option.value"
            />
          </el-select>

          <el-select v-if="selectedBackend !== 'replay'" v-model="selectedDuration" style="width: 140px">
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
            {{ monitoringCopy.startCollectionLabel }}
          </el-button>
          <el-button
            type="success"
            plain
            :loading="actionLoading.startDetection"
            :disabled="currentState !== 'COLLECTING'"
            @click="$emit('start-detection')"
          >
            {{ monitoringCopy.startDetectionLabel }}
          </el-button>
          <el-button
            v-if="blindTestEnabled"
            type="warning"
            plain
            :loading="actionLoading.startBlindTest"
            :disabled="currentState !== 'IDLE' || selectedBackend === 'replay'"
            @click="$emit('start-blind-test')"
          >
            {{ monitoringCopy.startBlindTestLabel }}
          </el-button>
          <el-button
            type="info"
            plain
            :loading="actionLoading.stopDetection"
            :disabled="currentState !== 'DETECTING'"
            @click="$emit('stop-detection')"
          >
            {{ monitoringCopy.stopDetectionLabel }}
          </el-button>
          <el-button
            type="danger"
            :loading="actionLoading.stopCollection"
            :disabled="currentState === 'IDLE'"
            @click="$emit('stop-collection')"
          >
            {{ monitoringCopy.stopCollectionLabel }}
          </el-button>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import { ElOption, ElSelect } from 'element-plus/es/components/select/index.mjs'
import {
  monitoringBackendOptions,
  monitoringCopy,
  monitoringDurationOptions,
  monitoringStateLabels
} from '@/modules/monitoring/monitoringView'

const props = defineProps({
  actionLoading: { type: Object, required: true },
  currentState: { type: String, required: true },
  onlineModels: { type: Array, default: () => [] },
  backendOptions: { type: Array, default: () => [] },
  blindTestEnabled: { type: Boolean, default: false },
  replayAssets: { type: Array, default: () => [] }
})

defineEmits([
  'start-collection',
  'start-detection',
  'start-blind-test',
  'stop-detection',
  'stop-collection'
])

const selectedBackend = defineModel('selectedBackend', { type: String, required: true })
const selectedDuration = defineModel('selectedDuration', { type: Number, required: true })
const selectedModel = defineModel('selectedModel', { type: String, required: true })
const selectedReplayAssetId = defineModel('selectedReplayAssetId', { default: null })

const currentStateText = computed(() => monitoringStateLabels[props.currentState] || props.currentState)
const currentBackendLabel = computed(() => {
  const option = props.backendOptions.find((item) => item.value === selectedBackend.value) ||
    monitoringBackendOptions.find((item) => item.value === selectedBackend.value)
  return option?.label || selectedBackend.value
})
const currentModelLabel = computed(() => {
  if (!selectedModel.value) return monitoringCopy.noSelection
  const option = props.onlineModels.find((item) => item.name === selectedModel.value)
  return option?.display_name || option?.name || selectedModel.value
})
</script>

<style scoped>
.monitoring-console {
  overflow: hidden;
}

.monitoring-console__summary {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}

.monitoring-console__pill {
  padding: 14px 16px;
  border-radius: 18px;
  background: linear-gradient(180deg, #f7fbfc 0%, #eef6f7 100%);
  border: 1px solid var(--line-soft);
}

.monitoring-console__pill span {
  display: block;
  font-size: 12px;
  color: var(--text-tertiary);
}

.monitoring-console__pill strong {
  display: block;
  margin-top: 8px;
  font-size: 18px;
  line-height: 1.35;
  color: var(--text-primary);
}

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

.monitoring-controls__actions {
  padding-top: 8px;
  border-top: 1px dashed var(--line-soft);
}

@media (max-width: 1480px) {
  .monitoring-console__summary {
    grid-template-columns: 1fr;
  }
}
</style>
