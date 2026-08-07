<template>
  <section class="section-card monitoring-console">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ monitoringCopy.controlTitle }}</div>
        <div class="section-card__desc">{{ monitoringCopy.controlDescription }}</div>
      </div>
    </div>
    <div class="section-card__body monitoring-console__body">
      <div class="monitoring-console__summary">
        <article class="monitoring-param-card">
          <div class="monitoring-param-card__label-row">
            <span class="status-dot" :class="currentStateDotClass"></span>
            <span class="monitoring-param-card__label">当前会话状态</span>
          </div>
          <strong class="monitoring-param-card__state">{{ currentStateText }}</strong>
          <small class="monitoring-param-card__meta">采集 / 检测生命周期</small>
        </article>

        <article class="monitoring-param-card">
          <div class="monitoring-param-card__label-row">
            <span class="status-dot status-dot--active"></span>
            <span class="monitoring-param-card__label">数据源</span>
          </div>
          <strong class="monitoring-param-card__value">{{ currentBackendLabel }}</strong>
          <small class="monitoring-param-card__meta">{{ currentBackendHint }}</small>
        </article>

        <article class="monitoring-param-card">
          <div class="monitoring-param-card__label-row">
            <span class="status-dot status-dot--pending"></span>
            <span class="monitoring-param-card__label">在线检测模型</span>
          </div>
          <strong class="monitoring-param-card__value" :title="currentModelDisplayName">
            {{ currentModelDisplayName }}
          </strong>
          <small v-if="currentModelRawName" class="monitoring-param-card__meta monitoring-param-card__meta--mono">
            {{ currentModelRawName }}
          </small>
          <small v-else class="monitoring-param-card__meta">请选择在线检测模型</small>
        </article>
      </div>

      <div class="monitoring-controls">
        <div class="monitoring-controls__inputs">
          <label class="monitoring-field">
            <span>数据源</span>
            <el-select v-model="selectedBackend" class="monitoring-control monitoring-control--backend">
              <el-option
                v-for="option in backendOptions"
                :key="option.value"
                :label="option.label"
                :value="option.value"
              />
            </el-select>
          </label>

          <label v-if="selectedBackend === 'px4'" class="monitoring-field">
            <span>采集时长</span>
            <el-select v-model="selectedDuration" class="monitoring-control monitoring-control--duration">
              <el-option
                v-for="option in monitoringDurationOptions"
                :key="option.value"
                :label="option.label"
                :value="option.value"
              />
            </el-select>
          </label>

          <label v-if="false /* MEC 推理位置暂时隐藏 */" class="monitoring-field">
            <span>{{ monitoringCopy.rflyInferenceLocationLabel }}</span>
            <el-select
              v-model="rflyInferenceLocation"
              class="monitoring-control monitoring-control--inference-location"
              data-test="rfly-inference-location"
            >
              <el-option
                v-for="option in rflyInferenceLocationOptions"
                :key="option.value"
                :label="option.label"
                :value="option.value"
              />
            </el-select>
          </label>

          <label class="monitoring-field monitoring-field--model">
            <span>在线模型</span>
            <el-select
              v-model="selectedModel"
              clearable
              class="monitoring-control monitoring-control--model"
              :placeholder="monitoringCopy.modelPlaceholder"
            >
              <el-option
                v-for="item in onlineModels"
                :key="item.name"
                :label="item.display_name || item.name"
                :value="item.name"
              />
            </el-select>
          </label>

          <label v-if="selectedBackend === 'replay'" class="monitoring-field monitoring-field--asset">
            <span>回放资产</span>
            <el-select
              v-model="selectedReplayAssetId"
              clearable
              filterable
              class="monitoring-control monitoring-control--asset"
              :placeholder="monitoringCopy.replayAssetPlaceholder"
            >
              <el-option
                v-for="asset in replayAssets"
                :key="asset.id"
                :label="`#${asset.id} ${asset.filename}`"
                :value="asset.id"
              />
            </el-select>
          </label>

          <label v-if="selectedBackend === 'rfly'" class="monitoring-field monitoring-field--rfly">
            <span>RflySim 路径</span>
            <el-input
              v-model="rflyPx4mavctrlPath"
              clearable
              class="monitoring-control monitoring-control--rfly-path"
              :placeholder="monitoringCopy.rflyPx4MavCtrlPlaceholder"
            />
          </label>
        </div>

        <div
          v-if="selectedBackend === 'rfly'"
          class="rfly-diagnostics"
          data-test="rfly-diagnostics-panel"
        >
          <div class="rfly-diagnostics__header">
            <div>
              <div class="rfly-diagnostics__title">{{ monitoringCopy.rflyDiagnosticsTitle }}</div>
              <div class="rfly-diagnostics__desc">{{ rflyDiagnosticsMessage }}</div>
            </div>
            <el-button
              plain
              size="small"
              :loading="rflyDiagnosticsLoading"
              data-test="rfly-diagnostics-button"
              @click="$emit('test-rfly-diagnostics')"
            >
              {{ monitoringCopy.rflyDiagnosticsAction }}
            </el-button>
          </div>
          <div class="rfly-diagnostics__checks">
            <span
              v-for="check in rflyDiagnosticsChecks"
              :key="check.key"
              class="rfly-diagnostics__check"
              :class="{ 'rfly-diagnostics__check--ok': check.ok }"
            >
              {{ check.label }}
            </span>
          </div>
        </div>

        <div class="monitoring-controls__actions">
          <el-button
            type="primary"
            :loading="actionLoading.startCollection"
            :disabled="isStartCollectionDisabled"
            data-test="start-collection"
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
import ElInput from 'element-plus/es/components/input/index.mjs'
import { ElOption, ElSelect } from 'element-plus/es/components/select/index.mjs'
import {
  monitoringBackendOptions,
  monitoringCopy,
  monitoringDurationOptions,
  rflyInferenceLocationOptions,
  monitoringStateLabels
} from '@/modules/monitoring/monitoringView'

const props = defineProps({
  actionLoading: { type: Object, required: true },
  currentState: { type: String, required: true },
  onlineModels: { type: Array, default: () => [] },
  backendOptions: { type: Array, default: () => [] },
  blindTestEnabled: { type: Boolean, default: false },
  replayAssets: { type: Array, default: () => [] },
  rflyDiagnostics: { type: Object, default: null },
  rflyDiagnosticsLoading: { type: Boolean, default: false }
})

defineEmits([
  'start-collection',
  'start-detection',
  'start-blind-test',
  'stop-detection',
  'stop-collection',
  'test-rfly-diagnostics'
])

const selectedBackend = defineModel('selectedBackend', { type: String, required: true })
const selectedDuration = defineModel('selectedDuration', { type: Number, required: true })
const selectedModel = defineModel('selectedModel', { type: String, required: true })
const selectedReplayAssetId = defineModel('selectedReplayAssetId', { default: null })
const rflyPx4mavctrlPath = defineModel('rflyPx4mavctrlPath', { type: String, default: '' })
const rflyInferenceLocation = defineModel('rflyInferenceLocation', { type: String, default: 'center' })

const currentStateText = computed(() => monitoringStateLabels[props.currentState] || props.currentState)
const currentBackendLabel = computed(() => {
  const option = props.backendOptions.find((item) => item.value === selectedBackend.value) ||
    monitoringBackendOptions.find((item) => item.value === selectedBackend.value)
  return option?.label || selectedBackend.value
})

const currentBackendHint = computed(() => {
  if (selectedBackend.value === 'replay') return '离线 CSV / ULG 资产回放'
  if (selectedBackend.value === 'rfly') return 'RflySim 实时遥测链路'
  return '当前遥测数据入口'
})

const currentModel = computed(() => {
  if (!selectedModel.value) return monitoringCopy.noSelection
  return props.onlineModels.find((item) => item.name === selectedModel.value) || {
    name: selectedModel.value,
    display_name: selectedModel.value
  }
})

const currentModelDisplayName = computed(() => {
  if (!selectedModel.value) return monitoringCopy.noSelection
  return currentModel.value.display_name || currentModel.value.name || selectedModel.value
})

const currentModelRawName = computed(() => {
  if (!selectedModel.value) return ''
  return currentModel.value.name || selectedModel.value
})

const currentStateDotClass = computed(() => {
  if (props.currentState === 'COLLECTING') return 'status-dot--active'
  if (props.currentState === 'DETECTING') return 'status-dot--ok'
  if (props.currentState === 'BLIND_TEST') return 'status-dot--pending'
  return 'status-dot--offline'
})

const rflyDiagnosticsMessage = computed(() => {
  return props.rflyDiagnostics?.summary?.message || monitoringCopy.rflyDiagnosticsEmpty
})

const rflyDiagnosticsChecks = computed(() => {
  const checks = props.rflyDiagnostics?.checks || {}
  return Object.entries(checks).map(([key, check]) => ({
    key,
    label: check.label || check.message || key,
    ok: Boolean(check.ok)
  }))
})

const isStartCollectionDisabled = computed(() => {
  if (props.currentState !== 'IDLE') return true
  if (selectedBackend.value !== 'rfly') return false
  return props.rflyDiagnostics?.ready !== true
})
</script>

<style scoped>
.monitoring-console {
  overflow: hidden;
}

.monitoring-console__body {
  padding: var(--space-lg);
}

.monitoring-console__summary {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--space-md);
  margin-bottom: var(--space-md);
}

.monitoring-param-card {
  min-height: 74px;
  padding: 12px 14px;
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  border: 1px solid var(--line-soft);
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 5px;
}

.monitoring-param-card__label-row {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.monitoring-param-card__label {
  font-size: 11px;
  font-weight: 700;
  color: var(--text-tertiary);
}

.monitoring-param-card__state,
.monitoring-param-card__value {
  min-width: 0;
  font-size: 16px;
  line-height: 1.25;
  font-weight: 800;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.monitoring-param-card__meta {
  min-width: 0;
  font-size: 11px;
  line-height: 1.3;
  color: var(--text-tertiary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.monitoring-param-card__meta--mono {
  font-family: var(--font-mono);
}

.monitoring-controls {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}

.monitoring-controls__inputs {
  display: grid;
  grid-template-columns: 150px 140px minmax(220px, 1fr) minmax(260px, 1.1fr);
  gap: var(--space-md);
  align-items: end;
  padding: 12px;
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  border: 1px solid var(--line-soft);
}

.monitoring-field {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.monitoring-field span {
  font-size: 11px;
  color: var(--text-tertiary);
  font-weight: 700;
}

.monitoring-field--model,
.monitoring-field--asset,
.monitoring-field--rfly {
  min-width: 0;
}

.monitoring-control {
  width: 100%;
}

.rfly-diagnostics {
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
  padding: var(--space-md);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  border: 1px solid var(--line-soft);
}

.rfly-diagnostics__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-md);
}

.rfly-diagnostics__title {
  font-size: 13px;
  font-weight: 800;
  color: var(--text-primary);
}

.rfly-diagnostics__desc {
  margin-top: 4px;
  font-size: 12px;
  color: var(--text-secondary);
}

.rfly-diagnostics__checks {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.rfly-diagnostics__check {
  padding: 3px 8px;
  border-radius: var(--radius-sm);
  background: var(--status-danger-soft);
  color: var(--status-danger);
  font-size: 11px;
  font-weight: 700;
}

.rfly-diagnostics__check--ok {
  background: var(--status-ok-soft);
  color: var(--status-ok);
}

.monitoring-controls__actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: var(--space-sm);
  flex-wrap: wrap;
  padding-top: var(--space-md);
  border-top: 1px dashed var(--line-soft);
}

@media (max-width: 1320px) {
  .monitoring-controls__inputs {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 760px) {
  .monitoring-console__summary,
  .monitoring-controls__inputs {
    grid-template-columns: 1fr;
  }
}
</style>
