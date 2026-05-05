<template>
  <div class="diag-statusbar">
    <div class="diag-statusbar__item diag-statusbar__item--online">
      <span class="status-dot status-dot--ok"></span>
      <span class="diag-statusbar__value">系统在线</span>
    </div>

    <div class="diag-statusbar__divider"></div>

    <div class="diag-statusbar__item diag-statusbar__item--primary">
      <span class="diag-statusbar__label">{{ diagnosisCopy.stageLabel }}</span>
      <span class="diag-statusbar__value">{{ stage.label }}</span>
    </div>

    <div class="diag-statusbar__divider"></div>

    <div class="diag-statusbar__item diag-statusbar__item--asset">
      <span class="diag-statusbar__label">{{ diagnosisCopy.assetSummaryLabel }}</span>
      <span class="diag-statusbar__value" :title="selectedAssetName">{{ selectedAssetName }}</span>
    </div>

    <div class="diag-statusbar__divider"></div>

    <div class="diag-statusbar__item">
      <span class="diag-statusbar__label">{{ diagnosisCopy.taskIdLabel }}</span>
      <span class="diag-statusbar__value diag-statusbar__value--mono">{{ currentRun?.id ? `DG-${currentRun.id}` : '-' }}</span>
    </div>

    <div class="diag-statusbar__divider"></div>

    <div class="diag-statusbar__item">
      <span class="diag-statusbar__label">{{ diagnosisCopy.modelSummaryLabel }}</span>
      <span class="diag-statusbar__value" :title="selectedModelName">{{ selectedModelName }}</span>
    </div>

    <div class="diag-statusbar__divider"></div>

    <div class="diag-statusbar__item">
      <span class="diag-statusbar__label">{{ diagnosisCopy.taskTypeLabel }}</span>
      <span class="diag-statusbar__value diag-statusbar__value--mono">{{ currentRun?.task || 'offline_multiclass' }}</span>
    </div>

    <div class="diag-statusbar__divider"></div>

    <div class="diag-statusbar__item">
      <span class="diag-statusbar__label">{{ diagnosisCopy.linkedSessionLabel }}</span>
      <span class="diag-statusbar__value diag-statusbar__value--mono">{{ activeSessionId || '-' }}</span>
    </div>

    <div class="diag-statusbar__divider"></div>

    <div class="diag-statusbar__item">
      <span class="diag-statusbar__label">报告状态</span>
      <span class="diag-statusbar__cue">{{ reportStatusText }}</span>
    </div>

    <div class="diag-statusbar__divider"></div>

    <div class="diag-statusbar__item">
      <span class="diag-statusbar__label">最近更新</span>
      <span class="diag-statusbar__value diag-statusbar__value--mono">{{ latestUpdatedText }}</span>
    </div>
  </div>
</template>

<script setup>
import { diagnosisCopy } from '@/modules/diagnosis/diagnosisView'

defineProps({
  activeSessionId: {
    type: String,
    default: ''
  },
  currentRun: {
    type: Object,
    default: null
  },
  latestUpdatedText: {
    type: String,
    default: '暂无'
  },
  reportStatusText: {
    type: String,
    default: '待诊断'
  },
  selectedAssetName: {
    type: String,
    required: true
  },
  selectedModelName: {
    type: String,
    required: true
  },
  stage: {
    type: Object,
    required: true
  }
})
</script>

<style scoped>
.diag-statusbar {
  display: flex;
  align-items: center;
  flex-wrap: nowrap;
  gap: var(--space-sm);
  min-width: 0;
  padding: 10px 14px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  font-size: 12px;
  color: var(--text-secondary);
  overflow: hidden;
}

.diag-statusbar__item {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  flex-shrink: 0;
}

.diag-statusbar__item--asset {
  flex: 1 1 240px;
}

.diag-statusbar__item--primary .diag-statusbar__value {
  color: var(--status-ok);
}

.diag-statusbar__item--online {
  color: var(--status-ok);
}

.diag-statusbar__label {
  color: var(--text-tertiary);
  font-size: 11px;
  white-space: nowrap;
}

.diag-statusbar__value {
  min-width: 0;
  max-width: min(280px, 100%);
  color: var(--text-primary);
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.diag-statusbar__value--mono {
  font-family: var(--font-mono);
  font-size: 11px;
}

.diag-statusbar__cue {
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  background: var(--brand-soft);
  color: var(--brand);
  font-size: 11px;
  font-weight: 700;
  white-space: nowrap;
}

.diag-statusbar__divider {
  width: 1px;
  height: 14px;
  background: var(--line-strong);
  flex-shrink: 0;
}

@media (max-width: 1180px) {
  .diag-statusbar {
    flex-wrap: wrap;
  }

  .diag-statusbar__divider {
    display: none;
  }
}
</style>
