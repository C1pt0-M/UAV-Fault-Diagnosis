<template>
  <!-- 诊断状态摘要条：工业风紧凑布局，替代原大渐变 Hero -->
  <div class="diag-statusbar">
    <!-- 当前阶段 -->
    <div class="diag-statusbar__item diag-statusbar__item--primary">
      <span class="diag-statusbar__label">{{ diagnosisCopy.stageLabel }}</span>
      <span class="diag-statusbar__value">{{ stage.label }}</span>
    </div>

    <div class="diag-statusbar__divider"></div>

    <!-- 操作提示 -->
    <div class="diag-statusbar__item">
      <span class="diag-statusbar__cue">{{ stage.cue }}</span>
    </div>

    <div class="diag-statusbar__divider"></div>

    <!-- 关联资产 -->
    <div class="diag-statusbar__item">
      <span class="diag-statusbar__label">{{ diagnosisCopy.assetSummaryLabel }}</span>
      <span class="diag-statusbar__value">{{ selectedAssetName }}</span>
    </div>

    <div class="diag-statusbar__divider"></div>

    <!-- 关联会话 -->
    <div class="diag-statusbar__item">
      <span class="diag-statusbar__label">{{ diagnosisCopy.linkedSessionLabel }}</span>
      <span class="diag-statusbar__value diag-statusbar__value--mono">{{ activeSessionId || '-' }}</span>
    </div>

    <div class="diag-statusbar__divider"></div>

    <!-- 模型 -->
    <div class="diag-statusbar__item">
      <span class="diag-statusbar__label">{{ diagnosisCopy.modelSummaryLabel }}</span>
      <span class="diag-statusbar__value">{{ selectedModelName }}</span>
    </div>

    <div class="diag-statusbar__divider"></div>

    <!-- 任务类型 -->
    <div class="diag-statusbar__item">
      <span class="diag-statusbar__label">任务类型</span>
      <span class="diag-statusbar__value diag-statusbar__value--mono">{{ currentRun?.task || 'offline_multiclass' }}</span>
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
/* 诊断状态摘要条：与 system-statusbar 同一视觉风格 */
.diag-statusbar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: var(--space-sm);
  padding: var(--space-sm) var(--space-lg);
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  font-size: 12px;
  color: var(--text-secondary);
}

.diag-statusbar__item {
  display: flex;
  align-items: center;
  gap: 6px;
}

/* 当前阶段项：稍微突出 */
.diag-statusbar__item--primary .diag-statusbar__value {
  color: var(--brand);
  font-weight: 700;
}

.diag-statusbar__label {
  color: var(--text-tertiary);
  font-size: 11px;
  white-space: nowrap;
}

.diag-statusbar__value {
  color: var(--text-primary);
  font-weight: 600;
  white-space: nowrap;
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.diag-statusbar__value--mono {
  font-family: var(--font-mono);
  font-size: 11px;
}

/* 操作提示：使用 badge 风格 */
.diag-statusbar__cue {
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  background: var(--brand-soft);
  color: var(--brand);
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
}

.diag-statusbar__divider {
  width: 1px;
  height: 14px;
  background: var(--line-strong);
  flex-shrink: 0;
}
</style>
