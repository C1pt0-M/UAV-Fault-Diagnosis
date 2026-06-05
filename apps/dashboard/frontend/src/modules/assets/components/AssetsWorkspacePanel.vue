<template>
  <section class="section-card assets-workspace">
    <div class="section-card__header page-toolbar">
      <div class="header-info">
        <div class="header-title-row">
          <el-icon class="header-icon"><Files /></el-icon>
          <div class="section-card__title">{{ assetsCopy.title }}</div>
        </div>
        <div class="section-card__desc">{{ assetsCopy.description }}</div>
      </div>

      <div class="assets-actions">
        <el-tabs v-model="selectedKind" class="kind-tabs" @tab-change="$emit('refresh')">
          <el-tab-pane
            v-for="tab in assetKindTabs"
            :key="tab.value"
            :label="tab.label"
            :name="tab.value"
          />
        </el-tabs>

        <div class="button-group">
          <el-upload
            :show-file-list="false"
            accept=".ulg,.csv"
            multiple
            :http-request="(options) => $emit('upload', options)"
          >
            <el-button type="primary" :loading="uploading" class="action-btn" round>
              <el-icon><Upload /></el-icon>
              <span>{{ assetsCopy.uploadButton }}</span>
            </el-button>
          </el-upload>

          <el-button plain class="action-btn" round @click="$emit('open-reports-center')">
            <el-icon><Document /></el-icon>
            <span>{{ assetsCopy.reportsCenterAction }}</span>
          </el-button>
        </div>
      </div>
    </div>

    <div class="section-card__body no-padding">
      <div class="assets-summary-strip">
        <article
          v-for="item in assetSummaryItems"
          :key="item.label"
          class="assets-summary-strip__item"
          :class="`assets-summary-strip__item--${item.tone}`"
        >
          <span>{{ item.label }}</span>
          <strong>{{ item.value }}</strong>
        </article>
      </div>

      <el-table :data="assets" stripe v-loading="loading" :empty-text="assetsCopy.emptyText" class="asset-table">
        <el-table-column prop="id" label="ID" width="80" align="center" />
        <el-table-column prop="filename" label="文件名" min-width="240">
          <template #default="{ row }">
            <div class="filename-cell">
              <el-icon class="file-icon">
                <component :is="row.kind === 'ulg' ? Box : Memo" />
              </el-icon>
              <span>{{ row.filename }}</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="类型" width="100">
          <template #default="{ row }">
            <el-tag :type="row.kind === 'ulg' ? 'warning' : 'success'" effect="light" round size="small">
              {{ String(row.kind || '').toUpperCase() }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="大小" width="110">
          <template #default="{ row }">
            <span class="size-text">{{ formatAssetSize(row.size_bytes || 0) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="预处理" width="120">
          <template #default="{ row }">
            <div class="status-chip" :class="{ 'is-done': row.meta?.prepared_csv_path }">
              <span class="dot"></span>
              <span>{{ row.meta?.prepared_csv_path ? '已就绪' : '待处理' }}</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="上传时间" width="180" />
        <el-table-column label="操作" width="280" fixed="right">
          <template #default="{ row }">
            <div class="table-actions">
              <el-button
                v-for="action in createAssetActionItems(row)"
                :key="action.key"
                link
                :type="action.type"
                :loading="resolveActionLoading(action.key, row)"
                @click="emitAssetAction(action.key, row)"
              >
                {{ action.label }}
              </el-button>
            </div>
          </template>
        </el-table-column>
      </el-table>

      <div v-if="total > pageSize" class="table-pagination">
        <el-pagination
          background
          layout="prev, pager, next"
          :current-page="currentPage"
          :page-size="pageSize"
          :total="total"
          @current-change="$emit('update:current-page', $event)"
        />
      </div>
    </div>
  </section>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElIcon from 'element-plus/es/components/icon/index.mjs'
import ElPagination from 'element-plus/es/components/pagination/index.mjs'
import ElTable from 'element-plus/es/components/table/index.mjs'
import { ElTableColumn } from 'element-plus/es/components/table/index.mjs'
import { ElTabPane, ElTabs } from 'element-plus/es/components/tabs/index.mjs'
import ElTag from 'element-plus/es/components/tag/index.mjs'
import ElUpload from 'element-plus/es/components/upload/index.mjs'
import { Files, Upload, Document, Box, Memo } from '@element-plus/icons-vue'
import { computed } from 'vue'
import { assetKindTabs, assetsCopy, createAssetActionItems, formatAssetSize } from '@/modules/assets/assetsView'

const props = defineProps({
  assets: { type: Array, default: () => [] },
  currentPage: { type: Number, default: 1 },
  pageSize: { type: Number, default: 10 },
  total: { type: Number, default: 0 },
  loading: { type: Boolean, default: false },
  uploading: { type: Boolean, default: false },
  preparingMap: { type: Object, required: true },
  deletingMap: { type: Object, required: true }
})

const emit = defineEmits([
  'refresh', 'upload', 'download', 'open-reports-center',
  'open-reports', 'prepare', 'show-detail', 'remove'
])

const selectedKind = defineModel('activeKind', {
  type: String,
  required: true
})

const assetSummaryItems = computed(() => {
  const csvCount = props.assets.filter((item) => item.kind === 'csv').length
  const ulgCount = props.assets.filter((item) => item.kind === 'ulg').length
  const preparedCount = props.assets.filter((item) => Boolean(item.meta?.prepared_csv_path)).length
  const pendingCount = props.assets.length - preparedCount

  return [
    { label: '当前筛选总数', value: props.total, tone: 'primary' },
    { label: '当前页 CSV', value: csvCount, tone: 'ok' },
    { label: '当前页 ULG', value: ulgCount, tone: 'warn' },
    { label: '当前页已就绪', value: preparedCount, tone: 'ok' },
    { label: '当前页待处理', value: pendingCount, tone: 'offline' }
  ]
})

function resolveActionLoading(actionKey, row) {
  if (actionKey === 'prepare') return Boolean(props.preparingMap[row.id])
  if (actionKey === 'remove') return Boolean(props.deletingMap[row.id])
  return false
}

function emitAssetAction(actionKey, row) {
  const eventMap = {
    download: 'download',
    reports: 'open-reports',
    prepare: 'prepare',
    detail: 'show-detail',
    remove: 'remove'
  }
  emit(eventMap[actionKey], row)
}
</script>

<style scoped>
.assets-workspace {
  overflow: hidden;
}

.header-info { display: flex; flex-direction: column; gap: 4px; }
.header-title-row { display: flex; align-items: center; gap: 12px; }
.header-icon { font-size: 20px; color: var(--brand); }
.assets-actions { display: flex; align-items: center; gap: 24px; }
.kind-tabs {
  padding: 8px var(--space-md) 0;
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  border: 1px solid var(--line-soft);
}
.kind-tabs :deep(.el-tabs__header) { margin-bottom: 0; }
.button-group { display: flex; align-items: center; gap: 12px; }
.action-btn { padding: 8px 16px; display: flex; align-items: center; gap: 8px; }
.no-padding { padding: 0 !important; }
.assets-summary-strip {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 8px;
  padding: 12px 20px;
  border-bottom: 1px solid var(--line-soft);
  background: var(--bg-panel);
}
.assets-summary-strip__item {
  min-width: 0;
  padding: 9px 12px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
}
.assets-summary-strip__item span {
  display: block;
  color: var(--text-tertiary);
  font-size: 11px;
  font-weight: 800;
}
.assets-summary-strip__item strong {
  display: block;
  margin-top: 4px;
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 18px;
  line-height: 1;
}
.assets-summary-strip__item--primary { border-left: 3px solid var(--brand); }
.assets-summary-strip__item--ok { border-left: 3px solid var(--status-ok); }
.assets-summary-strip__item--warn { border-left: 3px solid var(--status-warn); }
.assets-summary-strip__item--offline { border-left: 3px solid var(--status-offline); }
.asset-table { border-radius: 0; }
.table-pagination {
  display: flex;
  justify-content: flex-end;
  padding: 12px 18px 14px;
  border-top: 1px solid var(--line-soft);
  background: var(--bg-panel-raised);
}
.filename-cell { display: flex; align-items: center; gap: 9px; font-weight: 600; }
.file-icon { font-size: 15px; color: var(--text-tertiary); }
.size-text { font-family: var(--font-mono); font-size: 12px; color: var(--text-secondary); }
.status-chip { display: inline-flex; align-items: center; gap: 6px; padding: 3px 9px; border-radius: var(--radius-sm); background: var(--bg-panel-raised); font-size: 12px; color: var(--text-secondary); }
.status-chip .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text-tertiary); }
.status-chip.is-done { background: var(--status-ok-soft); color: var(--status-ok); }
.status-chip.is-done .dot { background: var(--status-ok); box-shadow: 0 0 8px rgba(22, 163, 74, 0.4); }
.table-actions { display: inline-flex; align-items: center; gap: 8px; white-space: nowrap; }

:deep(.assets-workspace .el-table__row td) {
  padding-top: 10px;
  padding-bottom: 10px;
}

:deep(.assets-workspace .el-table__header th) {
  padding-top: 9px;
  padding-bottom: 9px;
  color: var(--text-secondary);
  font-weight: 800;
}

@media (max-width: 1320px) {
  .assets-summary-strip {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 860px) {
  .assets-actions {
    align-items: flex-start;
    flex-direction: column;
  }

  .assets-summary-strip {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
