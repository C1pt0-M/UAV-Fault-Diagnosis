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
        <el-table-column label="操作" width="340" fixed="right">
          <template #default="{ row }">
            <div class="table-actions">
              <el-button link type="primary" @click="$emit('download', row)">下载</el-button>
              <el-divider direction="vertical" />
              <el-button link type="success" @click="$emit('open-reports', row)">报告</el-button>
              <el-divider direction="vertical" />
              <el-button
                v-if="row.kind === 'ulg'"
                link
                type="warning"
                :loading="preparingMap[row.id]"
                @click="$emit('prepare', row)"
              >处理</el-button>
              <el-divider v-if="row.kind === 'ulg'" direction="vertical" />
              <el-button link type="info" @click="$emit('show-detail', row)">详情</el-button>
              <el-divider direction="vertical" />
              <el-button
                link
                type="danger"
                :loading="deletingMap[row.id]"
                @click="$emit('remove', row)"
              >删除</el-button>
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
import ElDivider from 'element-plus/es/components/divider/index.mjs'
import ElIcon from 'element-plus/es/components/icon/index.mjs'
import ElPagination from 'element-plus/es/components/pagination/index.mjs'
import ElTable from 'element-plus/es/components/table/index.mjs'
import { ElTableColumn } from 'element-plus/es/components/table/index.mjs'
import { ElTabPane, ElTabs } from 'element-plus/es/components/tabs/index.mjs'
import ElTag from 'element-plus/es/components/tag/index.mjs'
import ElUpload from 'element-plus/es/components/upload/index.mjs'
import { Files, Upload, Document, Box, Memo } from '@element-plus/icons-vue'
import { assetKindTabs, assetsCopy, formatAssetSize } from '@/modules/assets/assetsView'

defineProps({
  assets: { type: Array, default: () => [] },
  currentPage: { type: Number, default: 1 },
  pageSize: { type: Number, default: 10 },
  total: { type: Number, default: 0 },
  loading: { type: Boolean, default: false },
  uploading: { type: Boolean, default: false },
  preparingMap: { type: Object, required: true },
  deletingMap: { type: Object, required: true }
})

defineEmits([
  'refresh', 'upload', 'download', 'open-reports-center',
  'open-reports', 'prepare', 'show-detail', 'remove'
])

const selectedKind = defineModel('activeKind', {
  type: String,
  required: true
})
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
.asset-table { border-radius: 0 0 16px 16px; }
.table-pagination {
  display: flex;
  justify-content: flex-end;
  padding: 16px 20px 20px;
  border-top: 1px solid var(--line-soft);
  background: var(--bg-panel-raised);
}
.filename-cell { display: flex; align-items: center; gap: 10px; font-weight: 500; }
.file-icon { font-size: 16px; color: var(--text-tertiary); }
.size-text { font-family: var(--font-mono); font-size: 12px; color: var(--text-secondary); }
.status-chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: var(--radius-sm); background: var(--bg-panel-raised); font-size: 12px; color: var(--text-secondary); }
.status-chip .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text-tertiary); }
.status-chip.is-done { background: var(--status-ok-soft); color: var(--status-ok); }
.status-chip.is-done .dot { background: var(--status-ok); box-shadow: 0 0 8px rgba(22, 163, 74, 0.4); }
.table-actions { display: flex; align-items: center; gap: 4px; }

:deep(.assets-workspace .el-table__row td) {
  padding-top: 12px;
  padding-bottom: 12px;
}
</style>
