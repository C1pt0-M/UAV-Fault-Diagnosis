<template>
  <section class="section-card">
    <div class="section-card__header page-toolbar">
      <div>
        <div class="section-card__title">{{ assetsCopy.title }}</div>
        <div class="section-card__desc">{{ assetsCopy.description }}</div>
      </div>

      <div class="assets-actions">
        <el-tabs v-model="selectedKind" @tab-change="$emit('refresh')">
          <el-tab-pane
            v-for="tab in assetKindTabs"
            :key="tab.value"
            :label="tab.label"
            :name="tab.value"
          />
        </el-tabs>

        <el-upload
          :show-file-list="false"
          accept=".ulg,.csv"
          multiple
          :http-request="(options) => $emit('upload', options)"
        >
          <el-button type="primary" :loading="uploading">
            {{ assetsCopy.uploadButton }}
          </el-button>
        </el-upload>

        <el-button plain @click="$emit('open-reports-center')">
          {{ assetsCopy.reportsCenterAction }}
        </el-button>
      </div>
    </div>

    <div class="section-card__body">
      <el-table :data="assets" stripe v-loading="loading" :empty-text="assetsCopy.emptyText">
        <el-table-column prop="id" label="ID" width="90" />
        <el-table-column prop="filename" label="文件名" min-width="260" />
        <el-table-column label="类型" width="110">
          <template #default="{ row }">
            <el-tag :type="row.kind === 'ulg' ? 'warning' : 'success'">
              {{ String(row.kind || '').toUpperCase() }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="大小" width="120">
          <template #default="{ row }">
            {{ formatAssetSize(row.size_bytes || 0) }}
          </template>
        </el-table-column>
        <el-table-column label="预处理" width="140">
          <template #default="{ row }">
            <el-tag :type="row.meta?.prepared_csv_path ? 'success' : 'info'">
              {{ row.meta?.prepared_csv_path ? assetsCopy.preparingDone : assetsCopy.preparingPending }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="上传时间" width="180" />
        <el-table-column label="操作" width="360">
          <template #default="{ row }">
            <el-button link type="primary" @click="$emit('download', row)">
              {{ assetsCopy.downloadAction }}
            </el-button>
            <el-button link type="success" @click="$emit('open-reports', row)">
              {{ assetsCopy.reportAction }}
            </el-button>
            <el-button
              v-if="row.kind === 'ulg'"
              link
              type="warning"
              :loading="preparingMap[row.id]"
              @click="$emit('prepare', row)"
            >
              {{ assetsCopy.prepareAction }}
            </el-button>
            <el-button link type="info" @click="$emit('show-detail', row)">
              {{ assetsCopy.detailAction }}
            </el-button>
            <el-button
              link
              type="danger"
              :loading="deletingMap[row.id]"
              @click="$emit('remove', row)"
            >
              {{ assetsCopy.removeAction }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </section>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
import { ElTabPane, ElTabs } from 'element-plus/es/components/tabs/index.mjs'
import { ElTable, ElTableColumn } from 'element-plus/es/components/table/index.mjs'
import ElTag from 'element-plus/es/components/tag/index.mjs'
import ElUpload from 'element-plus/es/components/upload/index.mjs'
import { assetKindTabs, assetsCopy, formatAssetSize } from '@/modules/assets/assetsView'

defineProps({
  assets: {
    type: Array,
    default: () => []
  },
  loading: {
    type: Boolean,
    default: false
  },
  uploading: {
    type: Boolean,
    default: false
  },
  preparingMap: {
    type: Object,
    required: true
  },
  deletingMap: {
    type: Object,
    required: true
  }
})

defineEmits([
  'refresh',
  'upload',
  'download',
  'open-reports-center',
  'open-reports',
  'prepare',
  'show-detail',
  'remove'
])

const selectedKind = defineModel('activeKind', {
  type: String,
  required: true
})
</script>

<style scoped>
.assets-actions {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.assets-actions :deep(.el-tabs__header) {
  margin-bottom: 0;
}
</style>
