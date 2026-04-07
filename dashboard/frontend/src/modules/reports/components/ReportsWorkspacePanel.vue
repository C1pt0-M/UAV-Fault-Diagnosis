<template>
  <section class="section-card reports-workspace">
    <div class="section-card__header page-toolbar">
      <div>
        <div class="section-card__title">{{ reportsCopy.title }}</div>
        <div class="section-card__desc">{{ reportsCopy.description }}</div>
      </div>

      <div class="reports-toolbar">
        <el-select
          v-model="reportFilters.assetId"
          clearable
          filterable
          style="width: 220px"
          :placeholder="reportsCopy.assetFilterPlaceholder"
        >
          <el-option
            v-for="asset in assetOptions"
            :key="asset.id"
            :label="`#${asset.id} ${asset.filename}`"
            :value="asset.id"
          />
        </el-select>

        <el-select
          v-model="reportFilters.label"
          clearable
          style="width: 220px"
          :placeholder="reportsCopy.labelFilterPlaceholder"
        >
          <el-option
            v-for="option in reportLabelOptions"
            :key="option.value"
            :label="option.label"
            :value="option.value"
          />
        </el-select>

        <el-button @click="$emit('reset')">
          {{ reportsCopy.resetButton }}
        </el-button>
        <el-button type="primary" @click="$emit('refresh')">
          {{ reportsCopy.refreshButton }}
        </el-button>
      </div>
    </div>

    <div class="section-card__body">
      <el-table :data="reports" stripe v-loading="loading" :empty-text="reportsCopy.emptyText">
        <el-table-column prop="id" label="报告 ID" width="100" />
        <el-table-column prop="run_id" label="任务 ID" width="100" />
        <el-table-column label="资产" min-width="220">
          <template #default="{ row }">
            #{{ row.asset_id || '-' }} {{ assetNameMap.get(row.asset_id) || '-' }}
          </template>
        </el-table-column>
        <el-table-column prop="model_name" label="模型" width="220" />
        <el-table-column prop="prediction_label" label="诊断结论" width="140" />
        <el-table-column label="置信度" width="120">
          <template #default="{ row }">
            {{ `${(Number(row.confidence || 0) * 100).toFixed(2)}%` }}
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="生成时间" width="180" />
        <el-table-column label="操作" width="280">
          <template #default="{ row }">
            <el-button link type="primary" @click="$emit('show-detail', row)">
              {{ reportsCopy.detailAction }}
            </el-button>
            <el-button link type="warning" @click="$emit('open-ai', row)">
              {{ reportsCopy.aiAction }}
            </el-button>
            <el-button link type="success" @click="$emit('download', row)">
              {{ reportsCopy.downloadAction }}
            </el-button>
            <el-button link type="danger" @click="$emit('remove', row)">
              {{ reportsCopy.removeAction }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </section>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
import { ElOption, ElSelect } from 'element-plus/es/components/select/index.mjs'
import { ElTable, ElTableColumn } from 'element-plus/es/components/table/index.mjs'
import { reportLabelOptions, reportsCopy } from '@/modules/reports/reportsView'

defineProps({
  assetOptions: { type: Array, default: () => [] },
  assetNameMap: { type: Object, required: true },
  loading: { type: Boolean, default: false },
  reports: { type: Array, default: () => [] }
})

defineEmits(['refresh', 'reset', 'show-detail', 'open-ai', 'download', 'remove'])

const reportFilters = defineModel('filters', {
  required: true
})
</script>

<style scoped>
.reports-workspace {
  overflow: hidden;
}

.reports-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  padding: 12px 14px;
  border-radius: 18px;
  background: linear-gradient(180deg, #f7fbfc 0%, #eef6f7 100%);
  border: 1px solid var(--line-soft);
}

:deep(.reports-workspace .el-table__row td) {
  padding-top: 12px;
  padding-bottom: 12px;
}
</style>
