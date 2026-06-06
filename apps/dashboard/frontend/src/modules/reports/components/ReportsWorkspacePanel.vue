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
          class="reports-filter-select"
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
          class="reports-filter-select"
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
      <div class="reports-summary-strip">
        <article
          v-for="item in reportSummaryItems"
          :key="item.label"
          class="reports-summary-strip__item"
          :class="`reports-summary-strip__item--${item.tone}`"
        >
          <span>{{ item.label }}</span>
          <strong>{{ item.value }}</strong>
        </article>
      </div>

      <el-table
        :data="reports"
        stripe
        v-loading="loading"
        :empty-text="reportsCopy.emptyText"
        class="reports-table"
      >
        <el-table-column prop="id" label="报告 ID" width="100" />
        <el-table-column prop="run_id" label="任务 ID" width="100" />
        <el-table-column label="资产" min-width="220">
          <template #default="{ row }">
            #{{ row.asset_id || '-' }} {{ assetNameMap.get(row.asset_id) || '-' }}
          </template>
        </el-table-column>
        <el-table-column label="模型" width="180">
          <template #default="{ row }">
            <span class="report-model-name" :title="row.model_name || ''">
              {{ getReportModelDisplayName(row.model_name) }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="诊断结论" width="140">
          <template #default="{ row }">
            {{ getReportLabelText(row.prediction_label) }}
          </template>
        </el-table-column>
        <el-table-column label="置信度" width="120">
          <template #default="{ row }">
            {{ formatReportConfidence(row.confidence) }}
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="生成时间" width="180" />
        <el-table-column label="操作" width="250">
          <template #default="{ row }">
            <div class="reports-actions">
              <el-button
                v-for="action in createReportActionItems()"
                :key="action.key"
                link
                :type="action.type"
                :class="{ 'reports-action--danger': action.key === 'remove' }"
                @click="emitReportAction(action.key, row)"
              >
                {{ action.label }}
              </el-button>
            </div>
          </template>
        </el-table-column>
      </el-table>

      <div v-if="pagination.total > pagination.perPage" class="reports-pagination">
        <span>{{ reportsCopy.paginationTotal(pagination.total) }}</span>
        <el-pagination
          background
          layout="sizes, prev, pager, next"
          :current-page="pagination.page"
          :page-size="pagination.perPage"
          :page-sizes="[10, 20, 50, 100]"
          :total="pagination.total"
          @current-change="$emit('page-change', $event)"
          @size-change="$emit('page-size-change', $event)"
        />
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElPagination from 'element-plus/es/components/pagination/index.mjs'
import { ElOption, ElSelect } from 'element-plus/es/components/select/index.mjs'
import { ElTable, ElTableColumn } from 'element-plus/es/components/table/index.mjs'
import {
  createReportActionItems,
  createReportSummaryItems,
  formatReportConfidence,
  getReportLabelText,
  getReportModelDisplayName,
  reportLabelOptions,
  reportsCopy
} from '@/modules/reports/reportsView'

const props = defineProps({
  assetOptions: { type: Array, default: () => [] },
  assetNameMap: { type: Object, required: true },
  loading: { type: Boolean, default: false },
  pagination: {
    type: Object,
    default: () => ({ page: 1, perPage: 20, total: 0, pages: 0 })
  },
  reports: { type: Array, default: () => [] }
})

const emit = defineEmits([
  'refresh',
  'reset',
  'show-detail',
  'open-ai',
  'remove',
  'page-change',
  'page-size-change'
])

const reportFilters = defineModel('filters', {
  required: true
})

const reportSummaryItems = computed(() => createReportSummaryItems(props.reports, props.pagination))

function emitReportAction(actionKey, row) {
  const eventMap = {
    detail: 'show-detail',
    ai: 'open-ai',
    remove: 'remove'
  }
  emit(eventMap[actionKey], row)
}
</script>

<style scoped>
.reports-workspace {
  overflow: hidden;
}

.reports-toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  padding: 10px 12px;
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  border: 1px solid var(--line-soft);
}

.reports-filter-select {
  width: 220px;
}

.reports-summary-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin-bottom: 12px;
}

.reports-summary-strip__item {
  min-width: 0;
  padding: 9px 12px;
  border: 1px solid var(--line-soft);
  border-left: 3px solid var(--brand);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
}

.reports-summary-strip__item span {
  display: block;
  color: var(--text-tertiary);
  font-size: 11px;
  font-weight: 800;
}

.reports-summary-strip__item strong {
  display: block;
  margin-top: 4px;
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 17px;
  line-height: 1;
}

.reports-summary-strip__item--ok {
  border-left-color: var(--status-ok);
}

.reports-summary-strip__item--warning {
  border-left-color: var(--status-warn);
}

.reports-summary-strip__item--ai {
  border-left-color: var(--status-pending);
}

.report-model-name {
  display: inline-block;
  max-width: 140px;
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: middle;
  white-space: nowrap;
  font-weight: 700;
  color: var(--text-primary);
}

.reports-actions {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
}

.reports-pagination {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 14px;
  color: var(--text-secondary);
  font-size: 12px;
}

:deep(.reports-workspace .el-table__row td) {
  padding-top: 10px;
  padding-bottom: 10px;
}

:deep(.reports-workspace .el-table__header th) {
  padding-top: 9px;
  padding-bottom: 9px;
  color: var(--text-secondary);
  font-weight: 800;
}

@media (max-width: 1180px) {
  .reports-summary-strip {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 860px) {
  .page-toolbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .reports-filter-select {
    width: min(100%, 260px);
  }

  .reports-pagination {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
