<template>
  <div class="page-stack">
    <AssetsWorkspacePanel
      v-model:active-kind="activeKind"
      :assets="paginatedAssets"
      :current-page="currentPage"
      :page-size="pageSize"
      :total="totalAssets"
      :loading="loading"
      :uploading="uploading"
      :preparing-map="preparingMap"
      :deleting-map="deletingMap"
      @update:current-page="setCurrentPage"
      @refresh="refreshAssets"
      @upload="handleUpload"
      @download="downloadAsset"
      @open-reports-center="openReportsCenter"
      @open-reports="openReports"
      @prepare="prepareAsset"
      @show-detail="showDetail"
      @remove="removeAsset"
    />

    <AssetsAnalysisPanel
      v-model:analysis-asset-id="analysisAssetId"
      v-model:analysis-fields="analysisFields"
      v-model:analysis-start-t="analysisStartT"
      v-model:analysis-end-t="analysisEndT"
      v-model:analysis-max-points="analysisMaxPoints"
      :assets="assets"
      :annotation-items="analysisAnnotations"
      :field-options="analysisFieldOptions"
      :chart-times="analysisChartTimes"
      :chart-series="analysisChartSeries"
      :chart-fault-prob="analysisChartFaultProb"
      :loading-fields="analysisLoadingFields"
      :loading-series="analysisLoadingSeries"
      :error-message="analysisErrorMessage"
      @reload-fields="loadAnalysisFields"
      @load-series="loadAnalysisSeries"
    />

    <AssetDetailDialog
      v-model:visible="detailVisible"
      :detail-json="detailJson"
    />
  </div>
</template>

<script setup>
import AssetsAnalysisPanel from '@/modules/assets/components/AssetsAnalysisPanel.vue'
import AssetDetailDialog from '@/modules/assets/components/AssetDetailDialog.vue'
import AssetsWorkspacePanel from '@/modules/assets/components/AssetsWorkspacePanel.vue'
import { useAssetsWorkspace } from '@/modules/assets/composables/useAssetsWorkspace'

const {
  activeKind,
  analysisAssetId,
  analysisChartFaultProb,
  analysisChartSeries,
  analysisChartTimes,
  analysisEndT,
  analysisErrorMessage,
  analysisAnnotations,
  analysisFieldOptions,
  analysisFields,
  analysisLoadingFields,
  analysisLoadingSeries,
  analysisMaxPoints,
  analysisStartT,
  assets,
  currentPage,
  deletingMap,
  detailJson,
  detailVisible,
  downloadAsset,
  handleUpload,
  loadAssets,
  loadAnalysisFields,
  loadAnalysisSeries,
  loading,
  openReports,
  openReportsCenter,
  paginatedAssets,
  pageSize,
  prepareAsset,
  preparingMap,
  removeAsset,
  refreshAssets,
  setCurrentPage,
  showDetail,
  totalAssets,
  uploading
} = useAssetsWorkspace()
</script>
