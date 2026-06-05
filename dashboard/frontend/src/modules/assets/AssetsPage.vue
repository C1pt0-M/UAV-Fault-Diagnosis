<template>
  <div class="page-stack">
    <AssetsWorkspacePanel
      v-model:active-kind="activeKind"
      :assets="assets"
      :loading="loading"
      :uploading="uploading"
      :preparing-map="preparingMap"
      :deleting-map="deletingMap"
      @refresh="loadAssets"
      @upload="handleUpload"
      @download="downloadAsset"
      @prepare="prepareAsset"
      @show-detail="showDetail"
      @remove="removeAsset"
    />

    <AssetDetailDialog
      v-model:visible="detailVisible"
      :detail-json="detailJson"
    />
  </div>
</template>

<script setup>
import AssetDetailDialog from '@/modules/assets/components/AssetDetailDialog.vue'
import AssetsWorkspacePanel from '@/modules/assets/components/AssetsWorkspacePanel.vue'
import { useAssetsWorkspace } from '@/modules/assets/composables/useAssetsWorkspace'

const {
  activeKind,
  assets,
  deletingMap,
  detailJson,
  detailVisible,
  downloadAsset,
  handleUpload,
  loadAssets,
  loading,
  prepareAsset,
  preparingMap,
  removeAsset,
  showDetail,
  uploading
} = useAssetsWorkspace()
</script>
